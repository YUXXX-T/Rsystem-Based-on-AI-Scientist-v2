import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "evaluation": {},
    "throughput_improvement_percentage": None,
}


def greedy_assignment(cost_matrix):
    n_agents, n_tasks = cost_matrix.shape
    row_ind, col_ind = [], []
    assigned_tasks, assigned_agents = set(), set()
    pairs = [(cost_matrix[i, j], i, j) for i in range(n_agents) for j in range(n_tasks)]
    pairs.sort()
    for cost, i, j in pairs:
        if i not in assigned_agents and j not in assigned_tasks:
            row_ind.append(i)
            col_ind.append(j)
            assigned_agents.add(i)
            assigned_tasks.add(j)
            if len(assigned_agents) == min(n_agents, n_tasks):
                break
    return np.array(row_ind), np.array(col_ind)


class WarehouseSimulator:
    def __init__(
        self, grid_size=20, n_agents=20, n_tasks=30, obstacles=None, hotspots=None
    ):
        self.grid_size, self.n_agents, self.n_tasks = grid_size, n_agents, n_tasks
        self.obstacles = obstacles if obstacles else set()
        self.hotspots = hotspots if hotspots else []
        self.congestion_events = 0
        self.reset()

    def _valid_position(self, pos):
        return tuple(pos) not in self.obstacles

    def _random_valid_position(self):
        for _ in range(100):
            pos = np.random.randint(0, self.grid_size, 2)
            if self._valid_position(pos):
                return pos
        return np.array([0, 0])

    def _hotspot_biased_position(self):
        if self.hotspots and np.random.random() < 0.6:
            hotspot = self.hotspots[np.random.randint(len(self.hotspots))]
            offset = np.random.randint(-2, 3, 2)
            pos = np.clip(hotspot + offset, 0, self.grid_size - 1)
            if self._valid_position(pos):
                return pos
        return self._random_valid_position()

    def reset(self):
        self.agent_positions = np.array(
            [self._random_valid_position() for _ in range(self.n_agents)]
        )
        self.agent_velocities = np.zeros((self.n_agents, 2))
        self.task_positions = np.array(
            [self._hotspot_biased_position() for _ in range(self.n_tasks)]
        )
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed = 0, 0
        self.congestion_events = 0
        return self.get_state()

    def get_state(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        task_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        for i, pos in enumerate(self.task_positions):
            if self.task_active[i]:
                task_grid[pos[0], pos[1]] += 1
        return np.stack([agent_grid, task_grid], axis=0)

    def count_congestion_events(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        return np.sum(agent_grid > 2)

    def compute_congestion(self, lookahead=5):
        congestion = np.zeros((self.grid_size, self.grid_size))
        for i, agent_pos in enumerate(self.agent_positions):
            target_idx = self.agent_targets[i]
            if target_idx >= 0 and self.task_active[target_idx]:
                path = self._get_path(agent_pos, self.task_positions[target_idx])
                for step, pos in enumerate(path[:lookahead]):
                    congestion[pos[0], pos[1]] += 1.0 / (step + 1)
        return congestion

    def _get_path(self, start, end):
        path, current = [tuple(start)], np.array(start)
        max_steps = self.grid_size * 2
        for _ in range(max_steps):
            if np.array_equal(current, end):
                break
            diff = end - current
            moved = False
            for d in [0, 1]:
                if diff[d] != 0:
                    new_pos = current.copy()
                    new_pos[d] += np.sign(diff[d])
                    if self._valid_position(new_pos):
                        current = new_pos
                        moved = True
                        break
            if not moved:
                break
            path.append(tuple(current))
        return path

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if task_idx >= 0 and self.task_active[task_idx]:
                self.agent_targets[agent_idx] = task_idx

    def step(self):
        self.time_step += 1
        self.congestion_events += self.count_congestion_events()
        for i in range(self.n_agents):
            target_idx = self.agent_targets[i]
            if target_idx >= 0 and self.task_active[target_idx]:
                target_pos = self.task_positions[target_idx]
                diff = target_pos - self.agent_positions[i]
                if np.sum(np.abs(diff)) <= 1:
                    self.task_active[target_idx] = False
                    self.tasks_completed += 1
                    self.agent_targets[i] = -1
                else:
                    move = np.zeros(2, dtype=int)
                    for d in [0, 1]:
                        if diff[d] != 0:
                            new_pos = self.agent_positions[i].copy()
                            new_pos[d] += np.sign(diff[d])
                            if self._valid_position(new_pos):
                                move[d] = np.sign(diff[d])
                                break
                    self.agent_positions[i] = np.clip(
                        self.agent_positions[i] + move, 0, self.grid_size - 1
                    )
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                self.task_positions[idx] = self._hotspot_biased_position()
                self.task_active[idx] = True
        return self.get_state()


def create_warehouse_configs():
    configs = []
    # Sparse warehouse
    obstacles = set()
    for i in range(5, 15, 5):
        for j in range(3, 17, 4):
            obstacles.add((i, j))
    configs.append((obstacles, [np.array([5, 5]), np.array([15, 15])], "sparse"))

    # Bottleneck warehouse
    obstacles = set()
    for i in range(20):
        if i not in [9, 10]:
            obstacles.add((i, 10))
    configs.append(
        (
            obstacles,
            [
                np.array([5, 5]),
                np.array([5, 15]),
                np.array([15, 5]),
                np.array([15, 15]),
            ],
            "bottleneck",
        )
    )

    return configs


class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))


def generate_training_data(
    n_samples=300, grid_size=20, n_agents=20, n_tasks=30, obstacles=None, hotspots=None
):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks, obstacles, hotspots)
        state = sim.get_state()
        active_tasks = np.where(sim.task_active)[0]
        sim.assign_tasks(
            [(i, active_tasks[i]) for i in range(min(n_agents, len(active_tasks)))]
        )
        congestion = sim.compute_congestion(lookahead=5)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(state)
        congestions.append(congestion)
    return np.array(states), np.array(congestions)


def distance_based_assignment(agent_positions, task_positions, task_active):
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    cost_matrix = np.array(
        [
            [np.sum(np.abs(ap - task_positions[ti])) for ti in active_indices]
            for ap in agent_positions
        ]
    )
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def congestion_aware_assignment(
    agent_positions, task_positions, task_active, congestion_map, congestion_weight=0.5
):
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    grid_size = congestion_map.shape[0]
    cost_matrix = np.zeros((len(agent_positions), len(active_indices)))
    for i, agent_pos in enumerate(agent_positions):
        for j, task_idx in enumerate(active_indices):
            task_pos = task_positions[task_idx]
            distance = np.sum(np.abs(agent_pos - task_pos))
            path_congestion, current = 0, agent_pos.copy()
            for _ in range(grid_size * 2):
                if np.array_equal(current, task_pos):
                    break
                diff = task_pos - current
                if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    current[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    current[1] += np.sign(diff[1])
                current = np.clip(current, 0, grid_size - 1)
                path_congestion += congestion_map[current[0], current[1]]
            cost_matrix[i, j] = distance + congestion_weight * path_congestion
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def train_predictor(
    model,
    train_states,
    train_congestions,
    val_states,
    val_congestions,
    epochs=50,
    batch_size=32,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_states_t = torch.FloatTensor(train_states).to(device)
    train_congestions_t = torch.FloatTensor(train_congestions).unsqueeze(1).to(device)
    val_states_t = torch.FloatTensor(val_states).to(device)
    val_congestions_t = torch.FloatTensor(val_congestions).unsqueeze(1).to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_states))
        train_loss, n_batches = 0, 0
        for i in range(0, len(train_states), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_states_t[idx]), train_congestions_t[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_states_t), val_congestions_t).item()
        experiment_data["training"]["losses"]["train"].append(train_loss)
        experiment_data["training"]["losses"]["val"].append(val_loss)
        experiment_data["training"]["epochs"].append(epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    return model


def evaluate_method(method, model, sim_params, n_episodes=5, max_steps=100):
    throughputs, congestion_list = [], []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(**sim_params)
        for step in range(max_steps):
            state = sim.get_state()
            unassigned = [
                i
                for i in range(sim.n_agents)
                if sim.agent_targets[i] < 0 or not sim.task_active[sim.agent_targets[i]]
            ]
            if len(unassigned) > 0 and np.any(sim.task_active):
                if method == "distance":
                    assignment = distance_based_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                    )
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        congestion_pred = model(state_tensor).squeeze().cpu().numpy()
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        congestion_pred,
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assignment])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        congestion_list.append(sim.congestion_events)
    return np.mean(throughputs), np.std(throughputs), np.mean(congestion_list)


# Generate training data
print("Generating training data...")
configs = create_warehouse_configs()
all_states, all_congestions = [], []
for obstacles, hotspots, name in configs:
    states, congestions = generate_training_data(
        n_samples=200, obstacles=obstacles, hotspots=hotspots
    )
    all_states.append(states)
    all_congestions.append(congestions)

train_states = np.concatenate([s[:160] for s in all_states])
train_congestions = np.concatenate([c[:160] for c in all_states])
train_congestions = np.concatenate([c[:160] for c in all_congestions])
val_states = np.concatenate([s[160:] for s in all_states])
val_congestions = np.concatenate([c[160:] for c in all_congestions])

print(f"Training samples: {len(train_states)}, Validation samples: {len(val_states)}")

# Train model
model = CongestionPredictor(grid_size=20, hidden_dim=32)
model = train_predictor(
    model, train_states, train_congestions, val_states, val_congestions, epochs=50
)

# Evaluate
print("\nEvaluating methods...")
agent_counts = [15, 20, 25, 30]
all_improvements = []

for obstacles, hotspots, name in configs:
    experiment_data["evaluation"][name] = {
        "distance": [],
        "cma": [],
        "improvement": [],
        "n_agents": agent_counts,
    }
    print(f"\n--- {name} warehouse ---")
    for n_agents in agent_counts:
        sim_params = {
            "grid_size": 20,
            "n_agents": n_agents,
            "n_tasks": 30,
            "obstacles": obstacles,
            "hotspots": hotspots,
        }
        dist_tp, dist_std, _ = evaluate_method("distance", None, sim_params)
        cma_tp, cma_std, _ = evaluate_method("cma", model, sim_params)
        improvement = (cma_tp - dist_tp) / dist_tp * 100 if dist_tp > 0 else 0
        experiment_data["evaluation"][name]["distance"].append(dist_tp)
        experiment_data["evaluation"][name]["cma"].append(cma_tp)
        experiment_data["evaluation"][name]["improvement"].append(improvement)
        all_improvements.append(improvement)
        print(
            f"  Agents={n_agents}: Distance={dist_tp:.1f}, CMA={cma_tp:.1f}, Improvement={improvement:.1f}%"
        )

throughput_improvement_percentage = np.mean(all_improvements)
experiment_data["throughput_improvement_percentage"] = throughput_improvement_percentage
print(f"\nthroughput_improvement_percentage = {throughput_improvement_percentage:.2f}%")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0].plot(experiment_data["training"]["losses"]["val"], label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].legend()

for name in experiment_data["evaluation"]:
    data = experiment_data["evaluation"][name]
    axes[1].plot(data["n_agents"], data["cma"], "o-", label=f"CMA-{name}")
    axes[1].plot(
        data["n_agents"], data["distance"], "x--", alpha=0.5, label=f"Dist-{name}"
    )
axes[1].set_xlabel("Agents")
axes[1].set_ylabel("Throughput")
axes[1].set_title("Throughput Comparison")
axes[1].legend()

for name in experiment_data["evaluation"]:
    axes[2].plot(
        experiment_data["evaluation"][name]["n_agents"],
        experiment_data["evaluation"][name]["improvement"],
        "o-",
        label=name,
    )
axes[2].axhline(y=0, color="r", linestyle="--")
axes[2].set_xlabel("Agents")
axes[2].set_ylabel("Improvement (%)")
axes[2].set_title("Throughput Improvement")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Results saved to {working_dir}")
