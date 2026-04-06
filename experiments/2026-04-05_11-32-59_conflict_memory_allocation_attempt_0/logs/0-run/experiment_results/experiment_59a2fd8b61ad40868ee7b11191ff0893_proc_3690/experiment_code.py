import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)

experiment_data = {
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "datasets": {},
    "cpr_tracking": {},
    "best_model_metrics": {},
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
        self,
        grid_size=20,
        n_agents=20,
        n_tasks=30,
        obstacles=None,
        hotspots=None,
        memory_decay=0.95,
    ):
        self.grid_size, self.n_agents, self.n_tasks = grid_size, n_agents, n_tasks
        self.obstacles = obstacles if obstacles else set()
        self.hotspots = hotspots if hotspots else []
        self.memory_decay = memory_decay
        self.conflict_memory = np.zeros((grid_size, grid_size))
        self.congestion_events = 0
        self.reset()

    def _valid_position(self, pos):
        return (
            tuple(pos) not in self.obstacles
            and 0 <= pos[0] < self.grid_size
            and 0 <= pos[1] < self.grid_size
        )

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
        self.task_positions = np.array(
            [self._hotspot_biased_position() for _ in range(self.n_tasks)]
        )
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed, self.congestion_events = 0, 0, 0
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

    def get_state_with_memory(self):
        state = self.get_state()
        memory_norm = self.conflict_memory / (self.conflict_memory.max() + 1e-6)
        return np.concatenate([state, memory_norm[np.newaxis, :, :]], axis=0)

    def count_congestion_events(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        congested_cells = np.where(agent_grid > 1)
        for x, y in zip(congested_cells[0], congested_cells[1]):
            self.conflict_memory[x, y] += agent_grid[x, y] - 1
        self.conflict_memory *= self.memory_decay
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
        for _ in range(self.grid_size * 2):
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


# HuggingFace-inspired datasets
def create_warehouse_hf1(grid_size=20):
    """Inspired by warehouse-10-20-10 benchmark"""
    obstacles = set()
    for i in range(4, 16, 4):
        for j in range(2, grid_size - 2):
            if j % 4 != 0:
                obstacles.add((i, j))
    hotspots = [
        np.array([2, 5]),
        np.array([2, 15]),
        np.array([18, 5]),
        np.array([18, 15]),
    ]
    return obstacles, hotspots, "warehouse_hf1"


def create_warehouse_hf2(grid_size=20):
    """Inspired by room-64-64-8 benchmark - room-like structure"""
    obstacles = set()
    for i in [6, 13]:
        for j in range(grid_size):
            if j not in [3, 4, 15, 16]:
                obstacles.add((i, j))
    for j in [6, 13]:
        for i in range(grid_size):
            if i not in [3, 4, 15, 16]:
                obstacles.add((i, j))
    hotspots = [
        np.array([3, 3]),
        np.array([3, 16]),
        np.array([16, 3]),
        np.array([16, 16]),
        np.array([10, 10]),
    ]
    return obstacles, hotspots, "warehouse_hf2"


def create_warehouse_hf3(grid_size=20):
    """Inspired by random-32-32-20 benchmark - scattered obstacles"""
    np.random.seed(123)
    obstacles = set()
    for _ in range(60):
        pos = (np.random.randint(2, grid_size - 2), np.random.randint(2, grid_size - 2))
        obstacles.add(pos)
    np.random.seed(42)
    hotspots = [
        np.array([5, 5]),
        np.array([5, 14]),
        np.array([14, 5]),
        np.array([14, 14]),
    ]
    return obstacles, hotspots, "warehouse_hf3"


class CongestionPredictorWithMemory(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))


def generate_training_data(
    n_samples=300, grid_size=20, n_agents=20, n_tasks=30, obstacles=None, hotspots=None
):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks, obstacles, hotspots)
        for _ in range(10):
            sim.step()
        state = sim.get_state_with_memory()
        active_tasks = np.where(sim.task_active)[0]
        sim.assign_tasks(
            [
                (i, active_tasks[i % len(active_tasks)])
                for i in range(min(n_agents, len(active_tasks)))
            ]
        )
        congestion = sim.compute_congestion(lookahead=8)
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
    agent_positions,
    task_positions,
    task_active,
    congestion_map,
    obstacles,
    congestion_weight=0.5,
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
                moved = False
                for d in [0, 1]:
                    if diff[d] != 0:
                        new_pos = current.copy()
                        new_pos[d] += np.sign(diff[d])
                        if (
                            tuple(new_pos) not in obstacles
                            and 0 <= new_pos[0] < grid_size
                            and 0 <= new_pos[1] < grid_size
                        ):
                            current = new_pos
                            moved = True
                            break
                if not moved:
                    if diff[0] != 0:
                        current[0] += np.sign(diff[0])
                    elif diff[1] != 0:
                        current[1] += np.sign(diff[1])
                current = np.clip(current, 0, grid_size - 1)
                path_congestion += congestion_map[int(current[0]), int(current[1])]
            cost_matrix[i, j] = distance + congestion_weight * path_congestion
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def evaluate_method(method, model, sim_params, n_episodes=5, max_steps=100):
    throughputs, congestion_list = [], []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(**sim_params)
        for step in range(max_steps):
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
                    state = sim.get_state_with_memory()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        congestion_pred = model(state_tensor).squeeze().cpu().numpy()
                    combined = congestion_pred + 0.3 * sim.conflict_memory / (
                        sim.conflict_memory.max() + 1e-6
                    )
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        combined,
                        sim.obstacles,
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assignment])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        congestion_list.append(sim.congestion_events)
    return (
        np.mean(throughputs),
        np.std(throughputs),
        np.mean(congestion_list),
        np.std(congestion_list),
    )


# Generate data
print("Creating three HuggingFace-inspired warehouse datasets...")
datasets = [create_warehouse_hf1(), create_warehouse_hf2(), create_warehouse_hf3()]

all_states, all_congestions = [], []
for obstacles, hotspots, name in datasets:
    print(f"  Generating data for: {name}")
    states, congestions = generate_training_data(
        n_samples=250, obstacles=obstacles, hotspots=hotspots
    )
    all_states.append(states)
    all_congestions.append(congestions)

train_states = np.concatenate([s[:200] for s in all_states])
train_congestions = np.concatenate([c[:200] for c in all_congestions])
val_states = np.concatenate([s[200:] for s in all_states])
val_congestions = np.concatenate([c[200:] for c in all_congestions])
print(f"Training: {train_states.shape}, Validation: {val_states.shape}")

# Training
model = CongestionPredictorWithMemory(grid_size=20, hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs, batch_size = 50, 32

train_states_t = torch.FloatTensor(train_states).to(device)
train_congestions_t = torch.FloatTensor(train_congestions).unsqueeze(1).to(device)
val_states_t = torch.FloatTensor(val_states).to(device)
val_congestions_t = torch.FloatTensor(val_congestions).unsqueeze(1).to(device)

best_val_loss, best_model_state = float("inf"), None
print("\nTraining with temporal conflict memory...")
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

model.load_state_dict(best_model_state)
print(f"\nLoaded best model with val_loss = {best_val_loss:.4f}")

# Evaluation with CPR tracking
print("\n" + "=" * 60)
print("EVALUATION WITH CONFLICT PREVENTION RATE (CPR)")
print("=" * 60)

agent_counts = [15, 25, 35]
all_improvements, all_cprs = [], []

for obstacles, hotspots, name in datasets:
    experiment_data["datasets"][name] = {
        "distance": {"throughput": [], "congestion": []},
        "cma": {"throughput": [], "congestion": []},
        "improvement": [],
        "cpr": [],
        "n_agents": agent_counts,
    }
    print(f"\n--- {name} ---")
    for n_agents in agent_counts:
        sim_params = {
            "grid_size": 20,
            "n_agents": n_agents,
            "n_tasks": 30,
            "obstacles": obstacles,
            "hotspots": hotspots,
        }
        dist_tp, dist_std, dist_cong, _ = evaluate_method("distance", None, sim_params)
        cma_tp, cma_std, cma_cong, _ = evaluate_method("cma", model, sim_params)
        improvement = (cma_tp - dist_tp) / dist_tp * 100 if dist_tp > 0 else 0
        cpr = (dist_cong - cma_cong) / dist_cong * 100 if dist_cong > 0 else 0
        experiment_data["datasets"][name]["distance"]["throughput"].append(dist_tp)
        experiment_data["datasets"][name]["distance"]["congestion"].append(dist_cong)
        experiment_data["datasets"][name]["cma"]["throughput"].append(cma_tp)
        experiment_data["datasets"][name]["cma"]["congestion"].append(cma_cong)
        experiment_data["datasets"][name]["improvement"].append(improvement)
        experiment_data["datasets"][name]["cpr"].append(cpr)
        all_improvements.append(improvement)
        all_cprs.append(cpr)
        print(f"  Agents={n_agents}: TP Imp={improvement:.1f}%, CPR={cpr:.1f}%")

overall_improvement = np.mean(all_improvements)
overall_cpr = np.mean(all_cprs)
experiment_data["overall_improvement"] = overall_improvement
experiment_data["overall_cpr"] = overall_cpr

print("\n" + "=" * 60)
print(f"OVERALL THROUGHPUT IMPROVEMENT: {overall_improvement:.2f}%")
print(f"OVERALL CONFLICT PREVENTION RATE (CPR): {overall_cpr:.2f}%")
print("=" * 60)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0, 0].plot(experiment_data["training"]["losses"]["val"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training Curves")
axes[0, 0].legend()

dataset_names = [d[2] for d in datasets]
x = np.arange(len(dataset_names))
axes[0, 1].bar(
    x - 0.2,
    [np.mean(experiment_data["datasets"][d]["improvement"]) for d in dataset_names],
    0.4,
    label="Throughput Imp %",
)
axes[0, 1].bar(
    x + 0.2,
    [np.mean(experiment_data["datasets"][d]["cpr"]) for d in dataset_names],
    0.4,
    label="CPR %",
)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(dataset_names, fontsize=9)
axes[0, 1].set_ylabel("Percentage")
axes[0, 1].set_title("Improvement & CPR by Dataset")
axes[0, 1].legend()
axes[0, 1].axhline(0, color="r", linestyle="--")

for i, name in enumerate(dataset_names):
    data = experiment_data["datasets"][name]
    axes[1, 0].plot(
        data["n_agents"], data["cma"]["throughput"], "o-", label=f"CMA-{name[:10]}"
    )
    axes[1, 0].plot(data["n_agents"], data["distance"]["throughput"], "x--", alpha=0.5)
axes[1, 0].set_xlabel("Agents")
axes[1, 0].set_ylabel("Throughput")
axes[1, 0].set_title("Throughput Comparison")
axes[1, 0].legend(fontsize=8)

for i, name in enumerate(dataset_names):
    data = experiment_data["datasets"][name]
    axes[1, 1].plot(data["n_agents"], data["cpr"], "o-", label=name[:10])
axes[1, 1].set_xlabel("Agents")
axes[1, 1].set_ylabel("CPR (%)")
axes[1, 1].set_title("Conflict Prevention Rate")
axes[1, 1].legend(fontsize=8)
axes[1, 1].axhline(0, color="r", linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cma_temporal_memory_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
print(f"throughput_improvement_percentage = {overall_improvement:.2f}%")
print(f"conflict_prevention_rate = {overall_cpr:.2f}%")
