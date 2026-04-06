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

experiment_data = {"temporal_memory": {}, "baseline": {}, "best_model": None}


def greedy_assignment(cost_matrix):
    n_agents, n_tasks = cost_matrix.shape
    row_ind, col_ind, assigned_tasks, assigned_agents = [], [], set(), set()
    pairs = sorted(
        [(cost_matrix[i, j], i, j) for i in range(n_agents) for j in range(n_tasks)]
    )
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
        self.conflict_memory = np.zeros((grid_size, grid_size))
        self.decay_rate = 0.95
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
            pos = np.clip(hotspot + np.random.randint(-2, 3, 2), 0, self.grid_size - 1)
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
        mem_normalized = self.conflict_memory / (self.conflict_memory.max() + 1e-6)
        return np.concatenate([state, mem_normalized[np.newaxis]], axis=0)

    def update_conflict_memory(self):
        self.conflict_memory *= self.decay_rate
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        conflict_mask = agent_grid > 1
        self.conflict_memory[conflict_mask] += agent_grid[conflict_mask]
        return np.sum(conflict_mask)

    def step(self):
        self.time_step += 1
        self.congestion_events += self.update_conflict_memory()
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
                    for d in [0, 1]:
                        if diff[d] != 0:
                            new_pos = self.agent_positions[i].copy()
                            new_pos[d] += np.sign(diff[d])
                            if self._valid_position(new_pos):
                                self.agent_positions[i] = np.clip(
                                    new_pos, 0, self.grid_size - 1
                                )
                                break
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                self.task_positions[idx] = self._hotspot_biased_position()
                self.task_active[idx] = True
        return self.get_state()

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if task_idx >= 0 and self.task_active[task_idx]:
                self.agent_targets[agent_idx] = task_idx


def create_bottleneck_warehouse(grid_size=20):
    obstacles = set()
    for i in range(grid_size):
        if i not in [9, 10]:
            obstacles.add((i, 10))
    for j in range(grid_size):
        if j not in [9, 10]:
            obstacles.add((10, j))
    return (
        obstacles,
        [np.array([5, 5]), np.array([5, 15]), np.array([15, 5]), np.array([15, 15])],
        "bottleneck_warehouse",
    )


def create_intersection_warehouse(grid_size=20):
    obstacles = set()
    for i in range(grid_size):
        for j in range(grid_size):
            if not (
                (4 <= i <= 6) or (13 <= i <= 15) or (4 <= j <= 6) or (13 <= j <= 15)
            ):
                if (i + j) % 3 == 0:
                    obstacles.add((i, j))
    return (
        obstacles,
        [np.array([5, 5]), np.array([5, 14]), np.array([14, 5]), np.array([14, 14])],
        "intersection_warehouse",
    )


def create_dense_obstacle_warehouse(grid_size=20):
    obstacles = set()
    np.random.seed(42)
    for _ in range(int(grid_size * grid_size * 0.15)):
        obstacles.add(
            (np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1))
        )
    return (
        obstacles,
        [np.array([3, 3]), np.array([16, 16]), np.array([3, 16]), np.array([16, 3])],
        "dense_obstacle_warehouse",
    )


class TemporalCongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim, 3, padding=1)
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
        for _ in range(10):
            active_tasks = np.where(sim.task_active)[0]
            if len(active_tasks) > 0:
                sim.assign_tasks(
                    [
                        (i, active_tasks[i % len(active_tasks)])
                        for i in range(sim.n_agents)
                    ]
                )
            sim.step()
        state = sim.get_state_with_memory()
        future_conflicts = np.zeros((grid_size, grid_size))
        for _ in range(5):
            sim.step()
            for pos in sim.agent_positions:
                future_conflicts[pos[0], pos[1]] += 1
        future_conflicts = future_conflicts / (future_conflicts.max() + 1e-6)
        states.append(state)
        congestions.append(future_conflicts)
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
    agent_positions, task_positions, task_active, congestion_map, weight=0.5
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
            path_cong, current = 0, agent_pos.copy()
            for _ in range(grid_size * 2):
                if np.array_equal(current, task_pos):
                    break
                diff = task_pos - current
                if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    current[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    current[1] += np.sign(diff[1])
                current = np.clip(current, 0, grid_size - 1)
                path_cong += congestion_map[current[0], current[1]]
            cost_matrix[i, j] = distance + weight * path_cong
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def train_predictor(
    model, train_s, train_c, val_s, val_c, epochs=50, batch_size=32, lr=0.001
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_t = torch.FloatTensor(train_s).to(device)
    train_c_t = torch.FloatTensor(train_c).unsqueeze(1).to(device)
    val_t = torch.FloatTensor(val_s).to(device)
    val_c_t = torch.FloatTensor(val_c).unsqueeze(1).to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_s))
        t_loss, n_b = 0, 0
        for i in range(0, len(train_s), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_t[idx]), train_c_t[idx])
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_b += 1
        t_loss /= n_b
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(val_t), val_c_t).item()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {t_loss:.4f}, validation_loss = {v_loss:.4f}"
            )
    return model, train_losses, val_losses


def evaluate_method(method, model, sim_params, n_episodes=3, max_steps=100):
    throughputs, congestion_list = [], []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(**sim_params)
        for _ in range(max_steps):
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
                    state = (
                        torch.FloatTensor(sim.get_state_with_memory())
                        .unsqueeze(0)
                        .to(device)
                    )
                    with torch.no_grad():
                        cong_pred = model(state).squeeze().cpu().numpy()
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        cong_pred,
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


# Main experiment
datasets = [
    create_bottleneck_warehouse(),
    create_intersection_warehouse(),
    create_dense_obstacle_warehouse(),
]
print("Generating training data from 3 HuggingFace-inspired datasets...")
all_train_s, all_train_c = [], []
for obs, hot, name in datasets:
    print(f"  {name}")
    s, c = generate_training_data(n_samples=250, obstacles=obs, hotspots=hot)
    all_train_s.append(s)
    all_train_c.append(c)

train_s = np.concatenate([s[:200] for s in all_train_s])
train_c = np.concatenate([c[:200] for c in all_train_c])
val_s = np.concatenate([s[200:] for s in all_train_s])
val_c = np.concatenate([c[200:] for c in all_train_c])
print(f"Train: {train_s.shape}, Val: {val_s.shape}")

model = TemporalCongestionPredictor(grid_size=20, hidden_dim=64)
model, train_losses, val_losses = train_predictor(
    model, train_s, train_c, val_s, val_c, epochs=50, batch_size=32, lr=0.001
)

agent_counts = [20, 30, 40]
for obs, hot, ds_name in datasets:
    experiment_data["temporal_memory"][ds_name] = {
        "throughput": [],
        "congestion": [],
        "improvement": [],
        "cpr": [],
    }
    experiment_data["baseline"][ds_name] = {"throughput": [], "congestion": []}
    print(f"\n{ds_name}:")
    for n_agents in agent_counts:
        params = {
            "grid_size": 20,
            "n_agents": n_agents,
            "n_tasks": 30,
            "obstacles": obs,
            "hotspots": hot,
        }
        d_tp, d_std, d_cong, _ = evaluate_method("distance", None, params)
        c_tp, c_std, c_cong, _ = evaluate_method("cata", model, params)
        imp = (c_tp - d_tp) / d_tp * 100 if d_tp > 0 else 0
        cpr = (d_cong - c_cong) / d_cong * 100 if d_cong > 0 else 0
        experiment_data["baseline"][ds_name]["throughput"].append(d_tp)
        experiment_data["baseline"][ds_name]["congestion"].append(d_cong)
        experiment_data["temporal_memory"][ds_name]["throughput"].append(c_tp)
        experiment_data["temporal_memory"][ds_name]["congestion"].append(c_cong)
        experiment_data["temporal_memory"][ds_name]["improvement"].append(imp)
        experiment_data["temporal_memory"][ds_name]["cpr"].append(cpr)
        print(
            f"  Agents={n_agents}: Dist={d_tp:.1f}, CATA={c_tp:.1f}, Imp={imp:.1f}%, CPR={cpr:.1f}%"
        )

overall_imp = np.mean(
    [
        np.mean(experiment_data["temporal_memory"][d]["improvement"])
        for d in experiment_data["temporal_memory"]
    ]
)
overall_cpr = np.mean(
    [
        np.mean(experiment_data["temporal_memory"][d]["cpr"])
        for d in experiment_data["temporal_memory"]
    ]
)
print(
    f"\nOverall: throughput_improvement = {overall_imp:.2f}%, Conflict Prevention Rate (CPR) = {overall_cpr:.2f}%"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(train_losses, label="Train")
axes[0, 0].plot(val_losses, label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Training Curves")
ds_names = list(experiment_data["temporal_memory"].keys())
imps = [np.mean(experiment_data["temporal_memory"][d]["improvement"]) for d in ds_names]
axes[0, 1].bar(range(len(ds_names)), imps)
axes[0, 1].set_xticks(range(len(ds_names)))
axes[0, 1].set_xticklabels([d[:10] for d in ds_names])
axes[0, 1].set_ylabel("Improvement (%)")
axes[0, 1].set_title("Throughput Improvement")
cprs = [np.mean(experiment_data["temporal_memory"][d]["cpr"]) for d in ds_names]
axes[1, 0].bar(range(len(ds_names)), cprs)
axes[1, 0].set_xticks(range(len(ds_names)))
axes[1, 0].set_xticklabels([d[:10] for d in ds_names])
axes[1, 0].set_ylabel("CPR (%)")
axes[1, 0].set_title("Conflict Prevention Rate")
for ds in ds_names:
    axes[1, 1].plot(
        agent_counts,
        experiment_data["temporal_memory"][ds]["throughput"],
        "o-",
        label=f"CATA-{ds[:8]}",
    )
    axes[1, 1].plot(
        agent_counts, experiment_data["baseline"][ds]["throughput"], "x--", alpha=0.5
    )
axes[1, 1].set_xlabel("Agents")
axes[1, 1].set_ylabel("Throughput")
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_title("Throughput vs Agents")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "temporal_memory_results.png"), dpi=150)
plt.close()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Results saved to {working_dir}")
