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

experiment_data = {
    "methods": {},
    "best_method": None,
    "best_improvement": None,
    "best_cpr": None,
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
        self.conflict_history = np.zeros((grid_size, grid_size))
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
        self.task_positions = np.array(
            [self._hotspot_biased_position() for _ in range(self.n_tasks)]
        )
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed, self.congestion_events = 0, 0, 0
        self.conflict_history = np.zeros((self.grid_size, self.grid_size))
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

    def count_and_record_conflicts(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        conflicts = agent_grid > 1
        self.conflict_history = 0.95 * self.conflict_history + conflicts.astype(float)
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
            for d in [0, 1]:
                if diff[d] != 0:
                    new_pos = current.copy()
                    new_pos[d] += np.sign(diff[d])
                    if self._valid_position(new_pos):
                        current = new_pos
                        break
            path.append(tuple(current))
        return path

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if task_idx >= 0 and self.task_active[task_idx]:
                self.agent_targets[agent_idx] = task_idx

    def step(self):
        self.time_step += 1
        self.congestion_events += self.count_and_record_conflicts()
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


def create_sparse_warehouse(grid_size=20):
    obstacles = set()
    for i in range(5, 15, 5):
        for j in range(3, grid_size - 3, 4):
            obstacles.add((i, j))
    return obstacles, [np.array([5, 5]), np.array([15, 15])], "sparse_warehouse"


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


def create_aisle_warehouse(grid_size=20):
    obstacles = set()
    for i in range(2, grid_size - 2, 3):
        for j in range(grid_size):
            if j not in [0, 1, grid_size - 2, grid_size - 1, 9, 10]:
                obstacles.add((i, j))
    return (
        obstacles,
        [np.array([1, 10]), np.array([grid_size - 2, 10])],
        "aisle_warehouse",
    )


class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64, use_memory=False):
        super().__init__()
        in_channels = 3 if use_memory else 2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))


class GNNCongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=32):
        super().__init__()
        self.grid_size = grid_size
        self.node_embed = nn.Linear(3, hidden_dim)
        self.conv1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        node_feats = self.relu(self.node_embed(x_flat))
        for conv in [self.conv1, self.conv2]:
            agg = torch.zeros_like(node_feats)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                shifted = torch.roll(
                    node_feats.reshape(B, H, W, -1), shifts=(di, dj), dims=(1, 2)
                ).reshape(B, H * W, -1)
                agg = agg + shifted
            agg = agg / 4.0
            node_feats = self.relu(conv(torch.cat([node_feats, agg], dim=-1)))
        out = torch.sigmoid(self.output(node_feats)).reshape(B, 1, H, W)
        return out


def generate_training_data(
    n_samples=200, grid_size=20, n_agents=20, n_tasks=30, obstacles=None, hotspots=None
):
    states, congestions, memories = [], [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks, obstacles, hotspots)
        for _ in range(10):
            sim.step()
        state = sim.get_state()
        active_tasks = np.where(sim.task_active)[0]
        sim.assign_tasks(
            [
                (i, active_tasks[i % len(active_tasks)])
                for i in range(min(n_agents, len(active_tasks)))
            ]
        )
        congestion = sim.compute_congestion(lookahead=5)
        congestion = congestion / (congestion.max() + 1e-6)
        memory = sim.conflict_history / (sim.conflict_history.max() + 1e-6)
        states.append(state)
        congestions.append(congestion)
        memories.append(memory)
    return np.array(states), np.array(congestions), np.array(memories)


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
    memory_map=None,
    congestion_weight=0.5,
    memory_weight=0.3,
):
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    grid_size = congestion_map.shape[0]
    cost_matrix = np.zeros((len(agent_positions), len(active_indices)))
    combined_map = congestion_map + (
        memory_weight * memory_map if memory_map is not None else 0
    )
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
                path_congestion += combined_map[current[0], current[1]]
            cost_matrix[i, j] = distance + congestion_weight * path_congestion
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def train_predictor(
    model, train_data, val_data, epochs=40, batch_size=32, lr=0.001, use_memory=False
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_states, train_cong, train_mem = train_data
    val_states, val_cong, val_mem = val_data
    if use_memory:
        train_x = np.concatenate([train_states, train_mem[:, np.newaxis]], axis=1)
        val_x = np.concatenate([val_states, val_mem[:, np.newaxis]], axis=1)
    else:
        train_x, val_x = train_states, val_states
    train_x_t = torch.FloatTensor(train_x).to(device)
    train_y_t = torch.FloatTensor(train_cong).unsqueeze(1).to(device)
    val_x_t = torch.FloatTensor(val_x).to(device)
    val_y_t = torch.FloatTensor(val_cong).unsqueeze(1).to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        train_loss, n_batches = 0, 0
        for i in range(0, len(train_x), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_x_t[idx]), train_y_t[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x_t), val_y_t).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )
    return model, train_losses, val_losses


def evaluate_method(
    method_name, model, sim_params, n_episodes=3, max_steps=100, use_memory=False
):
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
            if unassigned and np.any(sim.task_active):
                if method_name == "distance":
                    assignment = distance_based_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                    )
                else:
                    if use_memory:
                        memory = sim.conflict_history / (
                            sim.conflict_history.max() + 1e-6
                        )
                        state_with_mem = np.concatenate(
                            [state, memory[np.newaxis]], axis=0
                        )
                        state_tensor = (
                            torch.FloatTensor(state_with_mem).unsqueeze(0).to(device)
                        )
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        congestion_pred = model(state_tensor).squeeze().cpu().numpy()
                    mem_map = (
                        sim.conflict_history / (sim.conflict_history.max() + 1e-6)
                        if use_memory
                        else None
                    )
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        congestion_pred,
                        mem_map,
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
print("Generating training data...")
datasets = [
    create_sparse_warehouse(),
    create_bottleneck_warehouse(),
    create_dense_obstacle_warehouse(),
    create_intersection_warehouse(),
    create_aisle_warehouse(),
]
all_states, all_cong, all_mem = [], [], []
for obstacles, hotspots, name in datasets:
    print(f"  {name}")
    s, c, m = generate_training_data(
        n_samples=150, obstacles=obstacles, hotspots=hotspots
    )
    all_states.append(s)
    all_cong.append(c)
    all_mem.append(m)

train_states = np.concatenate([s[:120] for s in all_states])
train_cong = np.concatenate([c[:120] for c in all_cong])
train_mem = np.concatenate([m[:120] for m in all_mem])
val_states = np.concatenate([s[120:] for s in all_states])
val_cong = np.concatenate([c[120:] for c in all_cong])
val_mem = np.concatenate([m[120:] for m in all_mem])
print(f"Train: {train_states.shape}, Val: {val_states.shape}")

# Train and evaluate methods
methods = [
    ("CNN_baseline", False, CongestionPredictor),
    ("CNN_memory", True, CongestionPredictor),
    ("GNN_memory", True, GNNCongestionPredictor),
]
agent_counts = [20, 30, 40]

for method_name, use_mem, ModelClass in methods:
    print(f"\n{'='*50}\nTraining {method_name}\n{'='*50}")
    model = (
        ModelClass(
            grid_size=20,
            hidden_dim=64 if "CNN" in method_name else 32,
            use_memory=use_mem,
        )
        if "CNN" in method_name
        else ModelClass(grid_size=20, hidden_dim=32)
    )
    model, train_losses, val_losses = train_predictor(
        model,
        (train_states, train_cong, train_mem),
        (val_states, val_cong, val_mem),
        epochs=40,
        use_memory=use_mem,
    )
    experiment_data["methods"][method_name] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "datasets": {},
    }

    all_improvements, all_cprs = [], []
    for obstacles, hotspots, ds_name in datasets:
        experiment_data["methods"][method_name]["datasets"][ds_name] = {
            "throughput": [],
            "congestion": [],
            "improvement": [],
            "cpr": [],
        }
        for n_agents in agent_counts:
            sim_params = {
                "grid_size": 20,
                "n_agents": n_agents,
                "n_tasks": 30,
                "obstacles": obstacles,
                "hotspots": hotspots,
            }
            dist_tp, _, dist_cong, _ = evaluate_method("distance", None, sim_params)
            cma_tp, _, cma_cong, _ = evaluate_method(
                method_name, model, sim_params, use_memory=use_mem
            )
            imp = (cma_tp - dist_tp) / dist_tp * 100 if dist_tp > 0 else 0
            cpr = (dist_cong - cma_cong) / dist_cong * 100 if dist_cong > 0 else 0
            experiment_data["methods"][method_name]["datasets"][ds_name][
                "throughput"
            ].append(cma_tp)
            experiment_data["methods"][method_name]["datasets"][ds_name][
                "congestion"
            ].append(cma_cong)
            experiment_data["methods"][method_name]["datasets"][ds_name][
                "improvement"
            ].append(imp)
            experiment_data["methods"][method_name]["datasets"][ds_name]["cpr"].append(
                cpr
            )
            all_improvements.append(imp)
            all_cprs.append(cpr)
            print(f"  {ds_name}, agents={n_agents}: Imp={imp:.1f}%, CPR={cpr:.1f}%")
    experiment_data["methods"][method_name]["overall_improvement"] = np.mean(
        all_improvements
    )
    experiment_data["methods"][method_name]["overall_cpr"] = np.mean(all_cprs)
    print(
        f"{method_name}: Overall Improvement={np.mean(all_improvements):.2f}%, CPR={np.mean(all_cprs):.2f}%"
    )

best_method = max(
    experiment_data["methods"],
    key=lambda m: experiment_data["methods"][m]["overall_improvement"],
)
experiment_data["best_method"] = best_method
experiment_data["best_improvement"] = experiment_data["methods"][best_method][
    "overall_improvement"
]
experiment_data["best_cpr"] = experiment_data["methods"][best_method]["overall_cpr"]

print(
    f"\n{'='*70}\nBest method: {best_method}, Improvement: {experiment_data['best_improvement']:.2f}%, CPR: {experiment_data['best_cpr']:.2f}%"
)
print(f"throughput_improvement_percentage = {experiment_data['best_improvement']:.2f}%")

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for m in experiment_data["methods"]:
    axes[0, 0].plot(experiment_data["methods"][m]["val_losses"], label=m)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Val Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Validation Loss")

method_names = list(experiment_data["methods"].keys())
axes[0, 1].bar(
    method_names,
    [experiment_data["methods"][m]["overall_improvement"] for m in method_names],
)
axes[0, 1].set_ylabel("Improvement (%)")
axes[0, 1].set_title("Overall Throughput Improvement")

axes[0, 2].bar(
    method_names, [experiment_data["methods"][m]["overall_cpr"] for m in method_names]
)
axes[0, 2].set_ylabel("CPR (%)")
axes[0, 2].set_title("Conflict Prevention Rate")

ds_names = [d[2] for d in datasets]
x = np.arange(len(ds_names))
width = 0.25
for i, m in enumerate(method_names):
    axes[1, 0].bar(
        x + i * width,
        [
            np.mean(experiment_data["methods"][m]["datasets"][d]["improvement"])
            for d in ds_names
        ],
        width,
        label=m,
    )
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels([d[:8] for d in ds_names], rotation=45)
axes[1, 0].legend()
axes[1, 0].set_title("Improvement by Dataset")

for i, m in enumerate(method_names):
    axes[1, 1].bar(
        x + i * width,
        [
            np.mean(experiment_data["methods"][m]["datasets"][d]["cpr"])
            for d in ds_names
        ],
        width,
        label=m,
    )
axes[1, 1].set_xticks(x + width)
axes[1, 1].set_xticklabels([d[:8] for d in ds_names], rotation=45)
axes[1, 1].legend()
axes[1, 1].set_title("CPR by Dataset")

for m in method_names:
    axes[1, 2].plot(
        agent_counts,
        [
            np.mean(
                [
                    experiment_data["methods"][m]["datasets"][d]["throughput"][i]
                    for d in ds_names
                ]
            )
            for i in range(len(agent_counts))
        ],
        "o-",
        label=m,
    )
axes[1, 2].set_xlabel("Agents")
axes[1, 2].set_ylabel("Avg Throughput")
axes[1, 2].legend()
axes[1, 2].set_title("Throughput vs Agents")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "temporal_memory_gnn_results.png"), dpi=150)
plt.close()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
