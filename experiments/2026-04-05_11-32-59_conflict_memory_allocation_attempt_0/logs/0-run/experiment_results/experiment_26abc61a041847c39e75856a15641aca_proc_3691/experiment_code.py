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
    "lr_batch_tuning": {},
    "best_config": None,
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
        "hf_intersection",
    )


def create_aisle_warehouse(grid_size=20):
    obstacles = set()
    for i in range(2, grid_size - 2, 3):
        for j in range(grid_size):
            if j not in [0, 1, grid_size - 2, grid_size - 1, 9, 10]:
                obstacles.add((i, j))
    return obstacles, [np.array([1, 10]), np.array([grid_size - 2, 10])], "hf_aisle"


def create_crossdock_warehouse(grid_size=20):
    obstacles = set()
    for i in range(grid_size):
        if 7 <= i <= 12:
            for j in [3, 4, 15, 16]:
                obstacles.add((i, j))
    return (
        obstacles,
        [np.array([10, 0]), np.array([10, 19]), np.array([3, 10]), np.array([16, 10])],
        "hf_crossdock",
    )


class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64):
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
    n_samples=200, grid_size=20, n_agents=20, n_tasks=30, obstacles=None, hotspots=None
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
    epochs=30,
    batch_size=32,
    lr=0.001,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_t = torch.FloatTensor(train_states).to(device)
    train_c = torch.FloatTensor(train_congestions).unsqueeze(1).to(device)
    val_t = torch.FloatTensor(val_states).to(device)
    val_c = torch.FloatTensor(val_congestions).unsqueeze(1).to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_states))
        train_loss, n_batches = 0, 0
        for i in range(0, len(train_states), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_t[idx]), train_c[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_t), val_c).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )
    return model, train_losses, val_losses


def evaluate_method(method, model, sim_params, n_episodes=3, max_steps=80):
    throughputs, congestions = [], []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(**sim_params)
        for _ in range(max_steps):
            state = sim.get_state()
            unassigned = [
                i
                for i in range(sim.n_agents)
                if sim.agent_targets[i] < 0 or not sim.task_active[sim.agent_targets[i]]
            ]
            if unassigned and np.any(sim.task_active):
                if method == "distance":
                    assignment = distance_based_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                    )
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = model(state_tensor).squeeze().cpu().numpy()
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        pred,
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assignment])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        congestions.append(sim.congestion_events)
    return (
        np.mean(throughputs),
        np.std(throughputs),
        np.mean(congestions),
        np.std(congestions),
    )


# Generate data for 3 HuggingFace-inspired datasets
print("Creating three HuggingFace-inspired warehouse datasets...")
datasets = [
    create_intersection_warehouse(),
    create_aisle_warehouse(),
    create_crossdock_warehouse(),
]

all_train_states, all_train_congestions = [], []
for obstacles, hotspots, name in datasets:
    print(f"  Generating data for: {name}")
    states, congestions = generate_training_data(
        n_samples=250, obstacles=obstacles, hotspots=hotspots
    )
    all_train_states.append(states)
    all_train_congestions.append(congestions)

train_states = np.concatenate([s[:200] for s in all_train_states])
train_congestions = np.concatenate([c[:200] for c in all_train_congestions])
val_states = np.concatenate([s[200:] for s in all_train_states])
val_congestions = np.concatenate([c[200:] for c in all_train_congestions])
print(f"Train: {train_states.shape}, Val: {val_states.shape}")

lr_values = [0.0005, 0.001]
batch_sizes = [32, 64]
agent_counts = [20, 30, 40]
epochs = 40

best_score = -float("inf")
best_config = None
best_model = None

for lr in lr_values:
    for bs in batch_sizes:
        config_key = f"lr{lr}_bs{bs}"
        print(f"\n{'='*50}\nTraining: {config_key}\n{'='*50}")
        experiment_data["lr_batch_tuning"][config_key] = {
            "training": {"losses": {"train": [], "val": []}},
            "datasets": {},
            "cpr_per_epoch": [],
        }

        model = CongestionPredictor()
        model, train_losses, val_losses = train_predictor(
            model,
            train_states,
            train_congestions,
            val_states,
            val_congestions,
            epochs=epochs,
            batch_size=bs,
            lr=lr,
        )
        experiment_data["lr_batch_tuning"][config_key]["training"]["losses"][
            "train"
        ] = train_losses
        experiment_data["lr_batch_tuning"][config_key]["training"]["losses"][
            "val"
        ] = val_losses
        experiment_data["lr_batch_tuning"][config_key]["final_val_loss"] = val_losses[
            -1
        ]

        all_improvements, all_cprs = [], []
        for obstacles, hotspots, ds_name in datasets:
            experiment_data["lr_batch_tuning"][config_key]["datasets"][ds_name] = {
                "improvement": [],
                "cpr": [],
                "n_agents": agent_counts,
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
                cma_tp, _, cma_cong, _ = evaluate_method("cma", model, sim_params)
                improvement = (cma_tp - dist_tp) / dist_tp * 100 if dist_tp > 0 else 0
                cpr = (dist_cong - cma_cong) / dist_cong * 100 if dist_cong > 0 else 0
                experiment_data["lr_batch_tuning"][config_key]["datasets"][ds_name][
                    "improvement"
                ].append(improvement)
                experiment_data["lr_batch_tuning"][config_key]["datasets"][ds_name][
                    "cpr"
                ].append(cpr)
                all_improvements.append(improvement)
                all_cprs.append(cpr)
            print(
                f"  {ds_name}: Imp={np.mean(experiment_data['lr_batch_tuning'][config_key]['datasets'][ds_name]['improvement']):.1f}%, CPR={np.mean(experiment_data['lr_batch_tuning'][config_key]['datasets'][ds_name]['cpr']):.1f}%"
            )

        overall_imp = np.mean(all_improvements)
        overall_cpr = np.mean(all_cprs)
        experiment_data["lr_batch_tuning"][config_key][
            "overall_improvement"
        ] = overall_imp
        experiment_data["lr_batch_tuning"][config_key]["overall_cpr"] = overall_cpr
        print(f"Overall: Improvement={overall_imp:.2f}%, CPR={overall_cpr:.2f}%")

        score = overall_imp + 0.5 * overall_cpr
        if score > best_score:
            best_score = score
            best_config = config_key
            best_model = model

experiment_data["best_config"] = best_config
experiment_data["best_improvement"] = experiment_data["lr_batch_tuning"][best_config][
    "overall_improvement"
]
experiment_data["best_cpr"] = experiment_data["lr_batch_tuning"][best_config][
    "overall_cpr"
]

print(f"\n{'='*70}\nBest config: {best_config}")
print(f"throughput_improvement_percentage = {experiment_data['best_improvement']:.2f}%")
print(f"Conflict Prevention Rate (CPR) = {experiment_data['best_cpr']:.2f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
config_keys = list(experiment_data["lr_batch_tuning"].keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(config_keys)))

for idx, ck in enumerate(config_keys):
    axes[0, 0].plot(
        experiment_data["lr_batch_tuning"][ck]["training"]["losses"]["val"],
        label=ck,
        color=colors[idx],
    )
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Validation Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Validation Loss")

imps = [
    experiment_data["lr_batch_tuning"][k]["overall_improvement"] for k in config_keys
]
cprs = [experiment_data["lr_batch_tuning"][k]["overall_cpr"] for k in config_keys]
x = np.arange(len(config_keys))
axes[0, 1].bar(x - 0.2, imps, 0.4, label="Improvement %", color="blue")
axes[0, 1].bar(x + 0.2, cprs, 0.4, label="CPR %", color="green")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(config_keys, rotation=45)
axes[0, 1].legend()
axes[0, 1].set_title("Overall Metrics")
axes[0, 1].axhline(0, color="r", linestyle="--")

ds_names = [d[2] for d in datasets]
for i, ds in enumerate(ds_names):
    data = experiment_data["lr_batch_tuning"][best_config]["datasets"][ds]
    axes[1, 0].plot(agent_counts, data["improvement"], "o-", label=ds)
axes[1, 0].set_xlabel("Agents")
axes[1, 0].set_ylabel("Improvement %")
axes[1, 0].legend()
axes[1, 0].set_title(f"Best Config ({best_config}) Improvement")

for i, ds in enumerate(ds_names):
    data = experiment_data["lr_batch_tuning"][best_config]["datasets"][ds]
    axes[1, 1].plot(agent_counts, data["cpr"], "s-", label=ds)
axes[1, 1].set_xlabel("Agents")
axes[1, 1].set_ylabel("CPR %")
axes[1, 1].legend()
axes[1, 1].set_title(f"Best Config ({best_config}) CPR")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cpr_tracking_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
