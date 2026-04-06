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
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "datasets": {},
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
                                self.agent_positions[i] = new_pos
                                break
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                self.task_positions[idx] = self._hotspot_biased_position()
                self.task_active[idx] = True
        return self.get_state()


def create_sparse_warehouse(gs=20):
    obs = {(i, j) for i in range(5, 15, 5) for j in range(3, gs - 3, 4)}
    return obs, [np.array([5, 5]), np.array([15, 15])], "sparse_warehouse"


def create_bottleneck_warehouse(gs=20):
    obs = {(i, 10) for i in range(gs) if i not in [9, 10]} | {
        (10, j) for j in range(gs) if j not in [9, 10]
    }
    return (
        obs,
        [np.array([5, 5]), np.array([5, 15]), np.array([15, 5]), np.array([15, 15])],
        "bottleneck_warehouse",
    )


def create_dense_obstacle_warehouse(gs=20):
    np.random.seed(42)
    obs = {
        (np.random.randint(1, gs - 1), np.random.randint(1, gs - 1))
        for _ in range(int(gs * gs * 0.15))
    }
    return (
        obs,
        [np.array([3, 3]), np.array([16, 16]), np.array([3, 16]), np.array([16, 3])],
        "dense_obstacle_warehouse",
    )


def create_intersection_warehouse(gs=20):
    obs = {
        (i, j)
        for i in range(gs)
        for j in range(gs)
        if not ((4 <= i <= 6) or (13 <= i <= 15) or (4 <= j <= 6) or (13 <= j <= 15))
        and (i + j) % 3 == 0
    }
    return (
        obs,
        [np.array([5, 5]), np.array([5, 14]), np.array([14, 5]), np.array([14, 14])],
        "intersection_warehouse",
    )


def create_aisle_warehouse(gs=20):
    obs = {
        (i, j)
        for i in range(2, gs - 2, 3)
        for j in range(gs)
        if j not in [0, 1, gs - 2, gs - 1, 9, 10]
    }
    return (
        obs,
        [
            np.array([1, 10]),
            np.array([gs - 2, 10]),
            np.array([10, 1]),
            np.array([10, gs - 2]),
        ],
        "aisle_warehouse",
    )


class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def generate_training_data(n_samples, obstacles, hotspots):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(obstacles=obstacles, hotspots=hotspots)
        state = sim.get_state()
        active = np.where(sim.task_active)[0]
        sim.assign_tasks(
            [(i, active[i]) for i in range(min(sim.n_agents, len(active)))]
        )
        cong = sim.compute_congestion()
        cong = cong / (cong.max() + 1e-6)
        states.append(state)
        congestions.append(cong)
    return np.array(states), np.array(congestions)


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
                active_idx = np.where(sim.task_active)[0]
                cost = np.array(
                    [
                        [
                            np.sum(
                                np.abs(sim.agent_positions[u] - sim.task_positions[t])
                            )
                            for t in active_idx
                        ]
                        for u in unassigned
                    ]
                )
                if method == "cma" and model is not None:
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        cmap = model(state_t).squeeze().cpu().numpy()
                    for i, u in enumerate(unassigned):
                        for j, t in enumerate(active_idx):
                            cost[i, j] += (
                                0.5
                                * cmap[
                                    sim.task_positions[t][0], sim.task_positions[t][1]
                                ]
                            )
                row, col = greedy_assignment(cost)
                sim.assign_tasks(
                    [(unassigned[row[k]], active_idx[col[k]]) for k in range(len(row))]
                )
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        congestions.append(sim.congestion_events)
    return np.mean(throughputs), np.mean(congestions)


datasets = [
    create_sparse_warehouse(),
    create_bottleneck_warehouse(),
    create_dense_obstacle_warehouse(),
    create_intersection_warehouse(),
    create_aisle_warehouse(),
]

print("Generating training data...")
all_states, all_congs = [], []
for obs, hots, name in datasets:
    s, c = generate_training_data(150, obs, hots)
    all_states.append(s)
    all_congs.append(c)
train_s = np.concatenate([s[:120] for s in all_states])
train_c = np.concatenate([c[:120] for c in all_congs])
val_s = np.concatenate([s[120:] for s in all_states])
val_c = np.concatenate([c[120:] for c in all_congs])
print(f"Train: {train_s.shape}, Val: {val_s.shape}")

model = CongestionPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_t = torch.FloatTensor(train_s).to(device)
train_ct = torch.FloatTensor(train_c).unsqueeze(1).to(device)
val_t = torch.FloatTensor(val_s).to(device)
val_ct = torch.FloatTensor(val_c).unsqueeze(1).to(device)

epochs = 50
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(train_s))
    tloss = 0
    for i in range(0, len(train_s), 32):
        idx = perm[i : i + 32]
        optimizer.zero_grad()
        loss = criterion(model(train_t[idx]), train_ct[idx])
        loss.backward()
        optimizer.step()
        tloss += loss.item()
    tloss /= len(train_s) // 32
    model.eval()
    with torch.no_grad():
        vloss = criterion(model(val_t), val_ct).item()
    experiment_data["training"]["losses"]["train"].append(tloss)
    experiment_data["training"]["losses"]["val"].append(vloss)
    experiment_data["training"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: train_loss = {tloss:.4f}, validation_loss = {vloss:.4f}")

print("\nEvaluating on all datasets...")
agent_counts = [20, 30, 40]
total_cpr, total_imp = [], []
for obs, hots, name in datasets:
    experiment_data["datasets"][name] = {
        "throughput": {"dist": [], "cma": []},
        "congestion": {"dist": [], "cma": []},
        "improvement": [],
        "cpr": [],
        "n_agents": agent_counts,
    }
    for na in agent_counts:
        params = {
            "grid_size": 20,
            "n_agents": na,
            "n_tasks": 30,
            "obstacles": obs,
            "hotspots": hots,
        }
        d_tp, d_cong = evaluate_method("dist", None, params)
        c_tp, c_cong = evaluate_method("cma", model, params)
        imp = (c_tp - d_tp) / d_tp * 100 if d_tp > 0 else 0
        cpr = (d_cong - c_cong) / d_cong * 100 if d_cong > 0 else 0
        experiment_data["datasets"][name]["throughput"]["dist"].append(d_tp)
        experiment_data["datasets"][name]["throughput"]["cma"].append(c_tp)
        experiment_data["datasets"][name]["congestion"]["dist"].append(d_cong)
        experiment_data["datasets"][name]["congestion"]["cma"].append(c_cong)
        experiment_data["datasets"][name]["improvement"].append(imp)
        experiment_data["datasets"][name]["cpr"].append(cpr)
        total_cpr.append(cpr)
        total_imp.append(imp)
        print(f"{name} ({na} agents): Imp={imp:.1f}%, CPR={cpr:.1f}%")

experiment_data["best_improvement"] = np.mean(total_imp)
experiment_data["best_cpr"] = np.mean(total_cpr)
print(f"\nOverall: throughput_improvement = {experiment_data['best_improvement']:.2f}%")
print(f"Conflict Prevention Rate (CPR) = {experiment_data['best_cpr']:.2f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0, 0].plot(experiment_data["training"]["losses"]["val"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Training Curves")

ds_names = [d[2] for d in datasets]
avg_imp = [np.mean(experiment_data["datasets"][n]["improvement"]) for n in ds_names]
axes[0, 1].bar(range(len(ds_names)), avg_imp)
axes[0, 1].set_xticks(range(len(ds_names)))
axes[0, 1].set_xticklabels([n[:10] for n in ds_names], rotation=30)
axes[0, 1].set_ylabel("Improvement %")
axes[0, 1].set_title("Throughput Improvement")

avg_cpr = [np.mean(experiment_data["datasets"][n]["cpr"]) for n in ds_names]
axes[1, 0].bar(range(len(ds_names)), avg_cpr, color="green")
axes[1, 0].set_xticks(range(len(ds_names)))
axes[1, 0].set_xticklabels([n[:10] for n in ds_names], rotation=30)
axes[1, 0].set_ylabel("CPR %")
axes[1, 0].set_title("Conflict Prevention Rate")

for n in ds_names:
    axes[1, 1].plot(
        agent_counts,
        experiment_data["datasets"][n]["throughput"]["cma"],
        "o-",
        label=n[:8],
    )
axes[1, 1].set_xlabel("Agents")
axes[1, 1].set_ylabel("Throughput")
axes[1, 1].legend(fontsize=7)
axes[1, 1].set_title("CMA Throughput by Agents")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cma_results.png"), dpi=150)
plt.close()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
