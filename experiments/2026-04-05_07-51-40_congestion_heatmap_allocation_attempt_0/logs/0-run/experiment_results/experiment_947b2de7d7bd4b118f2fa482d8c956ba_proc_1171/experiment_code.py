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
    "sparse_warehouse": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "throughput": {},
        "cpr": {},
    },
    "bottleneck_warehouse": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "throughput": {},
        "cpr": {},
    },
    "dense_warehouse": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "throughput": {},
        "cpr": {},
    },
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
}

SCENARIOS = {
    "sparse_warehouse": {
        "grid_size": 20,
        "obstacle_density": 0.05,
        "n_agents": 20,
        "n_tasks": 30,
    },
    "bottleneck_warehouse": {
        "grid_size": 20,
        "obstacle_density": 0.15,
        "n_agents": 25,
        "n_tasks": 35,
    },
    "dense_warehouse": {
        "grid_size": 20,
        "obstacle_density": 0.25,
        "n_agents": 30,
        "n_tasks": 40,
    },
}


def generate_obstacles(grid_size, density, scenario_type):
    obstacles = set()
    n_obstacles = int(grid_size * grid_size * density)
    if scenario_type == "bottleneck_warehouse":
        for i in range(grid_size // 3, 2 * grid_size // 3):
            if i != grid_size // 2:
                obstacles.add((i, grid_size // 2))
    while len(obstacles) < n_obstacles:
        x, y = np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1)
        obstacles.add((x, y))
    return obstacles


class WarehouseSimulator:
    def __init__(self, grid_size=20, n_agents=20, n_tasks=30, obstacles=None):
        self.grid_size, self.n_agents, self.n_tasks = grid_size, n_agents, n_tasks
        self.obstacles = obstacles or set()
        self.reset()

    def reset(self):
        self.agent_positions = self._random_free_positions(self.n_agents)
        self.agent_velocities = np.zeros((self.n_agents, 2))
        self.task_positions = self._random_free_positions(self.n_tasks)
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed, self.congestion_events = 0, 0, 0
        return self.get_state()

    def _random_free_positions(self, n):
        positions = []
        while len(positions) < n:
            pos = np.random.randint(0, self.grid_size, 2)
            if tuple(pos) not in self.obstacles:
                positions.append(pos)
        return np.array(positions)

    def get_state(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        task_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[pos[0], pos[1]] += 1
        for i, pos in enumerate(self.task_positions):
            if self.task_active[i]:
                task_grid[pos[0], pos[1]] += 1
        return np.stack([agent_grid, task_grid], axis=0)

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
            moves = []
            if diff[0] != 0:
                moves.append((np.sign(diff[0]), 0))
            if diff[1] != 0:
                moves.append((0, np.sign(diff[1])))
            moved = False
            for dx, dy in moves:
                new_pos = (current[0] + dx, current[1] + dy)
                if (
                    0 <= new_pos[0] < self.grid_size
                    and 0 <= new_pos[1] < self.grid_size
                    and new_pos not in self.obstacles
                ):
                    current = np.array(new_pos)
                    path.append(new_pos)
                    moved = True
                    break
            if not moved:
                break
        return path

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if task_idx >= 0 and self.task_active[task_idx]:
                self.agent_targets[agent_idx] = task_idx

    def step(self):
        self.time_step += 1
        cell_counts = {}
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
                    moves = []
                    if diff[0] != 0:
                        moves.append((np.sign(diff[0]), 0))
                    if diff[1] != 0:
                        moves.append((0, np.sign(diff[1])))
                    for dx, dy in moves:
                        new_pos = np.clip(
                            self.agent_positions[i] + [dx, dy], 0, self.grid_size - 1
                        )
                        if tuple(new_pos) not in self.obstacles:
                            self.agent_positions[i] = new_pos
                            break
            pos_key = tuple(self.agent_positions[i])
            cell_counts[pos_key] = cell_counts.get(pos_key, 0) + 1
        self.congestion_events += sum(1 for c in cell_counts.values() if c > 2)
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                new_pos = self._random_free_positions(1)[0]
                self.task_positions[idx] = new_pos
                self.task_active[idx] = True
        return self.get_state()


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


def generate_training_data(scenario_name, n_samples=500):
    cfg = SCENARIOS[scenario_name]
    obstacles = generate_obstacles(
        cfg["grid_size"], cfg["obstacle_density"], scenario_name
    )
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(
            cfg["grid_size"], cfg["n_agents"], cfg["n_tasks"], obstacles
        )
        state = sim.get_state()
        active_tasks = np.where(sim.task_active)[0]
        sim.assign_tasks(
            [
                (i, active_tasks[i % len(active_tasks)])
                for i in range(min(cfg["n_agents"], len(active_tasks)))
            ]
        )
        congestion = sim.compute_congestion(lookahead=5)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(state)
        congestions.append(congestion)
    return np.array(states), np.array(congestions), obstacles


def distance_assignment(agent_pos, task_pos, task_active, obstacles, grid_size):
    active_idx = np.where(task_active)[0]
    if len(active_idx) == 0:
        return []
    cost = np.array(
        [[np.sum(np.abs(ap - task_pos[ti])) for ti in active_idx] for ap in agent_pos]
    )
    row, col = greedy_assignment(cost)
    return [(row[i], active_idx[col[i]]) for i in range(len(row))]


def congestion_assignment(
    agent_pos, task_pos, task_active, cong_map, obstacles, grid_size, weight=0.5
):
    active_idx = np.where(task_active)[0]
    if len(active_idx) == 0:
        return []
    cost = np.zeros((len(agent_pos), len(active_idx)))
    for i, ap in enumerate(agent_pos):
        for j, ti in enumerate(active_idx):
            tp = task_pos[ti]
            dist = np.sum(np.abs(ap - tp))
            path_cong, cur = 0, ap.copy()
            for _ in range(grid_size * 2):
                if np.array_equal(cur, tp):
                    break
                diff = tp - cur
                if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    cur[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    cur[1] += np.sign(diff[1])
                cur = np.clip(cur, 0, grid_size - 1)
                path_cong += cong_map[cur[0], cur[1]]
            cost[i, j] = dist + weight * path_cong
    row, col = greedy_assignment(cost)
    return [(row[i], active_idx[col[i]]) for i in range(len(row))]


def evaluate(method, model, scenario_name, n_episodes=5, max_steps=100):
    cfg = SCENARIOS[scenario_name]
    obstacles = generate_obstacles(
        cfg["grid_size"], cfg["obstacle_density"], scenario_name
    )
    throughputs, congestion_totals = [], []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(
            cfg["grid_size"], cfg["n_agents"], cfg["n_tasks"], obstacles
        )
        for _ in range(max_steps):
            state = sim.get_state()
            unassigned = [
                i
                for i in range(cfg["n_agents"])
                if sim.agent_targets[i] < 0 or not sim.task_active[sim.agent_targets[i]]
            ]
            if unassigned and np.any(sim.task_active):
                if method == "distance":
                    assign = distance_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        obstacles,
                        cfg["grid_size"],
                    )
                else:
                    with torch.no_grad():
                        cong = (
                            model(torch.FloatTensor(state).unsqueeze(0).to(device))
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                    assign = congestion_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        cong,
                        obstacles,
                        cfg["grid_size"],
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assign])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        congestion_totals.append(sim.congestion_events)
    return (
        np.mean(throughputs),
        np.std(throughputs),
        np.mean(congestion_totals),
        np.std(congestion_totals),
    )


# Training
print("Generating combined training data from all scenarios...")
all_train, all_cong = [], []
for sc in SCENARIOS:
    s, c, _ = generate_training_data(sc, n_samples=400)
    all_train.append(s)
    all_cong.append(c)
train_states = np.concatenate(all_train)
train_cong = np.concatenate(all_cong)
perm = np.random.permutation(len(train_states))
split = int(0.8 * len(train_states))
train_s, val_s = train_states[perm[:split]], train_states[perm[split:]]
train_c, val_c = train_cong[perm[:split]], train_cong[perm[split:]]

print(f"Training samples: {len(train_s)}, Validation samples: {len(val_s)}")
model = CongestionPredictor(grid_size=20, hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_t = torch.FloatTensor(train_s).to(device)
train_ct = torch.FloatTensor(train_c).unsqueeze(1).to(device)
val_t = torch.FloatTensor(val_s).to(device)
val_ct = torch.FloatTensor(val_c).unsqueeze(1).to(device)

for epoch in range(30):
    model.train()
    perm_e = torch.randperm(len(train_t))
    t_loss, nb = 0, 0
    for i in range(0, len(train_t), 32):
        idx = perm_e[i : i + 32]
        optimizer.zero_grad()
        loss = criterion(model(train_t[idx]), train_ct[idx])
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        nb += 1
    t_loss /= nb
    model.eval()
    with torch.no_grad():
        v_loss = criterion(model(val_t), val_ct).item()
    experiment_data["training"]["losses"]["train"].append(t_loss)
    experiment_data["training"]["losses"]["val"].append(v_loss)
    experiment_data["training"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: train_loss = {t_loss:.4f}, validation_loss = {v_loss:.4f}")

# Evaluation on 3 scenarios
print("\n" + "=" * 60)
print("EVALUATION ON THREE WAREHOUSE SCENARIOS")
print("=" * 60)
eps = 1e-3
for sc in SCENARIOS:
    print(f"\n--- {sc} ---")
    dist_tp, dist_std, dist_cong, _ = evaluate("distance", None, sc)
    cata_tp, cata_std, cata_cong, _ = evaluate("congestion_aware", model, sc)
    cpr = (dist_cong - cata_cong) / (dist_cong + eps) * 100
    improvement = (cata_tp - dist_tp) / (dist_tp + eps) * 100
    experiment_data[sc]["throughput"] = {
        "distance": dist_tp,
        "cata": cata_tp,
        "improvement": improvement,
    }
    experiment_data[sc]["cpr"] = {
        "distance_cong": dist_cong,
        "cata_cong": cata_cong,
        "cpr": cpr,
    }
    print(
        f"Distance-based: Throughput={dist_tp:.2f}±{dist_std:.2f}, Congestion={dist_cong:.1f}"
    )
    print(f"CATA: Throughput={cata_tp:.2f}±{cata_std:.2f}, Congestion={cata_cong:.1f}")
    print(f"Improvement: {improvement:.2f}%, Congestion Prevention Rate: {cpr:.2f}%")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0, 0].plot(experiment_data["training"]["losses"]["val"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training Loss")
axes[0, 0].legend()

scenarios = list(SCENARIOS.keys())
dist_tps = [experiment_data[s]["throughput"]["distance"] for s in scenarios]
cata_tps = [experiment_data[s]["throughput"]["cata"] for s in scenarios]
x = np.arange(len(scenarios))
axes[0, 1].bar(x - 0.2, dist_tps, 0.4, label="Distance")
axes[0, 1].bar(x + 0.2, cata_tps, 0.4, label="CATA")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(scenarios, rotation=15)
axes[0, 1].set_ylabel("Throughput")
axes[0, 1].set_title("Throughput Comparison")
axes[0, 1].legend()

cprs = [experiment_data[s]["cpr"]["cpr"] for s in scenarios]
colors = ["green" if c > 0 else "red" for c in cprs]
axes[1, 0].bar(scenarios, cprs, color=colors, alpha=0.7)
axes[1, 0].axhline(0, color="black", linestyle="--")
axes[1, 0].set_ylabel("CPR (%)")
axes[1, 0].set_title("Congestion Prevention Rate")
axes[1, 0].tick_params(axis="x", rotation=15)

imps = [experiment_data[s]["throughput"]["improvement"] for s in scenarios]
axes[1, 1].bar(scenarios, imps, color="blue", alpha=0.7)
axes[1, 1].axhline(0, color="black", linestyle="--")
axes[1, 1].set_ylabel("Improvement (%)")
axes[1, 1].set_title("Throughput Improvement")
axes[1, 1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "three_scenario_evaluation.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
