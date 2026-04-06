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
    "methods_comparison": {},
    "conflict_memory_analysis": {},
    "training": {},
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
        self.conflict_memory = np.zeros((grid_size, grid_size))
        self.decay_rate = 0.9
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
        if self.hotspots and np.random.random() < 0.7:
            hotspot = self.hotspots[np.random.randint(len(self.hotspots))]
            offset = np.random.randint(-3, 4, 2)
            pos = np.clip(hotspot + offset, 0, self.grid_size - 1)
            if self._valid_position(pos):
                return pos
        return self._random_valid_position()

    def reset(self, keep_memory=False):
        self.agent_positions = np.array(
            [self._random_valid_position() for _ in range(self.n_agents)]
        )
        self.task_positions = np.array(
            [self._hotspot_biased_position() for _ in range(self.n_tasks)]
        )
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed, self.congestion_events = 0, 0, 0
        if not keep_memory:
            self.conflict_memory *= 0
        return self.get_state()

    def get_state(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        task_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[int(pos[0]), int(pos[1])] += 1
        for i, pos in enumerate(self.task_positions):
            if self.task_active[i]:
                task_grid[int(pos[0]), int(pos[1])] += 1
        return np.stack([agent_grid, task_grid], axis=0)

    def update_conflict_memory(self):
        self.conflict_memory *= self.decay_rate
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[int(pos[0]), int(pos[1])] += 1
        conflict_locations = agent_grid > 1
        self.conflict_memory += conflict_locations * (agent_grid - 1)

    def count_congestion_events(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.agent_positions:
            agent_grid[int(pos[0]), int(pos[1])] += 1
        return int(np.sum(agent_grid[agent_grid > 1] - 1))

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if (
                task_idx >= 0
                and task_idx < len(self.task_active)
                and self.task_active[task_idx]
            ):
                self.agent_targets[agent_idx] = task_idx

    def step(self):
        self.time_step += 1
        self.update_conflict_memory()
        self.congestion_events += self.count_congestion_events()
        for i in range(self.n_agents):
            target_idx = self.agent_targets[i]
            if (
                target_idx >= 0
                and target_idx < len(self.task_active)
                and self.task_active[target_idx]
            ):
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


class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        return torch.sigmoid(self.conv_out(x))


def generate_training_data(
    n_samples=200, grid_size=20, n_agents=25, n_tasks=30, obstacles=None, hotspots=None
):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks, obstacles, hotspots)
        for _ in range(15):
            active_tasks = np.where(sim.task_active)[0]
            if len(active_tasks) > 0:
                sim.assign_tasks(
                    [(i, active_tasks[i % len(active_tasks)]) for i in range(n_agents)]
                )
            sim.step()
        state = sim.get_state()
        mem_max = sim.conflict_memory.max()
        memory_norm = (
            sim.conflict_memory / (mem_max + 1e-6)
            if mem_max > 0
            else sim.conflict_memory
        )
        state_with_memory = np.concatenate([state, memory_norm[np.newaxis]], axis=0)

        # Create better congestion target - combination of current density and historical conflicts
        agent_grid = np.zeros((grid_size, grid_size))
        for pos in sim.agent_positions:
            agent_grid[int(pos[0]), int(pos[1])] += 1
        congestion = np.maximum(agent_grid - 1, 0) + 0.5 * memory_norm
        cong_max = congestion.max()
        congestion = congestion / (cong_max + 1e-6) if cong_max > 0 else congestion
        states.append(state_with_memory)
        congestions.append(congestion)
    return np.array(states, dtype=np.float32), np.array(congestions, dtype=np.float32)


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


def cma_assignment(
    agent_positions,
    task_positions,
    task_active,
    congestion_map,
    conflict_memory,
    cw=0.4,
    mw=0.3,
):
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    grid_size = congestion_map.shape[0]
    cost_matrix = np.zeros((len(agent_positions), len(active_indices)))
    mem_max = conflict_memory.max()
    memory_norm = conflict_memory / (mem_max + 1e-6) if mem_max > 0 else conflict_memory

    for i, ap in enumerate(agent_positions):
        for j, ti in enumerate(active_indices):
            tp = task_positions[ti]
            dist = np.sum(np.abs(ap - tp))
            path_cong, path_mem, cur = 0, 0, ap.copy().astype(float)
            steps = int(dist) + 1
            for _ in range(steps):
                if np.allclose(cur, tp):
                    break
                diff = tp - cur
                if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    cur[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    cur[1] += np.sign(diff[1])
                cur = np.clip(cur, 0, grid_size - 1)
                cx, cy = int(cur[0]), int(cur[1])
                path_cong += congestion_map[cx, cy]
                path_mem += memory_norm[cx, cy]
            cost_matrix[i, j] = dist + cw * path_cong + mw * path_mem
    row_ind, col_ind = greedy_assignment(cost_matrix)
    return [(row_ind[i], active_indices[col_ind[i]]) for i in range(len(row_ind))]


def train_predictor(
    model, train_s, train_c, val_s, val_c, epochs=60, batch_size=32, lr=0.002
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()
    train_t = torch.FloatTensor(train_s).to(device)
    train_ct = torch.FloatTensor(train_c).unsqueeze(1).to(device)
    val_t = torch.FloatTensor(val_s).to(device)
    val_ct = torch.FloatTensor(val_c).unsqueeze(1).to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_s))
        tloss, nb = 0, 0
        for i in range(0, len(train_s), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_t[idx]), train_ct[idx])
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            nb += 1
        tloss /= nb
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vloss = criterion(model(val_t), val_ct).item()
        train_losses.append(tloss)
        val_losses.append(vloss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: validation_loss = {vloss:.4f}")
    return model, train_losses, val_losses


def evaluate_method(method, model, sim_params, n_episodes=5, max_steps=100):
    throughputs, cong_events = [], []
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
                    assign = distance_based_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                    )
                else:
                    mem_max = sim.conflict_memory.max()
                    mem_norm = (
                        sim.conflict_memory / (mem_max + 1e-6)
                        if mem_max > 0
                        else sim.conflict_memory
                    )
                    state_mem = np.concatenate([state, mem_norm[np.newaxis]], axis=0)
                    st = torch.FloatTensor(state_mem).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = model(st).squeeze().cpu().numpy()
                    assign = cma_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        pred,
                        sim.conflict_memory,
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assign])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
        cong_events.append(sim.congestion_events)
    return (
        np.mean(throughputs),
        np.std(throughputs),
        np.mean(cong_events),
        np.std(cong_events),
    )


print("Creating datasets...")
datasets = [
    create_sparse_warehouse(),
    create_bottleneck_warehouse(),
    create_dense_obstacle_warehouse(),
]

all_train, all_cong = [], []
for obs, hs, name in datasets:
    print(f"  Generating: {name}")
    s, c = generate_training_data(n_samples=250, obstacles=obs, hotspots=hs)
    all_train.append(s)
    all_cong.append(c)

train_s = np.concatenate([s[:200] for s in all_train])
train_c = np.concatenate([c[:200] for c in all_cong])
val_s = np.concatenate([s[200:] for s in all_train])
val_c = np.concatenate([c[200:] for c in all_cong])
print(f"Train: {train_s.shape}, Val: {val_s.shape}")

print("\nTraining CMA model...")
model = CongestionPredictor(grid_size=20, hidden_dim=64)
model, tl, vl = train_predictor(
    model, train_s, train_c, val_s, val_c, epochs=60, batch_size=32, lr=0.002
)
experiment_data["training"] = {"train_loss": tl, "val_loss": vl}

agent_counts = [20, 30, 40]
print("\n" + "=" * 60)
print("EVALUATION WITH CPR TRACKING")
print("=" * 60)

for obs, hs, dname in datasets:
    experiment_data["methods_comparison"][dname] = {
        "distance": {"tp": [], "cong": []},
        "cma": {"tp": [], "cong": []},
        "improvement": [],
        "cpr": [],
        "n_agents": agent_counts,
    }
    print(f"\n{dname}:")
    for na in agent_counts:
        sp = {
            "grid_size": 20,
            "n_agents": na,
            "n_tasks": 30,
            "obstacles": obs,
            "hotspots": hs,
        }
        d_tp, _, d_cg, _ = evaluate_method(
            "distance", None, sp, n_episodes=5, max_steps=100
        )
        c_tp, _, c_cg, _ = evaluate_method(
            "cma", model, sp, n_episodes=5, max_steps=100
        )
        imp = (c_tp - d_tp) / d_tp * 100 if d_tp > 0 else 0
        cpr = (d_cg - c_cg) / d_cg * 100 if d_cg > 0 else 0
        experiment_data["methods_comparison"][dname]["distance"]["tp"].append(d_tp)
        experiment_data["methods_comparison"][dname]["distance"]["cong"].append(d_cg)
        experiment_data["methods_comparison"][dname]["cma"]["tp"].append(c_tp)
        experiment_data["methods_comparison"][dname]["cma"]["cong"].append(c_cg)
        experiment_data["methods_comparison"][dname]["improvement"].append(imp)
        experiment_data["methods_comparison"][dname]["cpr"].append(cpr)
        print(f"  Agents={na}: TP Imp={imp:.1f}%, CPR={cpr:.1f}%")

all_imps = [
    np.mean(experiment_data["methods_comparison"][d]["improvement"])
    for d in experiment_data["methods_comparison"]
]
all_cprs = [
    np.mean(experiment_data["methods_comparison"][d]["cpr"])
    for d in experiment_data["methods_comparison"]
]
print(f"\nOverall Improvement: {np.mean(all_imps):.2f}%")
print(f"Overall CPR: {np.mean(all_cprs):.2f}%")
print(f"throughput_improvement_percentage = {np.mean(all_imps):.2f}%")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].plot(tl, label="Train")
axes[0, 0].plot(vl, label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].set_title("Training Curves")

dnames = list(experiment_data["methods_comparison"].keys())
avg_imps = [
    np.mean(experiment_data["methods_comparison"][d]["improvement"]) for d in dnames
]
axes[0, 1].bar(range(len(dnames)), avg_imps)
axes[0, 1].set_xticks(range(len(dnames)))
axes[0, 1].set_xticklabels([d[:8] for d in dnames], rotation=45)
axes[0, 1].set_ylabel("Improvement (%)")
axes[0, 1].set_title("Throughput Improvement")
axes[0, 1].axhline(0, c="r", ls="--")

avg_cprs = [np.mean(experiment_data["methods_comparison"][d]["cpr"]) for d in dnames]
axes[0, 2].bar(range(len(dnames)), avg_cprs, color="green")
axes[0, 2].set_xticks(range(len(dnames)))
axes[0, 2].set_xticklabels([d[:8] for d in dnames], rotation=45)
axes[0, 2].set_ylabel("CPR (%)")
axes[0, 2].set_title("Conflict Prevention Rate")
axes[0, 2].axhline(0, c="r", ls="--")

for i, d in enumerate(dnames[:3]):
    data = experiment_data["methods_comparison"][d]
    axes[1, i].plot(agent_counts, data["cma"]["tp"], "o-", label="CMA")
    axes[1, i].plot(agent_counts, data["distance"]["tp"], "x--", label="Distance")
    axes[1, i].set_xlabel("Agents")
    axes[1, i].set_ylabel("Throughput")
    axes[1, i].legend()
    axes[1, i].set_title(d[:15])

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cma_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
