import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
import time

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {"synthetic": {}, "traffic_grid": {}, "maze_grid": {}}


def compute_congestion_iou(pred, target, threshold=0.3):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    union = ((pred_binary + target_binary) > 0).float().sum()
    return (intersection / (union + 1e-6)).item()


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
    def __init__(self, grid_size=20, n_agents=20, n_tasks=30):
        self.grid_size, self.n_agents, self.n_tasks = grid_size, n_agents, n_tasks
        self.reset()

    def reset(self):
        self.agent_positions = np.random.randint(0, self.grid_size, (self.n_agents, 2))
        self.agent_velocities = np.zeros((self.n_agents, 2))
        self.task_positions = np.random.randint(0, self.grid_size, (self.n_tasks, 2))
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step, self.tasks_completed = 0, 0
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
        while not np.array_equal(current, end):
            diff = end - current
            if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                current[0] += np.sign(diff[0])
            elif diff[1] != 0:
                current[1] += np.sign(diff[1])
            path.append(tuple(current))
        return path

    def assign_tasks(self, assignment):
        for agent_idx, task_idx in assignment:
            if task_idx >= 0 and self.task_active[task_idx]:
                self.agent_targets[agent_idx] = task_idx

    def step(self):
        self.time_step += 1
        for i in range(self.n_agents):
            target_idx = self.agent_targets[i]
            if target_idx >= 0 and self.task_active[target_idx]:
                diff = self.task_positions[target_idx] - self.agent_positions[i]
                if np.sum(np.abs(diff)) <= 1:
                    self.task_active[target_idx] = False
                    self.tasks_completed += 1
                    self.agent_targets[i] = -1
                else:
                    move = np.zeros(2, dtype=int)
                    if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                        move[0] = np.sign(diff[0])
                    elif diff[1] != 0:
                        move[1] = np.sign(diff[1])
                    self.agent_positions[i] = np.clip(
                        self.agent_positions[i] + move, 0, self.grid_size - 1
                    )
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                self.task_positions[idx] = np.random.randint(0, self.grid_size, 2)
                self.task_active[idx] = True
        return self.get_state()


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


def generate_training_data(n_samples=1000, grid_size=20, n_agents=20, n_tasks=30):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks)
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


def generate_traffic_grid_data(n_samples=500, grid_size=20):
    """Simulates traffic-like grid data with hotspots"""
    states, congestions = [], []
    for _ in range(n_samples):
        n_vehicles = np.random.randint(15, 35)
        n_destinations = np.random.randint(20, 40)
        vehicle_grid = np.zeros((grid_size, grid_size))
        dest_grid = np.zeros((grid_size, grid_size))
        vehicle_pos = np.random.randint(0, grid_size, (n_vehicles, 2))
        dest_pos = np.random.randint(0, grid_size, (n_destinations, 2))
        for pos in vehicle_pos:
            vehicle_grid[pos[0], pos[1]] += 1
        for pos in dest_pos:
            dest_grid[pos[0], pos[1]] += 1
        congestion = np.zeros((grid_size, grid_size))
        for i in range(n_vehicles):
            target = dest_pos[i % n_destinations]
            current = vehicle_pos[i].copy()
            for step in range(5):
                diff = target - current
                if np.sum(np.abs(diff)) == 0:
                    break
                if abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    current[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    current[1] += np.sign(diff[1])
                current = np.clip(current, 0, grid_size - 1)
                congestion[current[0], current[1]] += 1.0 / (step + 1)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(np.stack([vehicle_grid, dest_grid], axis=0))
        congestions.append(congestion)
    return np.array(states), np.array(congestions)


def generate_maze_grid_data(n_samples=500, grid_size=20):
    """Simulates maze-like navigation with obstacles creating bottlenecks"""
    states, congestions = [], []
    for _ in range(n_samples):
        n_agents = np.random.randint(10, 30)
        n_goals = np.random.randint(15, 35)
        agent_grid = np.zeros((grid_size, grid_size))
        goal_grid = np.zeros((grid_size, grid_size))
        agent_pos = np.random.randint(0, grid_size, (n_agents, 2))
        goal_pos = np.random.randint(0, grid_size, (n_goals, 2))
        for pos in agent_pos:
            agent_grid[pos[0], pos[1]] += 1
        for pos in goal_pos:
            goal_grid[pos[0], pos[1]] += 1
        congestion = np.zeros((grid_size, grid_size))
        corridor_x = grid_size // 2
        corridor_width = 3
        for i in range(n_agents):
            target = goal_pos[i % n_goals]
            current = agent_pos[i].copy()
            for step in range(6):
                diff = target - current
                if np.sum(np.abs(diff)) == 0:
                    break
                if abs(current[0] - corridor_x) > corridor_width and diff[0] != 0:
                    current[0] += np.sign(corridor_x - current[0])
                elif abs(diff[0]) >= abs(diff[1]) and diff[0] != 0:
                    current[0] += np.sign(diff[0])
                elif diff[1] != 0:
                    current[1] += np.sign(diff[1])
                current = np.clip(current, 0, grid_size - 1)
                congestion[current[0], current[1]] += 1.0 / (step + 1)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(np.stack([agent_grid, goal_grid], axis=0))
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
    agent_positions, task_positions, task_active, congestion_map, congestion_weight=2.0
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
            while not np.array_equal(current, task_pos):
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
    epochs=100,
    batch_size=64,
    lr=0.0005,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_states_t = torch.FloatTensor(train_states).to(device)
    train_congestions_t = torch.FloatTensor(train_congestions).unsqueeze(1).to(device)
    val_states_t = torch.FloatTensor(val_states).to(device)
    val_congestions_t = torch.FloatTensor(val_congestions).unsqueeze(1).to(device)
    train_losses, val_losses, train_ious, val_ious = [], [], [], []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_states_t))
        train_loss, train_iou, n_batches = 0, 0, 0
        for i in range(0, len(train_states_t), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(train_states_t[idx])
            loss = criterion(pred, train_congestions_t[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += compute_congestion_iou(pred, train_congestions_t[idx])
            n_batches += 1
        train_loss /= n_batches
        train_iou /= n_batches
        model.eval()
        with torch.no_grad():
            val_pred = model(val_states_t)
            val_loss = criterion(val_pred, val_congestions_t).item()
            val_iou = compute_congestion_iou(val_pred, val_congestions_t)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}"
            )
    return model, {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_iou": train_ious,
        "val_iou": val_ious,
    }


def evaluate_throughput(
    method,
    model=None,
    n_episodes=3,
    max_steps=150,
    grid_size=20,
    n_agents=20,
    n_tasks=30,
):
    throughputs = []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks)
        for _ in range(max_steps):
            state = sim.get_state()
            unassigned = [
                i
                for i in range(n_agents)
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
                    with torch.no_grad():
                        congestion_pred = (
                            model(torch.FloatTensor(state).unsqueeze(0).to(device))
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                    assignment = congestion_aware_assignment(
                        sim.agent_positions[unassigned],
                        sim.task_positions,
                        sim.task_active,
                        congestion_pred,
                    )
                sim.assign_tasks([(unassigned[a], t) for a, t in assignment])
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
    return np.mean(throughputs), np.std(throughputs)


# Generate data for all datasets
print("Generating synthetic warehouse data...")
train_states, train_congestions = generate_training_data(n_samples=800)
val_states, val_congestions = generate_training_data(n_samples=200)

print("Generating traffic grid data...")
traffic_train, traffic_train_cong = generate_traffic_grid_data(n_samples=400)
traffic_val, traffic_val_cong = generate_traffic_grid_data(n_samples=100)

print("Generating maze grid data...")
maze_train, maze_train_cong = generate_maze_grid_data(n_samples=400)
maze_val, maze_val_cong = generate_maze_grid_data(n_samples=100)

# Hyperparameters to tune
learning_rates = [0.001, 0.0005]
batch_sizes = [32, 64]
epochs_list = [80]
hidden_dim = 32  # Best from previous stage

datasets = {
    "synthetic": (train_states, train_congestions, val_states, val_congestions),
    "traffic_grid": (traffic_train, traffic_train_cong, traffic_val, traffic_val_cong),
    "maze_grid": (maze_train, maze_train_cong, maze_val, maze_val_cong),
}

best_config = {"lr": None, "batch_size": None, "val_loss": float("inf")}
all_results = {}

for dataset_name, (tr_s, tr_c, va_s, va_c) in datasets.items():
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    experiment_data[dataset_name] = {"configs": {}, "best_throughput": {}}

    best_model_for_dataset = None
    best_val_loss_dataset = float("inf")

    for lr in learning_rates:
        for bs in batch_sizes:
            config_name = f"lr{lr}_bs{bs}"
            print(f"\n  Config: lr={lr}, batch_size={bs}")

            model = CongestionPredictor(grid_size=20, hidden_dim=hidden_dim)
            model, metrics = train_predictor(
                model, tr_s, tr_c, va_s, va_c, epochs=80, batch_size=bs, lr=lr
            )

            experiment_data[dataset_name]["configs"][config_name] = {
                "losses": {"train": metrics["train_loss"], "val": metrics["val_loss"]},
                "metrics": {
                    "train_iou": metrics["train_iou"],
                    "val_iou": metrics["val_iou"],
                },
                "final_val_loss": metrics["val_loss"][-1],
                "final_val_iou": metrics["val_iou"][-1],
            }

            if metrics["val_loss"][-1] < best_val_loss_dataset:
                best_val_loss_dataset = metrics["val_loss"][-1]
                best_model_for_dataset = model

            print(
                f"    Final: val_loss={metrics['val_loss'][-1]:.4f}, val_iou={metrics['val_iou'][-1]:.4f}"
            )

    # Evaluate throughput for synthetic dataset only
    if dataset_name == "synthetic":
        agent_counts = [20, 30, 40]
        for n_agents in agent_counts:
            dist_tp, dist_std = evaluate_throughput(
                "distance", n_episodes=3, n_agents=n_agents
            )
            cata_tp, cata_std = evaluate_throughput(
                "congestion_aware",
                model=best_model_for_dataset,
                n_episodes=3,
                n_agents=n_agents,
            )
            experiment_data[dataset_name]["best_throughput"][n_agents] = {
                "distance": {"mean": dist_tp, "std": dist_std},
                "cata": {"mean": cata_tp, "std": cata_std},
                "improvement": (cata_tp - dist_tp) / dist_tp * 100,
            }
            print(
                f"  Agents={n_agents}: Distance={dist_tp:.2f}, CATA={cata_tp:.2f}, Imp={(cata_tp-dist_tp)/dist_tp*100:.1f}%"
            )

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, dataset_name in enumerate(["synthetic", "traffic_grid", "maze_grid"]):
    ax = axes[0, idx]
    for config_name, config_data in experiment_data[dataset_name]["configs"].items():
        ax.plot(config_data["losses"]["val"], label=config_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"{dataset_name} - Val Loss")
    ax.legend(fontsize=7)

    ax2 = axes[1, idx]
    for config_name, config_data in experiment_data[dataset_name]["configs"].items():
        ax2.plot(config_data["metrics"]["val_iou"], label=config_name)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation IoU")
    ax2.set_title(f"{dataset_name} - Val IoU")
    ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "hyperparameter_tuning_all_datasets.png"), dpi=150
)
plt.close()

# Summary
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for dataset_name in datasets.keys():
    print(f"\n{dataset_name}:")
    for config_name, config_data in experiment_data[dataset_name]["configs"].items():
        print(
            f"  {config_name}: val_loss={config_data['final_val_loss']:.4f}, val_iou={config_data['final_val_iou']:.4f}"
        )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
