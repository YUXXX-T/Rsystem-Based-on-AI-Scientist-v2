# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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

# Experiment data storage for hyperparameter tuning
experiment_data = {
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "hyperparam_tuning": {},
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
                target_pos = self.task_positions[target_idx]
                diff = target_pos - self.agent_positions[i]
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
                    self.agent_velocities[i] = move
        for idx in np.where(~self.task_active)[0]:
            if np.random.random() < 0.3:
                self.task_positions[idx] = np.random.randint(0, self.grid_size, 2)
                self.task_active[idx] = True
        return self.get_state()


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
    epochs=50,
    batch_size=32,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_states = torch.FloatTensor(train_states).to(device)
    train_congestions = torch.FloatTensor(train_congestions).unsqueeze(1).to(device)
    val_states = torch.FloatTensor(val_states).to(device)
    val_congestions = torch.FloatTensor(val_congestions).unsqueeze(1).to(device)
    n_train = len(train_states)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss, n_batches = 0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_states[idx]), train_congestions[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_states), val_congestions).item()
        experiment_data["training"]["losses"]["train"].append(train_loss)
        experiment_data["training"]["losses"]["val"].append(val_loss)
        experiment_data["training"]["epochs"].append(epoch)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
            )
    return model


def evaluate_throughput(
    method,
    model=None,
    n_episodes=10,
    max_steps=200,
    grid_size=20,
    n_agents=20,
    n_tasks=30,
):
    throughputs = []
    for _ in range(n_episodes):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks)
        for step in range(max_steps):
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
    return np.mean(throughputs), np.std(throughputs)


# Generate training data and train model
print("Generating training data...")
train_states, train_congestions = generate_training_data(n_samples=800)
val_states, val_congestions = generate_training_data(n_samples=200)
print(f"Training data shape: {train_states.shape}")

print("\nTraining congestion predictor...")
model = CongestionPredictor(grid_size=20, hidden_dim=64)
model = train_predictor(
    model, train_states, train_congestions, val_states, val_congestions, epochs=50
)

# Hyperparameter tuning for max_steps
max_steps_values = [100, 200, 300, 400, 500]
agent_counts = [20, 30, 40, 50]

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING: max_steps")
print("=" * 60)

for max_steps in max_steps_values:
    print(f"\n--- Testing max_steps = {max_steps} ---")
    experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"] = {
        "distance_based": {"throughput": [], "std": []},
        "congestion_aware": {"throughput": [], "std": []},
        "improvement": [],
        "n_agents": agent_counts,
    }

    for n_agents in agent_counts:
        dist_throughput, dist_std = evaluate_throughput(
            "distance", n_episodes=5, max_steps=max_steps, n_agents=n_agents
        )
        cata_throughput, cata_std = evaluate_throughput(
            "congestion_aware",
            model=model,
            n_episodes=5,
            max_steps=max_steps,
            n_agents=n_agents,
        )
        improvement = (
            (cata_throughput - dist_throughput) / dist_throughput * 100
            if dist_throughput > 0
            else 0
        )

        experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"][
            "distance_based"
        ]["throughput"].append(dist_throughput)
        experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"][
            "distance_based"
        ]["std"].append(dist_std)
        experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"][
            "congestion_aware"
        ]["throughput"].append(cata_throughput)
        experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"][
            "congestion_aware"
        ]["std"].append(cata_std)
        experiment_data["hyperparam_tuning"][f"max_steps_{max_steps}"][
            "improvement"
        ].append(improvement)

        print(
            f"  Agents={n_agents}: Dist={dist_throughput:.2f}±{dist_std:.2f}, CATA={cata_throughput:.2f}±{cata_std:.2f}, Imp={improvement:.1f}%"
        )

# Summary
print("\n" + "=" * 60)
print("SUMMARY: Average Improvement by max_steps")
print("=" * 60)
summary_data = {"max_steps": [], "avg_improvement": [], "avg_dist": [], "avg_cata": []}
for max_steps in max_steps_values:
    key = f"max_steps_{max_steps}"
    avg_imp = np.mean(experiment_data["hyperparam_tuning"][key]["improvement"])
    avg_dist = np.mean(
        experiment_data["hyperparam_tuning"][key]["distance_based"]["throughput"]
    )
    avg_cata = np.mean(
        experiment_data["hyperparam_tuning"][key]["congestion_aware"]["throughput"]
    )
    summary_data["max_steps"].append(max_steps)
    summary_data["avg_improvement"].append(avg_imp)
    summary_data["avg_dist"].append(avg_dist)
    summary_data["avg_cata"].append(avg_cata)
    print(
        f"max_steps={max_steps}: Avg Dist={avg_dist:.2f}, Avg CATA={avg_cata:.2f}, Avg Improvement={avg_imp:.1f}%"
    )

experiment_data["summary"] = summary_data

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
axes[0, 0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0, 0].plot(experiment_data["training"]["losses"]["val"], label="Validation")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Congestion Predictor Training")
axes[0, 0].legend()

# Throughput vs max_steps
axes[0, 1].plot(
    summary_data["max_steps"], summary_data["avg_dist"], "o-", label="Distance-based"
)
axes[0, 1].plot(
    summary_data["max_steps"], summary_data["avg_cata"], "s-", label="Congestion-aware"
)
axes[0, 1].set_xlabel("max_steps")
axes[0, 1].set_ylabel("Avg Throughput (tasks/min)")
axes[0, 1].set_title("Throughput vs max_steps")
axes[0, 1].legend()

# Improvement vs max_steps
axes[1, 0].bar(
    summary_data["max_steps"], summary_data["avg_improvement"], color="green", alpha=0.7
)
axes[1, 0].set_xlabel("max_steps")
axes[1, 0].set_ylabel("Avg Improvement (%)")
axes[1, 0].set_title("CATA Improvement over Distance-based")
axes[1, 0].axhline(y=0, color="r", linestyle="--")

# Throughput by agent count for each max_steps
colors = plt.cm.viridis(np.linspace(0, 1, len(max_steps_values)))
for idx, max_steps in enumerate(max_steps_values):
    key = f"max_steps_{max_steps}"
    axes[1, 1].plot(
        agent_counts,
        experiment_data["hyperparam_tuning"][key]["congestion_aware"]["throughput"],
        "o-",
        color=colors[idx],
        label=f"CATA (max_steps={max_steps})",
    )
axes[1, 1].set_xlabel("Number of Agents")
axes[1, 1].set_ylabel("Throughput (tasks/min)")
axes[1, 1].set_title("CATA Throughput by Agent Count")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "max_steps_tuning_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
