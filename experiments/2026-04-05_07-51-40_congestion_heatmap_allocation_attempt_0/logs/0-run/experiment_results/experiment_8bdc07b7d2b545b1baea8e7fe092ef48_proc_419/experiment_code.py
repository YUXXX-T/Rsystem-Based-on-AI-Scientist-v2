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

experiment_data = {"lookahead_tuning": {}}


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
                target_pos = self.task_positions[target_idx]
                path = self._get_path(agent_pos, target_pos)
                for step, pos in enumerate(path[:lookahead]):
                    congestion[pos[0], pos[1]] += 1.0 / (step + 1)
        return congestion

    def _get_path(self, start, end):
        path = [tuple(start)]
        current = np.array(start)
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


def generate_training_data(
    n_samples=1000, grid_size=20, n_agents=20, n_tasks=30, lookahead=5
):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks)
        state = sim.get_state()
        active_tasks = np.where(sim.task_active)[0]
        assignment = [
            (i, active_tasks[i]) for i in range(min(n_agents, len(active_tasks)))
        ]
        sim.assign_tasks(assignment)
        congestion = sim.compute_congestion(lookahead=lookahead)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(state)
        congestions.append(congestion)
    return np.array(states), np.array(congestions)


def distance_based_assignment(agent_positions, task_positions, task_active):
    n_agents = len(agent_positions)
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
    n_agents = len(agent_positions)
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    cost_matrix = np.zeros((n_agents, len(active_indices)))
    grid_size = congestion_map.shape[0]
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
    train_losses, val_losses = [], []

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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
    return model, train_losses, val_losses


def evaluate_throughput(
    method,
    model=None,
    n_episodes=5,
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
                assignment = [(unassigned[a], t) for a, t in assignment]
                sim.assign_tasks(assignment)
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
    return np.mean(throughputs), np.std(throughputs)


# Hyperparameter tuning for lookahead
lookahead_values = [2, 5, 8, 12]
agent_counts = [20, 30, 40, 50]
best_lookahead, best_avg_throughput = None, -1

print("=" * 60)
print("HYPERPARAMETER TUNING: Lookahead Parameter")
print("=" * 60)

for lookahead in lookahead_values:
    print(f"\n{'='*50}")
    print(f"Testing lookahead = {lookahead}")
    print(f"{'='*50}")

    experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "evaluation": {"distance_based": [], "congestion_aware": []},
        "lookahead_value": lookahead,
    }

    print(f"Generating training data with lookahead={lookahead}...")
    train_states, train_congestions = generate_training_data(
        n_samples=600, lookahead=lookahead
    )
    val_states, val_congestions = generate_training_data(
        n_samples=150, lookahead=lookahead
    )

    print(f"Training congestion predictor...")
    model = CongestionPredictor(grid_size=20, hidden_dim=64)
    model, train_losses, val_losses = train_predictor(
        model, train_states, train_congestions, val_states, val_congestions, epochs=30
    )

    experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"]["losses"][
        "train"
    ] = train_losses
    experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"]["losses"][
        "val"
    ] = val_losses

    print(f"\nEvaluating with lookahead={lookahead}...")
    throughputs_cata = []

    for n_agents in agent_counts:
        dist_throughput, dist_std = evaluate_throughput(
            "distance", n_episodes=3, n_agents=n_agents
        )
        cata_throughput, cata_std = evaluate_throughput(
            "congestion_aware", model=model, n_episodes=3, n_agents=n_agents
        )

        experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"]["evaluation"][
            "distance_based"
        ].append({"n_agents": n_agents, "mean": dist_throughput, "std": dist_std})
        experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"]["evaluation"][
            "congestion_aware"
        ].append({"n_agents": n_agents, "mean": cata_throughput, "std": cata_std})
        throughputs_cata.append(cata_throughput)

        improvement = (cata_throughput - dist_throughput) / dist_throughput * 100
        print(
            f"  Agents={n_agents}: Distance={dist_throughput:.2f}, CATA={cata_throughput:.2f}, Improvement={improvement:.1f}%"
        )

    avg_throughput = np.mean(throughputs_cata)
    experiment_data["lookahead_tuning"][f"lookahead_{lookahead}"][
        "avg_throughput"
    ] = avg_throughput

    if avg_throughput > best_avg_throughput:
        best_avg_throughput = avg_throughput
        best_lookahead = lookahead

    print(f"  Average CATA throughput: {avg_throughput:.2f}")

# Summary
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)
print(f"{'Lookahead':<12} {'Avg Throughput':<18} {'Final Val Loss':<15}")
print("-" * 45)
for lookahead in lookahead_values:
    key = f"lookahead_{lookahead}"
    avg_tp = experiment_data["lookahead_tuning"][key]["avg_throughput"]
    final_val_loss = experiment_data["lookahead_tuning"][key]["losses"]["val"][-1]
    marker = " <-- BEST" if lookahead == best_lookahead else ""
    print(f"{lookahead:<12} {avg_tp:<18.2f} {final_val_loss:<15.4f}{marker}")

print(
    f"\nBest lookahead value: {best_lookahead} with avg throughput: {best_avg_throughput:.2f}"
)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training losses for all lookahead values
for lookahead in lookahead_values:
    key = f"lookahead_{lookahead}"
    axes[0, 0].plot(
        experiment_data["lookahead_tuning"][key]["losses"]["train"],
        label=f"LA={lookahead}",
    )
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Training Loss")
axes[0, 0].set_title("Training Loss vs Lookahead")
axes[0, 0].legend()

# Plot 2: Validation losses for all lookahead values
for lookahead in lookahead_values:
    key = f"lookahead_{lookahead}"
    axes[0, 1].plot(
        experiment_data["lookahead_tuning"][key]["losses"]["val"],
        label=f"LA={lookahead}",
    )
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Validation Loss")
axes[0, 1].set_title("Validation Loss vs Lookahead")
axes[0, 1].legend()

# Plot 3: Throughput comparison across agent counts
x = np.arange(len(agent_counts))
width = 0.2
for idx, lookahead in enumerate(lookahead_values):
    key = f"lookahead_{lookahead}"
    throughputs = [
        e["mean"]
        for e in experiment_data["lookahead_tuning"][key]["evaluation"][
            "congestion_aware"
        ]
    ]
    axes[1, 0].bar(
        x + idx * width - 1.5 * width, throughputs, width, label=f"LA={lookahead}"
    )
axes[1, 0].set_xlabel("Number of Agents")
axes[1, 0].set_ylabel("Throughput (tasks/min)")
axes[1, 0].set_title("CATA Throughput by Lookahead")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(agent_counts)
axes[1, 0].legend()

# Plot 4: Average throughput per lookahead
avg_throughputs = [
    experiment_data["lookahead_tuning"][f"lookahead_{la}"]["avg_throughput"]
    for la in lookahead_values
]
colors = ["green" if la == best_lookahead else "steelblue" for la in lookahead_values]
axes[1, 1].bar([str(la) for la in lookahead_values], avg_throughputs, color=colors)
axes[1, 1].set_xlabel("Lookahead Value")
axes[1, 1].set_ylabel("Avg Throughput (tasks/min)")
axes[1, 1].set_title("Average Throughput vs Lookahead (Best in Green)")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "lookahead_tuning_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
