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

# Experiment data storage
experiment_data = {
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "evaluation": {
        "distance_based": {"throughput": []},
        "congestion_aware": {"throughput": []},
    },
}


def greedy_assignment(cost_matrix):
    """Greedy assignment as a replacement for Hungarian algorithm."""
    n_agents, n_tasks = cost_matrix.shape
    row_ind = []
    col_ind = []
    assigned_tasks = set()

    # Sort all agent-task pairs by cost
    pairs = []
    for i in range(n_agents):
        for j in range(n_tasks):
            pairs.append((cost_matrix[i, j], i, j))
    pairs.sort()

    assigned_agents = set()
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
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.reset()

    def reset(self):
        self.agent_positions = np.random.randint(0, self.grid_size, (self.n_agents, 2))
        self.agent_velocities = np.zeros((self.n_agents, 2))
        self.task_positions = np.random.randint(0, self.grid_size, (self.n_tasks, 2))
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self.agent_targets = [-1] * self.n_agents
        self.time_step = 0
        self.tasks_completed = 0
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

        inactive_tasks = np.where(~self.task_active)[0]
        for idx in inactive_tasks:
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
        x = torch.sigmoid(self.conv3(x))
        return x


def generate_training_data(n_samples=1000, grid_size=20, n_agents=20, n_tasks=30):
    states, congestions = [], []
    for _ in range(n_samples):
        sim = WarehouseSimulator(grid_size, n_agents, n_tasks)
        state = sim.get_state()
        active_tasks = np.where(sim.task_active)[0]
        assignment = [
            (i, active_tasks[i]) for i in range(min(n_agents, len(active_tasks)))
        ]
        sim.assign_tasks(assignment)
        congestion = sim.compute_congestion(lookahead=5)
        congestion = congestion / (congestion.max() + 1e-6)
        states.append(state)
        congestions.append(congestion)
    return np.array(states), np.array(congestions)


def distance_based_assignment(agent_positions, task_positions, task_active):
    n_agents = len(agent_positions)
    active_indices = np.where(task_active)[0]
    if len(active_indices) == 0:
        return []
    cost_matrix = np.zeros((n_agents, len(active_indices)))
    for i, agent_pos in enumerate(agent_positions):
        for j, task_idx in enumerate(active_indices):
            cost_matrix[i, j] = np.sum(np.abs(agent_pos - task_positions[task_idx]))
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
            path_congestion = 0
            current = agent_pos.copy()
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
            batch_states, batch_targets = train_states[idx], train_congestions[idx]
            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_targets)
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
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
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
                assignment = [(unassigned[a], t) for a, t in assignment]
                sim.assign_tasks(assignment)
            sim.step()
        throughputs.append(sim.tasks_completed / (max_steps / 60.0))
    return np.mean(throughputs), np.std(throughputs)


print("Generating training data...")
train_states, train_congestions = generate_training_data(n_samples=800)
val_states, val_congestions = generate_training_data(n_samples=200)
print(
    f"Training data shape: {train_states.shape}, Validation data shape: {val_states.shape}"
)

print("\nTraining congestion predictor...")
model = CongestionPredictor(grid_size=20, hidden_dim=64)
model = train_predictor(
    model, train_states, train_congestions, val_states, val_congestions, epochs=50
)

print("\nEvaluating methods...")
agent_counts = [20, 30, 40, 50]
results = {"distance": [], "congestion_aware": []}

for n_agents in agent_counts:
    print(f"\nTesting with {n_agents} agents...")
    dist_throughput, dist_std = evaluate_throughput(
        "distance", n_episodes=5, n_agents=n_agents
    )
    results["distance"].append((n_agents, dist_throughput, dist_std))
    experiment_data["evaluation"]["distance_based"]["throughput"].append(
        {"n_agents": n_agents, "mean": dist_throughput, "std": dist_std}
    )
    print(
        f"  Distance-based: throughput_tasks_per_minute = {dist_throughput:.2f} ± {dist_std:.2f}"
    )

    cata_throughput, cata_std = evaluate_throughput(
        "congestion_aware", model=model, n_episodes=5, n_agents=n_agents
    )
    results["congestion_aware"].append((n_agents, cata_throughput, cata_std))
    experiment_data["evaluation"]["congestion_aware"]["throughput"].append(
        {"n_agents": n_agents, "mean": cata_throughput, "std": cata_std}
    )
    print(
        f"  Congestion-aware: throughput_tasks_per_minute = {cata_throughput:.2f} ± {cata_std:.2f}"
    )

print("\n" + "=" * 50)
print("SUMMARY: Throughput (tasks per minute)")
print("=" * 50)
for i, n_agents in enumerate(agent_counts):
    dist_t, cata_t = results["distance"][i][1], results["congestion_aware"][i][1]
    improvement = (cata_t - dist_t) / dist_t * 100
    print(
        f"Agents={n_agents}: Distance={dist_t:.2f}, CATA={cata_t:.2f}, Improvement={improvement:.1f}%"
    )

final_dist = np.mean([r[1] for r in results["distance"]])
final_cata = np.mean([r[1] for r in results["congestion_aware"]])
print(f"\nOverall Average:")
print(f"  Distance-based throughput_tasks_per_minute: {final_dist:.2f}")
print(f"  Congestion-aware throughput_tasks_per_minute: {final_cata:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(experiment_data["training"]["losses"]["train"], label="Train")
axes[0].plot(experiment_data["training"]["losses"]["val"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Congestion Predictor Training")
axes[0].legend()

x = np.arange(len(agent_counts))
width = 0.35
axes[1].bar(
    x - width / 2,
    [r[1] for r in results["distance"]],
    width,
    yerr=[r[2] for r in results["distance"]],
    label="Distance-based",
    alpha=0.8,
)
axes[1].bar(
    x + width / 2,
    [r[1] for r in results["congestion_aware"]],
    width,
    yerr=[r[2] for r in results["congestion_aware"]],
    label="CATA",
    alpha=0.8,
)
axes[1].set_xlabel("Number of Agents")
axes[1].set_ylabel("Throughput (tasks/min)")
axes[1].set_title("Throughput Comparison")
axes[1].set_xticks(x)
axes[1].set_xticklabels(agent_counts)
axes[1].legend()

model.eval()
sample_state = torch.FloatTensor(val_states[0:1]).to(device)
with torch.no_grad():
    pred_congestion = model(sample_state).squeeze().cpu().numpy()
im = axes[2].imshow(pred_congestion, cmap="hot")
axes[2].set_title("Sample Predicted Congestion Heatmap")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cata_baseline_results.png"), dpi=150)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
