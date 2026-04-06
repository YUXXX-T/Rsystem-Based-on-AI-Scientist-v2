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
    "distance_based": {"throughput": [], "completion_times": [], "agent_counts": []},
    "congestion_aware_heuristic": {
        "throughput": [],
        "completion_times": [],
        "agent_counts": [],
    },
    "congestion_aware_learned": {
        "throughput": [],
        "completion_times": [],
        "agent_counts": [],
    },
    "predictor_training": {"losses": {"train": [], "val": []}, "epochs": []},
}


# Simple Warehouse Simulator
class WarehouseSimulator:
    def __init__(self, grid_size=20, num_agents=50, num_tasks=100):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.reset()

    def reset(self):
        # Random agent positions
        self.agent_positions = np.random.randint(
            0, self.grid_size, (self.num_agents, 2)
        )
        self.agent_targets = np.full((self.num_agents, 2), -1)  # -1 means no task
        self.agent_busy = np.zeros(self.num_agents, dtype=bool)

        # Generate tasks at random locations
        self.tasks = np.random.randint(0, self.grid_size, (self.num_tasks, 2))
        self.task_available = np.ones(self.num_tasks, dtype=bool)

        self.time_step = 0
        self.completed_tasks = 0
        self.task_completion_times = []
        self.task_start_times = {}

    def get_congestion_heatmap(self):
        """Compute actual congestion based on agent density"""
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for pos in self.agent_positions:
            x, y = pos
            # Add density in 3x3 neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        dist = abs(dx) + abs(dy)
                        heatmap[nx, ny] += 1.0 / (1 + dist)
        return heatmap / (heatmap.max() + 1e-6)

    def get_state_tensor(self):
        """Create state tensor for learned predictor: agent positions + task positions"""
        state = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        # Channel 0: agent positions
        for pos in self.agent_positions:
            state[0, pos[0], pos[1]] += 1.0
        # Channel 1: available task positions
        for i, pos in enumerate(self.tasks):
            if self.task_available[i]:
                state[1, pos[0], pos[1]] += 1.0
        # Normalize
        state[0] = state[0] / (state[0].max() + 1e-6)
        state[1] = state[1] / (state[1].max() + 1e-6)
        return state

    def compute_distance_matrix(self):
        """Compute Manhattan distance from each agent to each task"""
        dist_matrix = np.full((self.num_agents, self.num_tasks), np.inf)
        for i, agent_pos in enumerate(self.agent_positions):
            if not self.agent_busy[i]:
                for j, task_pos in enumerate(self.tasks):
                    if self.task_available[j]:
                        dist_matrix[i, j] = abs(agent_pos[0] - task_pos[0]) + abs(
                            agent_pos[1] - task_pos[1]
                        )
        return dist_matrix

    def compute_congestion_cost(self, agent_pos, task_pos, congestion_heatmap):
        """Estimate congestion cost along path"""
        x0, y0 = agent_pos
        x1, y1 = task_pos

        # Sample points along Manhattan path
        cost = 0.0
        steps = abs(x1 - x0) + abs(y1 - y0) + 1
        if steps <= 1:
            return cost

        # Simple path: first move in x, then in y
        path = []
        cx, cy = x0, y0
        while cx != x1:
            cx += 1 if x1 > cx else -1
            path.append((cx, cy))
        while cy != y1:
            cy += 1 if y1 > cy else -1
            path.append((cx, cy))

        for px, py in path:
            cost += congestion_heatmap[px, py]

        return cost / len(path) if path else 0.0

    def greedy_assignment(self, cost_matrix):
        """Greedy assignment when scipy not available"""
        assignments = []
        used_agents = set()
        used_tasks = set()

        # Flatten and sort by cost
        costs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < np.inf:
                    costs.append((cost_matrix[i, j], i, j))
        costs.sort()

        for cost, agent_idx, task_idx in costs:
            if agent_idx not in used_agents and task_idx not in used_tasks:
                assignments.append((agent_idx, task_idx))
                used_agents.add(agent_idx)
                used_tasks.add(task_idx)

        return assignments

    def allocate_tasks(
        self, method="distance", congestion_heatmap=None, congestion_weight=2.0
    ):
        """Allocate available tasks to free agents"""
        dist_matrix = self.compute_distance_matrix()

        if method == "distance":
            cost_matrix = dist_matrix.copy()
        elif method in ["congestion_heuristic", "congestion_learned"]:
            if congestion_heatmap is None:
                congestion_heatmap = self.get_congestion_heatmap()
            cost_matrix = dist_matrix.copy()
            for i in range(self.num_agents):
                if not self.agent_busy[i]:
                    for j in range(self.num_tasks):
                        if self.task_available[j]:
                            cong_cost = self.compute_congestion_cost(
                                self.agent_positions[i],
                                self.tasks[j],
                                congestion_heatmap,
                            )
                            cost_matrix[i, j] += (
                                congestion_weight * cong_cost * dist_matrix[i, j]
                            )
        else:
            cost_matrix = dist_matrix.copy()

        assignments = self.greedy_assignment(cost_matrix)

        for agent_idx, task_idx in assignments:
            self.agent_targets[agent_idx] = self.tasks[task_idx]
            self.agent_busy[agent_idx] = True
            self.task_available[task_idx] = False
            self.task_start_times[task_idx] = self.time_step

    def step(self):
        """Simulate one time step"""
        self.time_step += 1

        # Move agents toward their targets
        for i in range(self.num_agents):
            if self.agent_busy[i]:
                target = self.agent_targets[i]
                pos = self.agent_positions[i]

                # Simple movement: one step toward target
                if pos[0] < target[0]:
                    self.agent_positions[i, 0] += 1
                elif pos[0] > target[0]:
                    self.agent_positions[i, 0] -= 1
                elif pos[1] < target[1]:
                    self.agent_positions[i, 1] += 1
                elif pos[1] > target[1]:
                    self.agent_positions[i, 1] -= 1

                # Check if reached target
                if np.array_equal(self.agent_positions[i], target):
                    self.completed_tasks += 1
                    task_idx = np.where((self.tasks == target).all(axis=1))[0]
                    if len(task_idx) > 0:
                        completion_time = self.time_step - self.task_start_times.get(
                            task_idx[0], self.time_step
                        )
                        self.task_completion_times.append(completion_time)
                    self.agent_busy[i] = False
                    self.agent_targets[i] = [-1, -1]

    def run_episode(self, max_steps=500, method="distance", congestion_predictor=None):
        """Run a full episode"""
        self.reset()

        while self.time_step < max_steps and self.completed_tasks < self.num_tasks:
            # Get congestion heatmap
            if method == "congestion_learned" and congestion_predictor is not None:
                state = (
                    torch.FloatTensor(self.get_state_tensor()).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    congestion_heatmap = (
                        congestion_predictor(state).squeeze().cpu().numpy()
                    )
            elif method == "congestion_heuristic":
                congestion_heatmap = self.get_congestion_heatmap()
            else:
                congestion_heatmap = None

            # Allocate tasks every few steps
            if self.time_step % 5 == 0:
                self.allocate_tasks(
                    method=method, congestion_heatmap=congestion_heatmap
                )

            self.step()

        # Compute throughput (tasks per minute, assuming 1 step = 1 second)
        simulation_minutes = self.time_step / 60.0
        throughput = self.completed_tasks / max(simulation_minutes, 0.1)
        avg_completion_time = (
            np.mean(self.task_completion_times) if self.task_completion_times else 0
        )

        return {
            "throughput": throughput,
            "completed_tasks": self.completed_tasks,
            "avg_completion_time": avg_completion_time,
            "total_steps": self.time_step,
        }


# Simple CNN for congestion prediction
class CongestionPredictor(nn.Module):
    def __init__(self, grid_size=20):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x


# Generate training data for congestion predictor
def generate_training_data(num_samples=500, grid_size=20, num_agents_range=(20, 80)):
    states = []
    heatmaps = []

    for _ in range(num_samples):
        num_agents = np.random.randint(*num_agents_range)
        sim = WarehouseSimulator(
            grid_size=grid_size, num_agents=num_agents, num_tasks=100
        )

        # Run a few steps to get interesting states
        for _ in range(np.random.randint(10, 50)):
            sim.allocate_tasks(method="distance")
            sim.step()

        states.append(sim.get_state_tensor())
        heatmaps.append(sim.get_congestion_heatmap())

    return np.array(states), np.array(heatmaps)


# Train congestion predictor
def train_predictor(
    model, train_states, train_heatmaps, val_states, val_heatmaps, epochs=50
):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_states = torch.FloatTensor(train_states).to(device)
    train_heatmaps = torch.FloatTensor(train_heatmaps).unsqueeze(1).to(device)
    val_states = torch.FloatTensor(val_states).to(device)
    val_heatmaps = torch.FloatTensor(val_heatmaps).unsqueeze(1).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Mini-batch training
        batch_size = 32
        indices = np.random.permutation(len(train_states))[:batch_size]
        batch_x = train_states[indices]
        batch_y = train_heatmaps[indices]

        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_states)
            val_loss = criterion(val_pred, val_heatmaps).item()

        experiment_data["predictor_training"]["losses"]["train"].append(loss.item())
        experiment_data["predictor_training"]["losses"]["val"].append(val_loss)
        experiment_data["predictor_training"]["epochs"].append(epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {loss.item():.4f}, validation_loss = {val_loss:.4f}"
            )

    return model


# Main experiment
print("=" * 60)
print("CATA Baseline Experiment: Congestion-Aware Task Allocation")
print("=" * 60)

# Generate training data
print("\nGenerating training data for congestion predictor...")
train_states, train_heatmaps = generate_training_data(num_samples=400, grid_size=20)
val_states, val_heatmaps = generate_training_data(num_samples=100, grid_size=20)
print(f"Training samples: {len(train_states)}, Validation samples: {len(val_states)}")

# Train predictor
print("\nTraining congestion predictor...")
predictor = CongestionPredictor(grid_size=20).to(device)
predictor = train_predictor(
    predictor, train_states, train_heatmaps, val_states, val_heatmaps, epochs=50
)

# Run evaluation experiments
print("\n" + "=" * 60)
print("Running Evaluation Experiments")
print("=" * 60)

agent_counts = [20, 40, 60, 80]
num_trials = 5
methods = ["distance", "congestion_heuristic", "congestion_learned"]

results = {method: {"throughputs": [], "completion_times": []} for method in methods}

for num_agents in agent_counts:
    print(f"\n--- Testing with {num_agents} agents ---")

    for method in methods:
        throughputs = []
        completion_times = []

        for trial in range(num_trials):
            sim = WarehouseSimulator(grid_size=20, num_agents=num_agents, num_tasks=100)

            if method == "congestion_learned":
                result = sim.run_episode(
                    max_steps=500, method=method, congestion_predictor=predictor
                )
            else:
                result = sim.run_episode(max_steps=500, method=method)

            throughputs.append(result["throughput"])
            completion_times.append(result["avg_completion_time"])

        avg_throughput = np.mean(throughputs)
        avg_completion = np.mean(completion_times)

        results[method]["throughputs"].append(avg_throughput)
        results[method]["completion_times"].append(avg_completion)

        # Store in experiment data
        method_key = (
            f'congestion_aware_{method.split("_")[-1]}'
            if "congestion" in method
            else "distance_based"
        )
        if method == "distance":
            method_key = "distance_based"
        elif method == "congestion_heuristic":
            method_key = "congestion_aware_heuristic"
        else:
            method_key = "congestion_aware_learned"

        experiment_data[method_key]["throughput"].append(avg_throughput)
        experiment_data[method_key]["completion_times"].append(avg_completion)
        experiment_data[method_key]["agent_counts"].append(num_agents)

        print(
            f"  {method:25s}: throughput = {avg_throughput:.2f} tasks/min, avg_completion = {avg_completion:.2f} steps"
        )

# Print final summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print("\nThroughput (tasks per minute) by agent count:")
print(
    f"{'Agents':<10} {'Distance':<15} {'Heuristic':<15} {'Learned':<15} {'Improvement':<15}"
)
print("-" * 70)

for i, num_agents in enumerate(agent_counts):
    dist_tp = results["distance"]["throughputs"][i]
    heur_tp = results["congestion_heuristic"]["throughputs"][i]
    learn_tp = results["congestion_learned"]["throughputs"][i]
    best_improvement = max(heur_tp, learn_tp) / dist_tp - 1 if dist_tp > 0 else 0
    print(
        f"{num_agents:<10} {dist_tp:<15.2f} {heur_tp:<15.2f} {learn_tp:<15.2f} {best_improvement*100:<15.1f}%"
    )

# Primary metric: average throughput improvement
avg_dist_throughput = np.mean(results["distance"]["throughputs"])
avg_heur_throughput = np.mean(results["congestion_heuristic"]["throughputs"])
avg_learn_throughput = np.mean(results["congestion_learned"]["throughputs"])

print(f"\nAverage throughput across all agent counts:")
print(f"  Distance-based:         {avg_dist_throughput:.2f} tasks/min")
print(f"  Congestion (heuristic): {avg_heur_throughput:.2f} tasks/min")
print(f"  Congestion (learned):   {avg_learn_throughput:.2f} tasks/min")

best_cata_throughput = max(avg_heur_throughput, avg_learn_throughput)
improvement = (
    (best_cata_throughput / avg_dist_throughput - 1) * 100
    if avg_dist_throughput > 0
    else 0
)

print(
    f"\n*** PRIMARY METRIC: throughput_tasks_per_minute = {best_cata_throughput:.2f} ***"
)
print(f"*** Improvement over baseline: {improvement:.1f}% ***")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Throughput comparison
ax1 = axes[0]
x = np.arange(len(agent_counts))
width = 0.25
ax1.bar(
    x - width,
    results["distance"]["throughputs"],
    width,
    label="Distance-based",
    color="blue",
    alpha=0.7,
)
ax1.bar(
    x,
    results["congestion_heuristic"]["throughputs"],
    width,
    label="Heuristic CATA",
    color="green",
    alpha=0.7,
)
ax1.bar(
    x + width,
    results["congestion_learned"]["throughputs"],
    width,
    label="Learned CATA",
    color="red",
    alpha=0.7,
)
ax1.set_xlabel("Number of Agents")
ax1.set_ylabel("Throughput (tasks/min)")
ax1.set_title("Throughput Comparison")
ax1.set_xticks(x)
ax1.set_xticklabels(agent_counts)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Training loss
ax2 = axes[1]
ax2.plot(
    experiment_data["predictor_training"]["epochs"],
    experiment_data["predictor_training"]["losses"]["train"],
    label="Train Loss",
    color="blue",
)
ax2.plot(
    experiment_data["predictor_training"]["epochs"],
    experiment_data["predictor_training"]["losses"]["val"],
    label="Val Loss",
    color="red",
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_title("Congestion Predictor Training")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Sample congestion heatmap
ax3 = axes[2]
sim = WarehouseSimulator(grid_size=20, num_agents=50, num_tasks=100)
for _ in range(30):
    sim.allocate_tasks(method="distance")
    sim.step()
heatmap = sim.get_congestion_heatmap()
im = ax3.imshow(heatmap, cmap="hot", interpolation="nearest")
ax3.scatter(
    sim.agent_positions[:, 1],
    sim.agent_positions[:, 0],
    c="cyan",
    s=20,
    marker="o",
    label="Agents",
)
ax3.set_title("Sample Congestion Heatmap")
ax3.legend(loc="upper right")
plt.colorbar(im, ax=ax3, label="Congestion")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cata_baseline_results.png"), dpi=150)
plt.close()

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Save results summary
results_summary = {
    "agent_counts": agent_counts,
    "distance_throughputs": results["distance"]["throughputs"],
    "heuristic_throughputs": results["congestion_heuristic"]["throughputs"],
    "learned_throughputs": results["congestion_learned"]["throughputs"],
    "primary_metric_throughput": best_cata_throughput,
    "improvement_percent": improvement,
}
np.save(os.path.join(working_dir, "results_summary.npy"), results_summary)

print(f"\nResults saved to {working_dir}")
print("Experiment completed successfully!")
