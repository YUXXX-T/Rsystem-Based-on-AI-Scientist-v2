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
    "distance_based": {"throughput": [], "completion_times": []},
    "heuristic_congestion": {"throughput": [], "completion_times": []},
    "learned_congestion": {"throughput": [], "completion_times": []},
    "training": {"losses": {"train": [], "val": []}, "epochs": []},
    "agent_counts": [],
}


# --- Warehouse Simulator ---
class WarehouseSimulator:
    def __init__(self, grid_size=20, num_agents=30, num_tasks=50):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.reset()

    def reset(self):
        # Random agent positions
        self.agent_positions = np.random.randint(
            0, self.grid_size, size=(self.num_agents, 2)
        )
        self.agent_targets = np.full((self.num_agents, 2), -1)
        self.agent_assigned = np.zeros(self.num_agents, dtype=bool)

        # Random task positions (clustered to create congestion scenarios)
        self.tasks = []
        # Create some clustered tasks
        num_clusters = 3
        for _ in range(num_clusters):
            center = np.random.randint(2, self.grid_size - 2, size=2)
            cluster_size = self.num_tasks // num_clusters
            for _ in range(cluster_size):
                offset = np.random.randint(-3, 4, size=2)
                task_pos = np.clip(center + offset, 0, self.grid_size - 1)
                self.tasks.append(task_pos.tolist())
        # Add remaining tasks randomly
        while len(self.tasks) < self.num_tasks:
            self.tasks.append(np.random.randint(0, self.grid_size, size=2).tolist())

        self.tasks = [list(t) for t in self.tasks]
        self.task_assigned = np.zeros(len(self.tasks), dtype=bool)
        self.completed_tasks = 0
        self.timestep = 0
        self.total_completion_time = 0

    def get_congestion_heatmap(self, grid_resolution=5):
        """Compute actual congestion heatmap based on agent positions"""
        cell_size = self.grid_size // grid_resolution
        heatmap = np.zeros((grid_resolution, grid_resolution))
        for pos in self.agent_positions:
            cx = min(int(pos[0] // cell_size), grid_resolution - 1)
            cy = min(int(pos[1] // cell_size), grid_resolution - 1)
            heatmap[cx, cy] += 1
        # Normalize
        heatmap = heatmap / (self.num_agents + 1e-6)
        return heatmap

    def get_state_tensor(self, grid_resolution=5):
        """Get state as tensor for neural network input"""
        cell_size = self.grid_size // grid_resolution

        # Agent density channel
        agent_map = np.zeros((grid_resolution, grid_resolution))
        for pos in self.agent_positions:
            cx = min(int(pos[0] // cell_size), grid_resolution - 1)
            cy = min(int(pos[1] // cell_size), grid_resolution - 1)
            agent_map[cx, cy] += 1

        # Task density channel
        task_map = np.zeros((grid_resolution, grid_resolution))
        for i, task in enumerate(self.tasks):
            if not self.task_assigned[i]:
                cx = min(int(task[0] // cell_size), grid_resolution - 1)
                cy = min(int(task[1] // cell_size), grid_resolution - 1)
                task_map[cx, cy] += 1

        # Normalize
        agent_map = agent_map / (self.num_agents + 1e-6)
        task_map = task_map / (len(self.tasks) + 1e-6)

        state = np.stack([agent_map, task_map], axis=0)
        return state

    def step(self):
        """Move agents toward their targets"""
        self.timestep += 1

        for i in range(self.num_agents):
            if self.agent_assigned[i]:
                target = self.agent_targets[i]
                pos = self.agent_positions[i]

                # Move one step toward target (simple pathfinding)
                dx = np.sign(target[0] - pos[0])
                dy = np.sign(target[1] - pos[1])

                # Randomly choose x or y movement to avoid perfect synchronization
                if np.random.random() < 0.5 and dx != 0:
                    self.agent_positions[i][0] += dx
                elif dy != 0:
                    self.agent_positions[i][1] += dy
                elif dx != 0:
                    self.agent_positions[i][0] += dx

                # Check if reached target
                if np.array_equal(self.agent_positions[i], target):
                    self.completed_tasks += 1
                    self.agent_assigned[i] = False
                    # Mark task as done
                    for j, task in enumerate(self.tasks):
                        if np.array_equal(task, target) and self.task_assigned[j]:
                            self.task_assigned[j] = True  # Keep marked
                            break

    def get_unassigned_agents(self):
        return np.where(~self.agent_assigned)[0]

    def get_unassigned_tasks(self):
        return [i for i, assigned in enumerate(self.task_assigned) if not assigned]


# --- Assignment Strategies ---
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def greedy_assignment(cost_matrix):
    """Simple greedy assignment to avoid scipy dependency"""
    n_agents, n_tasks = cost_matrix.shape
    agent_assignment = {}
    assigned_tasks = set()

    # Flatten and sort by cost
    costs = []
    for i in range(n_agents):
        for j in range(n_tasks):
            costs.append((cost_matrix[i, j], i, j))
    costs.sort()

    for cost, agent, task in costs:
        if agent not in agent_assignment and task not in assigned_tasks:
            agent_assignment[agent] = task
            assigned_tasks.add(task)
        if len(agent_assignment) == min(n_agents, n_tasks):
            break

    return agent_assignment


def distance_based_allocation(sim):
    """Pure distance-based task allocation"""
    unassigned_agents = sim.get_unassigned_agents()
    unassigned_tasks = sim.get_unassigned_tasks()

    if len(unassigned_agents) == 0 or len(unassigned_tasks) == 0:
        return

    # Build cost matrix
    n_agents = len(unassigned_agents)
    n_tasks = len(unassigned_tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))

    for i, agent_idx in enumerate(unassigned_agents):
        for j, task_idx in enumerate(unassigned_tasks):
            cost_matrix[i, j] = manhattan_distance(
                sim.agent_positions[agent_idx], sim.tasks[task_idx]
            )

    # Greedy assignment
    assignment = greedy_assignment(cost_matrix)

    for agent_local, task_local in assignment.items():
        agent_idx = unassigned_agents[agent_local]
        task_idx = unassigned_tasks[task_local]
        sim.agent_targets[agent_idx] = np.array(sim.tasks[task_idx])
        sim.agent_assigned[agent_idx] = True
        sim.task_assigned[task_idx] = True


def heuristic_congestion_allocation(sim, congestion_weight=2.0):
    """Congestion-aware allocation using heuristic congestion estimation"""
    unassigned_agents = sim.get_unassigned_agents()
    unassigned_tasks = sim.get_unassigned_tasks()

    if len(unassigned_agents) == 0 or len(unassigned_tasks) == 0:
        return

    # Get congestion heatmap
    heatmap = sim.get_congestion_heatmap(grid_resolution=5)
    cell_size = sim.grid_size // 5

    # Build cost matrix with congestion penalty
    n_agents = len(unassigned_agents)
    n_tasks = len(unassigned_tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))

    for i, agent_idx in enumerate(unassigned_agents):
        for j, task_idx in enumerate(unassigned_tasks):
            # Base distance cost
            dist = manhattan_distance(
                sim.agent_positions[agent_idx], sim.tasks[task_idx]
            )

            # Congestion penalty along path (simplified: just at target)
            task_pos = sim.tasks[task_idx]
            cx = min(int(task_pos[0] // cell_size), 4)
            cy = min(int(task_pos[1] // cell_size), 4)
            congestion_penalty = heatmap[cx, cy] * congestion_weight * sim.grid_size

            cost_matrix[i, j] = dist + congestion_penalty

    # Greedy assignment
    assignment = greedy_assignment(cost_matrix)

    for agent_local, task_local in assignment.items():
        agent_idx = unassigned_agents[agent_local]
        task_idx = unassigned_tasks[task_local]
        sim.agent_targets[agent_idx] = np.array(sim.tasks[task_idx])
        sim.agent_assigned[agent_idx] = True
        sim.task_assigned[task_idx] = True


# --- Congestion Prediction CNN ---
class CongestionPredictor(nn.Module):
    def __init__(self, grid_resolution=5):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x.squeeze(1)


def learned_congestion_allocation(sim, model, congestion_weight=2.0):
    """Congestion-aware allocation using learned congestion prediction"""
    unassigned_agents = sim.get_unassigned_agents()
    unassigned_tasks = sim.get_unassigned_tasks()

    if len(unassigned_agents) == 0 or len(unassigned_tasks) == 0:
        return

    # Get predicted congestion heatmap
    state = sim.get_state_tensor(grid_resolution=5)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_heatmap = model(state_tensor).cpu().numpy()[0]

    cell_size = sim.grid_size // 5

    # Build cost matrix with predicted congestion penalty
    n_agents = len(unassigned_agents)
    n_tasks = len(unassigned_tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))

    for i, agent_idx in enumerate(unassigned_agents):
        for j, task_idx in enumerate(unassigned_tasks):
            dist = manhattan_distance(
                sim.agent_positions[agent_idx], sim.tasks[task_idx]
            )

            task_pos = sim.tasks[task_idx]
            cx = min(int(task_pos[0] // cell_size), 4)
            cy = min(int(task_pos[1] // cell_size), 4)
            congestion_penalty = (
                predicted_heatmap[cx, cy] * congestion_weight * sim.grid_size
            )

            cost_matrix[i, j] = dist + congestion_penalty

    assignment = greedy_assignment(cost_matrix)

    for agent_local, task_local in assignment.items():
        agent_idx = unassigned_agents[agent_local]
        task_idx = unassigned_tasks[task_local]
        sim.agent_targets[agent_idx] = np.array(sim.tasks[task_idx])
        sim.agent_assigned[agent_idx] = True
        sim.task_assigned[task_idx] = True


# --- Data Collection for Training ---
def collect_training_data(num_episodes=100, steps_per_episode=50):
    """Collect state-congestion pairs for supervised learning"""
    print("Collecting training data...")
    states = []
    congestion_labels = []

    for ep in range(num_episodes):
        sim = WarehouseSimulator(grid_size=20, num_agents=30, num_tasks=50)

        for step in range(steps_per_episode):
            # Record state before allocation
            state = sim.get_state_tensor(grid_resolution=5)

            # Allocate using distance-based (baseline)
            distance_based_allocation(sim)

            # Move agents
            sim.step()

            # Record congestion after movement (T+1 label)
            congestion = sim.get_congestion_heatmap(grid_resolution=5)

            states.append(state)
            congestion_labels.append(congestion)

    return np.array(states), np.array(congestion_labels)


# --- Training ---
def train_congestion_predictor(states, labels, epochs=50, batch_size=32):
    """Train the congestion prediction model"""
    print("Training congestion predictor...")

    # Split data
    n_samples = len(states)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_states = torch.FloatTensor(states[train_idx]).to(device)
    train_labels = torch.FloatTensor(labels[train_idx]).to(device)
    val_states = torch.FloatTensor(states[val_idx]).to(device)
    val_labels = torch.FloatTensor(labels[val_idx]).to(device)

    model = CongestionPredictor(grid_resolution=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(train_states))
        train_loss = 0
        n_batches = 0

        for i in range(0, len(train_states), batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_states = train_states[batch_idx]
            batch_labels = train_labels[batch_idx]

            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_states)
            val_loss = criterion(val_predictions, val_labels).item()

        experiment_data["training"]["losses"]["train"].append(train_loss)
        experiment_data["training"]["losses"]["val"].append(val_loss)
        experiment_data["training"]["epochs"].append(epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )

    return model


# --- Evaluation ---
def run_evaluation(allocation_fn, sim, max_steps=200, model=None):
    """Run simulation and measure throughput"""
    sim.reset()

    for step in range(max_steps):
        if model is not None:
            allocation_fn(sim, model)
        else:
            allocation_fn(sim)
        sim.step()

        # Check if all tasks done
        if sim.completed_tasks >= len(sim.tasks):
            break

    # Calculate throughput (tasks per minute, assuming 1 step = 1 second)
    time_minutes = sim.timestep / 60.0
    throughput = sim.completed_tasks / max(time_minutes, 1e-6)

    return throughput, sim.completed_tasks, sim.timestep


def evaluate_all_methods(model, agent_counts=[20, 30, 50, 70, 100]):
    """Compare all allocation methods across different agent densities"""
    results = defaultdict(lambda: defaultdict(list))

    for num_agents in agent_counts:
        print(f"\nEvaluating with {num_agents} agents...")
        experiment_data["agent_counts"].append(num_agents)

        num_tasks = int(num_agents * 1.5)

        # Multiple trials for each configuration
        for trial in range(5):
            # Distance-based
            sim = WarehouseSimulator(
                grid_size=20, num_agents=num_agents, num_tasks=num_tasks
            )
            throughput, completed, steps = run_evaluation(
                distance_based_allocation, sim
            )
            results["distance_based"][num_agents].append(throughput)

            # Heuristic congestion-aware
            sim = WarehouseSimulator(
                grid_size=20, num_agents=num_agents, num_tasks=num_tasks
            )
            throughput, completed, steps = run_evaluation(
                heuristic_congestion_allocation, sim
            )
            results["heuristic_congestion"][num_agents].append(throughput)

            # Learned congestion-aware
            sim = WarehouseSimulator(
                grid_size=20, num_agents=num_agents, num_tasks=num_tasks
            )
            throughput, completed, steps = run_evaluation(
                learned_congestion_allocation, sim, model=model
            )
            results["learned_congestion"][num_agents].append(throughput)

    return results


# --- Main Execution ---
print("=" * 60)
print("CATA: Congestion-Aware Task Allocation Experiment")
print("=" * 60)

# Collect training data
states, labels = collect_training_data(num_episodes=100, steps_per_episode=50)
print(f"Collected {len(states)} training samples")

# Train model
model = train_congestion_predictor(states, labels, epochs=50, batch_size=32)

# Evaluate all methods
print("\n" + "=" * 60)
print("Running evaluation across different agent densities...")
print("=" * 60)

agent_counts = [20, 30, 50, 70, 100]
results = evaluate_all_methods(model, agent_counts)

# Print results
print("\n" + "=" * 60)
print("RESULTS: Throughput (tasks per minute)")
print("=" * 60)

print(f"\n{'Agents':<10} {'Distance':<15} {'Heuristic':<15} {'Learned':<15}")
print("-" * 55)

for num_agents in agent_counts:
    dist_mean = np.mean(results["distance_based"][num_agents])
    heur_mean = np.mean(results["heuristic_congestion"][num_agents])
    learn_mean = np.mean(results["learned_congestion"][num_agents])

    experiment_data["distance_based"]["throughput"].append(dist_mean)
    experiment_data["heuristic_congestion"]["throughput"].append(heur_mean)
    experiment_data["learned_congestion"]["throughput"].append(learn_mean)

    print(f"{num_agents:<10} {dist_mean:<15.2f} {heur_mean:<15.2f} {learn_mean:<15.2f}")

# Calculate improvements
print("\n" + "=" * 60)
print("Improvement over Distance-Based Baseline")
print("=" * 60)

for i, num_agents in enumerate(agent_counts):
    dist_mean = np.mean(results["distance_based"][num_agents])
    heur_mean = np.mean(results["heuristic_congestion"][num_agents])
    learn_mean = np.mean(results["learned_congestion"][num_agents])

    heur_improvement = (heur_mean - dist_mean) / dist_mean * 100
    learn_improvement = (learn_mean - dist_mean) / dist_mean * 100

    print(
        f"{num_agents} agents: Heuristic {heur_improvement:+.1f}%, Learned {learn_improvement:+.1f}%"
    )

# Final throughput metric
final_throughput = np.mean(experiment_data["learned_congestion"]["throughput"])
print(f"\n{'=' * 60}")
print(f"PRIMARY METRIC - throughput_tasks_per_minute: {final_throughput:.2f}")
print(f"{'=' * 60}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --- Visualization ---
# Training loss curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(
    experiment_data["training"]["epochs"],
    experiment_data["training"]["losses"]["train"],
    label="Train",
)
plt.plot(
    experiment_data["training"]["epochs"],
    experiment_data["training"]["losses"]["val"],
    label="Validation",
)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Congestion Predictor Training")
plt.legend()
plt.grid(True)

# Throughput comparison
plt.subplot(1, 2, 2)
x = np.arange(len(agent_counts))
width = 0.25
plt.bar(
    x - width,
    experiment_data["distance_based"]["throughput"],
    width,
    label="Distance-based",
)
plt.bar(
    x,
    experiment_data["heuristic_congestion"]["throughput"],
    width,
    label="Heuristic CATA",
)
plt.bar(
    x + width,
    experiment_data["learned_congestion"]["throughput"],
    width,
    label="Learned CATA",
)
plt.xlabel("Number of Agents")
plt.ylabel("Throughput (tasks/min)")
plt.title("Throughput vs Agent Density")
plt.xticks(x, agent_counts)
plt.legend()
plt.grid(True, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cata_results.png"), dpi=150)
plt.close()

# Congestion heatmap visualization
sim = WarehouseSimulator(grid_size=20, num_agents=50, num_tasks=75)
state = sim.get_state_tensor(grid_resolution=5)
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    predicted = model(state_tensor).cpu().numpy()[0]
actual = sim.get_congestion_heatmap(grid_resolution=5)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(state[0], cmap="Blues")
plt.title("Agent Density")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(actual, cmap="Reds")
plt.title("Actual Congestion")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(predicted, cmap="Reds")
plt.title("Predicted Congestion")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "congestion_heatmaps.png"), dpi=150)
plt.close()

print(f"\nResults saved to {working_dir}")
print("Experiment completed successfully!")
