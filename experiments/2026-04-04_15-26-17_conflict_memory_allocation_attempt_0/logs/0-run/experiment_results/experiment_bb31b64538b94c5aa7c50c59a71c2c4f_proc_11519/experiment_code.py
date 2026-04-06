# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import heapq
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data tracking
experiment_data = {
    "cma_experiment": {
        "baseline_makespans": [],
        "cma_makespans": [],
        "improvements": [],
        "conflict_counts_baseline": [],
        "conflict_counts_cma": [],
        "assignment_changes": [],
    }
}


def create_warehouse_map(width=20, height=20):
    """Create warehouse with corridors and chokepoints."""
    grid = np.zeros((height, width), dtype=int)
    # Add shelf obstacles creating narrow corridors
    for row in [3, 7, 11, 15]:
        for col in range(2, width - 2):
            if col not in [5, 10, 15]:  # Leave gaps for corridors
                grid[row, col] = 1
    return grid


def get_neighbors(pos, grid):
    """Get valid neighboring positions."""
    h, w = grid.shape
    neighbors = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ny, nx = pos[0] + dy, pos[1] + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
            neighbors.append((ny, nx))
    return neighbors


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_with_reservations(start, goal, grid, reservations, max_time=100):
    """A* with space-time reservations. Returns path, conflicts encountered."""
    conflicts_encountered = []

    if start == goal:
        return [start], conflicts_encountered

    open_set = [(heuristic(start, goal), 0, start, [start])]
    closed = set()

    while open_set:
        f, t, pos, path = heapq.heappop(open_set)

        if t > max_time:
            continue

        if (pos, t) in closed:
            continue
        closed.add((pos, t))

        if pos == goal:
            return path, conflicts_encountered

        for npos in get_neighbors(pos, grid) + [pos]:  # Include wait action
            nt = t + 1

            # Check vertex conflict
            if (npos, nt) in reservations:
                conflicts_encountered.append(("vertex", npos, nt))
                continue

            # Check edge conflict (swap)
            if npos != pos:
                if (pos, nt) in reservations:
                    # Check if agent at npos is moving to pos
                    edge_key = (npos, pos, nt)
                    if edge_key in reservations.get("edges", set()):
                        conflicts_encountered.append(("edge", pos, npos, nt))
                        continue

            if (npos, nt) not in closed:
                new_path = path + [npos]
                heapq.heappush(
                    open_set, (nt + heuristic(npos, goal), nt, npos, new_path)
                )

    # Return partial path if no full path found
    return [start], conflicts_encountered


def prioritized_planning(agents_starts, agents_goals, grid, conflict_memory=None):
    """Plan paths for all agents with priority ordering. Returns paths and conflict info."""
    n_agents = len(agents_starts)
    reservations = {}
    reservations["edges"] = set()
    all_paths = []
    total_conflicts = []
    reservation_pressure = np.zeros_like(grid, dtype=float)

    for i in range(n_agents):
        path, conflicts = astar_with_reservations(
            agents_starts[i], agents_goals[i], grid, reservations
        )
        all_paths.append(path)
        total_conflicts.extend(conflicts)

        # Add reservations for this agent's path
        for t, pos in enumerate(path):
            reservations[(pos, t)] = i
            reservation_pressure[pos[0], pos[1]] += 1

            # Add edge reservation
            if t > 0:
                prev_pos = path[t - 1]
                if prev_pos != pos:
                    reservations["edges"].add((prev_pos, pos, t))

        # Extend final position reservation
        final_pos = path[-1]
        for t in range(len(path), len(path) + 10):
            reservations[(final_pos, t)] = i

    return all_paths, total_conflicts, reservation_pressure


def compute_makespan(paths):
    """Compute makespan (time for all agents to reach goals)."""
    return max(len(p) for p in paths) if paths else 0


def get_shortest_path_cells(start, goal, grid):
    """Get cells along shortest path for cost estimation."""
    path, _ = astar_with_reservations(start, goal, grid, {}, max_time=200)
    return path


def greedy_assignment(agents, tasks, cost_matrix):
    """Greedy assignment: each agent picks cheapest unassigned task."""
    n = len(agents)
    m = len(tasks)
    assignment = [-1] * n
    assigned_tasks = set()

    # Sort agents by their minimum cost
    agent_order = list(range(n))

    for agent in agent_order:
        best_task = -1
        best_cost = float("inf")
        for task in range(m):
            if task not in assigned_tasks:
                if cost_matrix[agent, task] < best_cost:
                    best_cost = cost_matrix[agent, task]
                    best_task = task
        if best_task >= 0:
            assignment[agent] = best_task
            assigned_tasks.add(best_task)

    return assignment


def hungarian_assignment(cost_matrix):
    """Simple Hungarian-like assignment using auction algorithm."""
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return []

    assignment = [-1] * n
    prices = np.zeros(m)
    epsilon = 1.0 / (n + 1)

    for _ in range(n * 10):  # Iterate until convergence
        unassigned = [i for i in range(n) if assignment[i] == -1]
        if not unassigned:
            break

        for agent in unassigned:
            # Find best and second-best tasks
            values = cost_matrix[agent] + prices
            sorted_idx = np.argsort(values)
            best_task = sorted_idx[0]
            best_value = values[best_task]

            if len(sorted_idx) > 1:
                second_value = values[sorted_idx[1]]
            else:
                second_value = best_value + epsilon

            # Check if task is taken
            current_owner = -1
            for i in range(n):
                if assignment[i] == best_task:
                    current_owner = i
                    break

            if current_owner >= 0:
                assignment[current_owner] = -1

            assignment[agent] = best_task
            prices[best_task] += second_value - best_value + epsilon

    # Fill any remaining unassigned
    assigned_tasks = set(a for a in assignment if a >= 0)
    for i in range(n):
        if assignment[i] == -1:
            for t in range(m):
                if t not in assigned_tasks:
                    assignment[i] = t
                    assigned_tasks.add(t)
                    break

    return assignment


class ConflictMemory:
    """Spatial conflict memory map."""

    def __init__(self, grid_shape, decay=0.9):
        self.memory = np.zeros(grid_shape, dtype=float)
        self.edge_memory = defaultdict(float)
        self.decay = decay
        self.update_count = 0

    def update(self, conflicts, reservation_pressure):
        """Update memory from conflicts and reservation pressure."""
        self.memory *= self.decay

        # Add vertex conflicts
        for conflict in conflicts:
            if conflict[0] == "vertex":
                pos = conflict[1]
                self.memory[pos[0], pos[1]] += 1.0
            elif conflict[0] == "edge":
                pos1, pos2 = conflict[1], conflict[2]
                self.edge_memory[(pos1, pos2)] += 1.0
                self.memory[pos1[0], pos1[1]] += 0.5
                self.memory[pos2[0], pos2[1]] += 0.5

        # Add reservation pressure (normalized)
        if reservation_pressure.max() > 0:
            normalized_pressure = reservation_pressure / reservation_pressure.max()
            self.memory += normalized_pressure * 0.3

        self.update_count += 1

    def get_path_penalty(self, path_cells, penalty_weight=1.0):
        """Compute penalty for a path based on conflict memory."""
        if not path_cells:
            return 0.0
        penalty = 0.0
        for cell in path_cells:
            penalty += self.memory[cell[0], cell[1]]
        return penalty * penalty_weight

    def freeze(self):
        """Return a frozen copy for evaluation."""
        frozen = ConflictMemory(self.memory.shape, self.decay)
        frozen.memory = self.memory.copy()
        frozen.edge_memory = self.edge_memory.copy()
        frozen.update_count = self.update_count
        return frozen


def compute_cost_matrix(agents, tasks, grid, conflict_memory=None, penalty_weight=2.0):
    """Compute assignment cost matrix with optional CMA penalties."""
    n_agents = len(agents)
    n_tasks = len(tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))

    for i, agent_pos in enumerate(agents):
        for j, task_pos in enumerate(tasks):
            # Base cost: path length
            path_cells = get_shortest_path_cells(agent_pos, task_pos, grid)
            base_cost = len(path_cells)

            # Add conflict memory penalty
            if conflict_memory is not None:
                penalty = conflict_memory.get_path_penalty(path_cells, penalty_weight)
                cost_matrix[i, j] = base_cost + penalty
            else:
                cost_matrix[i, j] = base_cost

    return cost_matrix


def run_episode(
    grid, n_agents, conflict_memory=None, use_cma=False, penalty_weight=2.0
):
    """Run a single episode with task allocation and path planning."""
    h, w = grid.shape

    # Generate random start positions (on free cells)
    free_cells = [(i, j) for i in range(h) for j in range(w) if grid[i, j] == 0]

    if len(free_cells) < n_agents * 2:
        raise ValueError("Not enough free cells for agents and tasks")

    np.random.shuffle(free_cells)
    agent_starts = free_cells[:n_agents]
    task_positions = free_cells[n_agents : n_agents * 2]

    # Compute cost matrix
    if use_cma and conflict_memory is not None:
        cost_matrix = compute_cost_matrix(
            agent_starts, task_positions, grid, conflict_memory, penalty_weight
        )
    else:
        cost_matrix = compute_cost_matrix(agent_starts, task_positions, grid)

    # Assignment
    assignment = hungarian_assignment(cost_matrix)

    # Get agent goals based on assignment
    agent_goals = []
    for i, task_idx in enumerate(assignment):
        if task_idx >= 0 and task_idx < len(task_positions):
            agent_goals.append(task_positions[task_idx])
        else:
            agent_goals.append(agent_starts[i])  # Stay in place if no task

    # Path planning
    paths, conflicts, reservation_pressure = prioritized_planning(
        agent_starts, agent_goals, grid
    )

    makespan = compute_makespan(paths)

    return {
        "makespan": makespan,
        "conflicts": conflicts,
        "reservation_pressure": reservation_pressure,
        "paths": paths,
        "assignment": assignment,
        "agent_starts": agent_starts,
        "task_positions": task_positions,
    }


def run_experiment(n_warmup=30, n_eval=50, n_agents=6, penalty_weight=2.5):
    """Run full CMA experiment."""
    print(f"\n{'='*60}")
    print(f"Running CMA Experiment")
    print(f"Warmup episodes: {n_warmup}, Eval episodes: {n_eval}")
    print(f"Agents: {n_agents}, Penalty weight: {penalty_weight}")
    print(f"{'='*60}\n")

    # Create warehouse map
    grid = create_warehouse_map(20, 20)
    print(f"Grid size: {grid.shape}, Free cells: {np.sum(grid == 0)}")

    # Initialize conflict memory
    conflict_memory = ConflictMemory(grid.shape, decay=0.95)

    # Warmup phase: build conflict memory
    print("\n--- Warmup Phase ---")
    warmup_conflicts_total = 0
    for ep in range(n_warmup):
        np.random.seed(ep)
        result = run_episode(grid, n_agents, conflict_memory=None, use_cma=False)

        # Update conflict memory
        conflict_memory.update(result["conflicts"], result["reservation_pressure"])
        warmup_conflicts_total += len(result["conflicts"])

        if (ep + 1) % 10 == 0:
            print(
                f"Warmup episode {ep+1}: makespan={result['makespan']}, "
                f"conflicts={len(result['conflicts'])}, "
                f"memory_max={conflict_memory.memory.max():.2f}"
            )

    print(f"\nWarmup complete. Total conflicts recorded: {warmup_conflicts_total}")
    print(
        f"Memory statistics: max={conflict_memory.memory.max():.3f}, "
        f"mean={conflict_memory.memory.mean():.3f}, "
        f"nonzero cells={np.sum(conflict_memory.memory > 0.01)}"
    )

    # Freeze memory for evaluation
    frozen_memory = conflict_memory.freeze()

    # Evaluation phase
    print("\n--- Evaluation Phase ---")
    baseline_makespans = []
    cma_makespans = []
    baseline_conflicts = []
    cma_conflicts = []
    assignment_changes = []

    for ep in range(n_eval):
        seed = n_warmup + ep

        # Baseline run
        np.random.seed(seed)
        baseline_result = run_episode(
            grid, n_agents, conflict_memory=None, use_cma=False
        )

        # CMA run (same seed for identical initial conditions)
        np.random.seed(seed)
        cma_result = run_episode(
            grid,
            n_agents,
            conflict_memory=frozen_memory,
            use_cma=True,
            penalty_weight=penalty_weight,
        )

        baseline_makespans.append(baseline_result["makespan"])
        cma_makespans.append(cma_result["makespan"])
        baseline_conflicts.append(len(baseline_result["conflicts"]))
        cma_conflicts.append(len(cma_result["conflicts"]))

        # Check if assignments differ
        assignments_differ = baseline_result["assignment"] != cma_result["assignment"]
        if isinstance(assignments_differ, np.ndarray):
            assignments_differ = assignments_differ.any()
        assignment_changes.append(int(assignments_differ))

        if (ep + 1) % 10 == 0:
            print(
                f"Eval episode {ep+1}: baseline_makespan={baseline_result['makespan']}, "
                f"cma_makespan={cma_result['makespan']}, "
                f"assignment_changed={assignments_differ}"
            )

    # Store results
    experiment_data["cma_experiment"]["baseline_makespans"] = baseline_makespans
    experiment_data["cma_experiment"]["cma_makespans"] = cma_makespans
    experiment_data["cma_experiment"]["conflict_counts_baseline"] = baseline_conflicts
    experiment_data["cma_experiment"]["conflict_counts_cma"] = cma_conflicts
    experiment_data["cma_experiment"]["assignment_changes"] = assignment_changes

    # Compute metrics
    baseline_mean = np.mean(baseline_makespans)
    cma_mean = np.mean(cma_makespans)

    improvements = [
        (b - c) / b if b > 0 else 0 for b, c in zip(baseline_makespans, cma_makespans)
    ]
    experiment_data["cma_experiment"]["improvements"] = improvements

    makespan_improvement_ratio = (
        (baseline_mean - cma_mean) / baseline_mean if baseline_mean > 0 else 0
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(
        f"Baseline average makespan: {baseline_mean:.2f} (std: {np.std(baseline_makespans):.2f})"
    )
    print(
        f"CMA average makespan:      {cma_mean:.2f} (std: {np.std(cma_makespans):.2f})"
    )
    print(
        f"Makespan improvement ratio: {makespan_improvement_ratio:.4f} ({makespan_improvement_ratio*100:.2f}%)"
    )
    print(f"Assignment change rate: {np.mean(assignment_changes)*100:.1f}%")
    print(f"Baseline avg conflicts: {np.mean(baseline_conflicts):.2f}")
    print(f"CMA avg conflicts: {np.mean(cma_conflicts):.2f}")

    return makespan_improvement_ratio, frozen_memory, grid


def visualize_results(conflict_memory, grid):
    """Visualize conflict memory and results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Grid layout
    ax1 = axes[0]
    ax1.imshow(grid, cmap="binary", origin="upper")
    ax1.set_title("Warehouse Layout (black=obstacle)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot 2: Conflict memory heatmap
    ax2 = axes[1]
    im = ax2.imshow(conflict_memory.memory, cmap="hot", origin="upper")
    ax2.set_title("Conflict Memory Map")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    plt.colorbar(im, ax=ax2)

    # Plot 3: Makespan comparison
    ax3 = axes[2]
    baseline_makespans = experiment_data["cma_experiment"]["baseline_makespans"]
    cma_makespans = experiment_data["cma_experiment"]["cma_makespans"]

    x = np.arange(len(baseline_makespans))
    ax3.plot(x, baseline_makespans, "b-", alpha=0.7, label="Baseline")
    ax3.plot(x, cma_makespans, "r-", alpha=0.7, label="CMA")
    ax3.axhline(
        np.mean(baseline_makespans),
        color="b",
        linestyle="--",
        label=f"Baseline mean: {np.mean(baseline_makespans):.1f}",
    )
    ax3.axhline(
        np.mean(cma_makespans),
        color="r",
        linestyle="--",
        label=f"CMA mean: {np.mean(cma_makespans):.1f}",
    )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Makespan")
    ax3.set_title("Makespan Comparison")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cma_results.png"), dpi=150)
    plt.close()
    print(f"\nVisualization saved to {os.path.join(working_dir, 'cma_results.png')}")


def run_parameter_sweep():
    """Sweep penalty weights to find best configuration."""
    print("\n" + "=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)

    penalty_weights = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for pw in penalty_weights:
        print(f"\nTesting penalty_weight = {pw}")
        improvement, _, _ = run_experiment(
            n_warmup=20, n_eval=30, n_agents=6, penalty_weight=pw
        )
        results.append((pw, improvement))
        print(f"  -> Improvement ratio: {improvement:.4f}")

    best_pw, best_improvement = max(results, key=lambda x: x[1])
    print(f"\nBest penalty weight: {best_pw} with improvement: {best_improvement:.4f}")

    return best_pw, results


# Main execution
print("=" * 60)
print("Conflict Memory Allocation (CMA) Experiment")
print("=" * 60)

# Run main experiment with default parameters
main_improvement, conflict_memory, grid = run_experiment(
    n_warmup=40, n_eval=60, n_agents=6, penalty_weight=2.5
)

# Visualize results
visualize_results(conflict_memory, grid)

# Run parameter sweep to find best configuration
best_pw, sweep_results = run_parameter_sweep()

# Final run with best parameters
print("\n" + "=" * 60)
print("FINAL RUN WITH BEST PARAMETERS")
print("=" * 60)
final_improvement, final_memory, final_grid = run_experiment(
    n_warmup=50, n_eval=80, n_agents=7, penalty_weight=best_pw
)

# Save experiment data
experiment_data["sweep_results"] = sweep_results
experiment_data["best_penalty_weight"] = best_pw
experiment_data["final_improvement"] = final_improvement

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")

# Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Main experiment makespan_improvement_ratio: {main_improvement:.4f}")
print(f"Final experiment makespan_improvement_ratio: {final_improvement:.4f}")
print(f"Best penalty weight from sweep: {best_pw}")
print(f"Target improvement range: 0.15-0.25 (15-25%)")

if final_improvement > 0:
    print(f"\nCMA shows POSITIVE improvement of {final_improvement*100:.2f}%")
else:
    print(
        f"\nCMA shows no improvement or negative result: {final_improvement*100:.2f}%"
    )
