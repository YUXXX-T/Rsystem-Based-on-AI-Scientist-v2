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

experiment_data = {
    "scaling_factor_tuning": {
        "baseline_makespans": [],
        "cma_makespans": [],
        "improvements": [],
        "conflict_counts_baseline": [],
        "conflict_counts_cma": [],
        "assignment_changes": [],
        "tuning_results": [],
    }
}


def create_warehouse_map(width=20, height=20):
    grid = np.zeros((height, width), dtype=int)
    for row in [3, 7, 11, 15]:
        for col in range(2, width - 2):
            if col not in [5, 10, 15]:
                grid[row, col] = 1
    return grid


def get_neighbors(pos, grid):
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
        for npos in get_neighbors(pos, grid) + [pos]:
            nt = t + 1
            if (npos, nt) in reservations:
                conflicts_encountered.append(("vertex", npos, nt))
                continue
            if npos != pos:
                if (pos, nt) in reservations:
                    edge_key = (npos, pos, nt)
                    if edge_key in reservations.get("edges", set()):
                        conflicts_encountered.append(("edge", pos, npos, nt))
                        continue
            if (npos, nt) not in closed:
                new_path = path + [npos]
                heapq.heappush(
                    open_set, (nt + heuristic(npos, goal), nt, npos, new_path)
                )
    return [start], conflicts_encountered


def prioritized_planning(agents_starts, agents_goals, grid, conflict_memory=None):
    n_agents = len(agents_starts)
    reservations = {"edges": set()}
    all_paths = []
    total_conflicts = []
    reservation_pressure = np.zeros_like(grid, dtype=float)
    for i in range(n_agents):
        path, conflicts = astar_with_reservations(
            agents_starts[i], agents_goals[i], grid, reservations
        )
        all_paths.append(path)
        total_conflicts.extend(conflicts)
        for t, pos in enumerate(path):
            reservations[(pos, t)] = i
            reservation_pressure[pos[0], pos[1]] += 1
            if t > 0:
                prev_pos = path[t - 1]
                if prev_pos != pos:
                    reservations["edges"].add((prev_pos, pos, t))
        final_pos = path[-1]
        for t in range(len(path), len(path) + 10):
            reservations[(final_pos, t)] = i
    return all_paths, total_conflicts, reservation_pressure


def compute_makespan(paths):
    return max(len(p) for p in paths) if paths else 0


def get_shortest_path_cells(start, goal, grid):
    path, _ = astar_with_reservations(start, goal, grid, {}, max_time=200)
    return path


def hungarian_assignment(cost_matrix):
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return []
    assignment = [-1] * n
    prices = np.zeros(m)
    epsilon = 1.0 / (n + 1)
    for _ in range(n * 10):
        unassigned = [i for i in range(n) if assignment[i] == -1]
        if not unassigned:
            break
        for agent in unassigned:
            values = cost_matrix[agent] + prices
            sorted_idx = np.argsort(values)
            best_task = sorted_idx[0]
            best_value = values[best_task]
            second_value = (
                values[sorted_idx[1]] if len(sorted_idx) > 1 else best_value + epsilon
            )
            current_owner = -1
            for i in range(n):
                if assignment[i] == best_task:
                    current_owner = i
                    break
            if current_owner >= 0:
                assignment[current_owner] = -1
            assignment[agent] = best_task
            prices[best_task] += second_value - best_value + epsilon
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
    def __init__(self, grid_shape, decay=0.9):
        self.memory = np.zeros(grid_shape, dtype=float)
        self.edge_memory = defaultdict(float)
        self.decay = decay
        self.update_count = 0

    def update(self, conflicts, reservation_pressure):
        self.memory *= self.decay
        for conflict in conflicts:
            if conflict[0] == "vertex":
                pos = conflict[1]
                self.memory[pos[0], pos[1]] += 1.0
            elif conflict[0] == "edge":
                pos1, pos2 = conflict[1], conflict[2]
                self.edge_memory[(pos1, pos2)] += 1.0
                self.memory[pos1[0], pos1[1]] += 0.5
                self.memory[pos2[0], pos2[1]] += 0.5
        if reservation_pressure.max() > 0:
            normalized_pressure = reservation_pressure / reservation_pressure.max()
            self.memory += normalized_pressure * 0.3
        self.update_count += 1

    def get_path_penalty(self, path_cells, penalty_weight=1.0):
        if not path_cells:
            return 0.0
        penalty = sum(self.memory[cell[0], cell[1]] for cell in path_cells)
        return penalty * penalty_weight

    def freeze(self):
        frozen = ConflictMemory(self.memory.shape, self.decay)
        frozen.memory = self.memory.copy()
        frozen.edge_memory = self.edge_memory.copy()
        frozen.update_count = self.update_count
        return frozen


def compute_cost_matrix_with_scaling(
    agents, tasks, grid, conflict_memory=None, penalty_weight=2.0, scaling_factor=1.0
):
    """Compute cost matrix with scaling factor that normalizes penalty relative to base cost."""
    n_agents, n_tasks = len(agents), len(tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))
    base_costs = []

    # First pass: compute all base costs
    path_cache = {}
    for i, agent_pos in enumerate(agents):
        for j, task_pos in enumerate(tasks):
            path_cells = get_shortest_path_cells(agent_pos, task_pos, grid)
            path_cache[(i, j)] = path_cells
            base_costs.append(len(path_cells))

    avg_base_cost = np.mean(base_costs) if base_costs else 1.0

    # Second pass: compute final costs with scaled penalties
    for i, agent_pos in enumerate(agents):
        for j, task_pos in enumerate(tasks):
            path_cells = path_cache[(i, j)]
            base_cost = len(path_cells)

            if conflict_memory is not None:
                raw_penalty = conflict_memory.get_path_penalty(path_cells, 1.0)
                # Normalize penalty relative to average base cost using scaling factor
                normalized_penalty = (
                    (raw_penalty / avg_base_cost)
                    * scaling_factor
                    * penalty_weight
                    * base_cost
                )
                cost_matrix[i, j] = base_cost + normalized_penalty
            else:
                cost_matrix[i, j] = base_cost
    return cost_matrix


def run_episode_with_scaling(
    grid,
    n_agents,
    conflict_memory=None,
    use_cma=False,
    penalty_weight=2.0,
    scaling_factor=1.0,
):
    h, w = grid.shape
    free_cells = [(i, j) for i in range(h) for j in range(w) if grid[i, j] == 0]
    if len(free_cells) < n_agents * 2:
        raise ValueError("Not enough free cells")
    np.random.shuffle(free_cells)
    agent_starts = free_cells[:n_agents]
    task_positions = free_cells[n_agents : n_agents * 2]

    if use_cma and conflict_memory is not None:
        cost_matrix = compute_cost_matrix_with_scaling(
            agent_starts,
            task_positions,
            grid,
            conflict_memory,
            penalty_weight,
            scaling_factor,
        )
    else:
        cost_matrix = compute_cost_matrix_with_scaling(
            agent_starts, task_positions, grid
        )

    assignment = hungarian_assignment(cost_matrix)
    agent_goals = []
    for i, task_idx in enumerate(assignment):
        if task_idx >= 0 and task_idx < len(task_positions):
            agent_goals.append(task_positions[task_idx])
        else:
            agent_goals.append(agent_starts[i])

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


def run_experiment_with_scaling(
    n_warmup=30, n_eval=50, n_agents=6, penalty_weight=2.5, scaling_factor=1.0
):
    grid = create_warehouse_map(20, 20)
    conflict_memory = ConflictMemory(grid.shape, decay=0.95)

    for ep in range(n_warmup):
        np.random.seed(ep)
        result = run_episode_with_scaling(
            grid, n_agents, conflict_memory=None, use_cma=False
        )
        conflict_memory.update(result["conflicts"], result["reservation_pressure"])

    frozen_memory = conflict_memory.freeze()
    baseline_makespans, cma_makespans = [], []
    baseline_conflicts, cma_conflicts = [], []
    assignment_changes = []

    for ep in range(n_eval):
        seed = n_warmup + ep
        np.random.seed(seed)
        baseline_result = run_episode_with_scaling(
            grid, n_agents, conflict_memory=None, use_cma=False
        )
        np.random.seed(seed)
        cma_result = run_episode_with_scaling(
            grid,
            n_agents,
            conflict_memory=frozen_memory,
            use_cma=True,
            penalty_weight=penalty_weight,
            scaling_factor=scaling_factor,
        )
        baseline_makespans.append(baseline_result["makespan"])
        cma_makespans.append(cma_result["makespan"])
        baseline_conflicts.append(len(baseline_result["conflicts"]))
        cma_conflicts.append(len(cma_result["conflicts"]))
        assignments_differ = baseline_result["assignment"] != cma_result["assignment"]
        if isinstance(assignments_differ, np.ndarray):
            assignments_differ = assignments_differ.any()
        assignment_changes.append(int(assignments_differ))

    baseline_mean = np.mean(baseline_makespans)
    cma_mean = np.mean(cma_makespans)
    improvement = (baseline_mean - cma_mean) / baseline_mean if baseline_mean > 0 else 0

    return {
        "improvement": improvement,
        "baseline_mean": baseline_mean,
        "cma_mean": cma_mean,
        "baseline_makespans": baseline_makespans,
        "cma_makespans": cma_makespans,
        "baseline_conflicts": baseline_conflicts,
        "cma_conflicts": cma_conflicts,
        "assignment_changes": assignment_changes,
        "frozen_memory": frozen_memory,
        "grid": grid,
    }


def hyperparameter_tuning():
    print("=" * 60)
    print("HYPERPARAMETER TUNING: Penalty Weight Scaling Factor")
    print("=" * 60)

    scaling_factors = [0.5, 1.0, 2.0, 3.0, 5.0]
    penalty_weights = [1.0, 2.0, 3.0, 5.0]

    tuning_results = []
    best_improvement = -float("inf")
    best_config = None

    for sf in scaling_factors:
        for pw in penalty_weights:
            print(f"\nTesting scaling_factor={sf}, penalty_weight={pw}")
            result = run_experiment_with_scaling(
                n_warmup=25, n_eval=40, n_agents=6, penalty_weight=pw, scaling_factor=sf
            )
            print(f"  Improvement: {result['improvement']*100:.2f}%")

            tuning_results.append(
                {
                    "scaling_factor": sf,
                    "penalty_weight": pw,
                    "improvement": result["improvement"],
                    "baseline_mean": result["baseline_mean"],
                    "cma_mean": result["cma_mean"],
                }
            )

            if result["improvement"] > best_improvement:
                best_improvement = result["improvement"]
                best_config = {"scaling_factor": sf, "penalty_weight": pw}

    print(f"\n{'='*60}")
    print(
        f"Best configuration: scaling_factor={best_config['scaling_factor']}, penalty_weight={best_config['penalty_weight']}"
    )
    print(f"Best improvement: {best_improvement*100:.2f}%")

    return tuning_results, best_config, best_improvement


def visualize_tuning_results(tuning_results, final_result):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Heatmap of improvements
    scaling_factors = sorted(set(r["scaling_factor"] for r in tuning_results))
    penalty_weights = sorted(set(r["penalty_weight"] for r in tuning_results))
    improvement_matrix = np.zeros((len(scaling_factors), len(penalty_weights)))

    for r in tuning_results:
        i = scaling_factors.index(r["scaling_factor"])
        j = penalty_weights.index(r["penalty_weight"])
        improvement_matrix[i, j] = r["improvement"] * 100

    im = axes[0, 0].imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
    axes[0, 0].set_xticks(range(len(penalty_weights)))
    axes[0, 0].set_xticklabels([f"{pw}" for pw in penalty_weights])
    axes[0, 0].set_yticks(range(len(scaling_factors)))
    axes[0, 0].set_yticklabels([f"{sf}" for sf in scaling_factors])
    axes[0, 0].set_xlabel("Penalty Weight")
    axes[0, 0].set_ylabel("Scaling Factor")
    axes[0, 0].set_title("Improvement (%) Heatmap")
    plt.colorbar(im, ax=axes[0, 0])

    # Plot 2: Line plot by scaling factor
    for sf in scaling_factors:
        improvements = [
            r["improvement"] * 100 for r in tuning_results if r["scaling_factor"] == sf
        ]
        axes[0, 1].plot(penalty_weights, improvements, "o-", label=f"SF={sf}")
    axes[0, 1].set_xlabel("Penalty Weight")
    axes[0, 1].set_ylabel("Improvement (%)")
    axes[0, 1].set_title("Improvement by Scaling Factor")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Conflict memory heatmap
    if final_result["frozen_memory"] is not None:
        im2 = axes[1, 0].imshow(
            final_result["frozen_memory"].memory, cmap="hot", origin="upper"
        )
        axes[1, 0].set_title("Conflict Memory Map (Final)")
        plt.colorbar(im2, ax=axes[1, 0])

    # Plot 4: Makespan comparison for final run
    x = np.arange(len(final_result["baseline_makespans"]))
    axes[1, 1].plot(
        x, final_result["baseline_makespans"], "b-", alpha=0.7, label="Baseline"
    )
    axes[1, 1].plot(x, final_result["cma_makespans"], "r-", alpha=0.7, label="CMA")
    axes[1, 1].axhline(
        np.mean(final_result["baseline_makespans"]),
        color="b",
        linestyle="--",
        label=f"Baseline mean: {np.mean(final_result['baseline_makespans']):.1f}",
    )
    axes[1, 1].axhline(
        np.mean(final_result["cma_makespans"]),
        color="r",
        linestyle="--",
        label=f"CMA mean: {np.mean(final_result['cma_makespans']):.1f}",
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Makespan")
    axes[1, 1].set_title("Final Run: Makespan Comparison")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "scaling_factor_tuning_results.png"), dpi=150)
    plt.close()
    print(
        f"\nVisualization saved to {os.path.join(working_dir, 'scaling_factor_tuning_results.png')}"
    )


# Main execution
print("=" * 60)
print("Penalty Weight Scaling Factor Hyperparameter Tuning")
print("=" * 60)

# Run hyperparameter tuning
tuning_results, best_config, best_improvement = hyperparameter_tuning()

# Run final experiment with best configuration
print(f"\n{'='*60}")
print("FINAL RUN WITH BEST CONFIGURATION")
print(f"{'='*60}")
final_result = run_experiment_with_scaling(
    n_warmup=50,
    n_eval=80,
    n_agents=7,
    penalty_weight=best_config["penalty_weight"],
    scaling_factor=best_config["scaling_factor"],
)

print(f"\nFinal Results:")
print(f"  Baseline mean makespan: {final_result['baseline_mean']:.2f}")
print(f"  CMA mean makespan: {final_result['cma_mean']:.2f}")
print(f"  Improvement: {final_result['improvement']*100:.2f}%")
print(
    f"  Assignment change rate: {np.mean(final_result['assignment_changes'])*100:.1f}%"
)

# Store results
experiment_data["scaling_factor_tuning"]["tuning_results"] = tuning_results
experiment_data["scaling_factor_tuning"]["best_config"] = best_config
experiment_data["scaling_factor_tuning"]["best_improvement"] = best_improvement
experiment_data["scaling_factor_tuning"]["baseline_makespans"] = final_result[
    "baseline_makespans"
]
experiment_data["scaling_factor_tuning"]["cma_makespans"] = final_result[
    "cma_makespans"
]
experiment_data["scaling_factor_tuning"]["improvements"] = [
    (b - c) / b if b > 0 else 0
    for b, c in zip(final_result["baseline_makespans"], final_result["cma_makespans"])
]
experiment_data["scaling_factor_tuning"]["conflict_counts_baseline"] = final_result[
    "baseline_conflicts"
]
experiment_data["scaling_factor_tuning"]["conflict_counts_cma"] = final_result[
    "cma_conflicts"
]
experiment_data["scaling_factor_tuning"]["assignment_changes"] = final_result[
    "assignment_changes"
]
experiment_data["scaling_factor_tuning"]["final_improvement"] = final_result[
    "improvement"
]

# Visualize results
visualize_tuning_results(tuning_results, final_result)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")

# Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Best scaling factor: {best_config['scaling_factor']}")
print(f"Best penalty weight: {best_config['penalty_weight']}")
print(f"Tuning best improvement: {best_improvement*100:.2f}%")
print(f"Final experiment improvement: {final_result['improvement']*100:.2f}%")
print(f"Target improvement range: 0.15-0.25 (15-25%)")

if final_result["improvement"] > 0:
    print(
        f"\nCMA with scaling shows POSITIVE improvement of {final_result['improvement']*100:.2f}%"
    )
else:
    print(f"\nCMA shows no improvement: {final_result['improvement']*100:.2f}%")
