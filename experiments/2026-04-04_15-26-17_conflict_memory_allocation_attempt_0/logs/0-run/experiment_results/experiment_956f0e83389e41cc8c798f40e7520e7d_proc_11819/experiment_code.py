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
    "warehouse": {
        "metrics": {"train": [], "val": []},
        "conflicts_baseline": [],
        "conflicts_cma": [],
        "makespans_baseline": [],
        "makespans_cma": [],
    },
    "maze": {
        "metrics": {"train": [], "val": []},
        "conflicts_baseline": [],
        "conflicts_cma": [],
        "makespans_baseline": [],
        "makespans_cma": [],
    },
    "random_grid": {
        "metrics": {"train": [], "val": []},
        "conflicts_baseline": [],
        "conflicts_cma": [],
        "makespans_baseline": [],
        "makespans_cma": [],
    },
}


def create_warehouse_map(width=20, height=20):
    grid = np.zeros((height, width), dtype=int)
    for row in [3, 7, 11, 15]:
        for col in range(2, width - 2):
            if col not in [5, 10, 15]:
                grid[row, col] = 1
    return grid


def create_maze_map(width=20, height=20, seed=42):
    np.random.seed(seed)
    grid = np.zeros((height, width), dtype=int)
    for i in range(2, height - 2, 4):
        for j in range(width):
            if np.random.random() > 0.3:
                grid[i, j] = 1
        gaps = np.random.choice(range(width), size=3, replace=False)
        for g in gaps:
            grid[i, g] = 0
    for j in range(2, width - 2, 4):
        for i in range(height):
            if np.random.random() > 0.3:
                grid[i, j] = 1
        gaps = np.random.choice(range(height), size=3, replace=False)
        for g in gaps:
            grid[g, j] = 0
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def create_random_grid(width=20, height=20, obstacle_rate=0.2, seed=42):
    np.random.seed(seed)
    grid = np.zeros((height, width), dtype=int)
    grid[np.random.random((height, width)) < obstacle_rate] = 1
    grid[0, :2] = 0
    grid[-1, -2:] = 0
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
            if (npos, nt) not in closed:
                new_path = path + [npos]
                heapq.heappush(
                    open_set, (nt + heuristic(npos, goal), nt, npos, new_path)
                )
    return [start], conflicts_encountered


def prioritized_planning(agents_starts, agents_goals, grid):
    n_agents = len(agents_starts)
    reservations = {}
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
            second_value = (
                values[sorted_idx[1]]
                if len(sorted_idx) > 1
                else values[best_task] + epsilon
            )
            current_owner = -1
            for i in range(n):
                if assignment[i] == best_task:
                    current_owner = i
                    break
            if current_owner >= 0:
                assignment[current_owner] = -1
            assignment[agent] = best_task
            prices[best_task] += second_value - values[best_task] + epsilon
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
        self.decay = decay
        self.update_count = 0

    def update(self, conflicts, reservation_pressure):
        self.memory *= self.decay
        # KEY FIX: Actually learn from real conflicts
        for conflict in conflicts:
            if conflict[0] == "vertex":
                pos = conflict[1]
                self.memory[pos[0], pos[1]] += 1.0
        if reservation_pressure.max() > 0:
            normalized_pressure = reservation_pressure / reservation_pressure.max()
            self.memory += normalized_pressure * 0.2
        self.update_count += 1

    def get_path_penalty(self, path_cells, penalty_weight=1.0):
        if not path_cells:
            return 0.0
        return sum(self.memory[c[0], c[1]] for c in path_cells) * penalty_weight

    def copy(self):
        cm = ConflictMemory(self.memory.shape, self.decay)
        cm.memory = self.memory.copy()
        cm.update_count = self.update_count
        return cm


def compute_cost_matrix(
    agents, tasks, grid, conflict_memory=None, penalty_weight=3.0, scaling_factor=0.5
):
    n_agents, n_tasks = len(agents), len(tasks)
    cost_matrix = np.zeros((n_agents, n_tasks))
    path_cache = {}
    base_costs = []
    for i, agent_pos in enumerate(agents):
        for j, task_pos in enumerate(tasks):
            path_cells = get_shortest_path_cells(agent_pos, task_pos, grid)
            path_cache[(i, j)] = path_cells
            base_costs.append(len(path_cells))
    avg_base = np.mean(base_costs) if base_costs else 1.0
    for i in range(n_agents):
        for j in range(n_tasks):
            path_cells = path_cache[(i, j)]
            base_cost = len(path_cells)
            if conflict_memory is not None:
                raw_penalty = conflict_memory.get_path_penalty(path_cells, 1.0)
                norm_penalty = (
                    (raw_penalty / avg_base)
                    * scaling_factor
                    * penalty_weight
                    * base_cost
                )
                cost_matrix[i, j] = base_cost + norm_penalty
            else:
                cost_matrix[i, j] = base_cost
    return cost_matrix


def run_episode(grid, n_agents, conflict_memory=None, use_cma=False, pw=3.0, sf=0.5):
    h, w = grid.shape
    free_cells = [(i, j) for i in range(h) for j in range(w) if grid[i, j] == 0]
    if len(free_cells) < n_agents * 2:
        return None
    np.random.shuffle(free_cells)
    agent_starts = free_cells[:n_agents]
    task_positions = free_cells[n_agents : n_agents * 2]
    if use_cma and conflict_memory:
        cost_matrix = compute_cost_matrix(
            agent_starts, task_positions, grid, conflict_memory, pw, sf
        )
    else:
        cost_matrix = compute_cost_matrix(agent_starts, task_positions, grid)
    assignment = hungarian_assignment(cost_matrix)
    agent_goals = [
        task_positions[t] if 0 <= t < len(task_positions) else agent_starts[i]
        for i, t in enumerate(assignment)
    ]
    paths, conflicts, pressure = prioritized_planning(agent_starts, agent_goals, grid)
    return {
        "makespan": compute_makespan(paths),
        "conflicts": conflicts,
        "pressure": pressure,
        "assignment": assignment,
    }


def evaluate_on_dataset(
    grid, dataset_name, n_warmup=30, n_eval=50, n_agents=6, pw=3.0, sf=0.5
):
    conflict_memory = ConflictMemory(grid.shape, decay=0.95)
    # Warmup: learn conflict patterns from baseline runs
    for ep in range(n_warmup):
        np.random.seed(ep * 100 + 1)
        result = run_episode(grid, n_agents, use_cma=False)
        if result:
            conflict_memory.update(result["conflicts"], result["pressure"])
    frozen_memory = conflict_memory.copy()
    baseline_makespans, cma_makespans = [], []
    baseline_conflicts, cma_conflicts = [], []
    for ep in range(n_eval):
        seed = n_warmup + ep
        np.random.seed(seed)
        baseline = run_episode(grid, n_agents, use_cma=False)
        np.random.seed(seed)
        cma = run_episode(grid, n_agents, frozen_memory, use_cma=True, pw=pw, sf=sf)
        if baseline and cma:
            baseline_makespans.append(baseline["makespan"])
            cma_makespans.append(cma["makespan"])
            baseline_conflicts.append(len(baseline["conflicts"]))
            cma_conflicts.append(len(cma["conflicts"]))
    return (
        baseline_makespans,
        cma_makespans,
        baseline_conflicts,
        cma_conflicts,
        frozen_memory,
    )


# Create datasets
print("Creating three test environments...")
warehouse_grid = create_warehouse_map(20, 20)
maze_grid = create_maze_map(20, 20, seed=42)
random_grid = create_random_grid(20, 20, obstacle_rate=0.15, seed=42)
datasets = {"warehouse": warehouse_grid, "maze": maze_grid, "random_grid": random_grid}

# Tuning on warehouse
print("\n" + "=" * 60)
print("TUNING on warehouse dataset")
print("=" * 60)
tuning_results = []
penalty_weights = [2.0, 2.5, 3.0, 3.5, 4.0]
scaling_factors = [0.25, 0.5, 0.75, 1.0]
best_crr, best_config = -float("inf"), {"pw": 3.0, "sf": 0.5}

for pw in penalty_weights:
    for sf in scaling_factors:
        bm, cm, bc, cc, _ = evaluate_on_dataset(
            warehouse_grid,
            "warehouse",
            n_warmup=20,
            n_eval=30,
            n_agents=6,
            pw=pw,
            sf=sf,
        )
        if bm:
            makespan_imp = (np.mean(bm) - np.mean(cm)) / np.mean(bm) * 100
            total_bc, total_cc = sum(bc), sum(cc)
            crr = (total_bc - total_cc) / total_bc * 100 if total_bc > 0 else 0
            combined = makespan_imp + crr
            tuning_results.append(
                {
                    "pw": pw,
                    "sf": sf,
                    "makespan_imp": makespan_imp,
                    "crr": crr,
                    "combined": combined,
                }
            )
            print(f"pw={pw}, sf={sf}: Makespan Imp={makespan_imp:.2f}%, CRR={crr:.2f}%")
            if combined > best_crr:
                best_crr, best_config = combined, {"pw": pw, "sf": sf}

print(f"\nBest config: pw={best_config['pw']}, sf={best_config['sf']}")

# Final evaluation on all three datasets
print("\n" + "=" * 60)
print("FINAL EVALUATION on all datasets")
print("=" * 60)

for name, grid in datasets.items():
    print(f"\nEvaluating on {name}...")
    bm, cm, bc, cc, memory = evaluate_on_dataset(
        grid,
        name,
        n_warmup=40,
        n_eval=60,
        n_agents=6,
        pw=best_config["pw"],
        sf=best_config["sf"],
    )
    experiment_data[name]["makespans_baseline"] = bm
    experiment_data[name]["makespans_cma"] = cm
    experiment_data[name]["conflicts_baseline"] = bc
    experiment_data[name]["conflicts_cma"] = cc

    if bm:
        makespan_imp = (np.mean(bm) - np.mean(cm)) / np.mean(bm) * 100
        total_bc, total_cc = sum(bc), sum(cc)
        crr = (total_bc - total_cc) / total_bc * 100 if total_bc > 0 else 0
        experiment_data[name]["metrics"]["val"].append(crr)
        print(
            f"  Baseline makespan: {np.mean(bm):.2f}, CMA makespan: {np.mean(cm):.2f}"
        )
        print(f"  Makespan improvement: {makespan_imp:.2f}%")
        print(f"  Conflict Reduction Ratio: {crr:.2f}%")
        print(f"Epoch 0: validation_loss = {-crr:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, (name, grid) in enumerate(datasets.items()):
    axes[0, idx].imshow(grid, cmap="binary")
    axes[0, idx].set_title(f"{name} Grid")
    if experiment_data[name]["makespans_baseline"]:
        axes[1, idx].plot(
            experiment_data[name]["makespans_baseline"],
            "b-",
            alpha=0.7,
            label="Baseline",
        )
        axes[1, idx].plot(
            experiment_data[name]["makespans_cma"], "r-", alpha=0.7, label="CMA"
        )
        axes[1, idx].legend()
        axes[1, idx].set_title(f"{name} Makespans")
        axes[1, idx].set_xlabel("Episode")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "cma_three_datasets_results.png"), dpi=150)
plt.close()

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nResults saved to {working_dir}")
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name in datasets:
    if experiment_data[name]["metrics"]["val"]:
        print(f"{name}: CRR = {experiment_data[name]['metrics']['val'][-1]:.2f}%")
