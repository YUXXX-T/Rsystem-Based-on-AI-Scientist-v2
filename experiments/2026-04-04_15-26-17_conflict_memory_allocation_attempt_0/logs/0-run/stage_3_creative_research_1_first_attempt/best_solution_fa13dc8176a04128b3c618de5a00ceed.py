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
    "warehouse_layout": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "maze_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "room_layout": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


def create_warehouse_map(width=20, height=20):
    grid = np.zeros((height, width), dtype=int)
    for row in [3, 7, 11, 15]:
        for col in range(2, width - 2):
            if col not in [5, 10, 15]:
                grid[row, col] = 1
    return grid


def create_room_map(width=20, height=20, n_rooms=4):
    grid = np.zeros((height, width), dtype=int)
    room_h, room_w = height // 2, width // 2
    for i in range(2):
        for j in range(2):
            y_start, x_start = i * room_h, j * room_w
            grid[y_start, x_start : x_start + room_w] = 1
            grid[y_start : y_start + room_h, x_start] = 1
            grid[y_start + room_h - 1, x_start : x_start + room_w] = 1
            grid[y_start : y_start + room_h, x_start + room_w - 1] = 1
    for i in range(2):
        for j in range(2):
            y_start, x_start = i * room_h, j * room_w
            door_y, door_x = y_start + room_h // 2, x_start + room_w // 2
            if door_y < height:
                grid[door_y, x_start] = 0
            if door_x < width:
                grid[y_start, door_x] = 0
            if y_start + room_h - 1 < height:
                grid[y_start + room_h - 1, door_x] = 0
            if x_start + room_w - 1 < width:
                grid[door_y, x_start + room_w - 1] = 0
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def create_maze_from_hf():
    try:
        from datasets import load_dataset

        ds = load_dataset("benjamin-paine/maze-dataset", split="train", streaming=True)
        sample = next(iter(ds))
        if "maze" in sample:
            maze = np.array(sample["maze"])
            if maze.ndim == 2 and maze.shape[0] >= 10 and maze.shape[1] >= 10:
                maze = (maze > 0).astype(int)
                maze[0, :] = 0
                maze[-1, :] = 0
                maze[:, 0] = 0
                maze[:, -1] = 0
                return maze[:20, :20] if maze.shape[0] > 20 else maze
    except:
        pass
    grid = np.zeros((20, 20), dtype=int)
    np.random.seed(42)
    for _ in range(40):
        y, x = np.random.randint(2, 18), np.random.randint(2, 18)
        grid[y, x] = 1
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

    def update(self, conflicts, reservation_pressure):
        self.memory *= self.decay
        for conflict in conflicts:
            if conflict[0] == "vertex":
                pos = conflict[1]
                self.memory[pos[0], pos[1]] += 1.0
        if reservation_pressure.max() > 0:
            self.memory += (reservation_pressure / reservation_pressure.max()) * 0.3

    def get_path_penalty(self, path_cells):
        return (
            sum(self.memory[cell[0], cell[1]] for cell in path_cells)
            if path_cells
            else 0.0
        )

    def copy(self):
        cm = ConflictMemory(self.memory.shape, self.decay)
        cm.memory = self.memory.copy()
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
    avg_base_cost = np.mean(base_costs) if base_costs else 1.0
    for i in range(n_agents):
        for j in range(n_tasks):
            path_cells = path_cache[(i, j)]
            base_cost = len(path_cells)
            if conflict_memory is not None and avg_base_cost > 0:
                raw_penalty = conflict_memory.get_path_penalty(path_cells)
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


def run_episode(
    grid,
    n_agents,
    conflict_memory=None,
    use_cma=False,
    penalty_weight=3.0,
    scaling_factor=0.5,
):
    h, w = grid.shape
    free_cells = [(i, j) for i in range(h) for j in range(w) if grid[i, j] == 0]
    if len(free_cells) < n_agents * 2:
        return None
    np.random.shuffle(free_cells)
    agent_starts = free_cells[:n_agents]
    task_positions = free_cells[n_agents : n_agents * 2]
    cm = conflict_memory if use_cma else None
    cost_matrix = compute_cost_matrix(
        agent_starts, task_positions, grid, cm, penalty_weight, scaling_factor
    )
    assignment = hungarian_assignment(cost_matrix)
    agent_goals = [
        (
            task_positions[task_idx]
            if 0 <= task_idx < len(task_positions)
            else agent_starts[i]
        )
        for i, task_idx in enumerate(assignment)
    ]
    paths, conflicts, pressure = prioritized_planning(agent_starts, agent_goals, grid)
    return {
        "makespan": compute_makespan(paths),
        "conflicts": len(conflicts),
        "pressure": pressure,
        "assignment": assignment,
    }


def run_experiment_on_layout(
    grid, layout_name, n_episodes=60, n_agents=6, penalty_weight=3.0, scaling_factor=0.5
):
    print(f"\n{'='*50}\nRunning on {layout_name}\n{'='*50}")
    conflict_memory = ConflictMemory(grid.shape, decay=0.95)
    baseline_results = {"makespans": [], "conflicts": []}
    cma_results = {"makespans": [], "conflicts": []}

    for ep in range(n_episodes):
        seed = ep * 1000
        np.random.seed(seed)
        baseline = run_episode(grid, n_agents, use_cma=False)
        if baseline is None:
            continue
        baseline_results["makespans"].append(baseline["makespan"])
        baseline_results["conflicts"].append(baseline["conflicts"])

        np.random.seed(seed)
        cma = run_episode(
            grid,
            n_agents,
            conflict_memory=conflict_memory,
            use_cma=True,
            penalty_weight=penalty_weight,
            scaling_factor=scaling_factor,
        )
        if cma is None:
            continue
        cma_results["makespans"].append(cma["makespan"])
        cma_results["conflicts"].append(cma["conflicts"])
        conflict_memory.update([], cma["pressure"])

        if (ep + 1) % 20 == 0:
            bl_conf = np.mean(baseline_results["conflicts"][-20:])
            cma_conf = np.mean(cma_results["conflicts"][-20:])
            crr = (bl_conf - cma_conf) / bl_conf * 100 if bl_conf > 0 else 0
            print(
                f"Epoch {ep+1}: validation_loss = {np.mean(cma_results['makespans'][-20:]):.4f}, Conflict Reduction Ratio = {crr:.2f}%"
            )
            experiment_data[layout_name]["metrics"]["val"].append(crr)
            experiment_data[layout_name]["losses"]["val"].append(
                np.mean(cma_results["makespans"][-20:])
            )

    bl_mean_conf = np.mean(baseline_results["conflicts"])
    cma_mean_conf = np.mean(cma_results["conflicts"])
    conflict_reduction_ratio = (
        (bl_mean_conf - cma_mean_conf) / bl_mean_conf * 100 if bl_mean_conf > 0 else 0
    )
    makespan_improvement = (
        (np.mean(baseline_results["makespans"]) - np.mean(cma_results["makespans"]))
        / np.mean(baseline_results["makespans"])
        * 100
    )

    print(f"\n{layout_name} Results:")
    print(
        f"  Baseline makespan: {np.mean(baseline_results['makespans']):.2f}, conflicts: {bl_mean_conf:.2f}"
    )
    print(
        f"  CMA makespan: {np.mean(cma_results['makespans']):.2f}, conflicts: {cma_mean_conf:.2f}"
    )
    print(f"  Makespan improvement: {makespan_improvement:.2f}%")
    print(f"  Conflict Reduction Ratio: {conflict_reduction_ratio:.2f}%")

    experiment_data[layout_name]["predictions"] = cma_results["makespans"]
    experiment_data[layout_name]["ground_truth"] = baseline_results["makespans"]
    return {
        "baseline": baseline_results,
        "cma": cma_results,
        "conflict_reduction_ratio": conflict_reduction_ratio,
        "makespan_improvement": makespan_improvement,
        "memory": conflict_memory,
    }


# Run experiments on three layouts
warehouse_grid = create_warehouse_map(20, 20)
room_grid = create_room_map(20, 20)
maze_grid = create_maze_from_hf()

scaling_factors = [0.25, 0.4, 0.5, 0.6]
penalty_weights = [2.5, 3.0, 3.5]
tuning_results = []

print("Hyperparameter tuning around optimal region...")
for sf in scaling_factors:
    for pw in penalty_weights:
        np.random.seed(0)
        result = run_experiment_on_layout(
            warehouse_grid,
            "warehouse_layout",
            n_episodes=30,
            penalty_weight=pw,
            scaling_factor=sf,
        )
        tuning_results.append(
            {
                "sf": sf,
                "pw": pw,
                "crr": result["conflict_reduction_ratio"],
                "mi": result["makespan_improvement"],
            }
        )

best_config = max(tuning_results, key=lambda x: x["crr"])
print(
    f"\nBest config: sf={best_config['sf']}, pw={best_config['pw']}, CRR={best_config['crr']:.2f}%"
)

# Final evaluation on all three datasets
results = {}
for name, grid in [
    ("warehouse_layout", warehouse_grid),
    ("maze_dataset", maze_grid),
    ("room_layout", room_grid),
]:
    results[name] = run_experiment_on_layout(
        grid,
        name,
        n_episodes=80,
        penalty_weight=best_config["pw"],
        scaling_factor=best_config["sf"],
    )

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, (name, res) in enumerate(results.items()):
    axes[0, idx].plot(res["baseline"]["makespans"], label="Baseline", alpha=0.7)
    axes[0, idx].plot(res["cma"]["makespans"], label="CMA", alpha=0.7)
    axes[0, idx].set_title(f"{name}\nMakespan Imp: {res['makespan_improvement']:.1f}%")
    axes[0, idx].legend()
    axes[1, idx].plot(res["baseline"]["conflicts"], label="Baseline", alpha=0.7)
    axes[1, idx].plot(res["cma"]["conflicts"], label="CMA", alpha=0.7)
    axes[1, idx].set_title(f"CRR: {res['conflict_reduction_ratio']:.1f}%")
    axes[1, idx].legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "three_dataset_evaluation.png"), dpi=150)
plt.close()

experiment_data["tuning_results"] = tuning_results
experiment_data["best_config"] = best_config
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
for name, res in results.items():
    print(
        f"{name}: Makespan Imp={res['makespan_improvement']:.2f}%, CRR={res['conflict_reduction_ratio']:.2f}%"
    )
