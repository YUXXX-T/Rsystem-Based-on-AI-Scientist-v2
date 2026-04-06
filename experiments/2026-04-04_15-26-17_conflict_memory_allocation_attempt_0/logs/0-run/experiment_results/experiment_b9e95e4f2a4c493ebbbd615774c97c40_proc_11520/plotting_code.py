import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    # Plot 1: Makespan Comparison Over Episodes
    try:
        plt.figure(figsize=(10, 6))
        cma_data = experiment_data["cma_experiment"]
        baseline_makespans = cma_data["baseline_makespans"]
        cma_makespans = cma_data["cma_makespans"]
        episodes = np.arange(len(baseline_makespans))

        plt.plot(episodes, baseline_makespans, "b-", alpha=0.7, label="Baseline")
        plt.plot(episodes, cma_makespans, "r-", alpha=0.7, label="CMA")
        plt.axhline(
            np.mean(baseline_makespans),
            color="b",
            linestyle="--",
            label=f"Baseline Mean: {np.mean(baseline_makespans):.2f}",
        )
        plt.axhline(
            np.mean(cma_makespans),
            color="r",
            linestyle="--",
            label=f"CMA Mean: {np.mean(cma_makespans):.2f}",
        )
        plt.xlabel("Episode")
        plt.ylabel("Makespan")
        plt.title(
            "CMA Experiment: Makespan Comparison Over Episodes\nBaseline vs Conflict Memory Allocation"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "cma_makespan_comparison.png"), dpi=150)
        plt.close()
        print("Saved: cma_makespan_comparison.png")
    except Exception as e:
        print(f"Error creating makespan comparison plot: {e}")
        plt.close()

    # Plot 2: Improvement Distribution
    try:
        plt.figure(figsize=(10, 6))
        improvements = cma_data["improvements"]
        plt.hist(improvements, bins=20, edgecolor="black", alpha=0.7, color="green")
        plt.axvline(
            np.mean(improvements),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(improvements)*100:.2f}%",
        )
        plt.xlabel("Improvement Ratio")
        plt.ylabel("Frequency")
        plt.title(
            "CMA Experiment: Distribution of Makespan Improvements\n(Positive = CMA better than Baseline)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "cma_improvement_distribution.png"), dpi=150
        )
        plt.close()
        print("Saved: cma_improvement_distribution.png")
    except Exception as e:
        print(f"Error creating improvement distribution plot: {e}")
        plt.close()

    # Plot 3: Conflict Counts Comparison
    try:
        plt.figure(figsize=(10, 6))
        baseline_conflicts = cma_data["conflict_counts_baseline"]
        cma_conflicts = cma_data["conflict_counts_cma"]
        episodes = np.arange(len(baseline_conflicts))

        plt.bar(
            episodes - 0.2,
            baseline_conflicts,
            width=0.4,
            label="Baseline",
            alpha=0.7,
            color="blue",
        )
        plt.bar(
            episodes + 0.2,
            cma_conflicts,
            width=0.4,
            label="CMA",
            alpha=0.7,
            color="red",
        )
        plt.xlabel("Episode")
        plt.ylabel("Conflict Count")
        plt.title(
            "CMA Experiment: Conflict Counts Comparison\nBaseline vs CMA per Episode"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(os.path.join(working_dir, "cma_conflict_counts.png"), dpi=150)
        plt.close()
        print("Saved: cma_conflict_counts.png")
    except Exception as e:
        print(f"Error creating conflict counts plot: {e}")
        plt.close()

    # Plot 4: Parameter Sweep Results
    try:
        plt.figure(figsize=(10, 6))
        sweep_results = experiment_data.get("sweep_results", [])
        if sweep_results:
            penalty_weights = [r[0] for r in sweep_results]
            improvements_sweep = [r[1] * 100 for r in sweep_results]
            best_pw = experiment_data.get("best_penalty_weight", None)

            plt.bar(
                range(len(penalty_weights)),
                improvements_sweep,
                color="steelblue",
                edgecolor="black",
            )
            plt.xticks(range(len(penalty_weights)), [str(pw) for pw in penalty_weights])
            plt.xlabel("Penalty Weight")
            plt.ylabel("Improvement (%)")
            plt.title(
                f"CMA Parameter Sweep: Effect of Penalty Weight on Improvement\nBest Penalty Weight: {best_pw}"
            )
            plt.grid(True, alpha=0.3, axis="y")
            plt.savefig(os.path.join(working_dir, "cma_parameter_sweep.png"), dpi=150)
            plt.close()
            print("Saved: cma_parameter_sweep.png")
    except Exception as e:
        print(f"Error creating parameter sweep plot: {e}")
        plt.close()

    # Plot 5: Assignment Changes Analysis
    try:
        plt.figure(figsize=(10, 6))
        assignment_changes = cma_data["assignment_changes"]
        change_rate = np.mean(assignment_changes) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Assignment changes over episodes
        ax1.bar(
            range(len(assignment_changes)),
            assignment_changes,
            color="purple",
            alpha=0.7,
        )
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Assignment Changed (0/1)")
        ax1.set_title("Assignment Changes per Episode")

        # Right: Pie chart of change rate
        ax2.pie(
            [change_rate, 100 - change_rate],
            labels=["Changed", "Same"],
            autopct="%1.1f%%",
            colors=["purple", "lightgray"],
        )
        ax2.set_title(f"Overall Assignment Change Rate: {change_rate:.1f}%")

        plt.suptitle(
            "CMA Experiment: Assignment Change Analysis\nHow often CMA chooses different task assignments than Baseline"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cma_assignment_changes.png"), dpi=150)
        plt.close()
        print("Saved: cma_assignment_changes.png")
    except Exception as e:
        print(f"Error creating assignment changes plot: {e}")
        plt.close()

    # Print summary metrics
    print("\n=== Experiment Summary ===")
    print(f"Baseline Mean Makespan: {np.mean(cma_data['baseline_makespans']):.2f}")
    print(f"CMA Mean Makespan: {np.mean(cma_data['cma_makespans']):.2f}")
    print(f"Mean Improvement: {np.mean(cma_data['improvements'])*100:.2f}%")
    print(f"Final Improvement: {experiment_data.get('final_improvement', 'N/A')}")
