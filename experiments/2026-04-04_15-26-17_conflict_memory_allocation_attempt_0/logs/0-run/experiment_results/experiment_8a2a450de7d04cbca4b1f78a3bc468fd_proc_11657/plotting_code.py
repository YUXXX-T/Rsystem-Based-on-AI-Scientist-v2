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
    data = experiment_data.get("scaling_factor_tuning", {})

    # Plot 1: Heatmap of improvements across scaling factors and penalty weights
    try:
        tuning_results = data.get("tuning_results", [])
        if tuning_results:
            scaling_factors = sorted(set(r["scaling_factor"] for r in tuning_results))
            penalty_weights = sorted(set(r["penalty_weight"] for r in tuning_results))
            improvement_matrix = np.zeros((len(scaling_factors), len(penalty_weights)))
            for r in tuning_results:
                i = scaling_factors.index(r["scaling_factor"])
                j = penalty_weights.index(r["penalty_weight"])
                improvement_matrix[i, j] = r["improvement"] * 100

            plt.figure(figsize=(10, 8))
            im = plt.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
            plt.xticks(range(len(penalty_weights)), [f"{pw}" for pw in penalty_weights])
            plt.yticks(range(len(scaling_factors)), [f"{sf}" for sf in scaling_factors])
            plt.xlabel("Penalty Weight")
            plt.ylabel("Scaling Factor")
            plt.title(
                "MAPF Scaling Factor Tuning: Improvement (%) Heatmap\nCMA vs Baseline Makespan Reduction"
            )
            plt.colorbar(im, label="Improvement (%)")
            for i in range(len(scaling_factors)):
                for j in range(len(penalty_weights)):
                    plt.text(
                        j,
                        i,
                        f"{improvement_matrix[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_scaling_tuning_heatmap.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

    # Plot 2: Line plot of improvement by scaling factor
    try:
        tuning_results = data.get("tuning_results", [])
        if tuning_results:
            scaling_factors = sorted(set(r["scaling_factor"] for r in tuning_results))
            penalty_weights = sorted(set(r["penalty_weight"] for r in tuning_results))

            plt.figure(figsize=(10, 6))
            for sf in scaling_factors:
                improvements = [
                    r["improvement"] * 100
                    for r in tuning_results
                    if r["scaling_factor"] == sf
                ]
                plt.plot(
                    penalty_weights,
                    improvements,
                    "o-",
                    label=f"SF={sf}",
                    linewidth=2,
                    markersize=8,
                )
            plt.xlabel("Penalty Weight")
            plt.ylabel("Improvement (%)")
            plt.title(
                "MAPF: Improvement vs Penalty Weight by Scaling Factor\nHigher is Better"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_improvement_by_scaling_factor.png"),
                dpi=150,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating line plot: {e}")
        plt.close()

    # Plot 3: Makespan comparison over episodes
    try:
        baseline_makespans = data.get("baseline_makespans", [])
        cma_makespans = data.get("cma_makespans", [])
        if baseline_makespans and cma_makespans:
            plt.figure(figsize=(12, 5))
            x = np.arange(len(baseline_makespans))
            plt.plot(x, baseline_makespans, "b-", alpha=0.7, label="Baseline")
            plt.plot(x, cma_makespans, "r-", alpha=0.7, label="CMA (Conflict Memory)")
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
                "MAPF: Makespan Comparison - Baseline vs CMA\nLower Makespan is Better"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_makespan_comparison.png"), dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Error creating makespan plot: {e}")
        plt.close()

    # Plot 4: Conflict counts comparison
    try:
        baseline_conflicts = data.get("conflict_counts_baseline", [])
        cma_conflicts = data.get("conflict_counts_cma", [])
        if baseline_conflicts and cma_conflicts:
            plt.figure(figsize=(10, 5))
            x = np.arange(len(baseline_conflicts))
            plt.bar(x - 0.2, baseline_conflicts, 0.4, label="Baseline", alpha=0.7)
            plt.bar(x + 0.2, cma_conflicts, 0.4, label="CMA", alpha=0.7)
            plt.xlabel("Episode")
            plt.ylabel("Conflict Count")
            plt.title(
                "MAPF: Conflict Counts per Episode\nBaseline vs CMA (Conflict Memory Approach)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "mapf_conflict_counts.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating conflict plot: {e}")
        plt.close()

    # Plot 5: Summary statistics
    try:
        best_config = data.get("best_config", {})
        final_improvement = data.get("final_improvement", 0)
        assignment_changes = data.get("assignment_changes", [])

        plt.figure(figsize=(10, 6))
        metrics = [
            "Final Improvement (%)",
            "Assignment Change Rate (%)",
            "Best Scaling Factor",
            "Best Penalty Weight",
        ]
        values = [
            final_improvement * 100,
            np.mean(assignment_changes) * 100 if assignment_changes else 0,
            best_config.get("scaling_factor", 0),
            best_config.get("penalty_weight", 0),
        ]
        colors = ["green" if values[0] > 0 else "red", "blue", "orange", "purple"]
        plt.barh(metrics, values, color=colors, alpha=0.7)
        plt.xlabel("Value")
        plt.title(
            "MAPF Scaling Factor Tuning: Summary Statistics\nBest Configuration Results"
        )
        for i, v in enumerate(values):
            plt.text(v + 0.1, i, f"{v:.2f}", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "mapf_tuning_summary.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

print("Plotting completed.")
