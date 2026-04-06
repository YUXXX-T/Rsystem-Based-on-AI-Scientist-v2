import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2026-04-04_15-26-17_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_bb31b64538b94c5aa7c50c59a71c2c4f_proc_11519/experiment_data.npy",
    "experiments/2026-04-04_15-26-17_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_b9e95e4f2a4c493ebbbd615774c97c40_proc_11520/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        if experiment_data_path and "None" not in experiment_data_path:
            full_path = os.path.join(
                os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
            )
            if os.path.exists(full_path):
                experiment_data = np.load(full_path, allow_pickle=True).item()
                all_experiment_data.append(experiment_data)
                print(f"Loaded: {experiment_data_path}")
            else:
                print(f"File not found: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

if len(all_experiment_data) > 0:
    # Aggregate metrics across runs
    all_baseline_means = []
    all_cma_means = []
    all_improvement_means = []
    all_baseline_conflict_means = []
    all_cma_conflict_means = []
    all_change_rates = []

    for exp_data in all_experiment_data:
        cma_data = exp_data.get("cma_experiment", {})
        if cma_data:
            all_baseline_means.append(np.mean(cma_data.get("baseline_makespans", [])))
            all_cma_means.append(np.mean(cma_data.get("cma_makespans", [])))
            all_improvement_means.append(
                np.mean(cma_data.get("improvements", [])) * 100
            )
            all_baseline_conflict_means.append(
                np.mean(cma_data.get("conflict_counts_baseline", []))
            )
            all_cma_conflict_means.append(
                np.mean(cma_data.get("conflict_counts_cma", []))
            )
            all_change_rates.append(
                np.mean(cma_data.get("assignment_changes", [])) * 100
            )

    all_baseline_means = np.array(all_baseline_means)
    all_cma_means = np.array(all_cma_means)
    all_improvement_means = np.array(all_improvement_means)
    all_baseline_conflict_means = np.array(all_baseline_conflict_means)
    all_cma_conflict_means = np.array(all_cma_conflict_means)
    all_change_rates = np.array(all_change_rates)

    n_runs = len(all_baseline_means)

    # Plot 1: Aggregated Makespan Comparison with Error Bars
    try:
        plt.figure(figsize=(10, 6))
        methods = ["Baseline", "CMA"]
        means = [np.mean(all_baseline_means), np.mean(all_cma_means)]
        stderrs = [
            np.std(all_baseline_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
            np.std(all_cma_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
        ]

        bars = plt.bar(
            methods,
            means,
            yerr=stderrs,
            capsize=10,
            color=["blue", "red"],
            alpha=0.7,
            edgecolor="black",
        )
        plt.ylabel("Mean Makespan")
        plt.title(
            f"CMA Experiment: Aggregated Makespan Comparison\nMean ± SE across {n_runs} runs"
        )
        plt.legend([bars[0]], [f"SE bars (n={n_runs})"], loc="upper right")
        plt.grid(True, alpha=0.3, axis="y")

        for i, (m, s) in enumerate(zip(means, stderrs)):
            plt.text(i, m + s + 0.5, f"{m:.2f}±{s:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "aggregated_cma_makespan_comparison.png"), dpi=150
        )
        plt.close()
        print("Saved: aggregated_cma_makespan_comparison.png")
    except Exception as e:
        print(f"Error creating aggregated makespan comparison plot: {e}")
        plt.close()

    # Plot 2: Aggregated Improvement with Error Bars
    try:
        plt.figure(figsize=(10, 6))
        mean_improvement = np.mean(all_improvement_means)
        stderr_improvement = (
            np.std(all_improvement_means) / np.sqrt(n_runs) if n_runs > 1 else 0
        )

        plt.bar(
            ["CMA Improvement"],
            [mean_improvement],
            yerr=[stderr_improvement],
            capsize=10,
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
        plt.ylabel("Improvement (%)")
        plt.title(
            f"CMA Experiment: Aggregated Makespan Improvement\nMean ± SE across {n_runs} runs"
        )
        plt.text(
            0,
            mean_improvement + stderr_improvement + 0.5,
            f"{mean_improvement:.2f}% ± {stderr_improvement:.2f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "aggregated_cma_improvement.png"), dpi=150
        )
        plt.close()
        print("Saved: aggregated_cma_improvement.png")
    except Exception as e:
        print(f"Error creating aggregated improvement plot: {e}")
        plt.close()

    # Plot 3: Aggregated Conflict Counts with Error Bars
    try:
        plt.figure(figsize=(10, 6))
        methods = ["Baseline", "CMA"]
        conflict_means = [
            np.mean(all_baseline_conflict_means),
            np.mean(all_cma_conflict_means),
        ]
        conflict_stderrs = [
            np.std(all_baseline_conflict_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
            np.std(all_cma_conflict_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
        ]

        bars = plt.bar(
            methods,
            conflict_means,
            yerr=conflict_stderrs,
            capsize=10,
            color=["blue", "red"],
            alpha=0.7,
            edgecolor="black",
        )
        plt.ylabel("Mean Conflict Count")
        plt.title(
            f"CMA Experiment: Aggregated Conflict Counts\nMean ± SE across {n_runs} runs"
        )
        plt.legend([bars[0]], [f"SE bars (n={n_runs})"], loc="upper right")
        plt.grid(True, alpha=0.3, axis="y")

        for i, (m, s) in enumerate(zip(conflict_means, conflict_stderrs)):
            plt.text(i, m + s + 0.2, f"{m:.2f}±{s:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "aggregated_cma_conflict_counts.png"), dpi=150
        )
        plt.close()
        print("Saved: aggregated_cma_conflict_counts.png")
    except Exception as e:
        print(f"Error creating aggregated conflict counts plot: {e}")
        plt.close()

    # Plot 4: Summary Metrics Dashboard
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Subplot 1: Makespan comparison
        ax1 = axes[0]
        methods = ["Baseline", "CMA"]
        means = [np.mean(all_baseline_means), np.mean(all_cma_means)]
        stderrs = [
            np.std(all_baseline_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
            np.std(all_cma_means) / np.sqrt(n_runs) if n_runs > 1 else 0,
        ]
        ax1.bar(
            methods, means, yerr=stderrs, capsize=8, color=["blue", "red"], alpha=0.7
        )
        ax1.set_ylabel("Mean Makespan")
        ax1.set_title("Makespan Comparison")
        ax1.grid(True, alpha=0.3, axis="y")

        # Subplot 2: Improvement
        ax2 = axes[1]
        ax2.bar(
            ["Improvement"],
            [mean_improvement],
            yerr=[stderr_improvement],
            capsize=8,
            color="green",
            alpha=0.7,
        )
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("CMA Improvement")
        ax2.grid(True, alpha=0.3, axis="y")

        # Subplot 3: Assignment change rate
        ax3 = axes[2]
        mean_change = np.mean(all_change_rates)
        stderr_change = np.std(all_change_rates) / np.sqrt(n_runs) if n_runs > 1 else 0
        ax3.bar(
            ["Change Rate"],
            [mean_change],
            yerr=[stderr_change],
            capsize=8,
            color="purple",
            alpha=0.7,
        )
        ax3.set_ylabel("Assignment Change Rate (%)")
        ax3.set_title("Assignment Changes")
        ax3.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"CMA Experiment: Aggregated Summary Dashboard\n(Mean ± SE across {n_runs} runs)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "aggregated_cma_summary_dashboard.png"), dpi=150
        )
        plt.close()
        print("Saved: aggregated_cma_summary_dashboard.png")
    except Exception as e:
        print(f"Error creating summary dashboard: {e}")
        plt.close()

    # Print aggregated summary metrics
    print("\n=== Aggregated Experiment Summary ===")
    print(f"Number of runs: {n_runs}")
    print(
        f"Baseline Mean Makespan: {np.mean(all_baseline_means):.2f} ± {np.std(all_baseline_means)/np.sqrt(n_runs) if n_runs > 1 else 0:.2f} (SE)"
    )
    print(
        f"CMA Mean Makespan: {np.mean(all_cma_means):.2f} ± {np.std(all_cma_means)/np.sqrt(n_runs) if n_runs > 1 else 0:.2f} (SE)"
    )
    print(f"Mean Improvement: {mean_improvement:.2f}% ± {stderr_improvement:.2f}% (SE)")
    print(f"Mean Conflict (Baseline): {np.mean(all_baseline_conflict_means):.2f}")
    print(f"Mean Conflict (CMA): {np.mean(all_cma_conflict_means):.2f}")
    print(f"Mean Assignment Change Rate: {np.mean(all_change_rates):.2f}%")
else:
    print("No valid experiment data found to aggregate.")
