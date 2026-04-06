import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2026-04-04_15-26-17_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_f6cf09083b294923bda801222277ed5a_proc_11658/experiment_data.npy",
    "experiments/2026-04-04_15-26-17_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_4e7cfc76656942fb88cb2f36228f29e1_proc_11659/experiment_data.npy",
    "experiments/2026-04-04_15-26-17_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_8a2a450de7d04cbca4b1f78a3bc468fd_proc_11657/experiment_data.npy",
]

try:
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    print(f"Successfully loaded {len(all_experiment_data)} experiment files")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if all_experiment_data:
    # Plot 1: Aggregated Heatmap with Mean Improvement
    try:
        all_improvement_matrices = []
        scaling_factors = None
        penalty_weights = None

        for exp_data in all_experiment_data:
            data = exp_data.get("scaling_factor_tuning", {})
            tuning_results = data.get("tuning_results", [])
            if tuning_results:
                if scaling_factors is None:
                    scaling_factors = sorted(
                        set(r["scaling_factor"] for r in tuning_results)
                    )
                    penalty_weights = sorted(
                        set(r["penalty_weight"] for r in tuning_results)
                    )
                improvement_matrix = np.zeros(
                    (len(scaling_factors), len(penalty_weights))
                )
                for r in tuning_results:
                    i = scaling_factors.index(r["scaling_factor"])
                    j = penalty_weights.index(r["penalty_weight"])
                    improvement_matrix[i, j] = r["improvement"] * 100
                all_improvement_matrices.append(improvement_matrix)

        if all_improvement_matrices:
            mean_improvement = np.mean(all_improvement_matrices, axis=0)
            std_improvement = np.std(all_improvement_matrices, axis=0)
            se_improvement = std_improvement / np.sqrt(len(all_improvement_matrices))

            plt.figure(figsize=(10, 8))
            im = plt.imshow(mean_improvement, cmap="RdYlGn", aspect="auto")
            plt.xticks(range(len(penalty_weights)), [f"{pw}" for pw in penalty_weights])
            plt.yticks(range(len(scaling_factors)), [f"{sf}" for sf in scaling_factors])
            plt.xlabel("Penalty Weight")
            plt.ylabel("Scaling Factor")
            plt.title(
                f"MAPF Scaling Factor Tuning: Mean Improvement (%) Heatmap\nAggregated over {len(all_improvement_matrices)} runs (Mean ± SE)"
            )
            plt.colorbar(im, label="Mean Improvement (%)")
            for i in range(len(scaling_factors)):
                for j in range(len(penalty_weights)):
                    plt.text(
                        j,
                        i,
                        f"{mean_improvement[i, j]:.1f}%\n±{se_improvement[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_aggregated_scaling_tuning_heatmap.png"),
                dpi=150,
            )
            plt.close()
            print("Created aggregated heatmap plot")
    except Exception as e:
        print(f"Error creating aggregated heatmap plot: {e}")
        plt.close()

    # Plot 2: Aggregated Line Plot with Error Bars
    try:
        all_improvements_by_sf = {}
        scaling_factors = None
        penalty_weights = None

        for exp_data in all_experiment_data:
            data = exp_data.get("scaling_factor_tuning", {})
            tuning_results = data.get("tuning_results", [])
            if tuning_results:
                if scaling_factors is None:
                    scaling_factors = sorted(
                        set(r["scaling_factor"] for r in tuning_results)
                    )
                    penalty_weights = sorted(
                        set(r["penalty_weight"] for r in tuning_results)
                    )
                for sf in scaling_factors:
                    if sf not in all_improvements_by_sf:
                        all_improvements_by_sf[sf] = []
                    improvements = [
                        r["improvement"] * 100
                        for r in tuning_results
                        if r["scaling_factor"] == sf
                    ]
                    all_improvements_by_sf[sf].append(improvements)

        if all_improvements_by_sf:
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(scaling_factors)))
            for idx, sf in enumerate(scaling_factors):
                improvements_array = np.array(all_improvements_by_sf[sf])
                mean_imp = np.mean(improvements_array, axis=0)
                se_imp = np.std(improvements_array, axis=0) / np.sqrt(
                    len(improvements_array)
                )
                plt.errorbar(
                    penalty_weights,
                    mean_imp,
                    yerr=se_imp,
                    fmt="o-",
                    label=f"SF={sf} (Mean ± SE)",
                    linewidth=2,
                    markersize=8,
                    capsize=4,
                    color=colors[idx],
                )
            plt.xlabel("Penalty Weight")
            plt.ylabel("Improvement (%)")
            plt.title(
                f"MAPF: Mean Improvement vs Penalty Weight by Scaling Factor\nAggregated over {len(all_experiment_data)} runs with Standard Error Bars"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "mapf_aggregated_improvement_by_scaling_factor.png"
                ),
                dpi=150,
            )
            plt.close()
            print("Created aggregated line plot")
    except Exception as e:
        print(f"Error creating aggregated line plot: {e}")
        plt.close()

    # Plot 3: Aggregated Makespan Comparison with Shaded Error Regions
    try:
        all_baseline_makespans = []
        all_cma_makespans = []

        for exp_data in all_experiment_data:
            data = exp_data.get("scaling_factor_tuning", {})
            baseline_makespans = data.get("baseline_makespans", [])
            cma_makespans = data.get("cma_makespans", [])
            if baseline_makespans and cma_makespans:
                all_baseline_makespans.append(baseline_makespans)
                all_cma_makespans.append(cma_makespans)

        if all_baseline_makespans and all_cma_makespans:
            min_len = min(
                min(len(b) for b in all_baseline_makespans),
                min(len(c) for c in all_cma_makespans),
            )
            baseline_array = np.array([b[:min_len] for b in all_baseline_makespans])
            cma_array = np.array([c[:min_len] for c in all_cma_makespans])

            mean_baseline = np.mean(baseline_array, axis=0)
            se_baseline = np.std(baseline_array, axis=0) / np.sqrt(len(baseline_array))
            mean_cma = np.mean(cma_array, axis=0)
            se_cma = np.std(cma_array, axis=0) / np.sqrt(len(cma_array))

            plt.figure(figsize=(12, 5))
            x = np.arange(min_len)
            plt.plot(
                x,
                mean_baseline,
                "b-",
                alpha=0.9,
                label=f"Baseline Mean: {np.mean(mean_baseline):.2f}",
                linewidth=2,
            )
            plt.fill_between(
                x,
                mean_baseline - se_baseline,
                mean_baseline + se_baseline,
                color="b",
                alpha=0.2,
                label="Baseline SE",
            )
            plt.plot(
                x,
                mean_cma,
                "r-",
                alpha=0.9,
                label=f"CMA Mean: {np.mean(mean_cma):.2f}",
                linewidth=2,
            )
            plt.fill_between(
                x,
                mean_cma - se_cma,
                mean_cma + se_cma,
                color="r",
                alpha=0.2,
                label="CMA SE",
            )
            plt.xlabel("Episode")
            plt.ylabel("Makespan")
            plt.title(
                f"MAPF: Aggregated Makespan Comparison - Baseline vs CMA\nMean with Shaded Standard Error ({len(all_baseline_makespans)} runs)"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_aggregated_makespan_comparison.png"),
                dpi=150,
            )
            plt.close()
            print("Created aggregated makespan comparison plot")
    except Exception as e:
        print(f"Error creating aggregated makespan plot: {e}")
        plt.close()

    # Plot 4: Aggregated Conflict Counts with Error Bars
    try:
        all_baseline_conflicts = []
        all_cma_conflicts = []

        for exp_data in all_experiment_data:
            data = exp_data.get("scaling_factor_tuning", {})
            baseline_conflicts = data.get("conflict_counts_baseline", [])
            cma_conflicts = data.get("conflict_counts_cma", [])
            if baseline_conflicts and cma_conflicts:
                all_baseline_conflicts.append(baseline_conflicts)
                all_cma_conflicts.append(cma_conflicts)

        if all_baseline_conflicts and all_cma_conflicts:
            min_len = min(
                min(len(b) for b in all_baseline_conflicts),
                min(len(c) for c in all_cma_conflicts),
            )
            baseline_array = np.array([b[:min_len] for b in all_baseline_conflicts])
            cma_array = np.array([c[:min_len] for c in all_cma_conflicts])

            mean_baseline = np.mean(baseline_array, axis=0)
            se_baseline = np.std(baseline_array, axis=0) / np.sqrt(len(baseline_array))
            mean_cma = np.mean(cma_array, axis=0)
            se_cma = np.std(cma_array, axis=0) / np.sqrt(len(cma_array))

            plt.figure(figsize=(10, 5))
            x = np.arange(min_len)
            width = 0.35
            plt.bar(
                x - width / 2,
                mean_baseline,
                width,
                yerr=se_baseline,
                label="Baseline (Mean ± SE)",
                alpha=0.7,
                capsize=3,
            )
            plt.bar(
                x + width / 2,
                mean_cma,
                width,
                yerr=se_cma,
                label="CMA (Mean ± SE)",
                alpha=0.7,
                capsize=3,
            )
            plt.xlabel("Episode")
            plt.ylabel("Conflict Count")
            plt.title(
                f"MAPF: Aggregated Conflict Counts per Episode\nBaseline vs CMA ({len(all_baseline_conflicts)} runs)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mapf_aggregated_conflict_counts.png"),
                dpi=150,
            )
            plt.close()
            print("Created aggregated conflict counts plot")
    except Exception as e:
        print(f"Error creating aggregated conflict plot: {e}")
        plt.close()

    # Plot 5: Aggregated Summary Statistics with Error Bars
    try:
        all_final_improvements = []
        all_assignment_changes_mean = []
        all_best_sf = []
        all_best_pw = []

        for exp_data in all_experiment_data:
            data = exp_data.get("scaling_factor_tuning", {})
            best_config = data.get("best_config", {})
            final_improvement = data.get("final_improvement", 0)
            assignment_changes = data.get("assignment_changes", [])

            all_final_improvements.append(final_improvement * 100)
            all_assignment_changes_mean.append(
                np.mean(assignment_changes) * 100 if assignment_changes else 0
            )
            all_best_sf.append(best_config.get("scaling_factor", 0))
            all_best_pw.append(best_config.get("penalty_weight", 0))

        metrics = [
            "Final Improvement (%)",
            "Assignment Change Rate (%)",
            "Best Scaling Factor",
            "Best Penalty Weight",
        ]
        means = [
            np.mean(all_final_improvements),
            np.mean(all_assignment_changes_mean),
            np.mean(all_best_sf),
            np.mean(all_best_pw),
        ]
        ses = [
            np.std(all_final_improvements) / np.sqrt(len(all_final_improvements)),
            np.std(all_assignment_changes_mean)
            / np.sqrt(len(all_assignment_changes_mean)),
            np.std(all_best_sf) / np.sqrt(len(all_best_sf)),
            np.std(all_best_pw) / np.sqrt(len(all_best_pw)),
        ]
        colors = ["green" if means[0] > 0 else "red", "blue", "orange", "purple"]

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(metrics))
        plt.barh(y_pos, means, xerr=ses, color=colors, alpha=0.7, capsize=5)
        plt.yticks(y_pos, metrics)
        plt.xlabel("Value")
        plt.title(
            f"MAPF Scaling Factor Tuning: Aggregated Summary Statistics\nMean ± SE over {len(all_experiment_data)} runs"
        )
        for i, (m, s) in enumerate(zip(means, ses)):
            plt.text(m + s + 0.5, i, f"{m:.2f} ± {s:.2f}", va="center")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_aggregated_tuning_summary.png"), dpi=150
        )
        plt.close()
        print("Created aggregated summary plot")

        # Print summary metrics
        print("\n=== Aggregated Results Summary ===")
        print(f"Number of runs: {len(all_experiment_data)}")
        print(
            f"Mean Final Improvement: {np.mean(all_final_improvements):.2f}% ± {np.std(all_final_improvements)/np.sqrt(len(all_final_improvements)):.2f}% SE"
        )
        print(
            f"Mean Assignment Change Rate: {np.mean(all_assignment_changes_mean):.2f}% ± {np.std(all_assignment_changes_mean)/np.sqrt(len(all_assignment_changes_mean)):.2f}% SE"
        )
    except Exception as e:
        print(f"Error creating aggregated summary plot: {e}")
        plt.close()

print("\nAggregated plotting completed.")
