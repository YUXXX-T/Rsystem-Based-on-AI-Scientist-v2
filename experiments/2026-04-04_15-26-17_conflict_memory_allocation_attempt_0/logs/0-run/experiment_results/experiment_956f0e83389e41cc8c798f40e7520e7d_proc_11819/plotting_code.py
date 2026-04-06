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
    datasets = ["warehouse", "maze", "random_grid"]

    # Plot 1: Makespan comparison across all datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, name in enumerate(datasets):
            data = experiment_data.get(name, {})
            baseline = data.get("makespans_baseline", [])
            cma = data.get("makespans_cma", [])
            if baseline and cma:
                x = np.arange(len(baseline))
                axes[idx].plot(x, baseline, "b-", alpha=0.7, label="Baseline")
                axes[idx].plot(x, cma, "r-", alpha=0.7, label="CMA")
                axes[idx].axhline(
                    np.mean(baseline), color="b", linestyle="--", alpha=0.5
                )
                axes[idx].axhline(np.mean(cma), color="r", linestyle="--", alpha=0.5)
                axes[idx].set_xlabel("Episode")
                axes[idx].set_ylabel("Makespan")
                axes[idx].set_title(
                    f'{name.replace("_", " ").title()}\nMean: Baseline={np.mean(baseline):.2f}, CMA={np.mean(cma):.2f}'
                )
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        plt.suptitle(
            "MAPF: Makespan Comparison - Baseline vs CMA Across All Datasets\nLower Makespan is Better",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_all_datasets_makespan_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating makespan comparison plot: {e}")
        plt.close()

    # Plot 2: Conflict counts comparison across all datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, name in enumerate(datasets):
            data = experiment_data.get(name, {})
            baseline = data.get("conflicts_baseline", [])
            cma = data.get("conflicts_cma", [])
            if baseline and cma:
                x = np.arange(len(baseline))
                width = 0.35
                axes[idx].bar(
                    x - width / 2,
                    baseline,
                    width,
                    label="Baseline",
                    alpha=0.7,
                    color="blue",
                )
                axes[idx].bar(
                    x + width / 2, cma, width, label="CMA", alpha=0.7, color="red"
                )
                axes[idx].set_xlabel("Episode")
                axes[idx].set_ylabel("Conflict Count")
                axes[idx].set_title(
                    f'{name.replace("_", " ").title()}\nTotal: Baseline={sum(baseline)}, CMA={sum(cma)}'
                )
                axes[idx].legend()
        plt.suptitle(
            "MAPF: Conflict Counts per Episode - Baseline vs CMA\nFewer Conflicts is Better",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_all_datasets_conflict_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating conflict comparison plot: {e}")
        plt.close()

    # Plot 3: Box plots for statistical comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        makespan_data, conflict_data, labels = [], [], []
        for name in datasets:
            data = experiment_data.get(name, {})
            baseline_m = data.get("makespans_baseline", [])
            cma_m = data.get("makespans_cma", [])
            baseline_c = data.get("conflicts_baseline", [])
            cma_c = data.get("conflicts_cma", [])
            if baseline_m and cma_m:
                makespan_data.extend([baseline_m, cma_m])
                conflict_data.extend([baseline_c, cma_c])
                labels.extend([f"{name[:4]}_BL", f"{name[:4]}_CMA"])

        bp1 = axes[0].boxplot(makespan_data, labels=labels, patch_artist=True)
        colors = ["lightblue", "lightcoral"] * 3
        for patch, color in zip(bp1["boxes"], colors):
            patch.set_facecolor(color)
        axes[0].set_ylabel("Makespan")
        axes[0].set_title(
            "Makespan Distribution\nLeft: Baseline, Right: CMA per Dataset"
        )
        axes[0].tick_params(axis="x", rotation=45)

        bp2 = axes[1].boxplot(conflict_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
        axes[1].set_ylabel("Conflict Count")
        axes[1].set_title(
            "Conflict Distribution\nLeft: Baseline, Right: CMA per Dataset"
        )
        axes[1].tick_params(axis="x", rotation=45)

        plt.suptitle("MAPF: Statistical Comparison Across All Datasets", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_all_datasets_boxplot_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating boxplot: {e}")
        plt.close()

    # Plot 4: Summary bar chart of improvements
    try:
        plt.figure(figsize=(10, 6))
        makespan_improvements, conflict_reductions = [], []
        for name in datasets:
            data = experiment_data.get(name, {})
            baseline_m = data.get("makespans_baseline", [])
            cma_m = data.get("makespans_cma", [])
            baseline_c = data.get("conflicts_baseline", [])
            cma_c = data.get("conflicts_cma", [])
            if baseline_m and cma_m:
                m_imp = (
                    (np.mean(baseline_m) - np.mean(cma_m)) / np.mean(baseline_m) * 100
                )
                c_red = (
                    (sum(baseline_c) - sum(cma_c)) / sum(baseline_c) * 100
                    if sum(baseline_c) > 0
                    else 0
                )
                makespan_improvements.append(m_imp)
                conflict_reductions.append(c_red)

        x = np.arange(len(datasets))
        width = 0.35
        plt.bar(
            x - width / 2,
            makespan_improvements,
            width,
            label="Makespan Improvement (%)",
            color="steelblue",
        )
        plt.bar(
            x + width / 2,
            conflict_reductions,
            width,
            label="Conflict Reduction (%)",
            color="coral",
        )
        plt.axhline(0, color="black", linewidth=0.5)
        plt.xlabel("Dataset")
        plt.ylabel("Improvement (%)")
        plt.title(
            "MAPF: CMA Performance Summary\nMakespan Improvement & Conflict Reduction Rate (Higher is Better)"
        )
        plt.xticks(x, [n.replace("_", " ").title() for n in datasets])
        plt.legend()
        for i, (m, c) in enumerate(zip(makespan_improvements, conflict_reductions)):
            plt.text(i - width / 2, m + 0.5, f"{m:.1f}%", ha="center", fontsize=9)
            plt.text(i + width / 2, c + 0.5, f"{c:.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_all_datasets_improvement_summary.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # Plot 5: CRR (Conflict Reduction Ratio) validation metrics
    try:
        plt.figure(figsize=(8, 5))
        crr_values = []
        for name in datasets:
            data = experiment_data.get(name, {})
            val_metrics = data.get("metrics", {}).get("val", [])
            if val_metrics:
                crr_values.append(val_metrics[-1])
            else:
                crr_values.append(0)

        colors = ["green" if v > 0 else "red" for v in crr_values]
        plt.bar(
            [n.replace("_", " ").title() for n in datasets],
            crr_values,
            color=colors,
            alpha=0.7,
        )
        plt.axhline(0, color="black", linewidth=0.5)
        plt.ylabel("Conflict Reduction Ratio (%)")
        plt.title(
            "MAPF: Final Validation CRR by Dataset\nGreen=Positive Improvement, Red=Negative"
        )
        for i, v in enumerate(crr_values):
            plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mapf_all_datasets_crr_validation.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating CRR plot: {e}")
        plt.close()

print("Plotting completed.")
