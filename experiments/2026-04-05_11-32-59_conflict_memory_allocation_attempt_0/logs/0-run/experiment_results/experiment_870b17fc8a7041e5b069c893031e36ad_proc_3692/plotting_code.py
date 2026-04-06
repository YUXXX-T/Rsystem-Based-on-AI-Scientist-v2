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
    datasets = list(experiment_data["temporal_memory"].keys())
    agent_counts = [20, 30, 40]

    # Plot 1: Throughput Improvement by Dataset
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(datasets))
        improvements = [
            np.mean(experiment_data["temporal_memory"][d]["improvement"])
            for d in datasets
        ]
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        bars = plt.bar(x, improvements, color=colors, alpha=0.7, edgecolor="black")
        plt.xticks(x, [d.replace("_warehouse", "") for d in datasets], rotation=15)
        plt.ylabel("Throughput Improvement (%)")
        plt.xlabel("Dataset")
        plt.title(
            "Throughput Improvement: CATA vs Distance-Based\nAcross Warehouse Datasets (Averaged over Agent Counts)"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{imp:.1f}%",
                ha="center",
                fontsize=10,
            )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_throughput_improvement_by_dataset.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput improvement plot: {e}")
        plt.close()

    # Plot 2: Conflict Prevention Rate (CPR) by Dataset
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(datasets))
        cprs = [np.mean(experiment_data["temporal_memory"][d]["cpr"]) for d in datasets]
        colors = ["blue" if cpr > 0 else "orange" for cpr in cprs]
        bars = plt.bar(x, cprs, color=colors, alpha=0.7, edgecolor="black")
        plt.xticks(x, [d.replace("_warehouse", "") for d in datasets], rotation=15)
        plt.ylabel("Conflict Prevention Rate (%)")
        plt.xlabel("Dataset")
        plt.title(
            "Conflict Prevention Rate (CPR): CATA vs Distance-Based\nAcross Warehouse Datasets (Averaged over Agent Counts)"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for i, (bar, cpr) in enumerate(zip(bars, cprs)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{cpr:.1f}%",
                ha="center",
                fontsize=10,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_cpr_by_dataset.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating CPR plot: {e}")
        plt.close()

    # Plot 3: Throughput vs Agent Count Comparison (All Datasets)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, ds in enumerate(datasets):
            cata_tp = experiment_data["temporal_memory"][ds]["throughput"]
            baseline_tp = experiment_data["baseline"][ds]["throughput"]
            axes[idx].plot(
                agent_counts,
                cata_tp,
                "o-",
                color="green",
                label="CATA (Congestion-Aware)",
                linewidth=2,
                markersize=8,
            )
            axes[idx].plot(
                agent_counts,
                baseline_tp,
                "s--",
                color="blue",
                label="Distance-Based",
                linewidth=2,
                markersize=8,
            )
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(
                f'{ds.replace("_warehouse", "").replace("_", " ").title()} Warehouse'
            )
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(agent_counts)
        plt.suptitle(
            "Throughput Comparison: CATA vs Distance-Based by Agent Count", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_throughput_vs_agents_all_datasets.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput vs agents plot: {e}")
        plt.close()

    # Plot 4: Congestion Comparison (All Datasets)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, ds in enumerate(datasets):
            cata_cong = experiment_data["temporal_memory"][ds]["congestion"]
            baseline_cong = experiment_data["baseline"][ds]["congestion"]
            x = np.arange(len(agent_counts))
            width = 0.35
            axes[idx].bar(
                x - width / 2,
                baseline_cong,
                width,
                label="Distance-Based",
                color="blue",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2, cata_cong, width, label="CATA", color="green", alpha=0.7
            )
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Congestion Events")
            axes[idx].set_title(
                f'{ds.replace("_warehouse", "").replace("_", " ").title()} Warehouse'
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(agent_counts)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        plt.suptitle("Congestion Events Comparison: Lower is Better", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_congestion_comparison_all_datasets.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion comparison plot: {e}")
        plt.close()

    # Plot 5: Summary Metrics Heatmap
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Improvement heatmap
        imp_matrix = np.array(
            [experiment_data["temporal_memory"][ds]["improvement"] for ds in datasets]
        )
        im1 = axes[0].imshow(imp_matrix, cmap="RdYlGn", aspect="auto")
        axes[0].set_xticks(range(len(agent_counts)))
        axes[0].set_xticklabels(agent_counts)
        axes[0].set_yticks(range(len(datasets)))
        axes[0].set_yticklabels([d.replace("_warehouse", "") for d in datasets])
        axes[0].set_xlabel("Number of Agents")
        axes[0].set_ylabel("Dataset")
        axes[0].set_title("Throughput Improvement (%)")
        for i in range(len(datasets)):
            for j in range(len(agent_counts)):
                axes[0].text(
                    j,
                    i,
                    f"{imp_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
        plt.colorbar(im1, ax=axes[0], label="Improvement (%)")

        # CPR heatmap
        cpr_matrix = np.array(
            [experiment_data["temporal_memory"][ds]["cpr"] for ds in datasets]
        )
        im2 = axes[1].imshow(cpr_matrix, cmap="Blues", aspect="auto")
        axes[1].set_xticks(range(len(agent_counts)))
        axes[1].set_xticklabels(agent_counts)
        axes[1].set_yticks(range(len(datasets)))
        axes[1].set_yticklabels([d.replace("_warehouse", "") for d in datasets])
        axes[1].set_xlabel("Number of Agents")
        axes[1].set_ylabel("Dataset")
        axes[1].set_title("Conflict Prevention Rate (%)")
        for i in range(len(datasets)):
            for j in range(len(agent_counts)):
                axes[1].text(
                    j,
                    i,
                    f"{cpr_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
        plt.colorbar(im2, ax=axes[1], label="CPR (%)")

        plt.suptitle(
            "Summary Heatmaps: Left - Throughput Improvement, Right - Conflict Prevention Rate",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_summary_heatmaps.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary heatmap plot: {e}")
        plt.close()

    # Plot 6: Overall Performance Summary
    try:
        plt.figure(figsize=(12, 6))
        overall_imp = np.mean(
            [
                np.mean(experiment_data["temporal_memory"][d]["improvement"])
                for d in datasets
            ]
        )
        overall_cpr = np.mean(
            [np.mean(experiment_data["temporal_memory"][d]["cpr"]) for d in datasets]
        )

        metrics = ["Throughput\nImprovement", "Conflict\nPrevention Rate"]
        values = [overall_imp, overall_cpr]
        colors = ["green", "blue"]

        bars = plt.bar(
            metrics, values, color=colors, alpha=0.7, edgecolor="black", width=0.5
        )
        plt.ylabel("Percentage (%)")
        plt.title(
            "Overall CATA Performance vs Distance-Based Baseline\nAveraged Across All Datasets and Agent Configurations"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.2f}%",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_overall_performance_summary.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating overall performance summary plot: {e}")
        plt.close()

print("All plots saved successfully.")
