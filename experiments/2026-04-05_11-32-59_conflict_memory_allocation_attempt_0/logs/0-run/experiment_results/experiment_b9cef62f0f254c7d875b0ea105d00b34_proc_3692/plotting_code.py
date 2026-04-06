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
    # Plot 1: Training and Validation Loss Curves
    try:
        plt.figure(figsize=(10, 6))
        epochs = experiment_data["training"]["epochs"]
        train_loss = experiment_data["training"]["losses"]["train"]
        val_loss = experiment_data["training"]["losses"]["val"]
        plt.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Task Assignment - All Datasets Combined"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_training_validation_curves.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput Improvement by Dataset
    try:
        plt.figure(figsize=(12, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        avg_improvements = [
            np.mean(experiment_data["datasets"][n]["improvement"])
            for n in dataset_names
        ]
        colors = ["green" if imp > 0 else "red" for imp in avg_improvements]
        bars = plt.bar(
            range(len(dataset_names)),
            avg_improvements,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        plt.xticks(
            range(len(dataset_names)),
            [n.replace("_", "\n") for n in dataset_names],
            fontsize=9,
        )
        plt.ylabel("Average Throughput Improvement (%)")
        plt.xlabel("Warehouse Configuration")
        plt.title(
            "Throughput Improvement: Congestion-Aware vs Distance-Based Assignment\nAveraged Across Agent Counts (20, 30, 40)"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for i, (bar, imp) in enumerate(zip(bars, avg_improvements)):
            plt.text(
                i,
                imp + 0.5,
                f"{imp:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
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

    # Plot 3: Conflict Prevention Rate (CPR) by Dataset
    try:
        plt.figure(figsize=(12, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        avg_cpr = [
            np.mean(experiment_data["datasets"][n]["cpr"]) for n in dataset_names
        ]
        colors = ["green" if cpr > 0 else "red" for cpr in avg_cpr]
        bars = plt.bar(
            range(len(dataset_names)),
            avg_cpr,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        plt.xticks(
            range(len(dataset_names)),
            [n.replace("_", "\n") for n in dataset_names],
            fontsize=9,
        )
        plt.ylabel("Conflict Prevention Rate (%)")
        plt.xlabel("Warehouse Configuration")
        plt.title(
            "Conflict Prevention Rate (CPR): Congestion Reduction by Dataset\nAveraged Across Agent Counts (20, 30, 40)"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for i, (bar, cpr) in enumerate(zip(bars, avg_cpr)):
            plt.text(
                i,
                cpr + 0.5,
                f"{cpr:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_cpr_by_dataset.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating CPR plot: {e}")
        plt.close()

    # Plot 4: Throughput Comparison (Distance vs CMA) for All Datasets
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        dataset_names = list(experiment_data["datasets"].keys())
        for idx, name in enumerate(dataset_names[:5]):
            data = experiment_data["datasets"][name]
            n_agents = data["n_agents"]
            dist_tp = data["throughput"]["dist"]
            cma_tp = data["throughput"]["cma"]
            x = np.arange(len(n_agents))
            width = 0.35
            axes[idx].bar(
                x - width / 2,
                dist_tp,
                width,
                label="Distance-Based",
                color="blue",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2,
                cma_tp,
                width,
                label="Congestion-Aware",
                color="green",
                alpha=0.7,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f'{name.replace("_", " ").title()}')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis="y")
        axes[5].text(
            0.5,
            0.5,
            f'Overall Improvement: {experiment_data["best_improvement"]:.2f}%\nOverall CPR: {experiment_data["best_cpr"]:.2f}%',
            ha="center",
            va="center",
            fontsize=14,
            transform=axes[5].transAxes,
        )
        axes[5].axis("off")
        plt.suptitle(
            "Throughput Comparison: Distance-Based vs Congestion-Aware Assignment\nLeft: Distance-Based (Blue), Right: Congestion-Aware (Green)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_throughput_comparison_all_datasets.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 5: Congestion Comparison Across Datasets
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        dataset_names = list(experiment_data["datasets"].keys())
        for idx, name in enumerate(dataset_names[:5]):
            data = experiment_data["datasets"][name]
            n_agents = data["n_agents"]
            dist_cong = data["congestion"]["dist"]
            cma_cong = data["congestion"]["cma"]
            x = np.arange(len(n_agents))
            width = 0.35
            axes[idx].bar(
                x - width / 2,
                dist_cong,
                width,
                label="Distance-Based",
                color="red",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2,
                cma_cong,
                width,
                label="Congestion-Aware",
                color="orange",
                alpha=0.7,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Congestion Events")
            axes[idx].set_title(f'{name.replace("_", " ").title()}')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis="y")
        axes[5].axis("off")
        plt.suptitle(
            "Congestion Comparison: Distance-Based vs Congestion-Aware Assignment\nLeft: Distance-Based (Red), Right: Congestion-Aware (Orange)",
            fontsize=12,
        )
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

    # Plot 6: Heatmap of Improvement by Dataset and Agent Count
    try:
        dataset_names = list(experiment_data["datasets"].keys())
        agent_counts = experiment_data["datasets"][dataset_names[0]]["n_agents"]
        improvement_matrix = np.array(
            [experiment_data["datasets"][n]["improvement"] for n in dataset_names]
        )
        plt.figure(figsize=(10, 8))
        im = plt.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Throughput Improvement (%)")
        plt.xticks(range(len(agent_counts)), agent_counts)
        plt.yticks(
            range(len(dataset_names)),
            [n.replace("_", " ").title() for n in dataset_names],
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Warehouse Configuration")
        plt.title(
            "Throughput Improvement Heatmap\nCongestion-Aware vs Distance-Based Assignment"
        )
        for i in range(len(dataset_names)):
            for j in range(len(agent_counts)):
                plt.text(
                    j,
                    i,
                    f"{improvement_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_heatmap.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

print("All plots saved successfully.")
