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
        train_loss = experiment_data["training"]["train_loss"]
        val_loss = experiment_data["training"]["val_loss"]
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            "CMA Model Training Curves\nCongestion Predictor for Warehouse Task Assignment"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_cma_training_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput Improvement by Dataset
    try:
        plt.figure(figsize=(10, 6))
        datasets = list(experiment_data["methods_comparison"].keys())
        avg_improvements = [
            np.mean(experiment_data["methods_comparison"][d]["improvement"])
            for d in datasets
        ]
        colors = ["green" if imp > 0 else "red" for imp in avg_improvements]

        bars = plt.bar(
            range(len(datasets)),
            avg_improvements,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        plt.xticks(
            range(len(datasets)), [d.replace("_", "\n") for d in datasets], fontsize=9
        )
        plt.ylabel("Average Throughput Improvement (%)")
        plt.xlabel("Warehouse Configuration")
        plt.title(
            "CMA vs Distance-Based: Throughput Improvement by Dataset\nWarehouse Congestion-Aware Task Assignment"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.grid(True, alpha=0.3, axis="y")

        for i, (bar, imp) in enumerate(zip(bars, avg_improvements)):
            plt.text(
                i, imp + 0.5, f"{imp:.1f}%", ha="center", va="bottom", fontweight="bold"
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
        plt.figure(figsize=(10, 6))
        datasets = list(experiment_data["methods_comparison"].keys())
        avg_cprs = [
            np.mean(experiment_data["methods_comparison"][d]["cpr"]) for d in datasets
        ]
        colors = ["green" if cpr > 0 else "red" for cpr in avg_cprs]

        bars = plt.bar(
            range(len(datasets)), avg_cprs, color=colors, alpha=0.7, edgecolor="black"
        )
        plt.xticks(
            range(len(datasets)), [d.replace("_", "\n") for d in datasets], fontsize=9
        )
        plt.ylabel("Conflict Prevention Rate (%)")
        plt.xlabel("Warehouse Configuration")
        plt.title(
            "CMA Conflict Prevention Rate by Dataset\nHigher is Better - Reduction in Congestion Events"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.grid(True, alpha=0.3, axis="y")

        for i, (bar, cpr) in enumerate(zip(bars, avg_cprs)):
            plt.text(
                i, cpr + 0.5, f"{cpr:.1f}%", ha="center", va="bottom", fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_cpr_by_dataset.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating CPR plot: {e}")
        plt.close()

    # Plot 4: Throughput Comparison by Agent Count for Each Dataset
    try:
        datasets = list(experiment_data["methods_comparison"].keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, dataset_name in enumerate(datasets):
            data = experiment_data["methods_comparison"][dataset_name]
            n_agents = data["n_agents"]
            dist_tp = data["distance"]["tp"]
            cma_tp = data["cma"]["tp"]

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
                x + width / 2, cma_tp, width, label="CMA", color="green", alpha=0.7
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f'{dataset_name.replace("_", " ").title()}')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Throughput Comparison: CMA vs Distance-Based by Agent Count", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_throughput_by_agents_all_datasets.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput by agents plot: {e}")
        plt.close()

    # Plot 5: Congestion Events Comparison
    try:
        datasets = list(experiment_data["methods_comparison"].keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, dataset_name in enumerate(datasets):
            data = experiment_data["methods_comparison"][dataset_name]
            n_agents = data["n_agents"]
            dist_cong = data["distance"]["cong"]
            cma_cong = data["cma"]["cong"]

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
                x + width / 2, cma_cong, width, label="CMA", color="green", alpha=0.7
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Congestion Events")
            axes[idx].set_title(f'{dataset_name.replace("_", " ").title()}')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Congestion Events Comparison: CMA vs Distance-Based (Lower is Better)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_congestion_events_all_datasets.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion events plot: {e}")
        plt.close()

    # Plot 6: Combined Metrics Summary
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        datasets = list(experiment_data["methods_comparison"].keys())

        # Training curve subplot
        train_loss = experiment_data["training"]["train_loss"]
        val_loss = experiment_data["training"]["val_loss"]
        axes[0, 0].plot(train_loss, "b-", label="Train", linewidth=2)
        axes[0, 0].plot(val_loss, "r-", label="Validation", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Improvement by dataset
        avg_imps = [
            np.mean(experiment_data["methods_comparison"][d]["improvement"])
            for d in datasets
        ]
        axes[0, 1].bar(
            range(len(datasets)),
            avg_imps,
            color=["green" if i > 0 else "red" for i in avg_imps],
            alpha=0.7,
        )
        axes[0, 1].set_xticks(range(len(datasets)))
        axes[0, 1].set_xticklabels([d[:8] for d in datasets], rotation=45)
        axes[0, 1].set_ylabel("Improvement (%)")
        axes[0, 1].set_title("Throughput Improvement by Dataset")
        axes[0, 1].axhline(0, c="r", ls="--")

        # CPR by dataset
        avg_cprs = [
            np.mean(experiment_data["methods_comparison"][d]["cpr"]) for d in datasets
        ]
        axes[1, 0].bar(range(len(datasets)), avg_cprs, color="green", alpha=0.7)
        axes[1, 0].set_xticks(range(len(datasets)))
        axes[1, 0].set_xticklabels([d[:8] for d in datasets], rotation=45)
        axes[1, 0].set_ylabel("CPR (%)")
        axes[1, 0].set_title("Conflict Prevention Rate by Dataset")
        axes[1, 0].axhline(0, c="r", ls="--")

        # Agent scaling for bottleneck (most challenging)
        data = experiment_data["methods_comparison"]["bottleneck_warehouse"]
        n_agents = data["n_agents"]
        axes[1, 1].plot(
            n_agents, data["cma"]["tp"], "go-", label="CMA", linewidth=2, markersize=8
        )
        axes[1, 1].plot(
            n_agents,
            data["distance"]["tp"],
            "bx--",
            label="Distance",
            linewidth=2,
            markersize=8,
        )
        axes[1, 1].set_xlabel("Number of Agents")
        axes[1, 1].set_ylabel("Throughput")
        axes[1, 1].set_title("Bottleneck Warehouse: Throughput vs Agents")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("CMA Warehouse Task Assignment - Summary Results", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_cma_summary_results.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

print("All plots saved successfully.")
