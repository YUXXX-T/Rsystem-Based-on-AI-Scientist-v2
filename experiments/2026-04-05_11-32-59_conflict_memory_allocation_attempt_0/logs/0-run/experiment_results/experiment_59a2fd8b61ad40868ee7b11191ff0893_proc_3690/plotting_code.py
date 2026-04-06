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
        plt.ylabel("Loss (MSE)")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Task Assignment with Temporal Memory"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_training_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Across All Datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        dataset_names = list(experiment_data["datasets"].keys())

        for idx, name in enumerate(dataset_names):
            data = experiment_data["datasets"][name]
            n_agents = data["n_agents"]
            x = np.arange(len(n_agents))
            width = 0.35

            axes[idx].bar(
                x - width / 2,
                data["distance"]["throughput"],
                width,
                label="Distance-Based",
                color="blue",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2,
                data["cma"]["throughput"],
                width,
                label="Congestion-Aware",
                color="green",
                alpha=0.7,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"{name}")
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(
            "Throughput Comparison: Distance-Based vs Congestion-Aware Assignment",
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

    # Plot 3: Congestion Events Comparison
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        dataset_names = list(experiment_data["datasets"].keys())

        for idx, name in enumerate(dataset_names):
            data = experiment_data["datasets"][name]
            n_agents = data["n_agents"]
            x = np.arange(len(n_agents))
            width = 0.35

            axes[idx].bar(
                x - width / 2,
                data["distance"]["congestion"],
                width,
                label="Distance-Based",
                color="red",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2,
                data["cma"]["congestion"],
                width,
                label="Congestion-Aware",
                color="green",
                alpha=0.7,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Congestion Events")
            axes[idx].set_title(f"{name}")
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle("Congestion Events Comparison (Lower is Better)", fontsize=12)
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

    # Plot 4: Improvement and CPR by Dataset
    try:
        plt.figure(figsize=(12, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        x = np.arange(len(dataset_names))
        width = 0.35

        avg_improvements = [
            np.mean(experiment_data["datasets"][d]["improvement"])
            for d in dataset_names
        ]
        avg_cprs = [
            np.mean(experiment_data["datasets"][d]["cpr"]) for d in dataset_names
        ]

        bars1 = plt.bar(
            x - width / 2,
            avg_improvements,
            width,
            label="Throughput Improvement %",
            color="blue",
            alpha=0.7,
            edgecolor="black",
        )
        bars2 = plt.bar(
            x + width / 2,
            avg_cprs,
            width,
            label="Conflict Prevention Rate %",
            color="green",
            alpha=0.7,
            edgecolor="black",
        )

        plt.xticks(x, dataset_names, fontsize=10)
        plt.ylabel("Percentage (%)")
        plt.xlabel("Dataset")
        plt.title(
            "Average Improvement and CPR by Warehouse Configuration\nLeft: Throughput Improvement, Right: Conflict Prevention Rate"
        )
        plt.legend()
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
        plt.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars1, avg_improvements):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                fontsize=9,
            )
        for bar, val in zip(bars2, avg_cprs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_cpr_by_dataset.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement/CPR plot: {e}")
        plt.close()

    # Plot 5: CPR Scaling with Agent Count
    try:
        plt.figure(figsize=(10, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        colors = ["blue", "green", "orange"]
        markers = ["o", "s", "^"]

        for idx, name in enumerate(dataset_names):
            data = experiment_data["datasets"][name]
            plt.plot(
                data["n_agents"],
                data["cpr"],
                marker=markers[idx],
                color=colors[idx],
                linewidth=2,
                markersize=10,
                label=name,
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Conflict Prevention Rate (%)")
        plt.title(
            "Conflict Prevention Rate vs Agent Density\nHigher CPR indicates better congestion avoidance"
        )
        plt.legend()
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_cpr_vs_agents.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating CPR vs agents plot: {e}")
        plt.close()

    # Plot 6: Overall Summary
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        overall_imp = experiment_data.get("overall_improvement", 0)
        overall_cpr = experiment_data.get("overall_cpr", 0)

        # Left: Overall metrics
        metrics = ["Throughput\nImprovement", "Conflict\nPrevention Rate"]
        values = [overall_imp, overall_cpr]
        colors = ["green" if v > 0 else "red" for v in values]

        bars = axes[0].bar(
            metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Percentage (%)")
        axes[0].set_title(
            "Overall Performance Metrics\nCongestion-Aware vs Distance-Based Assignment"
        )
        for bar, val in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.2f}%",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        axes[0].grid(True, alpha=0.3, axis="y")

        # Right: Dataset breakdown
        dataset_names = list(experiment_data["datasets"].keys())
        all_imps = []
        all_cprs = []
        labels = []
        for name in dataset_names:
            data = experiment_data["datasets"][name]
            for i, n_ag in enumerate(data["n_agents"]):
                all_imps.append(data["improvement"][i])
                all_cprs.append(data["cpr"][i])
                labels.append(f"{name[:8]}\n{n_ag}ag")

        x = np.arange(len(labels))
        width = 0.4
        axes[1].bar(
            x - width / 2,
            all_imps,
            width,
            label="Improvement %",
            color="blue",
            alpha=0.7,
        )
        axes[1].bar(
            x + width / 2, all_cprs, width, label="CPR %", color="green", alpha=0.7
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("Detailed Results by Dataset and Agent Count")
        axes[1].legend(fontsize=8)
        axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "warehouse_overall_summary.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating overall summary plot: {e}")
        plt.close()

print("All plots saved successfully.")
