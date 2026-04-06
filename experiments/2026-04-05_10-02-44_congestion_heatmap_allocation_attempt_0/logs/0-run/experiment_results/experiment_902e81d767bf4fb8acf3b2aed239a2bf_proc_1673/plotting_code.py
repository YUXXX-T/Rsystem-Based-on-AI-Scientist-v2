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
        plt.figure(figsize=(8, 6))
        epochs = experiment_data["training"]["epochs"]
        train_loss = experiment_data["training"]["losses"]["train"]
        val_loss = experiment_data["training"]["losses"]["val"]
        plt.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Robot Task Assignment"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "training_validation_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Across Datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = {
            "sparse_warehouse": "blue",
            "bottleneck_warehouse": "orange",
            "dense_obstacle_warehouse": "green",
        }
        for idx, dataset_name in enumerate(experiment_data["datasets"]):
            data = experiment_data["datasets"][dataset_name]
            n_agents = data["n_agents"]
            ax = axes[idx]
            ax.errorbar(
                n_agents,
                data["distance_based"]["throughput"],
                yerr=data["distance_based"]["std"],
                fmt="s--",
                color="gray",
                label="Distance-Based",
                capsize=3,
                linewidth=2,
            )
            ax.errorbar(
                n_agents,
                data["congestion_aware"]["throughput"],
                yerr=data["congestion_aware"]["std"],
                fmt="o-",
                color=colors[dataset_name],
                label="CATA",
                capsize=3,
                linewidth=2,
            )
            ax.set_xlabel("Number of Agents")
            ax.set_ylabel("Throughput (tasks/min)")
            ax.set_title(f'{dataset_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            "Throughput Comparison: CATA vs Distance-Based Assignment", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "throughput_comparison_all_datasets.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement Percentage by Agent Count
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(experiment_data["datasets"]["sparse_warehouse"]["n_agents"]))
        width = 0.25
        dataset_names = list(experiment_data["datasets"].keys())
        colors_list = ["steelblue", "coral", "seagreen"]
        for i, dataset_name in enumerate(dataset_names):
            data = experiment_data["datasets"][dataset_name]
            plt.bar(
                x + i * width,
                data["improvement"],
                width,
                label=dataset_name.replace("_", " ").title(),
                color=colors_list[i],
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput Improvement (%)")
        plt.title(
            "CATA Throughput Improvement Over Distance-Based Assignment\nAcross All Warehouse Datasets"
        )
        plt.xticks(
            x + width, experiment_data["datasets"]["sparse_warehouse"]["n_agents"]
        )
        plt.legend()
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "improvement_percentage_all_datasets.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Plot 4: Congestion Prevention Rate
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(experiment_data["datasets"]["sparse_warehouse"]["n_agents"]))
        width = 0.25
        colors_list = ["steelblue", "coral", "seagreen"]
        for i, dataset_name in enumerate(dataset_names):
            data = experiment_data["datasets"][dataset_name]
            plt.bar(
                x + i * width,
                data["congestion_prevention_rate"],
                width,
                label=dataset_name.replace("_", " ").title(),
                color=colors_list[i],
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Congestion Prevention Rate (%)")
        plt.title(
            "Congestion Prevention Rate: CATA vs Distance-Based\nAcross All Warehouse Datasets"
        )
        plt.xticks(
            x + width, experiment_data["datasets"]["sparse_warehouse"]["n_agents"]
        )
        plt.legend()
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "congestion_prevention_rate_all_datasets.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion prevention plot: {e}")
        plt.close()

    # Plot 5: Summary - Average Metrics by Dataset
    try:
        plt.figure(figsize=(10, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        avg_improvement = [
            np.mean(experiment_data["datasets"][d]["improvement"])
            for d in dataset_names
        ]
        avg_cpr = [
            np.mean(experiment_data["datasets"][d]["congestion_prevention_rate"])
            for d in dataset_names
        ]
        x = np.arange(len(dataset_names))
        width = 0.35
        plt.bar(
            x - width / 2,
            avg_improvement,
            width,
            label="Avg Throughput Improvement (%)",
            color="steelblue",
        )
        plt.bar(
            x + width / 2,
            avg_cpr,
            width,
            label="Avg Congestion Prevention Rate (%)",
            color="coral",
        )
        plt.xlabel("Warehouse Dataset")
        plt.ylabel("Percentage (%)")
        plt.title(
            "Summary: CATA Performance Metrics Averaged Across Agent Counts\nLeft: Throughput Improvement, Right: Congestion Prevention Rate"
        )
        plt.xticks(x, [d.replace("_", "\n").title() for d in dataset_names])
        plt.legend()
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "summary_metrics_all_datasets.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

print("All plots generated successfully.")
