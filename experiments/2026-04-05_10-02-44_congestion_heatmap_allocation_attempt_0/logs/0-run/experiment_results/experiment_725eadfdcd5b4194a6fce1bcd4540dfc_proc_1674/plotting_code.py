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

if experiment_data:
    # Plot 1: Training and Validation Loss Curves
    try:
        plt.figure(figsize=(8, 6))
        train_losses = experiment_data["training"]["losses"]["train"]
        val_losses = experiment_data["training"]["losses"]["val"]
        epochs = experiment_data["training"]["epochs"]
        plt.plot(epochs, train_losses, "b-", label="Train Loss")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Task Assignment Experiment"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "training_validation_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Across Datasets
    try:
        plt.figure(figsize=(10, 6))
        colors = {
            "sparse_warehouse": "blue",
            "bottleneck_warehouse": "orange",
            "dense_obstacle_warehouse": "green",
        }
        for dataset_name in experiment_data["datasets"]:
            data = experiment_data["datasets"][dataset_name]
            n_agents = data["n_agents"]
            plt.errorbar(
                n_agents,
                data["congestion_aware"]["throughput"],
                yerr=data["congestion_aware"]["std"],
                fmt="o-",
                color=colors[dataset_name],
                label=f"CATA-{dataset_name[:8]}",
                capsize=3,
            )
            plt.errorbar(
                n_agents,
                data["distance_based"]["throughput"],
                yerr=data["distance_based"]["std"],
                fmt="x--",
                color=colors[dataset_name],
                alpha=0.5,
                capsize=3,
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "Throughput Comparison: CATA (solid) vs Distance-Based (dashed)\nAcross Three Warehouse Datasets"
        )
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "throughput_comparison_all_datasets.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement Percentage by Dataset
    try:
        plt.figure(figsize=(10, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        x = np.arange(len(experiment_data["datasets"][dataset_names[0]]["n_agents"]))
        width = 0.25
        for i, dataset_name in enumerate(dataset_names):
            data = experiment_data["datasets"][dataset_name]
            plt.bar(
                x + i * width,
                data["improvement"],
                width,
                label=dataset_name.replace("_", " ").title(),
                color=colors[dataset_name],
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput Improvement (%)")
        plt.title(
            "CATA Throughput Improvement Over Distance-Based Assignment\nBy Dataset and Agent Count"
        )
        plt.xticks(x + width, experiment_data["datasets"][dataset_names[0]]["n_agents"])
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "improvement_percentage_by_dataset.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Plot 4: Congestion Prevention Rate
    try:
        plt.figure(figsize=(10, 6))
        for i, dataset_name in enumerate(dataset_names):
            data = experiment_data["datasets"][dataset_name]
            plt.bar(
                x + i * width,
                data["congestion_prevention_rate"],
                width,
                label=dataset_name.replace("_", " ").title(),
                color=colors[dataset_name],
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Congestion Prevention Rate (%)")
        plt.title(
            "Congestion Prevention Rate by Dataset\n(Reduction in Congestion Events: CATA vs Distance-Based)"
        )
        plt.xticks(x + width, experiment_data["datasets"][dataset_names[0]]["n_agents"])
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "congestion_prevention_rate_by_dataset.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion prevention plot: {e}")
        plt.close()

    # Plot 5: Summary Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        summary_imp = [
            np.mean(experiment_data["datasets"][d]["improvement"])
            for d in dataset_names
        ]
        summary_cpr = [
            np.mean(experiment_data["datasets"][d]["congestion_prevention_rate"])
            for d in dataset_names
        ]
        x_sum = np.arange(len(dataset_names))
        plt.bar(
            x_sum - 0.2,
            summary_imp,
            0.35,
            label="Avg Throughput Improvement",
            color="steelblue",
        )
        plt.bar(
            x_sum + 0.2,
            summary_cpr,
            0.35,
            label="Avg Congestion Prevention Rate",
            color="coral",
        )
        plt.xlabel("Warehouse Dataset")
        plt.ylabel("Percentage (%)")
        plt.title(
            "Summary: Average CATA Performance Improvement by Dataset\nCompared to Distance-Based Assignment"
        )
        plt.xticks(x_sum, ["Sparse", "Bottleneck", "Dense"])
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "summary_improvement_by_dataset.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    print("All plots saved successfully to working directory.")
