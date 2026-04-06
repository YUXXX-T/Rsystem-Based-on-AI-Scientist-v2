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
    colors = {
        "sparse_warehouse": "blue",
        "bottleneck_warehouse": "orange",
        "dense_obstacle_warehouse": "green",
    }

    # Plot 1: Training curves
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(
            experiment_data["training"]["epochs"],
            experiment_data["training"]["losses"]["train"],
            label="Train Loss",
            color="blue",
        )
        plt.plot(
            experiment_data["training"]["epochs"],
            experiment_data["training"]["losses"]["val"],
            label="Validation Loss",
            color="orange",
        )
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Congestion Predictor Training Curves\n(All Warehouse Datasets Combined)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "training_loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput comparison
    try:
        plt.figure(figsize=(10, 6))
        for dataset_name in experiment_data["datasets"]:
            data = experiment_data["datasets"][dataset_name]
            plt.plot(
                data["n_agents"],
                data["congestion_aware"]["throughput"],
                "o-",
                color=colors[dataset_name],
                label=f"CATA - {dataset_name}",
            )
            plt.plot(
                data["n_agents"],
                data["distance_based"]["throughput"],
                "s--",
                color=colors[dataset_name],
                alpha=0.5,
                label=f"Distance - {dataset_name}",
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "Throughput Comparison: CATA vs Distance-Based Assignment\nAcross Three Warehouse Datasets"
        )
        plt.legend(fontsize=8, loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "throughput_comparison_all_datasets.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement percentage by dataset
    try:
        plt.figure(figsize=(10, 6))
        agent_counts = list(experiment_data["datasets"].values())[0]["n_agents"]
        x = np.arange(len(agent_counts))
        width = 0.25
        for i, dataset_name in enumerate(experiment_data["datasets"]):
            data = experiment_data["datasets"][dataset_name]
            plt.bar(
                x + i * width,
                data["improvement"],
                width,
                label=dataset_name,
                color=colors[dataset_name],
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput Improvement (%)")
        plt.title(
            "CATA Throughput Improvement Over Distance-Based Method\nBy Warehouse Dataset and Agent Count"
        )
        plt.xticks(x + width, agent_counts)
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "throughput_improvement_by_dataset.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Plot 4: Congestion events comparison
    try:
        plt.figure(figsize=(10, 6))
        for dataset_name in experiment_data["datasets"]:
            data = experiment_data["datasets"][dataset_name]
            plt.plot(
                data["n_agents"],
                data["distance_based"]["congestion"],
                "s--",
                color=colors[dataset_name],
                alpha=0.5,
            )
            plt.plot(
                data["n_agents"],
                data["congestion_aware"]["congestion"],
                "o-",
                color=colors[dataset_name],
                label=f"{dataset_name}",
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Congestion Events")
        plt.title(
            "Congestion Events: CATA (solid) vs Distance-Based (dashed)\nAcross Three Warehouse Datasets"
        )
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "congestion_events_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion events plot: {e}")
        plt.close()

    # Plot 5: Summary metrics per dataset
    try:
        plt.figure(figsize=(10, 6))
        dataset_names = list(experiment_data["datasets"].keys())
        avg_improvements = [
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
            avg_improvements,
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
            "Average CATA Performance Metrics by Warehouse Dataset\n(Averaged Across All Agent Counts)"
        )
        plt.xticks(x, ["Sparse", "Bottleneck", "Dense"])
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "summary_metrics_by_dataset.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    print("All plots saved successfully to working directory.")
