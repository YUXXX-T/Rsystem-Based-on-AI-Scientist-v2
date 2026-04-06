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
    epoch_values = list(experiment_data["epoch_tuning"].keys())
    dataset_names = [
        "sparse_warehouse",
        "bottleneck_warehouse",
        "dense_obstacle_warehouse",
    ]
    colors_epochs = {30: "blue", 60: "orange"}

    # Plot 1: Training/Validation Loss Curves
    try:
        plt.figure(figsize=(10, 6))
        for epochs in epoch_values:
            data = experiment_data["epoch_tuning"][epochs]["training"]["losses"]
            plt.plot(
                data["train"],
                label=f"Train Loss (epochs={epochs})",
                color=colors_epochs[epochs],
                alpha=0.8,
            )
            plt.plot(
                data["val"],
                "--",
                label=f"Val Loss (epochs={epochs})",
                color=colors_epochs[epochs],
                alpha=0.8,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Task Assignment - Epoch Tuning"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "warehouse_training_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Overall Improvement by Epoch Count
    try:
        plt.figure(figsize=(8, 6))
        overall_improvements = [
            experiment_data["epoch_tuning"][e]["overall_improvement"]
            for e in epoch_values
        ]
        bars = plt.bar(
            range(len(epoch_values)),
            overall_improvements,
            color=[colors_epochs[e] for e in epoch_values],
        )
        plt.xticks(range(len(epoch_values)), [str(e) for e in epoch_values])
        plt.xlabel("Number of Training Epochs")
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.title(
            "Epoch Tuning: Overall Improvement Comparison\nCongestion-Aware vs Distance-Based Assignment"
        )
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        for i, v in enumerate(overall_improvements):
            plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=10)
        plt.savefig(
            os.path.join(working_dir, "warehouse_epoch_tuning_overall_improvement.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating overall improvement plot: {e}")
        plt.close()

    # Plot 3: Per-Dataset Improvement Comparison
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(dataset_names))
        width = 0.35
        for i, epochs in enumerate(epoch_values):
            avg_imps = [
                np.mean(
                    experiment_data["epoch_tuning"][epochs]["datasets"][d][
                        "improvement"
                    ]
                )
                for d in dataset_names
            ]
            ax.bar(
                x + i * width,
                avg_imps,
                width,
                label=f"Epochs={epochs}",
                color=colors_epochs[epochs],
            )
        ax.set_xticks(x + width * 0.5)
        ax.set_xticklabels(["Sparse", "Bottleneck", "Dense Obstacle"])
        ax.set_ylabel("Average Throughput Improvement (%)")
        ax.set_title(
            "Throughput Improvement by Dataset and Training Epochs\nWarehouse Task Assignment"
        )
        ax.legend()
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.savefig(
            os.path.join(working_dir, "warehouse_per_dataset_improvement.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating per-dataset improvement plot: {e}")
        plt.close()

    # Plot 4: Throughput Comparison for Best Model
    try:
        best_epochs = experiment_data["best_epochs"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors_ds = {
            "sparse_warehouse": "blue",
            "bottleneck_warehouse": "orange",
            "dense_obstacle_warehouse": "green",
        }
        for idx, dataset_name in enumerate(dataset_names):
            data = experiment_data["epoch_tuning"][best_epochs]["datasets"][
                dataset_name
            ]
            n_agents = data["n_agents"]
            ax = axes[idx]
            ax.errorbar(
                n_agents,
                data["congestion_aware"]["throughput"],
                yerr=data["congestion_aware"]["std"],
                fmt="o-",
                label="Congestion-Aware",
                color="green",
                capsize=3,
            )
            ax.errorbar(
                n_agents,
                data["distance_based"]["throughput"],
                yerr=data["distance_based"]["std"],
                fmt="s--",
                label="Distance-Based",
                color="red",
                capsize=3,
            )
            ax.set_xlabel("Number of Agents")
            ax.set_ylabel("Throughput (tasks/min)")
            ax.set_title(f'{dataset_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            f"Throughput Comparison (Best Model: epochs={best_epochs})\nLeft: Sparse, Middle: Bottleneck, Right: Dense",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison_best_model.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 5: Congestion Prevention Rate
    try:
        best_epochs = experiment_data["best_epochs"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(dataset_names))
        width = 0.25
        agent_counts = experiment_data["epoch_tuning"][best_epochs]["datasets"][
            dataset_names[0]
        ]["n_agents"]
        colors_agents = {20: "skyblue", 30: "steelblue", 40: "navy"}
        for i, n_agents in enumerate(agent_counts):
            cpr_values = [
                experiment_data["epoch_tuning"][best_epochs]["datasets"][d][
                    "congestion_prevention_rate"
                ][i]
                for d in dataset_names
            ]
            ax.bar(
                x + i * width,
                cpr_values,
                width,
                label=f"{n_agents} Agents",
                color=colors_agents[n_agents],
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Sparse", "Bottleneck", "Dense Obstacle"])
        ax.set_ylabel("Congestion Prevention Rate (%)")
        ax.set_title(
            f"Congestion Prevention Rate by Dataset (Best Model: epochs={best_epochs})\nWarehouse Task Assignment"
        )
        ax.legend()
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.savefig(
            os.path.join(working_dir, "warehouse_congestion_prevention_rate.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion prevention rate plot: {e}")
        plt.close()

    print(f"Best epochs: {experiment_data['best_epochs']}")
    print(f"Best improvement: {experiment_data['best_improvement']:.2f}%")
