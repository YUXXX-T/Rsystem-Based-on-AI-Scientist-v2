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
    epoch_values = [30, 60, 90, 120]
    dataset_names = [
        "sparse_warehouse",
        "bottleneck_warehouse",
        "dense_obstacle_warehouse",
    ]
    colors_epochs = {30: "blue", 60: "orange", 90: "green", 120: "red"}

    # Plot 1: Training curves for all epoch settings
    try:
        plt.figure(figsize=(12, 5))
        for i, epochs in enumerate(epoch_values):
            data = experiment_data["epoch_tuning"][epochs]["training"]["losses"]
            plt.subplot(1, 2, 1)
            plt.plot(
                data["train"], label=f"Epochs={epochs}", color=colors_epochs[epochs]
            )
            plt.subplot(1, 2, 2)
            plt.plot(data["val"], label=f"Epochs={epochs}", color=colors_epochs[epochs])
        plt.subplot(1, 2, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Curves")
        plt.legend()
        plt.suptitle(
            "Warehouse Task Assignment - Epoch Tuning: Training & Validation Curves"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_epoch_tuning_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # Plot 2: Final validation loss and overall improvement comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        final_val_losses = [
            experiment_data["epoch_tuning"][e]["final_val_loss"] for e in epoch_values
        ]
        overall_improvements = [
            experiment_data["epoch_tuning"][e]["overall_improvement"]
            for e in epoch_values
        ]

        axes[0].bar(
            range(len(epoch_values)),
            final_val_losses,
            color=[colors_epochs[e] for e in epoch_values],
        )
        axes[0].set_xticks(range(len(epoch_values)))
        axes[0].set_xticklabels(epoch_values)
        axes[0].set_xlabel("Training Epochs")
        axes[0].set_ylabel("Final Validation Loss")
        axes[0].set_title("Final Validation Loss by Epoch Count")

        axes[1].bar(
            range(len(epoch_values)),
            overall_improvements,
            color=[colors_epochs[e] for e in epoch_values],
        )
        axes[1].set_xticks(range(len(epoch_values)))
        axes[1].set_xticklabels(epoch_values)
        axes[1].set_xlabel("Training Epochs")
        axes[1].set_ylabel("Throughput Improvement (%)")
        axes[1].set_title("Overall Throughput Improvement by Epoch Count")
        axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.5)

        plt.suptitle(
            "Warehouse Epoch Tuning: Left - Validation Loss, Right - Improvement %"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_epoch_tuning_summary.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # Plot 3: Per-dataset improvement by epoch count
    try:
        plt.figure(figsize=(10, 5))
        x = np.arange(len(dataset_names))
        width = 0.2
        for i, epochs in enumerate(epoch_values):
            avg_imps = [
                np.mean(
                    experiment_data["epoch_tuning"][epochs]["datasets"][d][
                        "improvement"
                    ]
                )
                for d in dataset_names
            ]
            plt.bar(
                x + i * width,
                avg_imps,
                width,
                label=f"Epochs={epochs}",
                color=colors_epochs[epochs],
            )
        plt.xticks(x + width * 1.5, ["Sparse", "Bottleneck", "Dense Obstacle"])
        plt.xlabel("Warehouse Dataset")
        plt.ylabel("Average Throughput Improvement (%)")
        plt.title(
            "Warehouse Epoch Tuning: Throughput Improvement by Dataset and Training Epochs"
        )
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_epoch_tuning_improvement_by_dataset.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement by dataset plot: {e}")
        plt.close()

    # Plot 4: Congestion prevention rate by dataset and epochs
    try:
        plt.figure(figsize=(10, 5))
        x = np.arange(len(dataset_names))
        width = 0.2
        for i, epochs in enumerate(epoch_values):
            avg_cprs = [
                np.mean(
                    experiment_data["epoch_tuning"][epochs]["datasets"][d][
                        "congestion_prevention_rate"
                    ]
                )
                for d in dataset_names
            ]
            plt.bar(
                x + i * width,
                avg_cprs,
                width,
                label=f"Epochs={epochs}",
                color=colors_epochs[epochs],
            )
        plt.xticks(x + width * 1.5, ["Sparse", "Bottleneck", "Dense Obstacle"])
        plt.xlabel("Warehouse Dataset")
        plt.ylabel("Congestion Prevention Rate (%)")
        plt.title(
            "Warehouse Epoch Tuning: Congestion Prevention Rate by Dataset and Training Epochs"
        )
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_epoch_tuning_cpr_by_dataset.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating CPR plot: {e}")
        plt.close()

    # Plot 5: Best model throughput comparison across datasets
    try:
        best_epochs = experiment_data["best_epochs"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
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
            axes[idx].errorbar(
                n_agents,
                data["congestion_aware"]["throughput"],
                yerr=data["congestion_aware"]["std"],
                fmt="o-",
                label="Congestion-Aware",
                color="green",
                capsize=3,
            )
            axes[idx].errorbar(
                n_agents,
                data["distance_based"]["throughput"],
                yerr=data["distance_based"]["std"],
                fmt="s--",
                label="Distance-Based",
                color="red",
                capsize=3,
            )
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"{dataset_name.replace('_', ' ').title()}")
            axes[idx].legend(fontsize=8)

        plt.suptitle(
            f"Best Model (Epochs={best_epochs}) Throughput: Distance-Based vs Congestion-Aware"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_best_model_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating best model throughput plot: {e}")
        plt.close()

    print(f"Best epochs: {experiment_data['best_epochs']}")
    print(f"Best improvement: {experiment_data['best_improvement']:.2f}%")
