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
    # Plot 1: Training curves comparison
    try:
        plt.figure(figsize=(10, 6))
        for epochs in [50, 100]:
            config_key = f"epochs_{epochs}"
            train_losses = experiment_data[config_key]["training"]["losses"]["train"]
            val_losses = experiment_data[config_key]["training"]["losses"]["val"]
            epochs_range = range(len(train_losses))
            plt.plot(
                epochs_range,
                train_losses,
                label=f"{epochs} epochs (train)",
                linewidth=2,
            )
            plt.plot(
                epochs_range,
                val_losses,
                label=f"{epochs} epochs (val)",
                linestyle="--",
                alpha=0.7,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            "Warehouse Robot Coordination: Training/Validation Loss Curves\nCongestion Predictor Model"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "warehouse_training_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating training curves plot: {e}")
        plt.close()

    # Plot 2: Throughput comparison - Distance vs Congestion-Aware
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        agent_counts = [20, 30, 40, 50]
        x = np.arange(len(agent_counts))
        width = 0.35

        for idx, epochs in enumerate([50, 100]):
            config_key = f"epochs_{epochs}"
            dist_throughputs = [
                t["mean"]
                for t in experiment_data[config_key]["evaluation"]["distance_based"][
                    "throughput"
                ]
            ]
            dist_stds = [
                t["std"]
                for t in experiment_data[config_key]["evaluation"]["distance_based"][
                    "throughput"
                ]
            ]
            cata_throughputs = [
                t["mean"]
                for t in experiment_data[config_key]["evaluation"]["congestion_aware"][
                    "throughput"
                ]
            ]
            cata_stds = [
                t["std"]
                for t in experiment_data[config_key]["evaluation"]["congestion_aware"][
                    "throughput"
                ]
            ]

            axes[idx].bar(
                x - width / 2,
                dist_throughputs,
                width,
                yerr=dist_stds,
                label="Distance-based",
                alpha=0.8,
                capsize=3,
            )
            axes[idx].bar(
                x + width / 2,
                cata_throughputs,
                width,
                yerr=cata_stds,
                label="Congestion-Aware",
                alpha=0.8,
                capsize=3,
            )
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"Epochs={epochs}")
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(agent_counts)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Warehouse Robot Coordination: Throughput Comparison\nLeft: 50 Epochs, Right: 100 Epochs"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement percentage across agent counts
    try:
        plt.figure(figsize=(10, 6))
        agent_counts = [20, 30, 40, 50]
        x = np.arange(len(agent_counts))
        width = 0.35

        for i, epochs in enumerate([50, 100]):
            config_key = f"epochs_{epochs}"
            improvements = []
            for j in range(len(agent_counts)):
                dist = experiment_data[config_key]["evaluation"]["distance_based"][
                    "throughput"
                ][j]["mean"]
                cata = experiment_data[config_key]["evaluation"]["congestion_aware"][
                    "throughput"
                ][j]["mean"]
                improvements.append((cata - dist) / dist * 100)
            plt.bar(
                x + i * width - width / 2,
                improvements,
                width,
                label=f"{epochs} epochs",
                alpha=0.8,
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Improvement (%)")
        plt.title(
            "Warehouse Robot Coordination: CATA Improvement over Distance-based Method"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_percentage.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Plot 4: Final validation loss and average throughput summary
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epoch_configs = [50, 100]

        final_val_losses = [
            experiment_data[f"epochs_{e}"]["training"]["losses"]["val"][-1]
            for e in epoch_configs
        ]
        axes[0].bar(
            [str(e) for e in epoch_configs],
            final_val_losses,
            color=["steelblue", "coral"],
            alpha=0.8,
        )
        axes[0].set_xlabel("Training Epochs")
        axes[0].set_ylabel("Final Validation Loss")
        axes[0].set_title("Final Validation Loss Comparison")
        axes[0].grid(True, alpha=0.3, axis="y")

        avg_cata_throughputs = []
        for epochs in epoch_configs:
            config_key = f"epochs_{epochs}"
            throughputs = [
                t["mean"]
                for t in experiment_data[config_key]["evaluation"]["congestion_aware"][
                    "throughput"
                ]
            ]
            avg_cata_throughputs.append(np.mean(throughputs))
        axes[1].bar(
            [str(e) for e in epoch_configs],
            avg_cata_throughputs,
            color=["steelblue", "coral"],
            alpha=0.8,
        )
        axes[1].set_xlabel("Training Epochs")
        axes[1].set_ylabel("Avg Throughput (tasks/min)")
        axes[1].set_title("Average CATA Throughput")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Warehouse Robot Coordination: Hyperparameter Summary\nLeft: Validation Loss, Right: Average Throughput"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_hyperparameter_summary.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

print("All plots saved successfully.")
