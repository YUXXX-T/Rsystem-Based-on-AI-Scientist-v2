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
    n_samples_values = [400, 800, 1200, 1600, 2000]
    agent_counts = [20, 30, 40, 50]
    tuning_data = experiment_data.get("n_samples_tuning", {})

    # Plot 1: Training losses over epochs
    try:
        plt.figure(figsize=(10, 6))
        for n_samples in n_samples_values:
            key = f"n_samples_{n_samples}"
            if key in tuning_data and "losses" in tuning_data[key]:
                train_losses = tuning_data[key]["losses"]["train"]
                plt.plot(train_losses, label=f"n_samples={n_samples}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(
            "Warehouse Simulation: Training Loss vs Epoch\n(n_samples Hyperparameter Tuning)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_training_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Validation losses over epochs
    try:
        plt.figure(figsize=(10, 6))
        for n_samples in n_samples_values:
            key = f"n_samples_{n_samples}"
            if key in tuning_data and "losses" in tuning_data[key]:
                val_losses = tuning_data[key]["losses"]["val"]
                plt.plot(val_losses, label=f"n_samples={n_samples}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title(
            "Warehouse Simulation: Validation Loss vs Epoch\n(n_samples Hyperparameter Tuning)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_validation_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # Plot 3: Final validation loss and average improvement comparison
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        final_val_losses = [
            tuning_data[f"n_samples_{n}"]["final_val_loss"] for n in n_samples_values
        ]
        ax1.bar(range(len(n_samples_values)), final_val_losses, color="steelblue")
        ax1.set_xticks(range(len(n_samples_values)))
        ax1.set_xticklabels([str(n) for n in n_samples_values])
        ax1.set_xlabel("n_samples")
        ax1.set_ylabel("Final Validation Loss")
        ax1.set_title("Final Validation Loss vs n_samples")

        improvements = [
            tuning_data[f"n_samples_{n}"]["avg_improvement"] for n in n_samples_values
        ]
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        ax2.bar(range(len(n_samples_values)), improvements, color=colors)
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_xticks(range(len(n_samples_values)))
        ax2.set_xticklabels([str(n) for n in n_samples_values])
        ax2.set_xlabel("n_samples")
        ax2.set_ylabel("Avg Improvement (%)")
        ax2.set_title("Throughput Improvement vs n_samples")

        plt.suptitle(
            "Warehouse Simulation: n_samples Hyperparameter Analysis", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_n_samples_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()

    # Plot 4: Throughput comparison for best n_samples
    try:
        improvements = [
            tuning_data[f"n_samples_{n}"]["avg_improvement"] for n in n_samples_values
        ]
        best_n_samples = n_samples_values[np.argmax(improvements)]
        best_data = tuning_data[f"n_samples_{best_n_samples}"]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(agent_counts))
        width = 0.35
        dist_throughputs = [
            r["mean"] for r in best_data["throughput"]["distance_based"]
        ]
        cata_throughputs = [
            r["mean"] for r in best_data["throughput"]["congestion_aware"]
        ]
        dist_stds = [r["std"] for r in best_data["throughput"]["distance_based"]]
        cata_stds = [r["std"] for r in best_data["throughput"]["congestion_aware"]]

        plt.bar(
            x - width / 2,
            dist_throughputs,
            width,
            yerr=dist_stds,
            label="Distance-based",
            alpha=0.8,
            color="coral",
        )
        plt.bar(
            x + width / 2,
            cata_throughputs,
            width,
            yerr=cata_stds,
            label="CATA (Congestion-Aware)",
            alpha=0.8,
            color="steelblue",
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            f"Warehouse Simulation: Throughput Comparison\n(Best n_samples={best_n_samples})"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison_best.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 5: CATA throughput trends across n_samples
    try:
        plt.figure(figsize=(10, 6))
        for idx, n_agents in enumerate(agent_counts):
            cata_throughputs = [
                tuning_data[f"n_samples_{n}"]["throughput"]["congestion_aware"][idx][
                    "mean"
                ]
                for n in n_samples_values
            ]
            plt.plot(
                n_samples_values,
                cata_throughputs,
                marker="o",
                linewidth=2,
                label=f"{n_agents} agents",
            )
        plt.xlabel("n_samples (Training Data Size)")
        plt.ylabel("CATA Throughput (tasks/min)")
        plt.title(
            "Warehouse Simulation: CATA Throughput vs Training Data Size\n(Different Agent Counts)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_throughput_trends.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating CATA throughput trend plot: {e}")
        plt.close()

print("All plots generated successfully.")
