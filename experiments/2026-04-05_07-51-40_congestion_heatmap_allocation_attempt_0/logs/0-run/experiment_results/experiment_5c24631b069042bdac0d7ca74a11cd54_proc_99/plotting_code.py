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
        train_losses = experiment_data["training"]["losses"]["train"]
        val_losses = experiment_data["training"]["losses"]["val"]
        epochs = experiment_data["training"]["epochs"]

        plt.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2)
        plt.plot(
            epochs, val_losses, label="Validation Loss", color="orange", linewidth=2
        )
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Congestion Predictor Training Curves\nWarehouse Robot Coordination - CNN Model"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_training_loss_curves.png"), dpi=150
        )
        plt.close()
        print("Saved training loss curves plot")
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        dist_data = experiment_data["evaluation"]["distance_based"]["throughput"]
        cata_data = experiment_data["evaluation"]["congestion_aware"]["throughput"]

        agent_counts = [d["n_agents"] for d in dist_data]
        dist_means = [d["mean"] for d in dist_data]
        dist_stds = [d["std"] for d in dist_data]
        cata_means = [d["mean"] for d in cata_data]
        cata_stds = [d["std"] for d in cata_data]

        x = np.arange(len(agent_counts))
        width = 0.35

        plt.bar(
            x - width / 2,
            dist_means,
            width,
            yerr=dist_stds,
            label="Distance-based",
            alpha=0.8,
            color="steelblue",
            capsize=5,
        )
        plt.bar(
            x + width / 2,
            cata_means,
            width,
            yerr=cata_stds,
            label="Congestion-Aware (CATA)",
            alpha=0.8,
            color="coral",
            capsize=5,
        )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/minute)")
        plt.title(
            "Warehouse Task Assignment: Throughput Comparison\nDistance-based vs Congestion-Aware Assignment"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison.png"), dpi=150
        )
        plt.close()
        print("Saved throughput comparison plot")
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement Percentage Line Plot
    try:
        plt.figure(figsize=(8, 6))
        improvements = [
            (cata_means[i] - dist_means[i]) / dist_means[i] * 100
            for i in range(len(agent_counts))
        ]

        plt.plot(
            agent_counts,
            improvements,
            marker="o",
            linewidth=2,
            markersize=10,
            color="green",
        )
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.fill_between(agent_counts, 0, improvements, alpha=0.3, color="green")

        for i, (x, y) in enumerate(zip(agent_counts, improvements)):
            plt.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Improvement (%)")
        plt.title(
            "Warehouse Task Assignment: CATA Improvement over Distance-based\nPercentage Throughput Improvement by Agent Count"
        )
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_percentage.png"), dpi=150
        )
        plt.close()
        print("Saved improvement percentage plot")
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Print summary metrics
    print("\n=== Experiment Summary ===")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Average distance-based throughput: {np.mean(dist_means):.2f} tasks/min")
    print(f"Average CATA throughput: {np.mean(cata_means):.2f} tasks/min")
    print(f"Average improvement: {np.mean(improvements):.1f}%")
