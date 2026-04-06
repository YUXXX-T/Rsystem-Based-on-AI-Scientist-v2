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
    learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    best_lr = experiment_data.get("best_learning_rate", 0.001)

    # Plot 1: Training Loss Curves
    try:
        plt.figure(figsize=(8, 5))
        for lr in learning_rates:
            lr_key = f"lr_{lr}"
            if lr_key in experiment_data["learning_rate_tuning"]:
                train_losses = experiment_data["learning_rate_tuning"][lr_key][
                    "losses"
                ]["train"]
                plt.plot(train_losses, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title(
            "Warehouse Robot Coordination - Training Loss by Learning Rate\nCongestion Predictor CNN Training"
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

    # Plot 2: Validation Loss Curves
    try:
        plt.figure(figsize=(8, 5))
        for lr in learning_rates:
            lr_key = f"lr_{lr}"
            if lr_key in experiment_data["learning_rate_tuning"]:
                val_losses = experiment_data["learning_rate_tuning"][lr_key]["losses"][
                    "val"
                ]
                plt.plot(val_losses, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title(
            "Warehouse Robot Coordination - Validation Loss by Learning Rate\nCongestion Predictor CNN Training"
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

    # Plot 3: Final Validation Loss Comparison
    try:
        plt.figure(figsize=(8, 5))
        final_val_losses = [
            experiment_data["learning_rate_tuning"][f"lr_{lr}"]["final_val_loss"]
            for lr in learning_rates
        ]
        colors = ["green" if lr == best_lr else "steelblue" for lr in learning_rates]
        plt.bar(range(len(learning_rates)), final_val_losses, color=colors)
        plt.xticks(range(len(learning_rates)), [str(lr) for lr in learning_rates])
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Validation Loss (MSE)")
        plt.title(
            f"Warehouse Robot Coordination - Final Validation Loss Comparison\nBest LR: {best_lr} (shown in green)"
        )
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_lr_comparison_bar.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating LR comparison plot: {e}")
        plt.close()

    # Plot 4: Throughput Comparison
    try:
        plt.figure(figsize=(10, 6))
        dist_data = experiment_data["best_model_evaluation"]["distance_based"][
            "throughput"
        ]
        cata_data = experiment_data["best_model_evaluation"]["congestion_aware"][
            "throughput"
        ]
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
            capsize=5,
        )
        plt.bar(
            x + width / 2,
            cata_means,
            width,
            yerr=cata_stds,
            label=f"Congestion-Aware (lr={best_lr})",
            alpha=0.8,
            capsize=5,
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "Warehouse Robot Coordination - Throughput Comparison\nLeft: Distance-based Assignment, Right: Congestion-Aware Assignment"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    print(f"Plots saved to {working_dir}")
    print(
        f"Best learning rate: {best_lr}, Best validation loss: {experiment_data.get('best_val_loss', 'N/A'):.4f}"
    )
