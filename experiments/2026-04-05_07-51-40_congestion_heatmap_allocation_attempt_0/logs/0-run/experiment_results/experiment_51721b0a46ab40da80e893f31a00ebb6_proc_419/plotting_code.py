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
    n_tasks_values = [15, 30, 45, 60]
    agent_counts = [20, 30, 40]

    # Plot 1: Training and Validation Losses for each n_tasks
    try:
        plt.figure(figsize=(10, 6))
        for n_tasks in n_tasks_values:
            key = f"n_tasks_{n_tasks}"
            train_losses = experiment_data["n_tasks_tuning"][key]["training"]["losses"][
                "train"
            ]
            val_losses = experiment_data["n_tasks_tuning"][key]["training"]["losses"][
                "val"
            ]
            plt.plot(train_losses, label=f"n_tasks={n_tasks} (train)", linestyle="-")
            plt.plot(val_losses, label=f"n_tasks={n_tasks} (val)", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            "Warehouse CATA: Training and Validation Loss Curves\nCongestion Predictor across different n_tasks values"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_n_tasks_training_validation_loss.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Average Improvement by n_tasks
    try:
        plt.figure(figsize=(8, 6))
        avg_improvements = {}
        for n_tasks in n_tasks_values:
            key = f"n_tasks_{n_tasks}"
            improvements = [
                d["improvement"]
                for d in experiment_data["n_tasks_tuning"][key]["evaluation"][
                    "improvement"
                ]
            ]
            avg_improvements[n_tasks] = np.mean(improvements)
        plt.bar(
            range(len(n_tasks_values)),
            [avg_improvements[nt] for nt in n_tasks_values],
            color="steelblue",
        )
        plt.xticks(range(len(n_tasks_values)), n_tasks_values)
        plt.xlabel("Number of Tasks (n_tasks)")
        plt.ylabel("Average Improvement (%)")
        plt.title(
            "Warehouse CATA: Average Improvement over Distance-based Method\nby n_tasks Hyperparameter"
        )
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_n_tasks_avg_improvement.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating average improvement plot: {e}")
        plt.close()

    # Plot 3: Throughput Comparison for best n_tasks
    try:
        best_n_tasks = max(avg_improvements, key=avg_improvements.get)
        best_data = experiment_data["n_tasks_tuning"][f"n_tasks_{best_n_tasks}"][
            "evaluation"
        ]
        plt.figure(figsize=(8, 6))
        x = np.arange(len(agent_counts))
        width = 0.35
        dist_means = [d["mean"] for d in best_data["distance_based"]]
        cata_means = [d["mean"] for d in best_data["congestion_aware"]]
        plt.bar(
            x - width / 2,
            dist_means,
            width,
            label="Distance-based",
            alpha=0.8,
            color="coral",
        )
        plt.bar(
            x + width / 2,
            cata_means,
            width,
            label="Congestion-Aware (CATA)",
            alpha=0.8,
            color="teal",
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(f"Warehouse CATA: Throughput Comparison\nBest n_tasks={best_n_tasks}")
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_n_tasks_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 4: Improvement Heatmap
    try:
        plt.figure(figsize=(8, 6))
        improvement_matrix = np.zeros((len(n_tasks_values), len(agent_counts)))
        for i, n_tasks in enumerate(n_tasks_values):
            key = f"n_tasks_{n_tasks}"
            for j, agent_data in enumerate(
                experiment_data["n_tasks_tuning"][key]["evaluation"]["improvement"]
            ):
                improvement_matrix[i, j] = agent_data["improvement"]
        im = plt.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
        plt.xticks(range(len(agent_counts)), agent_counts)
        plt.yticks(range(len(n_tasks_values)), n_tasks_values)
        plt.xlabel("Number of Agents")
        plt.ylabel("Number of Tasks (n_tasks)")
        plt.title(
            "Warehouse CATA: Improvement (%) Heatmap\nCongestion-Aware vs Distance-based across configurations"
        )
        plt.colorbar(im, label="Improvement (%)")
        for i in range(len(n_tasks_values)):
            for j in range(len(agent_counts)):
                plt.text(
                    j,
                    i,
                    f"{improvement_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_n_tasks_improvement_heatmap.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

    print("All plots saved successfully.")
