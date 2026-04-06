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
        plt.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
        plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Warehouse Robot Task Allocation\nCongestion Predictor Training/Validation Loss"
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

    # Plot 2: Throughput vs max_steps Comparison
    try:
        plt.figure(figsize=(10, 6))
        summary = experiment_data["summary"]
        x = np.array(summary["max_steps"])
        plt.plot(
            x,
            summary["avg_dist"],
            "bo-",
            label="Distance-based",
            markersize=8,
            linewidth=2,
        )
        plt.plot(
            x,
            summary["avg_cata"],
            "gs-",
            label="Congestion-aware (CATA)",
            markersize=8,
            linewidth=2,
        )
        plt.xlabel("max_steps (Simulation Length)")
        plt.ylabel("Average Throughput (tasks/min)")
        plt.title(
            "Warehouse Robot Task Allocation\nThroughput Comparison: Distance-based vs Congestion-aware"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(x)
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_vs_maxsteps.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Improvement Percentage Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        summary = experiment_data["summary"]
        x = summary["max_steps"]
        improvements = summary["avg_improvement"]
        colors = ["green" if v >= 0 else "red" for v in improvements]
        plt.bar(x, improvements, color=colors, alpha=0.7, width=40)
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("max_steps (Simulation Length)")
        plt.ylabel("Average Improvement (%)")
        plt.title(
            "Warehouse Robot Task Allocation\nCATA Improvement over Distance-based Method by max_steps"
        )
        plt.xticks(x)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_by_maxsteps.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement bar chart: {e}")
        plt.close()

    # Plot 4: Throughput by Agent Count for Different max_steps
    try:
        plt.figure(figsize=(12, 7))
        max_steps_values = [100, 200, 300, 400, 500]
        colors = plt.cm.viridis(np.linspace(0, 1, len(max_steps_values)))
        for idx, max_steps in enumerate(max_steps_values):
            key = f"max_steps_{max_steps}"
            agent_counts = experiment_data["hyperparam_tuning"][key]["n_agents"]
            cata_throughput = experiment_data["hyperparam_tuning"][key][
                "congestion_aware"
            ]["throughput"]
            plt.plot(
                agent_counts,
                cata_throughput,
                "o-",
                color=colors[idx],
                label=f"max_steps={max_steps}",
                markersize=8,
                linewidth=2,
            )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "Warehouse Robot Task Allocation\nCATA Throughput by Agent Count Across Different max_steps"
        )
        plt.legend(title="Simulation Length")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_throughput_by_agents.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput by agent count plot: {e}")
        plt.close()

    # Plot 5: Heatmap of Improvement across max_steps and agent counts
    try:
        plt.figure(figsize=(10, 8))
        max_steps_values = [100, 200, 300, 400, 500]
        agent_counts = experiment_data["hyperparam_tuning"]["max_steps_100"]["n_agents"]
        improvement_matrix = []
        for max_steps in max_steps_values:
            key = f"max_steps_{max_steps}"
            improvement_matrix.append(
                experiment_data["hyperparam_tuning"][key]["improvement"]
            )
        improvement_matrix = np.array(improvement_matrix)
        im = plt.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Improvement (%)")
        plt.xticks(range(len(agent_counts)), agent_counts)
        plt.yticks(range(len(max_steps_values)), max_steps_values)
        plt.xlabel("Number of Agents")
        plt.ylabel("max_steps")
        plt.title(
            "Warehouse Robot Task Allocation\nCATA Improvement (%) Heatmap: max_steps vs Agent Count"
        )
        for i in range(len(max_steps_values)):
            for j in range(len(agent_counts)):
                plt.text(
                    j,
                    i,
                    f"{improvement_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        plt.savefig(
            os.path.join(working_dir, "warehouse_improvement_heatmap.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement heatmap: {e}")
        plt.close()

print("All plots saved to working directory.")
