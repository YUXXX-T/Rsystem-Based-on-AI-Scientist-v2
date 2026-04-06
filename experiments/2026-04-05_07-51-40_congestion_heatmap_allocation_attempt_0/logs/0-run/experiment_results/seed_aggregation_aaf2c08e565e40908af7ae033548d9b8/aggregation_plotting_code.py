import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
experiment_data_path_list = [
    "experiments/2026-04-05_07-51-40_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_87616954740d4edf8d3bfc1fd571290b_proc_418/experiment_data.npy",
    "experiments/2026-04-05_07-51-40_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_c9cc50aa82bc4a36a7bcf8f367f4ec9c_proc_416/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        if experiment_data_path is None or "None" in experiment_data_path:
            continue
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        if os.path.exists(full_path):
            experiment_data = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

print(f"Loaded {len(all_experiment_data)} experiment runs")

if len(all_experiment_data) > 0:
    # Plot 1: Aggregated Training and Validation Loss Curves with Standard Error
    try:
        plt.figure(figsize=(10, 6))
        all_train_losses = []
        all_val_losses = []
        epochs = None
        for exp_data in all_experiment_data:
            if epochs is None:
                epochs = np.array(exp_data["training"]["epochs"])
            all_train_losses.append(exp_data["training"]["losses"]["train"])
            all_val_losses.append(exp_data["training"]["losses"]["val"])

        all_train_losses = np.array(all_train_losses)
        all_val_losses = np.array(all_val_losses)

        train_mean = np.mean(all_train_losses, axis=0)
        train_se = np.std(all_train_losses, axis=0) / np.sqrt(len(all_experiment_data))
        val_mean = np.mean(all_val_losses, axis=0)
        val_se = np.std(all_val_losses, axis=0) / np.sqrt(len(all_experiment_data))

        plt.plot(epochs, train_mean, "b-", label="Train Loss (Mean)", linewidth=2)
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            color="blue",
            label="Train SE",
        )
        plt.plot(epochs, val_mean, "r-", label="Validation Loss (Mean)", linewidth=2)
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            alpha=0.3,
            color="red",
            label="Val SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            f"Warehouse Robot Task Allocation\nAggregated Training/Validation Loss (n={len(all_experiment_data)} runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_training_loss.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated training loss plot: {e}")
        plt.close()

    # Plot 2: Aggregated Throughput vs max_steps with Standard Error Bars
    try:
        plt.figure(figsize=(10, 6))
        all_avg_dist = []
        all_avg_cata = []
        max_steps = None
        for exp_data in all_experiment_data:
            if max_steps is None:
                max_steps = np.array(exp_data["summary"]["max_steps"])
            all_avg_dist.append(exp_data["summary"]["avg_dist"])
            all_avg_cata.append(exp_data["summary"]["avg_cata"])

        all_avg_dist = np.array(all_avg_dist)
        all_avg_cata = np.array(all_avg_cata)

        dist_mean = np.mean(all_avg_dist, axis=0)
        dist_se = np.std(all_avg_dist, axis=0) / np.sqrt(len(all_experiment_data))
        cata_mean = np.mean(all_avg_cata, axis=0)
        cata_se = np.std(all_avg_cata, axis=0) / np.sqrt(len(all_experiment_data))

        plt.errorbar(
            max_steps,
            dist_mean,
            yerr=dist_se,
            fmt="bo-",
            label="Distance-based (Mean ± SE)",
            markersize=8,
            linewidth=2,
            capsize=5,
        )
        plt.errorbar(
            max_steps,
            cata_mean,
            yerr=cata_se,
            fmt="gs-",
            label="CATA (Mean ± SE)",
            markersize=8,
            linewidth=2,
            capsize=5,
        )
        plt.xlabel("max_steps (Simulation Length)")
        plt.ylabel("Average Throughput (tasks/min)")
        plt.title(
            f"Warehouse Robot Task Allocation\nAggregated Throughput Comparison (n={len(all_experiment_data)} runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(max_steps)
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_throughput_vs_maxsteps.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated throughput plot: {e}")
        plt.close()

    # Plot 3: Aggregated Improvement Bar Chart with Standard Error
    try:
        plt.figure(figsize=(10, 6))
        all_improvements = []
        max_steps = None
        for exp_data in all_experiment_data:
            if max_steps is None:
                max_steps = exp_data["summary"]["max_steps"]
            all_improvements.append(exp_data["summary"]["avg_improvement"])

        all_improvements = np.array(all_improvements)
        improvement_mean = np.mean(all_improvements, axis=0)
        improvement_se = np.std(all_improvements, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        colors = ["green" if v >= 0 else "red" for v in improvement_mean]
        bars = plt.bar(
            max_steps,
            improvement_mean,
            color=colors,
            alpha=0.7,
            width=40,
            label="Mean Improvement",
        )
        plt.errorbar(
            max_steps,
            improvement_mean,
            yerr=improvement_se,
            fmt="none",
            ecolor="black",
            capsize=5,
            label="Standard Error",
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("max_steps (Simulation Length)")
        plt.ylabel("Average Improvement (%)")
        plt.title(
            f"Warehouse Robot Task Allocation\nAggregated CATA Improvement (n={len(all_experiment_data)} runs)"
        )
        plt.legend()
        plt.xticks(max_steps)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_improvement_by_maxsteps.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated improvement bar chart: {e}")
        plt.close()

    # Plot 4: Aggregated Throughput by Agent Count with Error Bands
    try:
        plt.figure(figsize=(12, 7))
        max_steps_values = [100, 200, 300, 400, 500]
        colors = plt.cm.viridis(np.linspace(0, 1, len(max_steps_values)))

        for idx, max_steps in enumerate(max_steps_values):
            key = f"max_steps_{max_steps}"
            all_throughputs = []
            agent_counts = None
            for exp_data in all_experiment_data:
                if agent_counts is None:
                    agent_counts = np.array(
                        exp_data["hyperparam_tuning"][key]["n_agents"]
                    )
                all_throughputs.append(
                    exp_data["hyperparam_tuning"][key]["congestion_aware"]["throughput"]
                )

            all_throughputs = np.array(all_throughputs)
            throughput_mean = np.mean(all_throughputs, axis=0)
            throughput_se = np.std(all_throughputs, axis=0) / np.sqrt(
                len(all_experiment_data)
            )

            plt.errorbar(
                agent_counts,
                throughput_mean,
                yerr=throughput_se,
                fmt="o-",
                color=colors[idx],
                label=f"max_steps={max_steps} (Mean ± SE)",
                markersize=8,
                linewidth=2,
                capsize=4,
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            f"Warehouse Robot Task Allocation\nAggregated CATA Throughput by Agent Count (n={len(all_experiment_data)} runs)"
        )
        plt.legend(title="Simulation Length")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_cata_throughput_by_agents.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated throughput by agent count plot: {e}")
        plt.close()

    # Plot 5: Aggregated Improvement Heatmap with Mean and SE
    try:
        plt.figure(figsize=(12, 9))
        max_steps_values = [100, 200, 300, 400, 500]
        agent_counts = all_experiment_data[0]["hyperparam_tuning"]["max_steps_100"][
            "n_agents"
        ]

        all_improvement_matrices = []
        for exp_data in all_experiment_data:
            improvement_matrix = []
            for max_steps in max_steps_values:
                key = f"max_steps_{max_steps}"
                improvement_matrix.append(
                    exp_data["hyperparam_tuning"][key]["improvement"]
                )
            all_improvement_matrices.append(np.array(improvement_matrix))

        all_improvement_matrices = np.array(all_improvement_matrices)
        improvement_mean = np.mean(all_improvement_matrices, axis=0)
        improvement_se = np.std(all_improvement_matrices, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        im = plt.imshow(improvement_mean, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Mean Improvement (%)")
        plt.xticks(range(len(agent_counts)), agent_counts)
        plt.yticks(range(len(max_steps_values)), max_steps_values)
        plt.xlabel("Number of Agents")
        plt.ylabel("max_steps")
        plt.title(
            f"Warehouse Robot Task Allocation\nAggregated CATA Improvement Heatmap (n={len(all_experiment_data)} runs)\nFormat: Mean ± SE"
        )

        for i in range(len(max_steps_values)):
            for j in range(len(agent_counts)):
                plt.text(
                    j,
                    i,
                    f"{improvement_mean[i, j]:.1f}%\n±{improvement_se[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_improvement_heatmap.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated improvement heatmap: {e}")
        plt.close()

print("All aggregated plots saved to working directory.")
