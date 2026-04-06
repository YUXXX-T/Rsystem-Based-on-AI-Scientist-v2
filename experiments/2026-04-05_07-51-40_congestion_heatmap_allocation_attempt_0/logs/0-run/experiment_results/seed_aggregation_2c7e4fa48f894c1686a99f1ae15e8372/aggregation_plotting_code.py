import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2026-04-05_07-51-40_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_5884b5198f5648469697d257203726d1_proc_100/experiment_data.npy",
    "experiments/2026-04-05_07-51-40_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_88f56061fbba482ab3a01d0a5ae45947_proc_97/experiment_data.npy",
    "experiments/2026-04-05_07-51-40_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_eb2439ee45a04e4891356560617deb8b_proc_98/experiment_data.npy",
]

try:
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    print(f"Successfully loaded {len(all_experiment_data)} experiment runs")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if len(all_experiment_data) > 0:
    # Plot 1: Aggregated Training and Validation Loss Curves with Standard Error
    try:
        plt.figure(figsize=(10, 6))

        all_train_losses = []
        all_val_losses = []
        epochs = None

        for exp_data in all_experiment_data:
            all_train_losses.append(exp_data["training"]["losses"]["train"])
            all_val_losses.append(exp_data["training"]["losses"]["val"])
            if epochs is None:
                epochs = exp_data["training"]["epochs"]

        all_train_losses = np.array(all_train_losses)
        all_val_losses = np.array(all_val_losses)

        train_mean = np.mean(all_train_losses, axis=0)
        train_se = np.std(all_train_losses, axis=0) / np.sqrt(len(all_experiment_data))
        val_mean = np.mean(all_val_losses, axis=0)
        val_se = np.std(all_val_losses, axis=0) / np.sqrt(len(all_experiment_data))

        plt.plot(
            epochs, train_mean, label="Train Loss (Mean)", color="blue", linewidth=2
        )
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="blue",
            alpha=0.2,
            label="Train SE",
        )
        plt.plot(
            epochs,
            val_mean,
            label="Validation Loss (Mean)",
            color="orange",
            linewidth=2,
        )
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="orange",
            alpha=0.2,
            label="Val SE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Aggregated Congestion Predictor Training Curves\nWarehouse Robot Coordination - CNN Model (Mean ± SE across 3 runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_training_loss_curves.png"),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated training loss curves plot")
    except Exception as e:
        print(f"Error creating aggregated training loss plot: {e}")
        plt.close()

    # Plot 2: Aggregated Throughput Comparison Bar Chart
    try:
        plt.figure(figsize=(12, 7))

        all_dist_means = []
        all_cata_means = []
        agent_counts = None

        for exp_data in all_experiment_data:
            dist_data = exp_data["evaluation"]["distance_based"]["throughput"]
            cata_data = exp_data["evaluation"]["congestion_aware"]["throughput"]

            if agent_counts is None:
                agent_counts = [d["n_agents"] for d in dist_data]

            all_dist_means.append([d["mean"] for d in dist_data])
            all_cata_means.append([d["mean"] for d in cata_data])

        all_dist_means = np.array(all_dist_means)
        all_cata_means = np.array(all_cata_means)

        dist_agg_mean = np.mean(all_dist_means, axis=0)
        dist_agg_se = np.std(all_dist_means, axis=0) / np.sqrt(len(all_experiment_data))
        cata_agg_mean = np.mean(all_cata_means, axis=0)
        cata_agg_se = np.std(all_cata_means, axis=0) / np.sqrt(len(all_experiment_data))

        x = np.arange(len(agent_counts))
        width = 0.35

        bars1 = plt.bar(
            x - width / 2,
            dist_agg_mean,
            width,
            yerr=dist_agg_se,
            label="Distance-based (Mean ± SE)",
            alpha=0.8,
            color="steelblue",
            capsize=5,
        )
        bars2 = plt.bar(
            x + width / 2,
            cata_agg_mean,
            width,
            yerr=cata_agg_se,
            label="Congestion-Aware CATA (Mean ± SE)",
            alpha=0.8,
            color="coral",
            capsize=5,
        )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/minute)")
        plt.title(
            "Aggregated Warehouse Task Assignment: Throughput Comparison\nDistance-based vs Congestion-Aware Assignment (Mean ± SE across 3 runs)"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated throughput comparison plot")
    except Exception as e:
        print(f"Error creating aggregated throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Aggregated Improvement Percentage Line Plot with Error Bars
    try:
        plt.figure(figsize=(10, 6))

        all_improvements = []
        for i in range(len(all_experiment_data)):
            improvements = [
                (all_cata_means[i][j] - all_dist_means[i][j])
                / all_dist_means[i][j]
                * 100
                for j in range(len(agent_counts))
            ]
            all_improvements.append(improvements)

        all_improvements = np.array(all_improvements)
        improvements_mean = np.mean(all_improvements, axis=0)
        improvements_se = np.std(all_improvements, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        plt.errorbar(
            agent_counts,
            improvements_mean,
            yerr=improvements_se,
            marker="o",
            linewidth=2,
            markersize=10,
            color="green",
            capsize=5,
            label="Mean ± SE",
        )
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.fill_between(
            agent_counts,
            improvements_mean - improvements_se,
            improvements_mean + improvements_se,
            alpha=0.2,
            color="green",
        )

        for i, (xval, yval) in enumerate(zip(agent_counts, improvements_mean)):
            plt.annotate(
                f"{yval:.1f}%",
                (xval, yval),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontsize=9,
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Improvement (%)")
        plt.title(
            "Aggregated CATA Improvement over Distance-based\nPercentage Throughput Improvement (Mean ± SE across 3 runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_improvement_percentage.png"
            ),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated improvement percentage plot")
    except Exception as e:
        print(f"Error creating aggregated improvement plot: {e}")
        plt.close()

    # Print aggregated summary metrics
    print("\n=== Aggregated Experiment Summary (across 3 runs) ===")
    print(f"Final training loss: {train_mean[-1]:.4f} ± {train_se[-1]:.4f} (mean ± SE)")
    print(f"Final validation loss: {val_mean[-1]:.4f} ± {val_se[-1]:.4f} (mean ± SE)")
    print(
        f"Average distance-based throughput: {np.mean(dist_agg_mean):.2f} ± {np.mean(dist_agg_se):.2f} tasks/min"
    )
    print(
        f"Average CATA throughput: {np.mean(cata_agg_mean):.2f} ± {np.mean(cata_agg_se):.2f} tasks/min"
    )
    print(
        f"Average improvement: {np.mean(improvements_mean):.1f}% ± {np.mean(improvements_se):.1f}%"
    )
