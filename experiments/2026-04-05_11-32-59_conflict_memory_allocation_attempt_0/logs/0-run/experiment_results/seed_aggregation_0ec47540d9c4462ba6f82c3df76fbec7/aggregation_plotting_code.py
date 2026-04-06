import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_6cfc2b64f7fa418a9d43d291d76b4322_proc_3691/experiment_data.npy",
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_485c27fa397045a79c8083fc1e7d6b39_proc_3693/experiment_data.npy",
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_711c47d81ea84db28b823b236b9229e3_proc_3690/experiment_data.npy",
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
    # Plot 1: Aggregated Training and Validation Loss Curves with Std Error
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(config_keys)))

        for idx, config_key in enumerate(config_keys):
            all_train_losses = []
            all_val_losses = []
            epochs = None

            for exp_data in all_experiment_data:
                if config_key in exp_data["lr_batch_tuning"]:
                    data = exp_data["lr_batch_tuning"][config_key]["training"]["losses"]
                    epochs = exp_data["lr_batch_tuning"][config_key]["training"][
                        "epochs"
                    ]
                    all_train_losses.append(data["train"])
                    all_val_losses.append(data["val"])

            if len(all_train_losses) > 0:
                train_mean = np.mean(all_train_losses, axis=0)
                train_se = np.std(all_train_losses, axis=0) / np.sqrt(
                    len(all_train_losses)
                )
                val_mean = np.mean(all_val_losses, axis=0)
                val_se = np.std(all_val_losses, axis=0) / np.sqrt(len(all_val_losses))

                axes[0].plot(
                    epochs, train_mean, label=config_key, color=colors[idx], alpha=0.8
                )
                axes[0].fill_between(
                    epochs,
                    train_mean - train_se,
                    train_mean + train_se,
                    color=colors[idx],
                    alpha=0.2,
                )
                axes[1].plot(
                    epochs, val_mean, label=config_key, color=colors[idx], alpha=0.8
                )
                axes[1].fill_between(
                    epochs,
                    val_mean - val_se,
                    val_mean + val_se,
                    color=colors[idx],
                    alpha=0.2,
                )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Aggregated Training Loss (Mean ± SE)")
        axes[0].legend(fontsize=6)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Aggregated Validation Loss (Mean ± SE)")
        axes[1].legend(fontsize=6)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Warehouse LR & Batch Tuning - Aggregated Loss Curves (n={len(all_experiment_data)} runs)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_loss_curves.png"), dpi=150
        )
        plt.close()
        print("Saved aggregated loss curves plot")
    except Exception as e:
        print(f"Error creating aggregated loss curves plot: {e}")
        plt.close()

    # Plot 2: Aggregated Overall Improvement Bar Chart with Error Bars
    try:
        plt.figure(figsize=(12, 6))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        mean_improvements = []
        se_improvements = []

        for config_key in config_keys:
            improvements = [
                exp["lr_batch_tuning"][config_key]["overall_improvement"]
                for exp in all_experiment_data
                if config_key in exp["lr_batch_tuning"]
            ]
            mean_improvements.append(np.mean(improvements))
            se_improvements.append(np.std(improvements) / np.sqrt(len(improvements)))

        bar_colors = ["green" if imp > 0 else "red" for imp in mean_improvements]
        x_pos = range(len(config_keys))

        bars = plt.bar(
            x_pos,
            mean_improvements,
            yerr=se_improvements,
            color=bar_colors,
            alpha=0.7,
            edgecolor="black",
            capsize=5,
            error_kw={"elinewidth": 2, "capthick": 2},
        )

        plt.xticks(x_pos, config_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.xlabel("Configuration (Learning Rate & Batch Size)")
        plt.title(
            f"Aggregated Overall Improvement by Configuration\nWarehouse Task Assignment (Mean ± SE, n={len(all_experiment_data)} runs)"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        best_idx = np.argmax(mean_improvements)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        plt.annotate(
            f"Best: {config_keys[best_idx]}",
            xy=(best_idx, mean_improvements[best_idx]),
            xytext=(
                best_idx,
                mean_improvements[best_idx] + se_improvements[best_idx] + 1,
            ),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        plt.legend(["Mean ± Standard Error"], loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_improvement_bar.png"),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated improvement bar chart")
    except Exception as e:
        print(f"Error creating aggregated improvement bar chart: {e}")
        plt.close()

    # Plot 3: Aggregated Heatmap with Mean and Std
    try:
        lr_values = [0.0005, 0.001, 0.002]
        batch_sizes = [16, 32, 64]
        mean_matrix = np.zeros((len(lr_values), len(batch_sizes)))
        std_matrix = np.zeros((len(lr_values), len(batch_sizes)))

        for i, lr in enumerate(lr_values):
            for j, bs in enumerate(batch_sizes):
                config_key = f"lr{lr}_bs{bs}"
                improvements = [
                    exp["lr_batch_tuning"][config_key]["overall_improvement"]
                    for exp in all_experiment_data
                    if config_key in exp.get("lr_batch_tuning", {})
                ]
                if len(improvements) > 0:
                    mean_matrix[i, j] = np.mean(improvements)
                    std_matrix[i, j] = np.std(improvements)

        plt.figure(figsize=(9, 7))
        im = plt.imshow(mean_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Mean Overall Improvement (%)")
        plt.xticks(range(len(batch_sizes)), batch_sizes)
        plt.yticks(range(len(lr_values)), lr_values)
        plt.xlabel("Batch Size")
        plt.ylabel("Learning Rate")
        plt.title(
            f"Aggregated Hyperparameter Grid Search Results\nWarehouse Task Assignment (Mean ± Std, n={len(all_experiment_data)} runs)"
        )

        for i in range(len(lr_values)):
            for j in range(len(batch_sizes)):
                plt.text(
                    j,
                    i,
                    f"{mean_matrix[i, j]:.1f}%\n±{std_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_heatmap.png"), dpi=150
        )
        plt.close()
        print("Saved aggregated heatmap")
    except Exception as e:
        print(f"Error creating aggregated heatmap: {e}")
        plt.close()

    # Plot 4: Aggregated Val Loss vs Improvement Scatter with Error Bars
    try:
        plt.figure(figsize=(10, 7))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        for idx, config_key in enumerate(config_keys):
            val_losses = [
                exp["lr_batch_tuning"][config_key]["final_val_loss"]
                for exp in all_experiment_data
                if config_key in exp["lr_batch_tuning"]
            ]
            improvements = [
                exp["lr_batch_tuning"][config_key]["overall_improvement"]
                for exp in all_experiment_data
                if config_key in exp["lr_batch_tuning"]
            ]

            mean_vl = np.mean(val_losses)
            se_vl = np.std(val_losses) / np.sqrt(len(val_losses))
            mean_imp = np.mean(improvements)
            se_imp = np.std(improvements) / np.sqrt(len(improvements))

            plt.errorbar(
                mean_vl,
                mean_imp,
                xerr=se_vl,
                yerr=se_imp,
                fmt="o",
                markersize=10,
                capsize=5,
                capthick=2,
                color=plt.cm.tab10(idx / len(config_keys)),
                label=config_key,
            )

        plt.xlabel("Final Validation Loss (Mean ± SE)")
        plt.ylabel("Overall Throughput Improvement (%) (Mean ± SE)")
        plt.title(
            f"Aggregated Validation Loss vs Throughput Improvement\nWarehouse Task Assignment (n={len(all_experiment_data)} runs)"
        )
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.legend(fontsize=7, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_val_loss_vs_improvement.png"
            ),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated val loss vs improvement plot")
    except Exception as e:
        print(f"Error creating aggregated val loss vs improvement plot: {e}")
        plt.close()

    # Plot 5: Individual Run Comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())
        x_pos = np.arange(len(config_keys))
        width = 0.25

        for run_idx, exp_data in enumerate(all_experiment_data):
            improvements = [
                exp_data["lr_batch_tuning"][k]["overall_improvement"]
                for k in config_keys
            ]
            axes[0].bar(
                x_pos + run_idx * width,
                improvements,
                width,
                label=f"Run {run_idx + 1}",
                alpha=0.7,
            )

        axes[0].set_xticks(x_pos + width)
        axes[0].set_xticklabels(config_keys, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Overall Improvement (%)")
        axes[0].set_title("Individual Run Comparison")
        axes[0].legend()
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[0].grid(True, alpha=0.3)

        # Box plot for distribution
        improvement_data = []
        for config_key in config_keys:
            improvements = [
                exp["lr_batch_tuning"][config_key]["overall_improvement"]
                for exp in all_experiment_data
            ]
            improvement_data.append(improvements)

        bp = axes[1].boxplot(improvement_data, labels=config_keys, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        axes[1].set_xticklabels(config_keys, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Overall Improvement (%)")
        axes[1].set_title("Distribution Across Runs (Box Plot)")
        axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Warehouse Task Assignment - Run Variability Analysis (n={len(all_experiment_data)} runs)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_run_variability_comparison.png"),
            dpi=150,
        )
        plt.close()
        print("Saved run variability comparison plot")
    except Exception as e:
        print(f"Error creating run variability plot: {e}")
        plt.close()

print("All aggregated plots saved successfully.")
