import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "None/experiment_data.npy",
        "None/experiment_data.npy",
        "None/experiment_data.npy",
    ]
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
            train_losses_all = []
            val_losses_all = []
            epochs = all_experiment_data[0]["lr_batch_tuning"][config_key]["training"][
                "epochs"
            ]

            for exp_data in all_experiment_data:
                train_losses_all.append(
                    exp_data["lr_batch_tuning"][config_key]["training"]["losses"][
                        "train"
                    ]
                )
                val_losses_all.append(
                    exp_data["lr_batch_tuning"][config_key]["training"]["losses"]["val"]
                )

            train_mean = np.mean(train_losses_all, axis=0)
            train_se = np.std(train_losses_all, axis=0) / np.sqrt(
                len(all_experiment_data)
            )
            val_mean = np.mean(val_losses_all, axis=0)
            val_se = np.std(val_losses_all, axis=0) / np.sqrt(len(all_experiment_data))

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
        axes[0].set_title("Training Loss (Mean ± SE across runs)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss (Mean ± SE across runs)")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Aggregated LR & Batch Size Tuning - Loss Curves (n={len(all_experiment_data)} runs)",
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

        improvements_all = []
        for exp_data in all_experiment_data:
            improvements_all.append(
                [
                    exp_data["lr_batch_tuning"][k]["overall_improvement"]
                    for k in config_keys
                ]
            )

        improvements_mean = np.mean(improvements_all, axis=0)
        improvements_se = np.std(improvements_all, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        bar_colors = ["green" if imp > 0 else "red" for imp in improvements_mean]

        x_pos = np.arange(len(config_keys))
        bars = plt.bar(
            x_pos,
            improvements_mean,
            color=bar_colors,
            alpha=0.7,
            edgecolor="black",
            label="Mean Improvement",
        )
        plt.errorbar(
            x_pos,
            improvements_mean,
            yerr=improvements_se,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
            label="Standard Error",
        )

        plt.xticks(x_pos, config_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.xlabel("Configuration (Learning Rate & Batch Size)")
        plt.title(
            f"Aggregated Overall Improvement by Configuration (n={len(all_experiment_data)} runs)\nWarehouse Task Assignment - Mean ± SE"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        best_idx = np.argmax(improvements_mean)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        plt.annotate(
            f"Best: {config_keys[best_idx]}\n{improvements_mean[best_idx]:.2f}±{improvements_se[best_idx]:.2f}%",
            xy=(best_idx, improvements_mean[best_idx]),
            xytext=(best_idx, improvements_mean[best_idx] + 3),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_improvement_by_config.png"),
            dpi=150,
        )
        plt.close()
        print("Saved aggregated improvement bar chart")
    except Exception as e:
        print(f"Error creating aggregated improvement plot: {e}")
        plt.close()

    # Plot 3: Aggregated Heatmap with Mean ± SE
    try:
        lr_values = [0.0005, 0.001, 0.002]
        batch_sizes = [16, 32, 64]
        improvement_matrices = []

        for exp_data in all_experiment_data:
            matrix = np.zeros((len(lr_values), len(batch_sizes)))
            for i, lr in enumerate(lr_values):
                for j, bs in enumerate(batch_sizes):
                    config_key = f"lr{lr}_bs{bs}"
                    if config_key in exp_data["lr_batch_tuning"]:
                        matrix[i, j] = exp_data["lr_batch_tuning"][config_key][
                            "overall_improvement"
                        ]
            improvement_matrices.append(matrix)

        mean_matrix = np.mean(improvement_matrices, axis=0)
        se_matrix = np.std(improvement_matrices, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        plt.figure(figsize=(10, 7))
        im = plt.imshow(mean_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Mean Overall Improvement (%)")
        plt.xticks(range(len(batch_sizes)), batch_sizes)
        plt.yticks(range(len(lr_values)), lr_values)
        plt.xlabel("Batch Size")
        plt.ylabel("Learning Rate")
        plt.title(
            f"Aggregated Hyperparameter Grid Search (n={len(all_experiment_data)} runs)\nWarehouse Task Assignment - Mean ± SE"
        )

        for i in range(len(lr_values)):
            for j in range(len(batch_sizes)):
                plt.text(
                    j,
                    i,
                    f"{mean_matrix[i, j]:.1f}%\n±{se_matrix[i, j]:.2f}",
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
        print(f"Error creating aggregated heatmap plot: {e}")
        plt.close()

    # Plot 4: Aggregated Validation Loss vs Improvement with Error Bars
    try:
        plt.figure(figsize=(10, 6))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        val_losses_all = []
        improvements_all = []
        for exp_data in all_experiment_data:
            val_losses_all.append(
                [exp_data["lr_batch_tuning"][k]["final_val_loss"] for k in config_keys]
            )
            improvements_all.append(
                [
                    exp_data["lr_batch_tuning"][k]["overall_improvement"]
                    for k in config_keys
                ]
            )

        val_mean = np.mean(val_losses_all, axis=0)
        val_se = np.std(val_losses_all, axis=0) / np.sqrt(len(all_experiment_data))
        imp_mean = np.mean(improvements_all, axis=0)
        imp_se = np.std(improvements_all, axis=0) / np.sqrt(len(all_experiment_data))

        colors = plt.cm.tab10(np.linspace(0, 1, len(config_keys)))

        for i, config in enumerate(config_keys):
            plt.errorbar(
                val_mean[i],
                imp_mean[i],
                xerr=val_se[i],
                yerr=imp_se[i],
                fmt="o",
                color=colors[i],
                markersize=10,
                capsize=4,
                capthick=1.5,
                ecolor=colors[i],
                alpha=0.8,
                label=config,
            )

        plt.xlabel("Final Validation Loss (Mean ± SE)")
        plt.ylabel("Overall Throughput Improvement (%) (Mean ± SE)")
        plt.title(
            f"Aggregated Validation Loss vs Throughput Improvement (n={len(all_experiment_data)} runs)\nWarehouse Task Assignment"
        )
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.legend(fontsize=8, loc="best")
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

    # Plot 5: Summary Statistics Table
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")

        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        val_losses_all = []
        improvements_all = []
        for exp_data in all_experiment_data:
            val_losses_all.append(
                [exp_data["lr_batch_tuning"][k]["final_val_loss"] for k in config_keys]
            )
            improvements_all.append(
                [
                    exp_data["lr_batch_tuning"][k]["overall_improvement"]
                    for k in config_keys
                ]
            )

        val_mean = np.mean(val_losses_all, axis=0)
        val_se = np.std(val_losses_all, axis=0) / np.sqrt(len(all_experiment_data))
        imp_mean = np.mean(improvements_all, axis=0)
        imp_se = np.std(improvements_all, axis=0) / np.sqrt(len(all_experiment_data))

        table_data = []
        for i, config in enumerate(config_keys):
            table_data.append(
                [
                    config,
                    f"{val_mean[i]:.4f} ± {val_se[i]:.4f}",
                    f"{imp_mean[i]:.2f} ± {imp_se[i]:.2f}%",
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=[
                "Configuration",
                "Final Val Loss (Mean ± SE)",
                "Improvement (Mean ± SE)",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.title(
            f"Aggregated Results Summary (n={len(all_experiment_data)} runs)\nWarehouse Congestion-Aware Task Assignment",
            fontsize=12,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_summary_table.png"), dpi=150
        )
        plt.close()
        print("Saved aggregated summary table")
    except Exception as e:
        print(f"Error creating summary table: {e}")
        plt.close()

print("All aggregated plots saved successfully.")
