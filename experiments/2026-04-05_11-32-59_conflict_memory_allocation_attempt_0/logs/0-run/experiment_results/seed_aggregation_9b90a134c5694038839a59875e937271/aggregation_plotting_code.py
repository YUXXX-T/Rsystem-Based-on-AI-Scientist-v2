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
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    print(f"Successfully loaded {len(all_experiment_data)} experiment files")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    # Fallback to local working directory
    try:
        experiment_data = np.load(
            os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
        ).item()
        all_experiment_data = [experiment_data, experiment_data, experiment_data]
        print("Loaded single experiment file as fallback")
    except Exception as e2:
        print(f"Error loading fallback data: {e2}")
        all_experiment_data = []

if len(all_experiment_data) > 0:
    # Plot 1: Aggregated Training and Validation Loss Curves with Mean and SE
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

            train_losses_all = np.array(train_losses_all)
            val_losses_all = np.array(val_losses_all)

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
        axes[0].set_title("Training Loss (Mean ± SE)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss (Mean ± SE)")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            "Aggregated LR & Batch Size Tuning - Loss Curves\n(Shaded regions show Standard Error)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_loss_curves.png"), dpi=150
        )
        plt.close()
        print("Created aggregated loss curves plot")
    except Exception as e:
        print(f"Error creating aggregated loss curves plot: {e}")
        plt.close()

    # Plot 2: Aggregated Overall Improvement Bar Chart with Error Bars
    try:
        plt.figure(figsize=(12, 6))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        improvements_all = []
        for exp_data in all_experiment_data:
            improvements = [
                exp_data["lr_batch_tuning"][k]["overall_improvement"]
                for k in config_keys
            ]
            improvements_all.append(improvements)

        improvements_all = np.array(improvements_all)
        mean_improvements = np.mean(improvements_all, axis=0)
        se_improvements = np.std(improvements_all, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        colors = ["green" if imp > 0 else "red" for imp in mean_improvements]
        x_pos = np.arange(len(config_keys))

        bars = plt.bar(
            x_pos,
            mean_improvements,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            yerr=se_improvements,
            capsize=5,
            error_kw={"elinewidth": 2, "capthick": 2},
        )

        plt.xticks(x_pos, config_keys, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.xlabel("Configuration (Learning Rate & Batch Size)")
        plt.title(
            "Aggregated Overall Improvement by Configuration\nWarehouse Task Assignment (Mean ± SE, n={})".format(
                len(all_experiment_data)
            )
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
                mean_improvements[best_idx] + se_improvements[best_idx] + 2,
            ),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        plt.legend(["Mean", "Standard Error"], loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_improvement_by_config.png"),
            dpi=150,
        )
        plt.close()
        print("Created aggregated improvement bar chart")
    except Exception as e:
        print(f"Error creating aggregated improvement plot: {e}")
        plt.close()

    # Plot 3: Aggregated Heatmap with Mean and SE annotations
    try:
        lr_values = [0.0005, 0.001, 0.002]
        batch_sizes = [16, 32, 64]
        improvement_matrices = []

        for exp_data in all_experiment_data:
            improvement_matrix = np.zeros((len(lr_values), len(batch_sizes)))
            for i, lr in enumerate(lr_values):
                for j, bs in enumerate(batch_sizes):
                    config_key = f"lr{lr}_bs{bs}"
                    if config_key in exp_data["lr_batch_tuning"]:
                        improvement_matrix[i, j] = exp_data["lr_batch_tuning"][
                            config_key
                        ]["overall_improvement"]
            improvement_matrices.append(improvement_matrix)

        improvement_matrices = np.array(improvement_matrices)
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
            "Aggregated Hyperparameter Grid Search Results\nWarehouse Task Assignment (Mean ± SE, n={})".format(
                len(all_experiment_data)
            )
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
            os.path.join(working_dir, "warehouse_aggregated_lr_batch_heatmap.png"),
            dpi=150,
        )
        plt.close()
        print("Created aggregated heatmap")
    except Exception as e:
        print(f"Error creating aggregated heatmap plot: {e}")
        plt.close()

    # Plot 4: Aggregated Dataset-specific Throughput Comparison
    try:
        best_config = all_experiment_data[0]["best_config"]
        datasets = list(
            all_experiment_data[0]["lr_batch_tuning"][best_config]["datasets"].keys()
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, dataset_name in enumerate(datasets[:5]):
            n_agents = all_experiment_data[0]["lr_batch_tuning"][best_config][
                "datasets"
            ][dataset_name]["n_agents"]

            dist_tp_all = []
            cata_tp_all = []
            for exp_data in all_experiment_data:
                data = exp_data["lr_batch_tuning"][best_config]["datasets"][
                    dataset_name
                ]
                dist_tp_all.append(data["distance_based"]["throughput"])
                cata_tp_all.append(data["congestion_aware"]["throughput"])

            dist_tp_all = np.array(dist_tp_all)
            cata_tp_all = np.array(cata_tp_all)

            dist_mean = np.mean(dist_tp_all, axis=0)
            dist_se = np.std(dist_tp_all, axis=0) / np.sqrt(len(all_experiment_data))
            cata_mean = np.mean(cata_tp_all, axis=0)
            cata_se = np.std(cata_tp_all, axis=0) / np.sqrt(len(all_experiment_data))

            x = np.arange(len(n_agents))
            width = 0.35
            axes[idx].bar(
                x - width / 2,
                dist_mean,
                width,
                label="Distance-Based (Mean)",
                color="blue",
                alpha=0.7,
                yerr=dist_se,
                capsize=3,
            )
            axes[idx].bar(
                x + width / 2,
                cata_mean,
                width,
                label="Congestion-Aware (Mean)",
                color="green",
                alpha=0.7,
                yerr=cata_se,
                capsize=3,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"{dataset_name}")
            axes[idx].legend(fontsize=7)
            axes[idx].grid(True, alpha=0.3)

        axes[5].axis("off")
        axes[5].text(
            0.5,
            0.5,
            f"Error bars show\nStandard Error\n(n={len(all_experiment_data)} runs)",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[5].transAxes,
        )
        plt.suptitle(
            f"Aggregated Throughput Comparison by Dataset - Best Config: {best_config}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_dataset_throughput.png"),
            dpi=150,
        )
        plt.close()
        print("Created aggregated dataset throughput comparison")
    except Exception as e:
        print(f"Error creating aggregated dataset comparison plot: {e}")
        plt.close()

    # Plot 5: Aggregated Validation Loss vs Overall Improvement with Error Bars
    try:
        plt.figure(figsize=(10, 6))
        config_keys = list(all_experiment_data[0]["lr_batch_tuning"].keys())

        val_losses_all = []
        improvements_all = []
        for exp_data in all_experiment_data:
            val_losses = [
                exp_data["lr_batch_tuning"][k]["final_val_loss"] for k in config_keys
            ]
            improvements = [
                exp_data["lr_batch_tuning"][k]["overall_improvement"]
                for k in config_keys
            ]
            val_losses_all.append(val_losses)
            improvements_all.append(improvements)

        val_losses_all = np.array(val_losses_all)
        improvements_all = np.array(improvements_all)

        mean_val_loss = np.mean(val_losses_all, axis=0)
        se_val_loss = np.std(val_losses_all, axis=0) / np.sqrt(len(all_experiment_data))
        mean_improvements = np.mean(improvements_all, axis=0)
        se_improvements = np.std(improvements_all, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        colors = plt.cm.tab10(np.linspace(0, 1, len(config_keys)))

        for i, config in enumerate(config_keys):
            plt.errorbar(
                mean_val_loss[i],
                mean_improvements[i],
                xerr=se_val_loss[i],
                yerr=se_improvements[i],
                fmt="o",
                color=colors[i],
                markersize=10,
                capsize=5,
                elinewidth=2,
                capthick=2,
                label=config if i < 5 else None,
            )
            plt.annotate(
                config,
                (mean_val_loss[i], mean_improvements[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

        plt.xlabel("Final Validation Loss (Mean ± SE)")
        plt.ylabel("Overall Throughput Improvement (%) (Mean ± SE)")
        plt.title(
            "Aggregated Validation Loss vs Throughput Improvement\nWarehouse Task Assignment (n={} runs)".format(
                len(all_experiment_data)
            )
        )
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right", fontsize=7, title="Configurations")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_val_loss_vs_improvement.png"
            ),
            dpi=150,
        )
        plt.close()
        print("Created aggregated val loss vs improvement plot")
    except Exception as e:
        print(f"Error creating aggregated val loss vs improvement plot: {e}")
        plt.close()

    print("All aggregated plots saved successfully.")
else:
    print("No experiment data available for plotting.")
