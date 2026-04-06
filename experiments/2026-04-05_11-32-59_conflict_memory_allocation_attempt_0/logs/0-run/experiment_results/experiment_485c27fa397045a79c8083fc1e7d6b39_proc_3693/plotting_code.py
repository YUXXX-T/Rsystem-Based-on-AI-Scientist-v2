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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        config_keys = list(experiment_data["lr_batch_tuning"].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(config_keys)))

        for idx, config_key in enumerate(config_keys):
            data = experiment_data["lr_batch_tuning"][config_key]["training"]["losses"]
            epochs = experiment_data["lr_batch_tuning"][config_key]["training"][
                "epochs"
            ]
            axes[0].plot(
                epochs, data["train"], label=config_key, color=colors[idx], alpha=0.8
            )
            axes[1].plot(
                epochs, data["val"], label=config_key, color=colors[idx], alpha=0.8
            )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curves for All Configurations")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss Curves for All Configurations")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("LR & Batch Size Tuning - Loss Curves", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_lr_batch_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # Plot 2: Overall Improvement Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        config_keys = list(experiment_data["lr_batch_tuning"].keys())
        overall_improvements = [
            experiment_data["lr_batch_tuning"][k]["overall_improvement"]
            for k in config_keys
        ]
        colors = ["green" if imp > 0 else "red" for imp in overall_improvements]

        bars = plt.bar(
            range(len(config_keys)),
            overall_improvements,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        plt.xticks(
            range(len(config_keys)), config_keys, rotation=45, ha="right", fontsize=9
        )
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.xlabel("Configuration (Learning Rate & Batch Size)")
        plt.title(
            "Overall Improvement by Configuration\nWarehouse Task Assignment - LR & Batch Size Tuning"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        best_config = experiment_data["best_config"]
        best_idx = config_keys.index(best_config)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        plt.annotate(
            f"Best: {best_config}",
            xy=(best_idx, overall_improvements[best_idx]),
            xytext=(best_idx, overall_improvements[best_idx] + 2),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_overall_improvement_by_config.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating overall improvement plot: {e}")
        plt.close()

    # Plot 3: Heatmap of Improvements (LR vs Batch Size)
    try:
        lr_values = [0.0005, 0.001, 0.002]
        batch_sizes = [16, 32, 64]
        improvement_matrix = np.zeros((len(lr_values), len(batch_sizes)))

        for i, lr in enumerate(lr_values):
            for j, bs in enumerate(batch_sizes):
                config_key = f"lr{lr}_bs{bs}"
                if config_key in experiment_data["lr_batch_tuning"]:
                    improvement_matrix[i, j] = experiment_data["lr_batch_tuning"][
                        config_key
                    ]["overall_improvement"]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Overall Improvement (%)")
        plt.xticks(range(len(batch_sizes)), batch_sizes)
        plt.yticks(range(len(lr_values)), lr_values)
        plt.xlabel("Batch Size")
        plt.ylabel("Learning Rate")
        plt.title(
            "Hyperparameter Grid Search Results\nWarehouse Congestion-Aware Task Assignment"
        )

        for i in range(len(lr_values)):
            for j in range(len(batch_sizes)):
                plt.text(
                    j,
                    i,
                    f"{improvement_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_lr_batch_heatmap.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

    # Plot 4: Dataset-specific Improvements for Best Config
    try:
        best_config = experiment_data["best_config"]
        datasets = list(
            experiment_data["lr_batch_tuning"][best_config]["datasets"].keys()
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, dataset_name in enumerate(datasets[:5]):
            data = experiment_data["lr_batch_tuning"][best_config]["datasets"][
                dataset_name
            ]
            n_agents = data["n_agents"]
            dist_tp = data["distance_based"]["throughput"]
            cata_tp = data["congestion_aware"]["throughput"]

            x = np.arange(len(n_agents))
            width = 0.35
            axes[idx].bar(
                x - width / 2,
                dist_tp,
                width,
                label="Distance-Based",
                color="blue",
                alpha=0.7,
            )
            axes[idx].bar(
                x + width / 2,
                cata_tp,
                width,
                label="Congestion-Aware",
                color="green",
                alpha=0.7,
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(n_agents)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"{dataset_name}")
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        axes[5].axis("off")
        plt.suptitle(
            f"Throughput Comparison by Dataset - Best Config: {best_config}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_dataset_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating dataset comparison plot: {e}")
        plt.close()

    # Plot 5: Final Validation Loss vs Overall Improvement
    try:
        plt.figure(figsize=(10, 6))
        config_keys = list(experiment_data["lr_batch_tuning"].keys())
        val_losses = [
            experiment_data["lr_batch_tuning"][k]["final_val_loss"] for k in config_keys
        ]
        improvements = [
            experiment_data["lr_batch_tuning"][k]["overall_improvement"]
            for k in config_keys
        ]

        scatter = plt.scatter(
            val_losses,
            improvements,
            c=range(len(config_keys)),
            cmap="tab10",
            s=150,
            edgecolors="black",
        )
        for i, config in enumerate(config_keys):
            plt.annotate(
                config,
                (val_losses[i], improvements[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        plt.xlabel("Final Validation Loss")
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.title(
            "Validation Loss vs Throughput Improvement\nWarehouse Task Assignment - Hyperparameter Tuning"
        )
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_val_loss_vs_improvement.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating val loss vs improvement plot: {e}")
        plt.close()

print("All plots saved successfully.")
