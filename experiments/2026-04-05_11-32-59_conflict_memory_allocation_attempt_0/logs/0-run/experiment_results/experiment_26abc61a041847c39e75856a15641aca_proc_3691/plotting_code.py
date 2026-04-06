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
            epochs = range(len(data["train"]))
            axes[0].plot(
                epochs, data["train"], label=config_key, color=colors[idx], alpha=0.8
            )
            axes[1].plot(
                epochs, data["val"], label=config_key, color=colors[idx], alpha=0.8
            )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curves for All Configurations")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss Curves for All Configurations")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            "LR & Batch Size Tuning - Loss Curves (HF Warehouse Datasets)", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_lr_batch_loss_curves.png"), dpi=150
        )
        plt.close()
        print("Created: warehouse_lr_batch_loss_curves.png")
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # Plot 2: Overall Improvement and CPR Bar Chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        config_keys = list(experiment_data["lr_batch_tuning"].keys())
        x = np.arange(len(config_keys))

        improvements = [
            experiment_data["lr_batch_tuning"][k]["overall_improvement"]
            for k in config_keys
        ]
        cprs = [
            experiment_data["lr_batch_tuning"][k]["overall_cpr"] for k in config_keys
        ]

        colors_imp = ["green" if imp > 0 else "red" for imp in improvements]
        colors_cpr = ["green" if c > 0 else "red" for c in cprs]

        bars1 = axes[0].bar(
            x, improvements, color=colors_imp, alpha=0.7, edgecolor="black"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(config_keys, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("Throughput Improvement (%)")
        axes[0].set_title("Overall Throughput Improvement by Config")
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[0].grid(True, alpha=0.3, axis="y")

        best_config = experiment_data["best_config"]
        if best_config in config_keys:
            best_idx = config_keys.index(best_config)
            bars1[best_idx].set_edgecolor("gold")
            bars1[best_idx].set_linewidth(3)

        bars2 = axes[1].bar(x, cprs, color=colors_cpr, alpha=0.7, edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(config_keys, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("Conflict Prevention Rate (%)")
        axes[1].set_title("Overall CPR by Config")
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[1].grid(True, alpha=0.3, axis="y")

        if best_config in config_keys:
            bars2[best_idx].set_edgecolor("gold")
            bars2[best_idx].set_linewidth(3)

        plt.suptitle(f"Overall Metrics Comparison (Best: {best_config})", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_overall_metrics_by_config.png"),
            dpi=150,
        )
        plt.close()
        print("Created: warehouse_overall_metrics_by_config.png")
    except Exception as e:
        print(f"Error creating overall metrics plot: {e}")
        plt.close()

    # Plot 3: Dataset-specific Improvement by Agent Count (Best Config)
    try:
        best_config = experiment_data["best_config"]
        datasets = list(
            experiment_data["lr_batch_tuning"][best_config]["datasets"].keys()
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        markers = ["o", "s", "^"]
        colors = ["blue", "green", "orange"]

        for idx, dataset_name in enumerate(datasets):
            data = experiment_data["lr_batch_tuning"][best_config]["datasets"][
                dataset_name
            ]
            n_agents = data["n_agents"]
            improvements = data["improvement"]

            axes[idx].plot(
                n_agents,
                improvements,
                marker=markers[idx],
                color=colors[idx],
                linewidth=2,
                markersize=10,
                label=dataset_name,
            )
            axes[idx].fill_between(
                n_agents, 0, improvements, alpha=0.3, color=colors[idx]
            )
            axes[idx].axhline(y=0, color="red", linestyle="--", linewidth=1)
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput Improvement (%)")
            axes[idx].set_title(f"Dataset: {dataset_name}")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(n_agents)

        plt.suptitle(
            f"Throughput Improvement by Agent Count - Best Config: {best_config}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_dataset_improvement_by_agents.png"),
            dpi=150,
        )
        plt.close()
        print("Created: warehouse_dataset_improvement_by_agents.png")
    except Exception as e:
        print(f"Error creating dataset improvement plot: {e}")
        plt.close()

    # Plot 4: CPR Comparison Across All Datasets (Best Config)
    try:
        best_config = experiment_data["best_config"]
        datasets = list(
            experiment_data["lr_batch_tuning"][best_config]["datasets"].keys()
        )

        plt.figure(figsize=(10, 6))
        markers = ["o-", "s-", "^-"]
        colors = ["blue", "green", "orange"]

        for idx, dataset_name in enumerate(datasets):
            data = experiment_data["lr_batch_tuning"][best_config]["datasets"][
                dataset_name
            ]
            n_agents = data["n_agents"]
            cprs = data["cpr"]
            plt.plot(
                n_agents,
                cprs,
                markers[idx],
                color=colors[idx],
                linewidth=2,
                markersize=10,
                label=dataset_name,
            )

        plt.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Baseline")
        plt.xlabel("Number of Agents")
        plt.ylabel("Conflict Prevention Rate (%)")
        plt.title(
            f"CPR Comparison Across Warehouse Layouts\nBest Config: {best_config}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_cpr_comparison_datasets.png"), dpi=150
        )
        plt.close()
        print("Created: warehouse_cpr_comparison_datasets.png")
    except Exception as e:
        print(f"Error creating CPR comparison plot: {e}")
        plt.close()

    # Plot 5: Heatmap of Improvements (LR vs Batch Size)
    try:
        lr_values = [0.0005, 0.001]
        batch_sizes = [32, 64]
        improvement_matrix = np.zeros((len(lr_values), len(batch_sizes)))
        cpr_matrix = np.zeros((len(lr_values), len(batch_sizes)))

        for i, lr in enumerate(lr_values):
            for j, bs in enumerate(batch_sizes):
                config_key = f"lr{lr}_bs{bs}"
                if config_key in experiment_data["lr_batch_tuning"]:
                    improvement_matrix[i, j] = experiment_data["lr_batch_tuning"][
                        config_key
                    ]["overall_improvement"]
                    cpr_matrix[i, j] = experiment_data["lr_batch_tuning"][config_key][
                        "overall_cpr"
                    ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im1, ax=axes[0], label="Improvement (%)")
        axes[0].set_xticks(range(len(batch_sizes)))
        axes[0].set_xticklabels(batch_sizes)
        axes[0].set_yticks(range(len(lr_values)))
        axes[0].set_yticklabels(lr_values)
        axes[0].set_xlabel("Batch Size")
        axes[0].set_ylabel("Learning Rate")
        axes[0].set_title("Throughput Improvement Heatmap")
        for i in range(len(lr_values)):
            for j in range(len(batch_sizes)):
                axes[0].text(
                    j,
                    i,
                    f"{improvement_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        im2 = axes[1].imshow(cpr_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im2, ax=axes[1], label="CPR (%)")
        axes[1].set_xticks(range(len(batch_sizes)))
        axes[1].set_xticklabels(batch_sizes)
        axes[1].set_yticks(range(len(lr_values)))
        axes[1].set_yticklabels(lr_values)
        axes[1].set_xlabel("Batch Size")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("CPR Heatmap")
        for i in range(len(lr_values)):
            for j in range(len(batch_sizes)):
                axes[1].text(
                    j,
                    i,
                    f"{cpr_matrix[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        plt.suptitle("Hyperparameter Grid Search - HF Warehouse Datasets", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_lr_batch_heatmap.png"), dpi=150
        )
        plt.close()
        print("Created: warehouse_lr_batch_heatmap.png")
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

    # Plot 6: Final Validation Loss vs Overall Improvement
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
        cprs = [
            experiment_data["lr_batch_tuning"][k]["overall_cpr"] for k in config_keys
        ]

        scatter = plt.scatter(
            val_losses,
            improvements,
            c=cprs,
            cmap="coolwarm",
            s=200,
            edgecolors="black",
            linewidth=2,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("CPR (%)")

        for i, config in enumerate(config_keys):
            plt.annotate(
                config,
                (val_losses[i], improvements[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                fontweight="bold",
            )

        plt.xlabel("Final Validation Loss")
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.title(
            "Validation Loss vs Improvement (colored by CPR)\nHF Warehouse Datasets - Hyperparameter Tuning"
        )
        plt.axhline(
            y=0, color="red", linestyle="--", alpha=0.5, label="Zero improvement"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_val_loss_vs_improvement.png"), dpi=150
        )
        plt.close()
        print("Created: warehouse_val_loss_vs_improvement.png")
    except Exception as e:
        print(f"Error creating val loss vs improvement plot: {e}")
        plt.close()

    print(f"\nBest config: {experiment_data['best_config']}")
    print(f"Best improvement: {experiment_data['best_improvement']:.2f}%")
    print(f"Best CPR: {experiment_data['best_cpr']:.2f}%")
    print("\nAll plots saved successfully.")
