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
    # Plot 1: Validation Loss curves for all datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        dataset_names = ["synthetic", "traffic_grid", "maze_grid"]
        for idx, dataset_name in enumerate(dataset_names):
            ax = axes[idx]
            for config_name, config_data in experiment_data[dataset_name][
                "configs"
            ].items():
                ax.plot(config_data["losses"]["val"], label=config_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title(f"{dataset_name} - Validation Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Hyperparameter Tuning: Validation Loss Curves Across Datasets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_validation_loss_curves.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # Plot 2: Validation IoU curves for all datasets
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, dataset_name in enumerate(dataset_names):
            ax = axes[idx]
            for config_name, config_data in experiment_data[dataset_name][
                "configs"
            ].items():
                ax.plot(config_data["metrics"]["val_iou"], label=config_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation IoU")
            ax.set_title(f"{dataset_name} - Validation IoU")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Hyperparameter Tuning: Validation IoU Curves Across Datasets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_validation_iou_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating validation IoU plot: {e}")
        plt.close()

    # Plot 3: Final metrics comparison bar chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        configs = list(experiment_data["synthetic"]["configs"].keys())
        x = np.arange(len(configs))
        width = 0.25

        for i, metric in enumerate(["final_val_loss", "final_val_iou"]):
            ax = axes[i]
            for j, dataset_name in enumerate(dataset_names):
                values = [
                    experiment_data[dataset_name]["configs"][c][metric] for c in configs
                ]
                ax.bar(x + j * width, values, width, label=dataset_name)
            ax.set_xlabel("Configuration")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'Final {metric.replace("final_val_", "").upper()} Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle("Hyperparameter Tuning: Final Metrics Comparison")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_final_metrics_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics bar chart: {e}")
        plt.close()

    # Plot 4: Throughput comparison for synthetic dataset
    try:
        if (
            "best_throughput" in experiment_data["synthetic"]
            and experiment_data["synthetic"]["best_throughput"]
        ):
            throughput_data = experiment_data["synthetic"]["best_throughput"]
            agent_counts = sorted(throughput_data.keys())

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Throughput values
            ax = axes[0]
            dist_means = [throughput_data[n]["distance"]["mean"] for n in agent_counts]
            dist_stds = [throughput_data[n]["distance"]["std"] for n in agent_counts]
            cata_means = [throughput_data[n]["cata"]["mean"] for n in agent_counts]
            cata_stds = [throughput_data[n]["cata"]["std"] for n in agent_counts]

            x = np.arange(len(agent_counts))
            width = 0.35
            ax.bar(
                x - width / 2,
                dist_means,
                width,
                yerr=dist_stds,
                label="Distance-based",
                capsize=5,
            )
            ax.bar(
                x + width / 2,
                cata_means,
                width,
                yerr=cata_stds,
                label="CATA",
                capsize=5,
            )
            ax.set_xlabel("Number of Agents")
            ax.set_ylabel("Throughput (tasks/min)")
            ax.set_title("Throughput Comparison: Distance vs CATA")
            ax.set_xticks(x)
            ax.set_xticklabels(agent_counts)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # Improvement percentage
            ax2 = axes[1]
            improvements = [throughput_data[n]["improvement"] for n in agent_counts]
            colors = ["green" if imp > 0 else "red" for imp in improvements]
            ax2.bar(x, improvements, color=colors)
            ax2.set_xlabel("Number of Agents")
            ax2.set_ylabel("Improvement (%)")
            ax2.set_title("CATA Improvement over Distance-based")
            ax2.set_xticks(x)
            ax2.set_xticklabels(agent_counts)
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis="y")

            plt.suptitle("Synthetic Dataset: Task Assignment Throughput Evaluation")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "synthetic_throughput_comparison.png"),
                dpi=150,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating throughput plot: {e}")
        plt.close()

    # Plot 5: Training vs Validation Loss for best config per dataset
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, dataset_name in enumerate(dataset_names):
            ax = axes[idx]
            best_config = min(
                experiment_data[dataset_name]["configs"].items(),
                key=lambda x: x[1]["final_val_loss"],
            )
            config_name, config_data = best_config
            ax.plot(config_data["losses"]["train"], label="Train Loss")
            ax.plot(config_data["losses"]["val"], label="Val Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{dataset_name}\nBest Config: {config_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle("Training vs Validation Loss for Best Configuration per Dataset")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_best_config_train_val_loss.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating train/val comparison plot: {e}")
        plt.close()

print("All plots saved successfully.")
