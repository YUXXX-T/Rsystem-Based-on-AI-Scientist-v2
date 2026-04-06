import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load all experiment data
experiment_data_path_list = [
    "experiments/2026-04-05_10-02-44_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_725eadfdcd5b4194a6fce1bcd4540dfc_proc_1674/experiment_data.npy",
    "experiments/2026-04-05_10-02-44_congestion_heatmap_allocation_attempt_0/logs/0-run/experiment_results/experiment_5cc5f7fd332a45568cec484850fba223_proc_1675/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        if experiment_data_path and "None" not in experiment_data_path:
            full_path = os.path.join(
                os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
            )
            if os.path.exists(full_path):
                experiment_data = np.load(full_path, allow_pickle=True).item()
                all_experiment_data.append(experiment_data)
    print(f"Successfully loaded {len(all_experiment_data)} experiment data files")
except Exception as e:
    print(f"Error loading experiment data: {e}")

if len(all_experiment_data) > 0:
    # Plot 1: Aggregated Training and Validation Loss Curves with Standard Error
    try:
        plt.figure(figsize=(10, 6))
        all_train_losses = [
            exp["training"]["losses"]["train"] for exp in all_experiment_data
        ]
        all_val_losses = [
            exp["training"]["losses"]["val"] for exp in all_experiment_data
        ]
        epochs = all_experiment_data[0]["training"]["epochs"]

        train_mean = np.mean(all_train_losses, axis=0)
        train_se = np.std(all_train_losses, axis=0) / np.sqrt(len(all_train_losses))
        val_mean = np.mean(all_val_losses, axis=0)
        val_se = np.std(all_val_losses, axis=0) / np.sqrt(len(all_val_losses))

        plt.plot(epochs, train_mean, "b-", label="Train Loss (Mean)")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="blue",
            alpha=0.2,
            label="Train SE",
        )
        plt.plot(epochs, val_mean, "r-", label="Validation Loss (Mean)")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="red",
            alpha=0.2,
            label="Val SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            f"Aggregated Training Curves (n={len(all_experiment_data)} runs)\nWarehouse Task Assignment - Mean ± Standard Error"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "aggregated_training_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated training loss plot: {e}")
        plt.close()

    # Plot 2: Aggregated Throughput Comparison with Mean and SE
    try:
        plt.figure(figsize=(12, 7))
        colors = {
            "sparse_warehouse": "blue",
            "bottleneck_warehouse": "orange",
            "dense_obstacle_warehouse": "green",
        }
        dataset_names = list(all_experiment_data[0]["datasets"].keys())
        n_agents = all_experiment_data[0]["datasets"][dataset_names[0]]["n_agents"]

        for dataset_name in dataset_names:
            cata_throughputs = np.array(
                [
                    exp["datasets"][dataset_name]["congestion_aware"]["throughput"]
                    for exp in all_experiment_data
                ]
            )
            dist_throughputs = np.array(
                [
                    exp["datasets"][dataset_name]["distance_based"]["throughput"]
                    for exp in all_experiment_data
                ]
            )

            cata_mean = np.mean(cata_throughputs, axis=0)
            cata_se = np.std(cata_throughputs, axis=0) / np.sqrt(
                len(all_experiment_data)
            )
            dist_mean = np.mean(dist_throughputs, axis=0)
            dist_se = np.std(dist_throughputs, axis=0) / np.sqrt(
                len(all_experiment_data)
            )

            plt.errorbar(
                n_agents,
                cata_mean,
                yerr=cata_se,
                fmt="o-",
                color=colors[dataset_name],
                label=f"CATA-{dataset_name[:8]} (Mean±SE)",
                capsize=4,
                linewidth=2,
            )
            plt.errorbar(
                n_agents,
                dist_mean,
                yerr=dist_se,
                fmt="x--",
                color=colors[dataset_name],
                alpha=0.5,
                capsize=4,
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            f"Aggregated Throughput Comparison (n={len(all_experiment_data)} runs)\nCATA (solid) vs Distance-Based (dashed) - Mean ± SE"
        )
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "aggregated_throughput_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated throughput plot: {e}")
        plt.close()

    # Plot 3: Aggregated Improvement Percentage with Error Bars
    try:
        plt.figure(figsize=(12, 7))
        x = np.arange(len(n_agents))
        width = 0.25

        for i, dataset_name in enumerate(dataset_names):
            improvements = np.array(
                [
                    exp["datasets"][dataset_name]["improvement"]
                    for exp in all_experiment_data
                ]
            )
            imp_mean = np.mean(improvements, axis=0)
            imp_se = np.std(improvements, axis=0) / np.sqrt(len(all_experiment_data))

            plt.bar(
                x + i * width,
                imp_mean,
                width,
                yerr=imp_se,
                capsize=3,
                label=f'{dataset_name.replace("_", " ").title()} (Mean±SE)',
                color=colors[dataset_name],
            )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput Improvement (%)")
        plt.title(
            f"Aggregated CATA Improvement (n={len(all_experiment_data)} runs)\nMean ± Standard Error by Dataset"
        )
        plt.xticks(x + width, n_agents)
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "aggregated_improvement_percentage.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated improvement plot: {e}")
        plt.close()

    # Plot 4: Aggregated Summary with Mean and SE
    try:
        plt.figure(figsize=(10, 6))
        summary_imp_all = []
        summary_cpr_all = []

        for exp in all_experiment_data:
            summary_imp_all.append(
                [np.mean(exp["datasets"][d]["improvement"]) for d in dataset_names]
            )
            summary_cpr_all.append(
                [
                    np.mean(exp["datasets"][d]["congestion_prevention_rate"])
                    for d in dataset_names
                ]
            )

        summary_imp_mean = np.mean(summary_imp_all, axis=0)
        summary_imp_se = np.std(summary_imp_all, axis=0) / np.sqrt(
            len(all_experiment_data)
        )
        summary_cpr_mean = np.mean(summary_cpr_all, axis=0)
        summary_cpr_se = np.std(summary_cpr_all, axis=0) / np.sqrt(
            len(all_experiment_data)
        )

        x_sum = np.arange(len(dataset_names))
        plt.bar(
            x_sum - 0.2,
            summary_imp_mean,
            0.35,
            yerr=summary_imp_se,
            capsize=4,
            label="Avg Throughput Improvement (Mean±SE)",
            color="steelblue",
        )
        plt.bar(
            x_sum + 0.2,
            summary_cpr_mean,
            0.35,
            yerr=summary_cpr_se,
            capsize=4,
            label="Avg Congestion Prevention Rate (Mean±SE)",
            color="coral",
        )

        plt.xlabel("Warehouse Dataset")
        plt.ylabel("Percentage (%)")
        plt.title(
            f"Aggregated Summary (n={len(all_experiment_data)} runs)\nCATA Performance - Mean ± Standard Error"
        )
        plt.xticks(x_sum, ["Sparse", "Bottleneck", "Dense"])
        plt.legend()
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "aggregated_summary_improvement.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated summary plot: {e}")
        plt.close()

    print("All aggregated plots saved successfully to working directory.")
else:
    print("No valid experiment data loaded.")
