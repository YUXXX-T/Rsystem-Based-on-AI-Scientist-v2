import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_05a4fd05aedc45c291f8e5822613a3ba_proc_2789/experiment_data.npy",
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_0a6f24c413d54e3395b2bbd6096561b3_proc_2788/experiment_data.npy",
    "experiments/2026-04-05_11-32-59_conflict_memory_allocation_attempt_0/logs/0-run/experiment_results/experiment_a7e3a483043b4c778e26b237791aca3b_proc_2790/experiment_data.npy",
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
    dataset_names = [
        "sparse_warehouse",
        "bottleneck_warehouse",
        "dense_obstacle_warehouse",
    ]
    epoch_values = list(all_experiment_data[0]["epoch_tuning"].keys())
    colors_epochs = {30: "blue", 60: "orange"}

    # Plot 1: Aggregated Training/Validation Loss Curves with Standard Error
    try:
        plt.figure(figsize=(12, 6))
        for epochs in epoch_values:
            train_losses_all = [
                exp["epoch_tuning"][epochs]["training"]["losses"]["train"]
                for exp in all_experiment_data
            ]
            val_losses_all = [
                exp["epoch_tuning"][epochs]["training"]["losses"]["val"]
                for exp in all_experiment_data
            ]

            min_len = min(len(tl) for tl in train_losses_all)
            train_losses_all = np.array([tl[:min_len] for tl in train_losses_all])
            val_losses_all = np.array([vl[:min_len] for vl in val_losses_all])

            train_mean = np.mean(train_losses_all, axis=0)
            train_se = np.std(train_losses_all, axis=0) / np.sqrt(
                len(all_experiment_data)
            )
            val_mean = np.mean(val_losses_all, axis=0)
            val_se = np.std(val_losses_all, axis=0) / np.sqrt(len(all_experiment_data))

            x = np.arange(min_len)
            plt.plot(
                x,
                train_mean,
                label=f"Train Loss Mean (epochs={epochs})",
                color=colors_epochs[epochs],
                alpha=0.8,
            )
            plt.fill_between(
                x,
                train_mean - train_se,
                train_mean + train_se,
                color=colors_epochs[epochs],
                alpha=0.2,
            )
            plt.plot(
                x,
                val_mean,
                "--",
                label=f"Val Loss Mean (epochs={epochs})",
                color=colors_epochs[epochs],
                alpha=0.8,
            )
            plt.fill_between(
                x,
                val_mean - val_se,
                val_mean + val_se,
                color=colors_epochs[epochs],
                alpha=0.1,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(
            "Aggregated Training Curves (Mean ± SE across runs)\nWarehouse Task Assignment - Conflict Memory Allocation"
        )
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_training_curves.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated training curves plot: {e}")
        plt.close()

    # Plot 2: Aggregated Overall Improvement by Epoch Count with Error Bars
    try:
        plt.figure(figsize=(8, 6))
        overall_improvements_all = {
            e: [
                exp["epoch_tuning"][e]["overall_improvement"]
                for exp in all_experiment_data
            ]
            for e in epoch_values
        }
        means = [np.mean(overall_improvements_all[e]) for e in epoch_values]
        stderrs = [
            np.std(overall_improvements_all[e]) / np.sqrt(len(all_experiment_data))
            for e in epoch_values
        ]

        bars = plt.bar(
            range(len(epoch_values)),
            means,
            color=[colors_epochs[e] for e in epoch_values],
            yerr=stderrs,
            capsize=5,
        )
        plt.xticks(range(len(epoch_values)), [str(e) for e in epoch_values])
        plt.xlabel("Number of Training Epochs")
        plt.ylabel("Overall Throughput Improvement (%)")
        plt.title(
            "Aggregated Epoch Tuning: Overall Improvement (Mean ± SE)\nCongestion-Aware vs Distance-Based Assignment"
        )
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        for i, (m, s) in enumerate(zip(means, stderrs)):
            plt.text(i, m + s + 0.5, f"{m:.2f}±{s:.2f}%", ha="center", fontsize=9)
        plt.legend(["Zero reference", "Mean ± SE"], loc="upper right")
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_overall_improvement.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated overall improvement plot: {e}")
        plt.close()

    # Plot 3: Aggregated Per-Dataset Improvement Comparison with Error Bars
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dataset_names))
        width = 0.35

        for i, epochs in enumerate(epoch_values):
            avg_imps_all = []
            for d in dataset_names:
                imps = [
                    np.mean(exp["epoch_tuning"][epochs]["datasets"][d]["improvement"])
                    for exp in all_experiment_data
                ]
                avg_imps_all.append(imps)

            means = [np.mean(imps) for imps in avg_imps_all]
            stderrs = [
                np.std(imps) / np.sqrt(len(all_experiment_data))
                for imps in avg_imps_all
            ]

            ax.bar(
                x + i * width,
                means,
                width,
                label=f"Epochs={epochs} (Mean ± SE)",
                color=colors_epochs[epochs],
                yerr=stderrs,
                capsize=4,
            )

        ax.set_xticks(x + width * 0.5)
        ax.set_xticklabels(["Sparse", "Bottleneck", "Dense Obstacle"])
        ax.set_ylabel("Average Throughput Improvement (%)")
        ax.set_title(
            "Aggregated Throughput Improvement by Dataset (Mean ± SE)\nWarehouse Task Assignment - Conflict Memory Allocation"
        )
        ax.legend()
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_per_dataset_improvement.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated per-dataset improvement plot: {e}")
        plt.close()

    # Plot 4: Aggregated Throughput Comparison
    try:
        best_epochs = all_experiment_data[0]["best_epochs"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, dataset_name in enumerate(dataset_names):
            n_agents = all_experiment_data[0]["epoch_tuning"][best_epochs]["datasets"][
                dataset_name
            ]["n_agents"]
            ax = axes[idx]

            ca_throughputs = np.array(
                [
                    exp["epoch_tuning"][best_epochs]["datasets"][dataset_name][
                        "congestion_aware"
                    ]["throughput"]
                    for exp in all_experiment_data
                ]
            )
            db_throughputs = np.array(
                [
                    exp["epoch_tuning"][best_epochs]["datasets"][dataset_name][
                        "distance_based"
                    ]["throughput"]
                    for exp in all_experiment_data
                ]
            )

            ca_mean = np.mean(ca_throughputs, axis=0)
            ca_se = np.std(ca_throughputs, axis=0) / np.sqrt(len(all_experiment_data))
            db_mean = np.mean(db_throughputs, axis=0)
            db_se = np.std(db_throughputs, axis=0) / np.sqrt(len(all_experiment_data))

            ax.errorbar(
                n_agents,
                ca_mean,
                yerr=ca_se,
                fmt="o-",
                label="Congestion-Aware (Mean ± SE)",
                color="green",
                capsize=3,
            )
            ax.errorbar(
                n_agents,
                db_mean,
                yerr=db_se,
                fmt="s--",
                label="Distance-Based (Mean ± SE)",
                color="red",
                capsize=3,
            )
            ax.set_xlabel("Number of Agents")
            ax.set_ylabel("Throughput (tasks/min)")
            ax.set_title(f'{dataset_name.replace("_", " ").title()}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Aggregated Throughput Comparison (epochs={best_epochs}, n={len(all_experiment_data)} runs)\nLeft: Sparse, Middle: Bottleneck, Right: Dense",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_aggregated_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated throughput comparison plot: {e}")
        plt.close()

    # Plot 5: Aggregated Congestion Prevention Rate with Error Bars
    try:
        best_epochs = all_experiment_data[0]["best_epochs"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(dataset_names))
        width = 0.25
        agent_counts = all_experiment_data[0]["epoch_tuning"][best_epochs]["datasets"][
            dataset_names[0]
        ]["n_agents"]
        colors_agents = {20: "skyblue", 30: "steelblue", 40: "navy"}

        for i, n_agents in enumerate(agent_counts):
            cpr_all = []
            for d in dataset_names:
                cprs = [
                    exp["epoch_tuning"][best_epochs]["datasets"][d][
                        "congestion_prevention_rate"
                    ][i]
                    for exp in all_experiment_data
                ]
                cpr_all.append(cprs)

            means = [np.mean(cprs) for cprs in cpr_all]
            stderrs = [
                np.std(cprs) / np.sqrt(len(all_experiment_data)) for cprs in cpr_all
            ]

            ax.bar(
                x + i * width,
                means,
                width,
                label=f"{n_agents} Agents (Mean ± SE)",
                color=colors_agents[n_agents],
                yerr=stderrs,
                capsize=3,
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Sparse", "Bottleneck", "Dense Obstacle"])
        ax.set_ylabel("Congestion Prevention Rate (%)")
        ax.set_title(
            f"Aggregated Congestion Prevention Rate (epochs={best_epochs}, n={len(all_experiment_data)} runs)\nWarehouse Task Assignment - Conflict Memory Allocation"
        )
        ax.legend()
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.savefig(
            os.path.join(
                working_dir, "warehouse_aggregated_congestion_prevention_rate.png"
            ),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated congestion prevention rate plot: {e}")
        plt.close()

    # Print aggregated metrics
    print("\n=== Aggregated Results Summary ===")
    best_epochs_list = [exp["best_epochs"] for exp in all_experiment_data]
    best_improvements = [exp["best_improvement"] for exp in all_experiment_data]
    print(f"Best epochs across runs: {best_epochs_list}")
    print(
        f"Best improvement - Mean: {np.mean(best_improvements):.2f}%, SE: {np.std(best_improvements)/np.sqrt(len(all_experiment_data)):.2f}%"
    )

    for epochs in epoch_values:
        overall_imps = [
            exp["epoch_tuning"][epochs]["overall_improvement"]
            for exp in all_experiment_data
        ]
        print(
            f"Epochs={epochs} - Overall Improvement Mean: {np.mean(overall_imps):.2f}%, SE: {np.std(overall_imps)/np.sqrt(len(all_experiment_data)):.2f}%"
        )
