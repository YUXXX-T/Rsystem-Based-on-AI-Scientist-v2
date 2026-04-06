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
    methods = list(experiment_data["methods"].keys())
    datasets = list(experiment_data["methods"][methods[0]]["datasets"].keys())

    # Plot 1: Training and Validation Loss Curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for idx, method in enumerate(methods):
            train_losses = experiment_data["methods"][method]["train_losses"]
            val_losses = experiment_data["methods"][method]["val_losses"]
            epochs = range(len(train_losses))
            axes[0].plot(
                epochs, train_losses, label=method, color=colors[idx], alpha=0.8
            )
            axes[1].plot(epochs, val_losses, label=method, color=colors[idx], alpha=0.8)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            "Congestion Predictor Training - All Methods Comparison", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_training_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # Plot 2: Overall Method Performance Comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        overall_imp = [
            experiment_data["methods"][m]["overall_improvement"] for m in methods
        ]
        overall_cpr = [experiment_data["methods"][m]["overall_cpr"] for m in methods]

        colors = ["green" if imp > 0 else "red" for imp in overall_imp]
        bars1 = axes[0].bar(
            methods, overall_imp, color=colors, alpha=0.7, edgecolor="black"
        )
        axes[0].set_ylabel("Overall Throughput Improvement (%)")
        axes[0].set_title("Throughput Improvement by Method")
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)

        best_method = experiment_data["best_method"]
        best_idx = methods.index(best_method)
        bars1[best_idx].set_edgecolor("gold")
        bars1[best_idx].set_linewidth(3)

        colors2 = ["blue" if cpr > 0 else "red" for cpr in overall_cpr]
        axes[1].bar(methods, overall_cpr, color=colors2, alpha=0.7, edgecolor="black")
        axes[1].set_ylabel("Conflict Prevention Rate (%)")
        axes[1].set_title("Conflict Prevention Rate by Method")
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)

        plt.suptitle(
            f"Overall Performance Comparison\nBest Method: {best_method}", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_overall_method_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating overall comparison plot: {e}")
        plt.close()

    # Plot 3: Dataset-specific Improvement Heatmap
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        agent_counts = [20, 30, 40]

        imp_matrix = np.zeros((len(methods), len(datasets)))
        cpr_matrix = np.zeros((len(methods), len(datasets)))

        for i, method in enumerate(methods):
            for j, ds in enumerate(datasets):
                imp_matrix[i, j] = np.mean(
                    experiment_data["methods"][method]["datasets"][ds]["improvement"]
                )
                cpr_matrix[i, j] = np.mean(
                    experiment_data["methods"][method]["datasets"][ds]["cpr"]
                )

        im1 = axes[0].imshow(imp_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im1, ax=axes[0], label="Improvement (%)")
        axes[0].set_xticks(range(len(datasets)))
        axes[0].set_xticklabels([d[:12] for d in datasets], rotation=45, ha="right")
        axes[0].set_yticks(range(len(methods)))
        axes[0].set_yticklabels(methods)
        axes[0].set_title("Throughput Improvement by Method & Dataset")
        for i in range(len(methods)):
            for j in range(len(datasets)):
                axes[0].text(
                    j,
                    i,
                    f"{imp_matrix[i,j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        im2 = axes[1].imshow(cpr_matrix, cmap="RdYlBu", aspect="auto")
        plt.colorbar(im2, ax=axes[1], label="CPR (%)")
        axes[1].set_xticks(range(len(datasets)))
        axes[1].set_xticklabels([d[:12] for d in datasets], rotation=45, ha="right")
        axes[1].set_yticks(range(len(methods)))
        axes[1].set_yticklabels(methods)
        axes[1].set_title("Conflict Prevention Rate by Method & Dataset")
        for i in range(len(methods)):
            for j in range(len(datasets)):
                axes[1].text(
                    j,
                    i,
                    f"{cpr_matrix[i,j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.suptitle("Performance Heatmaps Across All Warehouse Types", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_performance_heatmap.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap plot: {e}")
        plt.close()

    # Plot 4: Throughput vs Agent Count for Each Dataset
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        agent_counts = [20, 30, 40]

        for idx, ds in enumerate(datasets[:5]):
            for method in methods:
                throughputs = experiment_data["methods"][method]["datasets"][ds][
                    "throughput"
                ]
                axes[idx].plot(
                    agent_counts,
                    throughputs,
                    "o-",
                    label=method,
                    linewidth=2,
                    markersize=6,
                )
            axes[idx].set_xlabel("Number of Agents")
            axes[idx].set_ylabel("Throughput (tasks/min)")
            axes[idx].set_title(f"{ds}")
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        axes[5].axis("off")
        plt.suptitle("Throughput Scaling with Agent Count by Dataset", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_vs_agents.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput vs agents plot: {e}")
        plt.close()

    # Plot 5: Improvement and CPR by Agent Count (Best Method)
    try:
        best_method = experiment_data["best_method"]
        agent_counts = [20, 30, 40]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ds in datasets:
            improvements = experiment_data["methods"][best_method]["datasets"][ds][
                "improvement"
            ]
            axes[0].plot(agent_counts, improvements, "o-", label=ds[:12], linewidth=2)
        axes[0].set_xlabel("Number of Agents")
        axes[0].set_ylabel("Throughput Improvement (%)")
        axes[0].set_title(f"Improvement vs Agent Count ({best_method})")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="black", linestyle="--")

        for ds in datasets:
            cprs = experiment_data["methods"][best_method]["datasets"][ds]["cpr"]
            axes[1].plot(agent_counts, cprs, "o-", label=ds[:12], linewidth=2)
        axes[1].set_xlabel("Number of Agents")
        axes[1].set_ylabel("Conflict Prevention Rate (%)")
        axes[1].set_title(f"CPR vs Agent Count ({best_method})")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linestyle="--")

        plt.suptitle(
            f"Best Method ({best_method}) Performance Across Datasets", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_best_method_scaling.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating best method scaling plot: {e}")
        plt.close()

print("All plots saved successfully.")
