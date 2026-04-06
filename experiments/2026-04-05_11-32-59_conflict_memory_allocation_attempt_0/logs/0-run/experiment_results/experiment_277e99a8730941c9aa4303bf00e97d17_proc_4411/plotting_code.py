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

# Plot 1: Training/Validation Loss Curves Comparison
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    baseline_train = experiment_data["baseline_lookahead"]["training"]["losses"][
        "train"
    ]
    baseline_val = experiment_data["baseline_lookahead"]["training"]["losses"]["val"]
    ablation_train = experiment_data["ablation_no_lookahead"]["training"]["losses"][
        "train"
    ]
    ablation_val = experiment_data["ablation_no_lookahead"]["training"]["losses"]["val"]
    epochs = range(len(baseline_train))

    axes[0].plot(epochs, baseline_train, "b-", label="Train")
    axes[0].plot(epochs, baseline_val, "b--", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Baseline (Lookahead=5)")
    axes[0].legend()

    axes[1].plot(epochs, ablation_train, "r-", label="Train")
    axes[1].plot(epochs, ablation_val, "r--", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Ablation (No Lookahead)")
    axes[1].legend()

    fig.suptitle(
        "Training Curves: Baseline vs No Lookahead Ablation\nWarehouse Congestion Prediction"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "ablation_training_curves.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating training curves plot: {e}")
    plt.close()

# Plot 2: Overall Improvement Comparison
try:
    plt.figure(figsize=(8, 6))
    baseline_imp = experiment_data["comparison"]["baseline_improvement"]
    ablation_imp = experiment_data["comparison"]["ablation_improvement"]
    bars = plt.bar(
        ["Baseline\n(Lookahead=5)", "Ablation\n(No Lookahead)"],
        [baseline_imp, ablation_imp],
        color=["blue", "red"],
        width=0.5,
    )
    plt.ylabel("Overall Improvement (%)")
    plt.title(
        "Ablation Study: Overall Throughput Improvement\nCongestion-Aware vs Distance-Based Assignment"
    )
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    for bar, val in zip(bars, [baseline_imp, ablation_imp]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "ablation_overall_improvement.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating overall improvement plot: {e}")
    plt.close()

# Plot 3: Per-Dataset Improvement Comparison
try:
    dataset_names = list(experiment_data["baseline_lookahead"]["datasets"].keys())
    baseline_imps = [
        np.mean(experiment_data["baseline_lookahead"]["datasets"][d]["improvement"])
        for d in dataset_names
    ]
    ablation_imps = [
        np.mean(experiment_data["ablation_no_lookahead"]["datasets"][d]["improvement"])
        for d in dataset_names
    ]

    x = np.arange(len(dataset_names))
    width = 0.35
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(
        x - width / 2,
        baseline_imps,
        width,
        label="Baseline (Lookahead=5)",
        color="blue",
    )
    bars2 = plt.bar(
        x + width / 2,
        ablation_imps,
        width,
        label="Ablation (No Lookahead)",
        color="red",
    )
    plt.xlabel("Dataset")
    plt.ylabel("Average Improvement (%)")
    plt.title(
        "Per-Dataset Improvement: Baseline vs No Lookahead Ablation\nCongestion-Aware Task Assignment"
    )
    plt.xticks(x, [d.replace("_", "\n") for d in dataset_names], fontsize=9)
    plt.legend()
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "ablation_per_dataset_improvement.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating per-dataset improvement plot: {e}")
    plt.close()

# Plot 4: Throughput by Agent Count for Selected Datasets
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    dataset_names = list(experiment_data["baseline_lookahead"]["datasets"].keys())
    agent_counts = experiment_data["baseline_lookahead"]["datasets"][dataset_names[0]][
        "n_agents"
    ]

    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[idx // 3, idx % 3]
        baseline_data = experiment_data["baseline_lookahead"]["datasets"][dataset_name]
        ablation_data = experiment_data["ablation_no_lookahead"]["datasets"][
            dataset_name
        ]

        ax.plot(
            agent_counts,
            baseline_data["congestion_aware"]["throughput"],
            "o-",
            label="Baseline CATA",
            color="blue",
        )
        ax.plot(
            agent_counts,
            ablation_data["congestion_aware"]["throughput"],
            "s-",
            label="No Lookahead CATA",
            color="red",
        )
        ax.plot(
            agent_counts,
            baseline_data["distance_based"]["throughput"],
            "x--",
            label="Distance-Based",
            color="gray",
        )
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Throughput (tasks/min)")
        ax.set_title(dataset_name.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Throughput Scaling by Agent Count: Ablation Study\nComparing Lookahead vs No Lookahead Congestion Prediction",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "ablation_throughput_by_agents.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating throughput by agents plot: {e}")
    plt.close()

# Plot 5: Lookahead Contribution Analysis
try:
    plt.figure(figsize=(10, 6))
    dataset_names = list(experiment_data["baseline_lookahead"]["datasets"].keys())
    contributions = [
        np.mean(experiment_data["baseline_lookahead"]["datasets"][d]["improvement"])
        - np.mean(
            experiment_data["ablation_no_lookahead"]["datasets"][d]["improvement"]
        )
        for d in dataset_names
    ]
    colors = ["green" if c > 0 else "orange" for c in contributions]
    bars = plt.bar(range(len(dataset_names)), contributions, color=colors)
    plt.xticks(
        range(len(dataset_names)),
        [d.replace("_", "\n") for d in dataset_names],
        fontsize=9,
    )
    plt.ylabel("Lookahead Contribution (%)")
    plt.title(
        "Lookahead Value Contribution by Dataset\nDifference: Baseline Improvement - Ablation Improvement"
    )
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    overall_contrib = experiment_data["comparison"]["lookahead_contribution"]
    plt.axhline(
        y=overall_contrib,
        color="purple",
        linestyle=":",
        label=f"Overall: {overall_contrib:.2f}%",
    )
    plt.legend()
    for bar, val in zip(bars, contributions):
        ypos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.5
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "ablation_lookahead_contribution.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating lookahead contribution plot: {e}")
    plt.close()

print("All plots saved successfully.")
