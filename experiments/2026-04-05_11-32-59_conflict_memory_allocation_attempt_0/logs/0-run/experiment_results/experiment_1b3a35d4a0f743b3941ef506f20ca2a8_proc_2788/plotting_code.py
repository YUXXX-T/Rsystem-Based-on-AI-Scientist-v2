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

# Plot 1: Training and Validation Loss Curves
try:
    plt.figure(figsize=(8, 5))
    epochs = experiment_data["training"]["epochs"]
    train_loss = experiment_data["training"]["losses"]["train"]
    val_loss = experiment_data["training"]["losses"]["val"]
    plt.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Congestion Predictor Training Curves\nWarehouse Robot Task Assignment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "warehouse_training_loss_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot 2: Throughput Comparison by Warehouse Type
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, name in enumerate(experiment_data["evaluation"].keys()):
        data = experiment_data["evaluation"][name]
        n_agents = data["n_agents"]
        axes[idx].plot(
            n_agents,
            data["cma"],
            "go-",
            label="Congestion-Aware (CMA)",
            linewidth=2,
            markersize=8,
        )
        axes[idx].plot(
            n_agents,
            data["distance"],
            "bx--",
            label="Distance-Based",
            linewidth=2,
            markersize=8,
        )
        axes[idx].set_xlabel("Number of Agents")
        axes[idx].set_ylabel("Throughput (tasks/min)")
        axes[idx].set_title(f"{name.capitalize()} Warehouse")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle(
        "Throughput Comparison: Distance-Based vs Congestion-Aware Assignment",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "warehouse_throughput_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating throughput comparison plot: {e}")
    plt.close()

# Plot 3: Throughput Improvement Percentage
try:
    plt.figure(figsize=(8, 5))
    for name in experiment_data["evaluation"].keys():
        data = experiment_data["evaluation"][name]
        plt.plot(
            data["n_agents"],
            data["improvement"],
            "o-",
            label=f"{name.capitalize()} Warehouse",
            linewidth=2,
            markersize=8,
        )
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.7, label="No Improvement")
    overall_improvement = experiment_data["throughput_improvement_percentage"]
    plt.axhline(
        y=overall_improvement,
        color="g",
        linestyle=":",
        alpha=0.7,
        label=f"Average: {overall_improvement:.1f}%",
    )
    plt.xlabel("Number of Agents")
    plt.ylabel("Throughput Improvement (%)")
    plt.title(
        f"CMA vs Distance-Based: Throughput Improvement\nOverall Average Improvement: {overall_improvement:.2f}%"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "warehouse_throughput_improvement.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating improvement plot: {e}")
    plt.close()

# Plot 4: Bar Chart Summary
try:
    plt.figure(figsize=(10, 5))
    warehouses = list(experiment_data["evaluation"].keys())
    x = np.arange(len(warehouses))
    width = 0.35
    avg_dist = [
        np.mean(experiment_data["evaluation"][w]["distance"]) for w in warehouses
    ]
    avg_cma = [np.mean(experiment_data["evaluation"][w]["cma"]) for w in warehouses]
    bars1 = plt.bar(
        x - width / 2, avg_dist, width, label="Distance-Based", color="steelblue"
    )
    bars2 = plt.bar(
        x + width / 2,
        avg_cma,
        width,
        label="Congestion-Aware (CMA)",
        color="forestgreen",
    )
    plt.xlabel("Warehouse Configuration")
    plt.ylabel("Average Throughput (tasks/min)")
    plt.title("Average Throughput by Warehouse Type and Assignment Method")
    plt.xticks(x, [w.capitalize() for w in warehouses])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    for bar in bars1 + bars2:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.savefig(
        os.path.join(working_dir, "warehouse_average_throughput_bar.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

print("All plots saved successfully.")
print(
    f"Overall throughput improvement: {experiment_data['throughput_improvement_percentage']:.2f}%"
)
