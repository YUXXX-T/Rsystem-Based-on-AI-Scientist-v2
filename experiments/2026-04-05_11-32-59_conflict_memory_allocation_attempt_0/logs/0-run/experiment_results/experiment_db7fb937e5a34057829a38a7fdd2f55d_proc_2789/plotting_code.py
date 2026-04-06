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
    plt.figure(figsize=(8, 6))
    epochs = experiment_data["training"]["epochs"]
    train_losses = experiment_data["training"]["losses"]["train"]
    val_losses = experiment_data["training"]["losses"]["val"]

    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        "Congestion Predictor Training Curves\n(Combined Sparse & Bottleneck Warehouse Data)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(working_dir, "training_validation_curves.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating training curves plot: {e}")
    plt.close()

# Plot 2: Throughput Comparison - Sparse Warehouse
try:
    plt.figure(figsize=(8, 6))
    data = experiment_data["evaluation"]["sparse_warehouse"]
    n_agents = data["n_agents"]

    plt.errorbar(
        n_agents,
        data["distance_based"]["throughput"],
        yerr=data["distance_based"]["std"],
        marker="o",
        linestyle="--",
        label="Distance-Based",
        capsize=5,
        color="blue",
    )
    plt.errorbar(
        n_agents,
        data["congestion_aware"]["throughput"],
        yerr=data["congestion_aware"]["std"],
        marker="s",
        linestyle="-",
        label="Congestion-Aware",
        capsize=5,
        color="green",
    )
    plt.xlabel("Number of Agents")
    plt.ylabel("Throughput (tasks/min)")
    plt.title(
        "Throughput Comparison: Sparse Warehouse\nDistance-Based vs Congestion-Aware Assignment"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "sparse_warehouse_throughput_comparison.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating sparse warehouse throughput plot: {e}")
    plt.close()

# Plot 3: Throughput Comparison - Bottleneck Warehouse
try:
    plt.figure(figsize=(8, 6))
    data = experiment_data["evaluation"]["bottleneck_warehouse"]
    n_agents = data["n_agents"]

    plt.errorbar(
        n_agents,
        data["distance_based"]["throughput"],
        yerr=data["distance_based"]["std"],
        marker="o",
        linestyle="--",
        label="Distance-Based",
        capsize=5,
        color="blue",
    )
    plt.errorbar(
        n_agents,
        data["congestion_aware"]["throughput"],
        yerr=data["congestion_aware"]["std"],
        marker="s",
        linestyle="-",
        label="Congestion-Aware",
        capsize=5,
        color="green",
    )
    plt.xlabel("Number of Agents")
    plt.ylabel("Throughput (tasks/min)")
    plt.title(
        "Throughput Comparison: Bottleneck Warehouse\nDistance-Based vs Congestion-Aware Assignment"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "bottleneck_warehouse_throughput_comparison.png"),
        dpi=150,
    )
    plt.close()
except Exception as e:
    print(f"Error creating bottleneck warehouse throughput plot: {e}")
    plt.close()

# Plot 4: Improvement Percentage Bar Chart
try:
    plt.figure(figsize=(10, 6))
    sparse_data = experiment_data["evaluation"]["sparse_warehouse"]
    bottleneck_data = experiment_data["evaluation"]["bottleneck_warehouse"]
    n_agents = sparse_data["n_agents"]

    x = np.arange(len(n_agents))
    width = 0.35

    bars1 = plt.bar(
        x - width / 2,
        sparse_data["improvement"],
        width,
        label="Sparse Warehouse",
        color="steelblue",
    )
    bars2 = plt.bar(
        x + width / 2,
        bottleneck_data["improvement"],
        width,
        label="Bottleneck Warehouse",
        color="darkorange",
    )

    plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    avg_improvement = experiment_data["throughput_improvement_percentage"]
    plt.axhline(
        y=avg_improvement,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Avg Improvement: {avg_improvement:.2f}%",
    )

    plt.xlabel("Number of Agents")
    plt.ylabel("Throughput Improvement (%)")
    plt.title(
        "Congestion-Aware Assignment Throughput Improvement Over Distance-Based\nBy Warehouse Type and Agent Count"
    )
    plt.xticks(x, n_agents)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(
        os.path.join(working_dir, "throughput_improvement_comparison.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating improvement bar chart: {e}")
    plt.close()

# Print summary metrics
try:
    print(f"\nExperiment Summary:")
    print(
        f"Overall Throughput Improvement: {experiment_data['throughput_improvement_percentage']:.2f}%"
    )
    print(
        f"Final Training Loss: {experiment_data['training']['losses']['train'][-1]:.4f}"
    )
    print(
        f"Final Validation Loss: {experiment_data['training']['losses']['val'][-1]:.4f}"
    )
except Exception as e:
    print(f"Error printing summary: {e}")
