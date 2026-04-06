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
    scenarios = ["sparse_warehouse", "bottleneck_warehouse", "dense_warehouse"]

    # Plot 1: Training and Validation Loss Curves
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data["training"]["epochs"]
        train_loss = experiment_data["training"]["losses"]["train"]
        val_loss = experiment_data["training"]["losses"]["val"]
        plt.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
        plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "Warehouse Robot Task Allocation\nCongestion Predictor Training/Validation Loss"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_training_loss_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        dist_tps = [experiment_data[s]["throughput"]["distance"] for s in scenarios]
        cata_tps = [experiment_data[s]["throughput"]["cata"] for s in scenarios]
        x = np.arange(len(scenarios))
        width = 0.35
        plt.bar(
            x - width / 2,
            dist_tps,
            width,
            label="Distance-based",
            color="steelblue",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            cata_tps,
            width,
            label="CATA (Congestion-aware)",
            color="forestgreen",
            alpha=0.8,
        )
        plt.xlabel("Warehouse Scenario")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "Warehouse Robot Task Allocation\nThroughput Comparison: Distance-based vs CATA"
        )
        plt.xticks(x, [s.replace("_", " ").title() for s in scenarios], rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_comparison.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 3: Congestion Prevention Rate (CPR) Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        cprs = [experiment_data[s]["cpr"]["cpr"] for s in scenarios]
        colors = ["green" if c > 0 else "red" for c in cprs]
        x = np.arange(len(scenarios))
        plt.bar(x, cprs, color=colors, alpha=0.7, edgecolor="black")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Warehouse Scenario")
        plt.ylabel("Congestion Prevention Rate (%)")
        plt.title(
            "Warehouse Robot Task Allocation\nCongestion Prevention Rate (CPR) by Scenario\n(Positive = CATA reduces congestion)"
        )
        plt.xticks(x, [s.replace("_", " ").title() for s in scenarios], rotation=15)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_congestion_prevention_rate.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating CPR bar chart: {e}")
        plt.close()

    # Plot 4: Throughput Improvement Percentage
    try:
        plt.figure(figsize=(10, 6))
        improvements = [
            experiment_data[s]["throughput"]["improvement"] for s in scenarios
        ]
        colors = ["green" if v >= 0 else "red" for v in improvements]
        x = np.arange(len(scenarios))
        plt.bar(x, improvements, color=colors, alpha=0.7, edgecolor="black")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Warehouse Scenario")
        plt.ylabel("Throughput Improvement (%)")
        plt.title(
            "Warehouse Robot Task Allocation\nCATA Throughput Improvement over Distance-based Method"
        )
        plt.xticks(x, [s.replace("_", " ").title() for s in scenarios], rotation=15)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_throughput_improvement.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement bar chart: {e}")
        plt.close()

    # Plot 5: Congestion Events Comparison
    try:
        plt.figure(figsize=(10, 6))
        dist_cong = [experiment_data[s]["cpr"]["distance_cong"] for s in scenarios]
        cata_cong = [experiment_data[s]["cpr"]["cata_cong"] for s in scenarios]
        x = np.arange(len(scenarios))
        width = 0.35
        plt.bar(
            x - width / 2,
            dist_cong,
            width,
            label="Distance-based",
            color="coral",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            cata_cong,
            width,
            label="CATA (Congestion-aware)",
            color="lightgreen",
            alpha=0.8,
        )
        plt.xlabel("Warehouse Scenario")
        plt.ylabel("Total Congestion Events")
        plt.title(
            "Warehouse Robot Task Allocation\nCongestion Events Comparison: Distance-based vs CATA"
        )
        plt.xticks(x, [s.replace("_", " ").title() for s in scenarios], rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_congestion_events_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating congestion events plot: {e}")
        plt.close()

    # Plot 6: Combined Summary (2x2 subplot)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Warehouse Robot Task Allocation - Three Scenario Evaluation Summary",
            fontsize=14,
            fontweight="bold",
        )

        # Training loss
        axes[0, 0].plot(
            experiment_data["training"]["epochs"],
            experiment_data["training"]["losses"]["train"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[0, 0].plot(
            experiment_data["training"]["epochs"],
            experiment_data["training"]["losses"]["val"],
            "r-",
            label="Val",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("MSE Loss")
        axes[0, 0].set_title("Congestion Predictor Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Throughput comparison
        x = np.arange(len(scenarios))
        dist_tps = [experiment_data[s]["throughput"]["distance"] for s in scenarios]
        cata_tps = [experiment_data[s]["throughput"]["cata"] for s in scenarios]
        axes[0, 1].bar(x - 0.2, dist_tps, 0.4, label="Distance", color="steelblue")
        axes[0, 1].bar(x + 0.2, cata_tps, 0.4, label="CATA", color="forestgreen")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([s.split("_")[0].title() for s in scenarios])
        axes[0, 1].set_ylabel("Throughput (tasks/min)")
        axes[0, 1].set_title("Throughput Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # CPR
        cprs = [experiment_data[s]["cpr"]["cpr"] for s in scenarios]
        colors = ["green" if c > 0 else "red" for c in cprs]
        axes[1, 0].bar(x, cprs, color=colors, alpha=0.7)
        axes[1, 0].axhline(0, color="black", linestyle="--")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([s.split("_")[0].title() for s in scenarios])
        axes[1, 0].set_ylabel("CPR (%)")
        axes[1, 0].set_title("Congestion Prevention Rate (Positive = Better)")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Improvement
        imps = [experiment_data[s]["throughput"]["improvement"] for s in scenarios]
        colors = ["green" if i > 0 else "red" for i in imps]
        axes[1, 1].bar(x, imps, color=colors, alpha=0.7)
        axes[1, 1].axhline(0, color="black", linestyle="--")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([s.split("_")[0].title() for s in scenarios])
        axes[1, 1].set_ylabel("Improvement (%)")
        axes[1, 1].set_title("Throughput Improvement (Positive = Better)")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "warehouse_three_scenario_summary.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

print("All plots saved to working directory.")
