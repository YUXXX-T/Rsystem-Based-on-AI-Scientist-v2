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
    lookahead_values = [2, 5, 8, 12]
    agent_counts = [20, 30, 40, 50]

    # Plot 1: Training losses for all lookahead values
    try:
        plt.figure(figsize=(10, 6))
        for la in lookahead_values:
            key = f"lookahead_{la}"
            train_losses = experiment_data["lookahead_tuning"][key]["losses"]["train"]
            plt.plot(train_losses, label=f"Lookahead={la}", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title(
            "Warehouse CATA: Training Loss vs Epoch\n(Lookahead Hyperparameter Tuning)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_training_loss_curves.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Validation losses for all lookahead values
    try:
        plt.figure(figsize=(10, 6))
        for la in lookahead_values:
            key = f"lookahead_{la}"
            val_losses = experiment_data["lookahead_tuning"][key]["losses"]["val"]
            plt.plot(val_losses, label=f"Lookahead={la}", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title(
            "Warehouse CATA: Validation Loss vs Epoch\n(Lookahead Hyperparameter Tuning)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_validation_loss_curves.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # Plot 3: Throughput comparison - Distance vs CATA for best lookahead
    try:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(agent_counts))
        width = 0.35
        best_la = max(
            lookahead_values,
            key=lambda la: experiment_data["lookahead_tuning"][f"lookahead_{la}"][
                "avg_throughput"
            ],
        )
        key = f"lookahead_{best_la}"
        dist_throughputs = [
            e["mean"]
            for e in experiment_data["lookahead_tuning"][key]["evaluation"][
                "distance_based"
            ]
        ]
        cata_throughputs = [
            e["mean"]
            for e in experiment_data["lookahead_tuning"][key]["evaluation"][
                "congestion_aware"
            ]
        ]
        plt.bar(
            x - width / 2,
            dist_throughputs,
            width,
            label="Distance-Based",
            color="steelblue",
        )
        plt.bar(
            x + width / 2,
            cata_throughputs,
            width,
            label="Congestion-Aware (CATA)",
            color="green",
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            f"Warehouse CATA: Throughput Comparison\n(Best Lookahead={best_la}, Distance-Based vs Congestion-Aware)"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_throughput_comparison.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating throughput comparison plot: {e}")
        plt.close()

    # Plot 4: Average throughput per lookahead (bar chart)
    try:
        plt.figure(figsize=(10, 6))
        avg_throughputs = [
            experiment_data["lookahead_tuning"][f"lookahead_{la}"]["avg_throughput"]
            for la in lookahead_values
        ]
        best_la = lookahead_values[np.argmax(avg_throughputs)]
        colors = ["green" if la == best_la else "steelblue" for la in lookahead_values]
        bars = plt.bar(
            [str(la) for la in lookahead_values],
            avg_throughputs,
            color=colors,
            edgecolor="black",
        )
        plt.xlabel("Lookahead Value")
        plt.ylabel("Average Throughput (tasks/min)")
        plt.title(
            f"Warehouse CATA: Average Throughput vs Lookahead\n(Best Lookahead={best_la} highlighted in Green)"
        )
        for bar, val in zip(bars, avg_throughputs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
            )
        plt.grid(True, alpha=0.3, axis="y")
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_lookahead_tuning_summary.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating lookahead summary plot: {e}")
        plt.close()

    # Plot 5: Improvement percentage across agent counts for all lookahead values
    try:
        plt.figure(figsize=(12, 6))
        for la in lookahead_values:
            key = f"lookahead_{la}"
            dist_tp = [
                e["mean"]
                for e in experiment_data["lookahead_tuning"][key]["evaluation"][
                    "distance_based"
                ]
            ]
            cata_tp = [
                e["mean"]
                for e in experiment_data["lookahead_tuning"][key]["evaluation"][
                    "congestion_aware"
                ]
            ]
            improvements = [(c - d) / d * 100 for c, d in zip(cata_tp, dist_tp)]
            plt.plot(
                agent_counts,
                improvements,
                marker="o",
                linewidth=2,
                label=f"Lookahead={la}",
            )
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Number of Agents")
        plt.ylabel("Improvement (%)")
        plt.title(
            "Warehouse CATA: Throughput Improvement (%) over Distance-Based\n(CATA vs Distance-Based across Lookahead Values)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(working_dir, "warehouse_cata_improvement_analysis.png"),
            dpi=150,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating improvement analysis plot: {e}")
        plt.close()

print("Plotting complete.")
