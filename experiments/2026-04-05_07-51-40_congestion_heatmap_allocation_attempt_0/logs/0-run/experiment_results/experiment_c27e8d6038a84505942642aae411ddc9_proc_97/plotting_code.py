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

# Plot 1: Throughput Comparison Bar Chart
try:
    plt.figure(figsize=(10, 6))
    agent_counts = experiment_data["distance_based"]["agent_counts"]
    dist_throughput = experiment_data["distance_based"]["throughput"]
    heur_throughput = experiment_data["congestion_aware_heuristic"]["throughput"]
    learn_throughput = experiment_data["congestion_aware_learned"]["throughput"]

    x = np.arange(len(agent_counts))
    width = 0.25
    plt.bar(
        x - width,
        dist_throughput,
        width,
        label="Distance-based",
        color="blue",
        alpha=0.7,
    )
    plt.bar(x, heur_throughput, width, label="Heuristic CATA", color="green", alpha=0.7)
    plt.bar(
        x + width, learn_throughput, width, label="Learned CATA", color="red", alpha=0.7
    )

    plt.xlabel("Number of Agents")
    plt.ylabel("Throughput (tasks/min)")
    plt.title(
        "CATA Warehouse Simulation: Throughput Comparison by Agent Count\n(Higher is better)"
    )
    plt.xticks(x, agent_counts)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cata_throughput_comparison.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating throughput comparison plot: {e}")
    plt.close()

# Plot 2: Congestion Predictor Training Loss Curves
try:
    plt.figure(figsize=(10, 6))
    epochs = experiment_data["predictor_training"]["epochs"]
    train_loss = experiment_data["predictor_training"]["losses"]["train"]
    val_loss = experiment_data["predictor_training"]["losses"]["val"]

    plt.plot(epochs, train_loss, label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CATA Congestion Predictor: Training and Validation Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cata_predictor_training_loss.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot 3: Completion Times Comparison
try:
    plt.figure(figsize=(10, 6))
    agent_counts = experiment_data["distance_based"]["agent_counts"]
    dist_times = experiment_data["distance_based"]["completion_times"]
    heur_times = experiment_data["congestion_aware_heuristic"]["completion_times"]
    learn_times = experiment_data["congestion_aware_learned"]["completion_times"]

    x = np.arange(len(agent_counts))
    width = 0.25
    plt.bar(
        x - width, dist_times, width, label="Distance-based", color="blue", alpha=0.7
    )
    plt.bar(x, heur_times, width, label="Heuristic CATA", color="green", alpha=0.7)
    plt.bar(x + width, learn_times, width, label="Learned CATA", color="red", alpha=0.7)

    plt.xlabel("Number of Agents")
    plt.ylabel("Average Completion Time (steps)")
    plt.title(
        "CATA Warehouse Simulation: Average Task Completion Times\n(Lower is better)"
    )
    plt.xticks(x, agent_counts)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "cata_completion_times_comparison.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating completion times plot: {e}")
    plt.close()

# Plot 4: Summary Metrics with Improvement Percentages
try:
    plt.figure(figsize=(12, 5))

    # Calculate averages
    avg_dist = np.mean(experiment_data["distance_based"]["throughput"])
    avg_heur = np.mean(experiment_data["congestion_aware_heuristic"]["throughput"])
    avg_learn = np.mean(experiment_data["congestion_aware_learned"]["throughput"])

    methods = ["Distance-based\n(Baseline)", "Heuristic CATA", "Learned CATA"]
    throughputs = [avg_dist, avg_heur, avg_learn]
    colors = ["blue", "green", "red"]

    bars = plt.bar(methods, throughputs, color=colors, alpha=0.7, edgecolor="black")

    # Add improvement percentages
    for i, (bar, tp) in enumerate(zip(bars, throughputs)):
        if i == 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{tp:.2f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
        else:
            improvement = ((tp / avg_dist) - 1) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{tp:.2f}\n({improvement:+.1f}%)",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Average Throughput (tasks/min)")
    plt.title(
        "CATA Warehouse Simulation: Overall Method Comparison\n(Average across all agent counts)"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cata_summary_comparison.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating summary comparison plot: {e}")
    plt.close()

print("All plots saved successfully!")
