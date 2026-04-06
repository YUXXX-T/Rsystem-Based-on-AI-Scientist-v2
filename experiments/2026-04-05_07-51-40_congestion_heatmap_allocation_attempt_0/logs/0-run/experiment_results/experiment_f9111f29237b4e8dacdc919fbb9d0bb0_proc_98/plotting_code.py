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
    # Plot 1: Training Loss Curves
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data["training"]["epochs"]
        train_loss = experiment_data["training"]["losses"]["train"]
        val_loss = experiment_data["training"]["losses"]["val"]

        plt.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
        plt.plot(epochs, val_loss, label="Validation Loss", color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("CATA Congestion Predictor Training\nTrain vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cata_training_loss_curves.png"), dpi=150)
        plt.close()
        print("Saved training loss curves plot")
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot 2: Throughput Comparison Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        agent_counts = experiment_data["agent_counts"]
        dist_throughput = experiment_data["distance_based"]["throughput"]
        heur_throughput = experiment_data["heuristic_congestion"]["throughput"]
        learn_throughput = experiment_data["learned_congestion"]["throughput"]

        x = np.arange(len(agent_counts))
        width = 0.25

        plt.bar(
            x - width, dist_throughput, width, label="Distance-based", color="steelblue"
        )
        plt.bar(x, heur_throughput, width, label="Heuristic CATA", color="coral")
        plt.bar(
            x + width,
            learn_throughput,
            width,
            label="Learned CATA",
            color="forestgreen",
        )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "CATA Warehouse Simulation: Throughput Comparison\nAcross Different Agent Densities"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "cata_throughput_comparison_bar.png"), dpi=150
        )
        plt.close()
        print("Saved throughput comparison bar chart")
    except Exception as e:
        print(f"Error creating throughput bar chart: {e}")
        plt.close()

    # Plot 3: Throughput Scaling Line Plot
    try:
        plt.figure(figsize=(10, 6))
        agent_counts = experiment_data["agent_counts"]

        plt.plot(
            agent_counts,
            experiment_data["distance_based"]["throughput"],
            "o-",
            label="Distance-based",
            color="steelblue",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            agent_counts,
            experiment_data["heuristic_congestion"]["throughput"],
            "s-",
            label="Heuristic CATA",
            color="coral",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            agent_counts,
            experiment_data["learned_congestion"]["throughput"],
            "^-",
            label="Learned CATA",
            color="forestgreen",
            linewidth=2,
            markersize=8,
        )

        plt.xlabel("Number of Agents")
        plt.ylabel("Throughput (tasks/min)")
        plt.title(
            "CATA Warehouse Simulation: Throughput Scaling\nPerformance vs Agent Density"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "cata_throughput_scaling_line.png"), dpi=150
        )
        plt.close()
        print("Saved throughput scaling line plot")
    except Exception as e:
        print(f"Error creating throughput scaling plot: {e}")
        plt.close()

    # Plot 4: Improvement over Baseline
    try:
        plt.figure(figsize=(10, 6))
        agent_counts = experiment_data["agent_counts"]
        dist_throughput = np.array(experiment_data["distance_based"]["throughput"])
        heur_throughput = np.array(
            experiment_data["heuristic_congestion"]["throughput"]
        )
        learn_throughput = np.array(experiment_data["learned_congestion"]["throughput"])

        heur_improvement = (heur_throughput - dist_throughput) / dist_throughput * 100
        learn_improvement = (learn_throughput - dist_throughput) / dist_throughput * 100

        x = np.arange(len(agent_counts))
        width = 0.35

        plt.bar(
            x - width / 2,
            heur_improvement,
            width,
            label="Heuristic CATA",
            color="coral",
        )
        plt.bar(
            x + width / 2,
            learn_improvement,
            width,
            label="Learned CATA",
            color="forestgreen",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.xlabel("Number of Agents")
        plt.ylabel("Improvement over Distance-Based (%)")
        plt.title(
            "CATA Warehouse Simulation: Percentage Improvement\nOver Distance-Based Baseline"
        )
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "cata_improvement_over_baseline.png"), dpi=150
        )
        plt.close()
        print("Saved improvement over baseline plot")
    except Exception as e:
        print(f"Error creating improvement plot: {e}")
        plt.close()

    # Print summary metrics
    print("\n=== Summary Metrics ===")
    print(f"Agent counts tested: {experiment_data['agent_counts']}")
    print(f"Final train loss: {experiment_data['training']['losses']['train'][-1]:.4f}")
    print(f"Final val loss: {experiment_data['training']['losses']['val'][-1]:.4f}")
    print(
        f"Mean Learned CATA throughput: {np.mean(experiment_data['learned_congestion']['throughput']):.2f} tasks/min"
    )
