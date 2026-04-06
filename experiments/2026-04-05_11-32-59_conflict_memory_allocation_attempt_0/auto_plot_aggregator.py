import os
import hashlib
import warnings

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
    }
)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)


def pretty_name(name):
    mapping = {
        "sparse_warehouse": "Sparse warehouse",
        "bottleneck_warehouse": "Bottleneck warehouse",
        "dense_obstacle_warehouse": "Dense obstacle warehouse",
        "distance_based": "Standard decoupled",
        "congestion_aware": "Conflict memory allocation",
    }
    return mapping.get(name, str(name).replace("_", " ").title())


def safe_savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_npy_object(path):
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            obj = obj.item()
        elif isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            obj = obj.reshape(()).item()
        return obj
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def file_md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover_npy_files(search_roots=None):
    if search_roots is None:
        search_roots = [".", "experiment_results", "logs", "working", "results", "outputs", "exp_results_npy_files"]
    found = []
    excluded = {"figures", ".git", "__pycache__", ".venv", "venv", "env", "node_modules"}
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in excluded]
            for fn in filenames:
                if fn.endswith(".npy"):
                    found.append(os.path.join(dirpath, fn))
    return sorted(set(found))


def dedupe_paths(paths):
    seen = set()
    unique = []
    for p in paths:
        try:
            h = file_md5(p)
        except Exception:
            continue
        if h in seen:
            continue
        seen.add(h)
        unique.append(p)
    return unique


def is_dict_like(obj):
    return isinstance(obj, dict)


def get_epoch_key(epoch_tuning, epoch):
    if epoch in epoch_tuning:
        return epoch
    if str(epoch) in epoch_tuning:
        return str(epoch)
    try:
        e = int(epoch)
        if e in epoch_tuning:
            return e
        if str(e) in epoch_tuning:
            return str(e)
    except Exception:
        pass
    return None


def get_sorted_epochs(epoch_tuning):
    out = []
    for k in epoch_tuning.keys():
        try:
            out.append(int(k))
        except Exception:
            pass
    return sorted(set(out))


def get_dataset_entry(epoch_entry, dataset_name):
    ds = epoch_entry.get("datasets", {})
    if dataset_name in ds:
        return ds[dataset_name]
    if str(dataset_name) in ds:
        return ds[str(dataset_name)]
    return None


def infer_agent_counts(ds_entry, fallback_len=None):
    if ds_entry is None:
        return list(range(fallback_len or 0))
    if "n_agents" in ds_entry and ds_entry["n_agents"] is not None:
        try:
            return list(ds_entry["n_agents"])
        except Exception:
            pass
    for method in ["distance_based", "congestion_aware"]:
        if method in ds_entry:
            for metric in ["throughput", "congestion", "std", "congestion_std"]:
                if metric in ds_entry[method]:
                    try:
                        return list(range(len(ds_entry[method][metric])))
                    except Exception:
                        pass
    for metric in ["improvement", "congestion_prevention_rate"]:
        if metric in ds_entry:
            try:
                return list(range(len(ds_entry[metric])))
            except Exception:
                pass
    return list(range(fallback_len or 0))


def as_float_array(x):
    try:
        arr = np.asarray(x, dtype=float).squeeze()
        return arr if arr.ndim > 0 else None
    except Exception:
        return None


def extract_series(ds_entry, method, metric):
    if ds_entry is None:
        return None
    if method in ds_entry and isinstance(ds_entry[method], dict) and metric in ds_entry[method]:
        return as_float_array(ds_entry[method][metric])
    if metric in ds_entry:
        return as_float_array(ds_entry[metric])
    return None


def annotate_heatmap(ax, mat, fmt="{:.1f}", fontsize=8):
    arr = np.asarray(mat, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    norm = mpl.colors.Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
    cmap = plt.get_cmap("viridis")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if lum < 0.5 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color=color)


def score_primary(path, obj):
    score = 0
    if not is_dict_like(obj):
        return score
    if "epoch_tuning" in obj:
        score += 100
        try:
            score += len(obj["epoch_tuning"])
        except Exception:
            pass
        if "best_epochs" in obj:
            score += 5
        if "best_improvement" in obj:
            score += 5
    if "experiment_results" in path:
        score += 3
    if "/logs/" not in path.replace("\\", "/"):
        score += 2
    return score


def select_primary_experiment(loaded):
    candidates = []
    for path, obj in loaded:
        if is_dict_like(obj) and "epoch_tuning" in obj:
            candidates.append((score_primary(path, obj), path, obj))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def build_heatmap_matrices(primary_obj, metric_name):
    epoch_tuning = primary_obj["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_obj.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    agent_counts = None
    matrices = []
    for d in dataset_names:
        ds_best = get_dataset_entry(best_entry, d)
        if ds_best is None:
            continue
        series0 = np.asarray(ds_best.get(metric_name, []), dtype=float)
        if series0.size == 0:
            continue
        if agent_counts is None:
            agent_counts = infer_agent_counts(ds_best, fallback_len=len(series0))
        rows = []
        for e in epochs_sorted:
            entry = epoch_tuning[get_epoch_key(epoch_tuning, e)]
            ds_entry = get_dataset_entry(entry, d)
            vals = np.asarray(ds_entry.get(metric_name, []), dtype=float) if ds_entry is not None else np.array([])
            row = np.full(len(agent_counts), np.nan, dtype=float)
            n = min(len(row), len(vals))
            if n > 0:
                row[:n] = vals[:n]
            rows.append(row)
        matrices.append(np.asarray(rows, dtype=float))
    return dataset_names, epochs_sorted, agent_counts, matrices, best_epoch, best_entry


def find_keyword_numeric_blocks(obj, keywords=("ablation", "adaptation", "shift", "sensitivity", "decay", "weight")):
    found = []

    def rec(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}/{k}" if path else str(k)
                rec(v, p)
        elif isinstance(x, (list, tuple, np.ndarray)):
            arr = as_float_array(x)
            if arr is not None and arr.size >= 2:
                low = path.lower()
                if any(k in low for k in keywords):
                    found.append((path, arr))

    rec(obj)
    return found


def mean_or_nan(x):
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan


def load_unique_dicts():
    all_paths = discover_npy_files()
    unique_paths = dedupe_paths(all_paths)
    print(f"[INFO] Found {len(all_paths)} numpy files; {len(unique_paths)} after deduplication.")
    loaded = []
    for p in unique_paths:
        obj = load_npy_object(p)
        if is_dict_like(obj):
            loaded.append((p, obj))
            print(f"[INFO] Loaded dict-like object: {p}")
    return loaded


loaded_dicts = load_unique_dicts()
primary_path, primary_data = select_primary_experiment(loaded_dicts)
if primary_data is None:
    print("[WARN] No primary experiment with epoch tuning found.")
else:
    print(f"[INFO] Using primary experiment data from: {primary_path}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")

    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    if not epochs_sorted:
        raise RuntimeError("No epochs found.")

    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_key = get_epoch_key(epoch_tuning, best_epoch)
    best_entry = epoch_tuning[best_key]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    fig.suptitle("Conflict memory allocation: hyperparameter tuning", y=1.02)

    colors = plt.cm.tab10(np.linspace(0, 1, max(4, len(epochs_sorted))))

    ax = axes[0]
    epoch_handles = []
    for i, e in enumerate(epochs_sorted):
        entry = epoch_tuning[get_epoch_key(epoch_tuning, e)]
        train = np.asarray(entry["training"]["losses"]["train"], dtype=float)
        val = np.asarray(entry["training"]["losses"]["val"], dtype=float)
        c = colors[i % len(colors)]
        ax.plot(train, color=c, alpha=0.9)
        ax.plot(val, color=c, linestyle="--", alpha=0.9)
        epoch_handles.append(Line2D([0], [0], color=c, lw=2.4, label=f"{e} epochs"))
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    despine(ax)
    leg1 = ax.legend(handles=epoch_handles, title="Training length", loc="upper right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=2.4, linestyle="-", label="Training loss"),
            Line2D([0], [0], color="black", lw=2.4, linestyle="--", label="Validation loss"),
        ],
        loc="lower left",
        frameon=True,
    )

    ax = axes[1]
    final_val_losses = [float(epoch_tuning[get_epoch_key(epoch_tuning, e)]["final_val_loss"]) for e in epochs_sorted]
    ax.plot(epochs_sorted, final_val_losses, marker="o", color="#1f77b4")
    ax.axvline(float(best_epoch), linestyle="--", color="0.35", alpha=0.7)
    ax.scatter([best_epoch], [best_entry["final_val_loss"]], s=80, color="#d62728", zorder=5, label="Selected setting")
    ax.set_title("Final validation loss")
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    despine(ax)
    ax.legend(loc="best", frameon=True)

    ax = axes[2]
    overall = [float(epoch_tuning[get_epoch_key(epoch_tuning, e)].get("overall_improvement", np.nan)) for e in epochs_sorted]
    best_impr = float(primary_data.get("best_improvement", np.nan))
    ax.plot(epochs_sorted, overall, marker="o", color="#2ca02c")
    ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.2)
    ax.axvline(float(best_epoch), linestyle="--", color="0.35", alpha=0.7)
    ax.scatter([best_epoch], [best_impr], s=80, color="#d62728", zorder=5, label="Best setting")
    ax.set_title("Throughput improvement")
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Improvement (%)")
    ax.grid(True)
    despine(ax)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_1_hyperparameter_tuning.png"))
    print("[INFO] Saved figure 1.")
except Exception as e:
    print(f"[WARN] Figure 1 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    dataset_names, epochs_sorted, agent_counts, matrices, best_epoch, best_entry = build_heatmap_matrices(primary_data, "improvement")
    if not matrices or agent_counts is None:
        raise RuntimeError("No heatmap matrices available.")

    finite = [m for m in matrices if np.isfinite(m).any()]
    vmin = min(np.nanmin(m) for m in finite)
    vmax = max(np.nanmax(m) for m in finite)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.7), sharex=True, sharey=True)
    fig.suptitle("Throughput improvement across training length and agent count", y=1.02)

    for i, ax in enumerate(axes):
        if i >= len(matrices):
            ax.axis("off")
            continue
        mat = matrices[i]
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(pretty_name(dataset_names[i]))
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Training epochs")
        ax.set_xticks(np.arange(len(agent_counts)))
        ax.set_xticklabels([str(a) for a in agent_counts])
        ax.set_yticks(np.arange(len(epochs_sorted)))
        ax.set_yticklabels([str(e) for e in epochs_sorted])
        annotate_heatmap(ax, mat, fmt="{:.1f}", fontsize=8)
        despine(ax)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Throughput improvement (%)")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_2_improvement_heatmaps.png"))
    print("[INFO] Saved figure 2.")
except Exception as e:
    print(f"[WARN] Figure 2 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    dataset_names, epochs_sorted, agent_counts, matrices, best_epoch, best_entry = build_heatmap_matrices(primary_data, "congestion_prevention_rate")
    if not matrices or agent_counts is None:
        raise RuntimeError("No heatmap matrices available.")

    finite = [m for m in matrices if np.isfinite(m).any()]
    vmin = min(np.nanmin(m) for m in finite)
    vmax = max(np.nanmax(m) for m in finite)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.7), sharex=True, sharey=True)
    fig.suptitle("Congestion prevention across training length and agent count", y=1.02)

    for i, ax in enumerate(axes):
        if i >= len(matrices):
            ax.axis("off")
            continue
        mat = matrices[i]
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(pretty_name(dataset_names[i]))
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Training epochs")
        ax.set_xticks(np.arange(len(agent_counts)))
        ax.set_xticklabels([str(a) for a in agent_counts])
        ax.set_yticks(np.arange(len(epochs_sorted)))
        ax.set_yticklabels([str(e) for e in epochs_sorted])
        annotate_heatmap(ax, mat, fmt="{:.1f}", fontsize=8)
        despine(ax)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Congestion prevention rate (%)")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_3_congestion_prevention_heatmaps.png"))
    print("[INFO] Saved figure 3.")
except Exception as e:
    print(f"[WARN] Figure 3 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    if len(dataset_names) == 0:
        raise RuntimeError("No datasets found.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)
    fig.suptitle(f"Throughput at {best_epoch} training epochs", y=1.02)

    for ax, d in zip(axes, dataset_names):
        ds = get_dataset_entry(best_entry, d)
        if ds is None:
            ax.axis("off")
            continue
        agents = np.asarray(infer_agent_counts(ds, fallback_len=len(ds["distance_based"]["throughput"])), dtype=float)
        for key, label, color, linestyle in [
            ("distance_based", "Standard decoupled", "#7f7f7f", "--"),
            ("congestion_aware", "Conflict memory allocation", "#1f77b4", "-"),
        ]:
            y = extract_series(ds, key, "throughput")
            yerr = extract_series(ds, key, "std")
            if y is None:
                continue
            if yerr is None:
                yerr = np.zeros_like(y)
            n = min(len(agents), len(y), len(yerr))
            ax.errorbar(agents[:n], y[:n], yerr=yerr[:n], marker="o", linestyle=linestyle, color=color, capsize=3, label=label)
        ax.set_title(pretty_name(d))
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Throughput\n(tasks per minute)")
        ax.grid(True)
        despine(ax)
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_4_throughput_comparison.png"))
    print("[INFO] Saved figure 4.")
except Exception as e:
    print(f"[WARN] Figure 4 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    if len(dataset_names) == 0:
        raise RuntimeError("No datasets found.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)
    fig.suptitle(f"Congestion at {best_epoch} training epochs", y=1.02)

    for ax, d in zip(axes, dataset_names):
        ds = get_dataset_entry(best_entry, d)
        if ds is None:
            ax.axis("off")
            continue
        agents = np.asarray(infer_agent_counts(ds, fallback_len=len(ds["distance_based"]["congestion"])), dtype=float)
        for key, label, color, linestyle in [
            ("distance_based", "Standard decoupled", "#7f7f7f", "--"),
            ("congestion_aware", "Conflict memory allocation", "#d62728", "-"),
        ]:
            y = extract_series(ds, key, "congestion")
            yerr = extract_series(ds, key, "congestion_std")
            if y is None:
                continue
            if yerr is None:
                yerr = np.zeros_like(y)
            n = min(len(agents), len(y), len(yerr))
            ax.errorbar(agents[:n], y[:n], yerr=yerr[:n], marker="o", linestyle=linestyle, color=color, capsize=3, label=label)
        ax.set_title(pretty_name(d))
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Congestion\n(events)")
        ax.grid(True)
        despine(ax)
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_5_congestion_comparison.png"))
    print("[INFO] Saved figure 5.")
except Exception as e:
    print(f"[WARN] Figure 5 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    if len(dataset_names) == 0:
        raise RuntimeError("No datasets found.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)
    fig.suptitle("Mean throughput improvement versus training length", y=1.02)

    for ax, d in zip(axes, dataset_names):
        mean_improvement = []
        for e in epochs_sorted:
            ds = get_dataset_entry(epoch_tuning[get_epoch_key(epoch_tuning, e)], d)
            vals = np.asarray(ds.get("improvement", []), dtype=float) if ds is not None else np.array([])
            mean_improvement.append(float(np.nanmean(vals)) if vals.size else np.nan)
        ax.plot(epochs_sorted, mean_improvement, marker="o", color="#1f77b4")
        ax.axvline(float(best_epoch), linestyle="--", color="0.35", alpha=0.7)
        ax.set_title(pretty_name(d))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Improvement (%)")
        ax.grid(True)
        despine(ax)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_6_training_length_improvement.png"))
    print("[INFO] Saved figure 6.")
except Exception as e:
    print(f"[WARN] Figure 6 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    if len(dataset_names) == 0:
        raise RuntimeError("No datasets found.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)
    fig.suptitle("Mean congestion prevention versus training length", y=1.02)

    for ax, d in zip(axes, dataset_names):
        mean_cpr = []
        for e in epochs_sorted:
            ds = get_dataset_entry(epoch_tuning[get_epoch_key(epoch_tuning, e)], d)
            vals = np.asarray(ds.get("congestion_prevention_rate", []), dtype=float) if ds is not None else np.array([])
            mean_cpr.append(float(np.nanmean(vals)) if vals.size else np.nan)
        ax.plot(epochs_sorted, mean_cpr, marker="o", color="#d62728")
        ax.axvline(float(best_epoch), linestyle="--", color="0.35", alpha=0.7)
        ax.set_title(pretty_name(d))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Prevention rate (%)")
        ax.grid(True)
        despine(ax)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_7_training_length_prevention.png"))
    print("[INFO] Saved figure 7.")
except Exception as e:
    print(f"[WARN] Figure 7 failed: {e}")


try:
    if primary_data is None or "epoch_tuning" not in primary_data:
        raise RuntimeError("Missing epoch tuning data.")
    epoch_tuning = primary_data["epoch_tuning"]
    epochs_sorted = get_sorted_epochs(epoch_tuning)
    best_epoch = primary_data.get("best_epochs", epochs_sorted[-1])
    best_entry = epoch_tuning[get_epoch_key(epoch_tuning, best_epoch)]
    dataset_names = list(best_entry.get("datasets", {}).keys())
    if len(dataset_names) == 0:
        raise RuntimeError("No datasets found.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    fig.suptitle("Tradeoff between throughput gain and congestion prevention", y=1.02)

    color_cycle = plt.cm.Set2(np.linspace(0, 1, 8))
    for ax, d in zip(axes, dataset_names):
        ds = get_dataset_entry(best_entry, d)
        if ds is None:
            ax.axis("off")
            continue
        imp = np.asarray(ds.get("improvement", []), dtype=float)
        cpr = np.asarray(ds.get("congestion_prevention_rate", []), dtype=float)
        agents = infer_agent_counts(ds, fallback_len=min(len(imp), len(cpr)))
        n = min(len(imp), len(cpr), len(agents))
        ax.plot(imp[:n], cpr[:n], color="#444444", alpha=0.7, linewidth=1.8)
        handles = []
        for i in range(n):
            c = color_cycle[i % len(color_cycle)]
            ax.scatter(imp[i], cpr[i], s=70, color=c, edgecolor="white", linewidth=0.8, zorder=5)
            ax.text(imp[i] + 0.3, cpr[i] + 0.3, str(agents[i]), fontsize=10)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    markeredgecolor="white",
                    markersize=8,
                    label=f"{agents[i]} agents",
                )
            )
        ax.set_title(pretty_name(d))
        ax.set_xlabel("Throughput improvement (%)")
        ax.set_ylabel("Congestion prevention rate (%)")
        ax.grid(True)
        despine(ax)
        if handles:
            ax.legend(handles=handles, title="Agent count", loc="best", frameon=True)

    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIG_DIR, "figure_8_tradeoff_analysis.png"))
    print("[INFO] Saved figure 8.")
except Exception as e:
    print(f"[WARN] Figure 8 failed: {e}")


try:
    if len(loaded_dicts) > 0:
        candidates = []
        primary_id = id(primary_data)
        for path, obj in loaded_dicts:
            if id(obj) == primary_id:
                continue
            blocks = find_keyword_numeric_blocks(obj)
            if blocks:
                candidates.append((path, obj, blocks))
        if candidates:
            candidates.sort(key=lambda t: len(t[2]), reverse=True)
            path, obj, blocks = candidates[0]
            blocks = blocks[:3]
            fig, axes = plt.subplots(1, len(blocks), figsize=(6.2 * len(blocks), 4.8))
            if len(blocks) == 1:
                axes = [axes]
            fig.suptitle(f"Supplementary summary from {os.path.basename(path).replace('_', ' ')}", y=1.02)
            for ax, (name, arr) in zip(axes, blocks):
                ax.plot(np.arange(len(arr)), arr, marker="o", color="#1f77b4")
                ax.set_title(name.replace("_", " "))
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
                ax.grid(True)
                despine(ax)
            fig.tight_layout()
            safe_savefig(fig, os.path.join(FIG_DIR, "figure_9_supplementary_summary.png"))
            print(f"[INFO] Saved supplementary figure from {path}")
except Exception as e:
    print(f"[WARN] Supplementary figure failed: {e}")

print(f"[DONE] Finished plotting. Output directory: {FIG_DIR}")