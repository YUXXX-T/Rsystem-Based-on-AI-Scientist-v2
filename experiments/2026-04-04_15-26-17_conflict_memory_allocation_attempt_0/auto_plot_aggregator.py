import os
import re
import json
import glob
import hashlib
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "figures"
MAX_FIGURES = 12
DPI = 300

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

THEMES = [
    "main comparison",
    "efficiency",
    "conflict analysis",
    "adaptation",
    "ablation",
    "generalization",
    "benchmark breakdown",
    "supplementary",
]

THEME_KEYWORDS = {
    "main comparison": ["cma", "baseline", "coupled", "makespan", "flowtime", "success", "performance", "evaluation", "comparison", "hungarian", "cbs"],
    "efficiency": ["runtime", "time", "compute", "computation", "speed", "overhead", "efficiency"],
    "conflict analysis": ["conflict", "histogram", "heatmap", "spatial", "temporal", "distribution", "map"],
    "adaptation": ["adapt", "shift", "online", "generalization", "drift", "update"],
    "ablation": ["ablation", "decay", "weight", "penalty", "sensitivity", "resolution", "vertex", "edge", "complexity"],
    "generalization": ["generalization", "transfer", "robust", "shift", "adapt"],
    "benchmark breakdown": ["warehouse", "random", "room", "benchmark", "scenario", "instance"],
    "supplementary": [],
}

METHOD_COLORS = {
    "standard decoupled": "#4C78A8",
    "cma histogram": "#F58518",
    "cma gnn": "#54A24B",
    "fully coupled": "#E45756",
    "coupled": "#E45756",
    "baseline": "#4C78A8",
    "histogram": "#F58518",
    "gnn": "#54A24B",
    "conflict": "#B279A2",
    "adapt": "#72B7B2",
    "runtime": "#9D755D",
    "success": "#59A14F",
    "makespan": "#EDC948",
    "flowtime": "#76B7B2",
    "decay": "#7F7F7F",
    "penalty": "#FF9DA6",
    "resolution": "#B07AA1",
    "vertex": "#9C755F",
    "edge": "#BAB0AC",
}

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def clean_label(s):
    s = str(s).replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_title(s):
    return clean_label(s).title()

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, filename):
    ensure_fig_dir()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return path

def maybe_legend(ax, ncol=1, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc=loc, ncol=ncol)

def infer_color(name, idx=0):
    low = clean_label(name).lower()
    for k, c in METHOD_COLORS.items():
        if k in low:
            return c
    palette = plt.get_cmap("tab10").colors
    return palette[idx % len(palette)]

def to_array(x):
    try:
        arr = np.asarray(x)
        if arr.dtype == object:
            return None
        if np.issubdtype(arr.dtype, np.number):
            return arr
        return None
    except Exception:
        return None

def load_npy(path):
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            if obj.shape == ():
                try:
                    return obj.item()
                except Exception:
                    return obj
            if obj.size == 1:
                try:
                    return obj.reshape(()).item()
                except Exception:
                    return obj
        return obj
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None

def find_json_summary_files():
    roots = [".", "./results", "./outputs", "./experiments", "./exp_results_npy_files"]
    found = []
    for root in roots:
        if os.path.isdir(root):
            found.extend(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    out = []
    seen = set()
    for p in found:
        base = os.path.basename(p).lower()
        if any(k in base for k in ["summary", "result", "experiment", "baseline", "research", "ablation"]):
            p = os.path.normpath(p)
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out

def recursive_extract_npy_paths(obj):
    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "exp_results_npy_files":
                if isinstance(v, str):
                    paths.append(v)
                elif isinstance(v, (list, tuple)):
                    paths.extend([x for x in v if isinstance(x, str)])
            else:
                paths.extend(recursive_extract_npy_paths(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            paths.extend(recursive_extract_npy_paths(v))
    elif isinstance(obj, str) and obj.endswith(".npy"):
        paths.append(obj)
    return paths

def recursive_extract_scalars(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(recursive_extract_scalars(v, key))
    elif isinstance(obj, (list, tuple)):
        arr = to_array(obj)
        if arr is not None and arr.ndim == 0:
            out.append((prefix, float(arr)))
        else:
            for i, v in enumerate(obj):
                out.extend(recursive_extract_scalars(v, f"{prefix}[{i}]"))
    else:
        if np.isscalar(obj) and not isinstance(obj, (str, bytes)):
            out.append((prefix, float(obj)))
    return out

def load_summary_metadata():
    npy_paths = []
    scalars = []
    for jp in find_json_summary_files():
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            npy_paths.extend(recursive_extract_npy_paths(data))
            scalars.extend(recursive_extract_scalars(data))
        except Exception:
            continue
    npy_paths = [os.path.normpath(p) for p in npy_paths if isinstance(p, str) and p.endswith(".npy") and os.path.exists(p)]
    dedup = []
    seen = set()
    for p in npy_paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup, scalars

def fallback_discover_npy():
    roots = [".", "./exp_results_npy_files", "./results", "./outputs", "./experiments"]
    found = []
    for root in roots:
        if os.path.isdir(root):
            found.extend(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True))
    return sorted(set(map(os.path.normpath, found)))

def flatten_numeric_leaves(obj, prefix=""):
    leaves = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            leaves.extend(flatten_numeric_leaves(v, p))
    elif isinstance(obj, (list, tuple)):
        arr = to_array(obj)
        if arr is not None:
            leaves.append((prefix, arr))
        else:
            for i, v in enumerate(obj):
                leaves.extend(flatten_numeric_leaves(v, f"{prefix}[{i}]"))
    else:
        arr = to_array(obj)
        if arr is not None:
            leaves.append((prefix, arr))
        elif np.isscalar(obj) and not isinstance(obj, (str, bytes)):
            leaves.append((prefix, np.asarray(obj)))
    return leaves

def object_signature(obj):
    try:
        if isinstance(obj, dict):
            parts = []
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
                parts.append(str(k))
                parts.append(object_signature(v))
            return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        arr = to_array(obj)
        if arr is None:
            return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()
        arr = np.asarray(arr)
        sample = arr.ravel()[:2048]
        payload = f"{arr.shape}|{arr.dtype}|".encode("utf-8") + sample.tobytes()
        return hashlib.md5(payload).hexdigest()
    except Exception:
        return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()

def theme_for_item(path, obj):
    text = clean_label(Path(path).stem).lower()
    if isinstance(obj, dict):
        text += " " + " ".join(clean_label(str(k)).lower() for k in obj.keys())
    elif isinstance(obj, np.ndarray):
        text += f" shape {obj.shape}"
    best = "supplementary"
    best_score = -1
    for theme in THEMES:
        kws = THEME_KEYWORDS.get(theme, [])
        score = sum(1 for kw in kws if kw in text)
        if theme == "main comparison" and any(x in text for x in ["cma", "baseline", "coupled"]):
            score += 2
        if theme == "conflict analysis" and any(x in text for x in ["heatmap", "histogram", "map"]):
            score += 2
        if score > best_score:
            best = theme
            best_score = score
    return best, best_score

def relevant_label_from_key(key):
    key = key.split(".")[-1]
    key = key.split("[")[0]
    return clean_label(key)

def summary_highlights_text(scalars, n=3):
    vals = []
    for k, v in scalars:
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append((k, float(v)))
    if not vals:
        return ""
    vals = sorted(vals, key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return " | ".join([f"{relevant_label_from_key(k)}: {v:.3g}" for k, v in vals])

def plot_series(ax, y, label, color=None):
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=2.2, marker="o", markersize=3.5, label=label, color=color)

def plot_hist(ax, arr, color="#4C78A8"):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    bins = min(30, max(5, int(np.sqrt(arr.size))))
    ax.hist(arr, bins=bins, color=color, alpha=0.85, edgecolor="white")
    ax.set_xlabel("value")
    ax.set_ylabel("count")

def plot_heatmap(ax, arr, cmap="viridis"):
    arr = np.asarray(arr, dtype=float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("column")
    ax.set_ylabel("row")

def plot_sorted_line(ax, arr, label, color=None):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = np.sort(arr)[::-1]
    x = np.arange(len(arr))
    ax.plot(x, arr, linewidth=2.2, label=label, color=color)
    ax.set_xlabel("rank")
    ax.set_ylabel("value")

def plot_cumulative(ax, arr, label, color=None):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        return
    c = np.cumsum(arr - np.min(arr))
    denom = np.max(c) if np.max(c) > 0 else 1.0
    c = c / denom
    x = np.arange(len(c))
    ax.plot(x, c, linewidth=2.2, label=label, color=color)
    ax.set_xlabel("index")
    ax.set_ylabel("normalized cumulative value")

def plot_autocorr(ax, arr, label, color=None):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < 3:
        ax.text(0.5, 0.5, "Insufficient length", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = arr - np.mean(arr)
    if np.std(x) < 1e-12:
        ax.text(0.5, 0.5, "Constant series", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] != 0:
        ac = ac / ac[0]
    lags = np.arange(min(len(ac), 60))
    ax.plot(lags, ac[:len(lags)], linewidth=2.2, label=label, color=color)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")

def plot_box(ax, arr, title=None):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        ax.text(0.5, 0.5, "Unsupported data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = [arr[:, i] for i in range(arr.shape[1])]
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.55)
    ax.set_xlabel("group")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)

def plot_row_means(ax, arr, label, color=None):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        return
    y = np.mean(arr, axis=1)
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=2.2, label=label, color=color)
    ax.set_xlabel("row index")
    ax.set_ylabel("row mean")

def plot_col_means(ax, arr, label, color=None):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        return
    y = np.mean(arr, axis=0)
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=2.2, label=label, color=color)
    ax.set_xlabel("column index")
    ax.set_ylabel("column mean")

def safe_correlation_matrix(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    cols = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.isfinite(col).all() and np.std(col) > 1e-12:
            cols.append(col)
    if len(cols) < 2:
        return None
    data = np.vstack(cols)
    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr

def plot_col_corr(ax, arr):
    corr = safe_correlation_matrix(arr)
    if corr is None:
        ax.text(0.5, 0.5, "Insufficient variation", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    plot_heatmap(ax, corr, cmap="coolwarm")
    ax.set_title("Column correlation")
    ax.set_xlabel("column")
    ax.set_ylabel("column")

def plot_data_overview(fig, axes, items):
    rank_counts = {}
    sizes = []
    themes = {}
    for it in items:
        themes[it["theme"]] = themes.get(it["theme"], 0) + 1
        obj = it["obj"]
        arr = None
        if isinstance(obj, dict):
            leaves = flatten_numeric_leaves(obj)
            arrays = [np.asarray(v) for _, v in leaves if np.asarray(v).size > 0]
            if arrays:
                arr = arrays[0]
        else:
            arr = to_array(obj)
        if arr is None:
            continue
        rank_counts[arr.ndim] = rank_counts.get(arr.ndim, 0) + 1
        sizes.append(arr.size)
    ax = axes[0]
    ranks = sorted(rank_counts.items(), key=lambda kv: kv[0])
    if ranks:
        x = [str(k) for k, _ in ranks]
        y = [v for _, v in ranks]
        ax.bar(x, y, color="#4C78A8", alpha=0.9)
        ax.set_xlabel("array rank")
        ax.set_ylabel("count")
    ax.set_title("Result structure by rank")
    style_axes(ax)
    ax = axes[1]
    if sizes:
        ax.hist(np.log10(np.asarray(sizes, dtype=float)), bins=min(20, max(5, int(np.sqrt(len(sizes))))), color="#F58518", alpha=0.88, edgecolor="white")
        ax.set_xlabel("log 10 array size")
        ax.set_ylabel("count")
    ax.set_title("Array size distribution")
    style_axes(ax)
    ax = axes[2]
    theme_items = sorted([(k, v) for k, v in themes.items() if v > 0], key=lambda kv: (-kv[1], kv[0]))[:8]
    if theme_items:
        x = [clean_label(k) for k, _ in theme_items]
        y = [v for _, v in theme_items]
        ax.barh(np.arange(len(x)), y, color="#54A24B", alpha=0.9)
        ax.set_yticks(np.arange(len(x)))
        ax.set_yticklabels(x)
        ax.invert_yaxis()
        ax.set_xlabel("count")
    ax.set_title("Theme coverage")
    style_axes(ax)

def generate_specs_for_item(item):
    path = item["path"]
    obj = item["obj"]
    theme = item["theme"]
    score = item["score"]
    sig = item["sig"]
    base = clean_title(Path(path).stem)
    specs = []

    if isinstance(obj, dict):
        leaves = flatten_numeric_leaves(obj)
        scalars = []
        vectors = []
        matrices = []
        for k, v in leaves:
            arr = np.asarray(v)
            if arr.ndim == 0:
                scalars.append((k, arr.reshape(())))
            elif arr.ndim == 1 and arr.size > 0:
                vectors.append((k, arr))
            elif arr.ndim == 2 and arr.size > 0:
                matrices.append((k, arr))

        if scalars:
            uid = hashlib.md5((sig + "|dict scalars").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "barh", "title": base, "data": {"scalars": scalars}, "uid": uid, "score": score + 4 + len(scalars)})
        if len(vectors) >= 2:
            uid = hashlib.md5((sig + "|dict multiline").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "multiline", "title": base, "data": {"vectors": vectors[:6]}, "uid": uid, "score": score + 5 + len(vectors)})
        for k, v in vectors[:2]:
            uid = hashlib.md5((sig + f"|dict line|{k}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "line", "title": base, "data": {"key": k, "vector": v}, "uid": uid, "score": score + 4 + min(2, v.size // 25)})
            if v.size >= 20:
                for kind in ["sorted", "hist", "cumulative", "autocorr"]:
                    uid = hashlib.md5((sig + f"|dict {kind}|{k}").encode()).hexdigest()
                    specs.append({"theme": theme, "kind": kind, "title": base, "data": {"key": k, "vector": v}, "uid": uid, "score": score + 2 + min(2, v.size // 50)})
        for k, m in matrices[:2]:
            uid = hashlib.md5((sig + f"|dict heatmap|{k}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "heatmap", "title": base, "data": {"key": k, "matrix": m}, "uid": uid, "score": score + 5 + int(max(m.shape) > 20)})
            uid = hashlib.md5((sig + f"|dict box|{k}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "box", "title": base, "data": {"key": k, "matrix": m}, "uid": uid, "score": score + 4})
            uid = hashlib.md5((sig + f"|dict rowcol|{k}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "rowcol", "title": base, "data": {"key": k, "matrix": m}, "uid": uid, "score": score + 4})
            uid = hashlib.md5((sig + f"|dict corr|{k}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "corr", "title": base, "data": {"key": k, "matrix": m}, "uid": uid, "score": score + 4})
            if m.shape[1] <= 8 or m.shape[0] <= 8:
                uid = hashlib.md5((sig + f"|dict profile|{k}").encode()).hexdigest()
                specs.append({"theme": theme, "kind": "profile", "title": base, "data": {"key": k, "matrix": m}, "uid": uid, "score": score + 4})

        if len(vectors) >= 2:
            a_k, a_v = vectors[0]
            b_k, b_v = vectors[1]
            if len(a_v) == len(b_v) and len(a_v) >= 4:
                uid = hashlib.md5((sig + f"|dict scatter|{a_k}|{b_k}").encode()).hexdigest()
                specs.append({"theme": theme, "kind": "scatter", "title": base, "data": {"x_key": a_k, "x": a_v, "y_key": b_k, "y": b_v}, "uid": uid, "score": score + 4})
        return specs

    arr = to_array(obj)
    if arr is None:
        return specs

    if arr.ndim == 0:
        uid = hashlib.md5((sig + "|scalar").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "barh", "title": base, "data": {"scalars": [(Path(path).stem, arr.reshape(()))]}, "uid": uid, "score": score + 4})
    elif arr.ndim == 1:
        uid = hashlib.md5((sig + "|line").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "line", "title": base, "data": {"key": Path(path).stem, "vector": arr}, "uid": uid, "score": score + 4 + min(2, arr.size // 25)})
        if arr.size >= 20:
            for kind in ["sorted", "hist", "cumulative", "autocorr"]:
                uid = hashlib.md5((sig + f"|{kind}").encode()).hexdigest()
                specs.append({"theme": theme, "kind": kind, "title": base, "data": {"key": Path(path).stem, "vector": arr}, "uid": uid, "score": score + 2 + min(2, arr.size // 50)})
    elif arr.ndim == 2:
        uid = hashlib.md5((sig + "|heatmap").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "heatmap", "title": base, "data": {"key": Path(path).stem, "matrix": arr}, "uid": uid, "score": score + 5 + int(max(arr.shape) > 20)})
        uid = hashlib.md5((sig + "|box").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "box", "title": base, "data": {"key": Path(path).stem, "matrix": arr}, "uid": uid, "score": score + 4})
        uid = hashlib.md5((sig + "|rowcol").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "rowcol", "title": base, "data": {"key": Path(path).stem, "matrix": arr}, "uid": uid, "score": score + 4})
        uid = hashlib.md5((sig + "|corr").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "corr", "title": base, "data": {"key": Path(path).stem, "matrix": arr}, "uid": uid, "score": score + 4})
        if arr.shape[1] <= 8 or arr.shape[0] <= 8:
            uid = hashlib.md5((sig + "|profile").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "profile", "title": base, "data": {"key": Path(path).stem, "matrix": arr}, "uid": uid, "score": score + 4})
    else:
        sl_count = min(3, arr.shape[0])
        for i in range(sl_count):
            uid = hashlib.md5((sig + f"|slice|{i}").encode()).hexdigest()
            specs.append({"theme": theme, "kind": "slice", "title": base, "data": {"array": arr, "slice_index": i}, "uid": uid, "score": score + 4 + sl_count})
        uid = hashlib.md5((sig + "|slice summary").encode()).hexdigest()
        specs.append({"theme": theme, "kind": "slice summary", "title": base, "data": {"array": arr}, "uid": uid, "score": score + 5 + sl_count})
    return specs

def render_spec_on_ax(ax, spec, panel_index=0):
    kind = spec["kind"]
    title = spec["title"]
    data = spec["data"]

    if kind == "barh":
        vals = [(k, float(v)) for k, v in data["scalars"] if np.isfinite(float(v))]
        vals = sorted(vals, key=lambda kv: abs(kv[1]), reverse=True)[:8]
        if not vals:
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        y = np.arange(len(vals))
        labels = [relevant_label_from_key(k) for k, _ in vals]
        values = [v for _, v in vals]
        colors = [infer_color(k, i) for i, (k, _) in enumerate(vals)]
        ax.barh(y, values, color=colors, alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("value")
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(v, i, f"  {v:.3g}", va="center", fontsize=10)
        style_axes(ax)
        return

    if kind == "line":
        v = np.asarray(data["vector"], dtype=float).reshape(-1)
        plot_series(ax, v, relevant_label_from_key(data["key"]), color=infer_color(data["key"], panel_index))
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.set_title(title)
        maybe_legend(ax)
        style_axes(ax)
        return

    if kind == "sorted":
        v = np.asarray(data["vector"], dtype=float).reshape(-1)
        plot_sorted_line(ax, v, relevant_label_from_key(data["key"]), color=infer_color(data["key"], panel_index))
        ax.set_title(title)
        maybe_legend(ax)
        style_axes(ax)
        return

    if kind == "hist":
        v = np.asarray(data["vector"], dtype=float).reshape(-1)
        plot_hist(ax, v, color=infer_color(data["key"], panel_index))
        ax.set_title(title)
        style_axes(ax)
        return

    if kind == "cumulative":
        v = np.asarray(data["vector"], dtype=float).reshape(-1)
        plot_cumulative(ax, v, relevant_label_from_key(data["key"]), color=infer_color(data["key"], panel_index))
        ax.set_title(title)
        maybe_legend(ax)
        style_axes(ax)
        return

    if kind == "autocorr":
        v = np.asarray(data["vector"], dtype=float).reshape(-1)
        plot_autocorr(ax, v, relevant_label_from_key(data["key"]), color=infer_color(data["key"], panel_index))
        ax.set_title(title)
        maybe_legend(ax)
        style_axes(ax)
        return

    if kind == "multiline":
        vectors = data["vectors"]
        for i, (k, v) in enumerate(vectors[:6]):
            plot_series(ax, v, relevant_label_from_key(k), color=infer_color(k, i))
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.set_title(title)
        maybe_legend(ax, ncol=2)
        style_axes(ax)
        return

    if kind == "heatmap":
        m = np.asarray(data["matrix"], dtype=float)
        plot_heatmap(ax, m, cmap="viridis")
        ax.set_title(title)
        style_axes(ax)
        return

    if kind == "box":
        m = np.asarray(data["matrix"], dtype=float)
        plot_box(ax, m, title=title)
        style_axes(ax)
        return

    if kind == "rowcol":
        m = np.asarray(data["matrix"], dtype=float)
        plot_row_means(ax, m, relevant_label_from_key(data["key"]), color=infer_color(data["key"], panel_index))
        ax.set_title(title)
        maybe_legend(ax)
        style_axes(ax)
        return

    if kind == "corr":
        m = np.asarray(data["matrix"], dtype=float)
        plot_col_corr(ax, m)
        ax.set_title(title)
        style_axes(ax)
        return

    if kind == "profile":
        m = np.asarray(data["matrix"], dtype=float)
        if m.ndim != 2:
            ax.text(0.5, 0.5, "Unsupported data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        if m.shape[1] <= 8 and m.shape[0] >= m.shape[1]:
            x = np.arange(m.shape[0])
            for j in range(m.shape[1]):
                ax.plot(x, m[:, j], linewidth=2.0, label=f"series {j + 1}")
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            maybe_legend(ax, ncol=2)
        else:
            x = np.arange(m.shape[1])
            for i in range(min(m.shape[0], 6)):
                ax.plot(x, m[i], linewidth=2.0, label=f"row {i + 1}")
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            maybe_legend(ax, ncol=2)
        ax.set_title(title)
        style_axes(ax)
        return

    if kind == "scatter":
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        ax.scatter(x, y, s=18, alpha=0.75, color=infer_color(data["y_key"], panel_index), edgecolors="white", linewidths=0.4)
        ax.set_xlabel(clean_label(data["x_key"]))
        ax.set_ylabel(clean_label(data["y_key"]))
        ax.set_title(title)
        style_axes(ax)
        return

    if kind == "slice":
        arr = np.asarray(data["array"], dtype=float)
        i = int(data["slice_index"])
        if arr.ndim < 3 or i >= arr.shape[0]:
            ax.text(0.5, 0.5, "Unsupported data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        sl = arr[i]
        if sl.ndim == 2:
            plot_heatmap(ax, sl, cmap="viridis")
            ax.set_title(f"{title} slice {i + 1}")
        elif sl.ndim == 1:
            plot_series(ax, sl, f"slice {i + 1}", color=infer_color(title, i))
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            maybe_legend(ax)
            ax.set_title(f"{title} slice {i + 1}")
        else:
            ax.text(0.5, 0.5, "Unsupported slice", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        style_axes(ax)
        return

    if kind == "slice summary":
        arr = np.asarray(data["array"], dtype=float)
        if arr.ndim < 3:
            ax.text(0.5, 0.5, "Unsupported data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        means = [float(np.mean(arr[i])) for i in range(min(3, arr.shape[0]))]
        ax.bar(np.arange(len(means)), means, color="#4C78A8", alpha=0.9)
        ax.set_xlabel("slice")
        ax.set_ylabel("mean value")
        ax.set_title(title)
        style_axes(ax)
        return

    ax.text(0.5, 0.5, "No renderable data", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

def render_multi_panel_figure(specs, filename, title):
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(5.9 * n, 4.8), squeeze=False)
    axes = axes[0]
    for i, spec in enumerate(specs):
        render_spec_on_ax(axes[i], spec, i)
        style_axes(axes[i])
    fig.suptitle(title, y=1.03)
    save_fig(fig, filename)

def select_specs(specs, max_count, kind_preference=None):
    if kind_preference is None:
        kind_preference = []
    kind_rank = {k: i for i, k in enumerate(kind_preference)}
    specs = sorted(specs, key=lambda s: (kind_rank.get(s["kind"], 999), -s["score"], s["uid"]))
    chosen = []
    seen = set()
    for s in specs:
        if s["uid"] in seen:
            continue
        chosen.append(s)
        seen.add(s["uid"])
        if len(chosen) >= max_count:
            break
    return chosen

def build_jobs(specs, scalars):
    jobs = []
    if scalars:
        title = "Key numbers from experiment summaries"
        ht = summary_highlights_text(scalars)
        if ht:
            title = f"{title}\n{ht}"
        jobs.append({"kind": "summary", "filename": "key numbers from summaries.png", "title": title, "specs": []})

    jobs.append({"kind": "overview", "filename": "data overview.png", "title": "Data structure overview", "specs": []})

    theme_spec_order = [
        ("main comparison", ["multiline", "line", "box", "heatmap", "profile", "barh"]),
        ("efficiency", ["line", "sorted", "hist", "cumulative", "autocorr"]),
        ("conflict analysis", ["heatmap", "corr", "rowcol", "profile", "hist", "line"]),
        ("adaptation", ["line", "cumulative", "scatter", "sorted", "hist"]),
        ("ablation", ["heatmap", "line", "hist", "box", "corr"]),
        ("generalization", ["scatter", "line", "cumulative", "hist"]),
        ("benchmark breakdown", ["barh", "multiline", "box", "line", "heatmap"]),
    ]

    spec_by_theme = {}
    for s in specs:
        spec_by_theme.setdefault(s["theme"], []).append(s)

    used = set()
    for theme, pref in theme_spec_order:
        pool = [s for s in spec_by_theme.get(theme, []) if s["uid"] not in used]
        if not pool:
            continue
        chosen = select_specs(pool, 3, kind_preference=pref)
        if chosen:
            for s in chosen:
                used.add(s["uid"])
            title = clean_title(theme)
            jobs.append({"kind": "theme", "filename": f"{theme.replace(' ', '_')}.png", "title": title, "specs": chosen})

    remaining = [s for s in specs if s["uid"] not in used]
    remaining = sorted(remaining, key=lambda s: (-s["score"], s["theme"], s["uid"]))

    supplementary_specs = select_specs(remaining, 3, kind_preference=["multiline", "heatmap", "line", "hist", "box", "cumulative", "corr", "profile", "sorted"])
    if supplementary_specs:
        jobs.append({"kind": "supplementary", "filename": "supplementary results.png", "title": "Supplementary results", "specs": supplementary_specs})

    if len(jobs) > MAX_FIGURES:
        jobs = jobs[:MAX_FIGURES]

    return jobs

def main():
    ensure_fig_dir()

    npy_paths, scalars = load_summary_metadata()
    if not npy_paths:
        npy_paths = fallback_discover_npy()

    items = []
    seen_paths = set()
    for path in npy_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            obj = load_npy(path)
            if obj is None:
                continue
            theme, score = theme_for_item(path, obj)
            sig = object_signature(obj)
            items.append({"path": path, "obj": obj, "theme": theme, "score": score, "sig": sig})
        except Exception as e:
            warnings.warn(f"Could not process {path}: {e}")

    if not items:
        print("[info] No .npy experiment files were found from the provided summaries or on disk.")
        print("[info] No figures were generated.")
        return

    specs = []
    for item in items:
        try:
            specs.extend(generate_specs_for_item(item))
        except Exception as e:
            warnings.warn(f"Spec extraction failed for {item['path']}: {e}")

    dedup_specs = []
    seen = set()
    for s in specs:
        key = (s["uid"], s["kind"], s["theme"])
        if key in seen:
            continue
        seen.add(key)
        dedup_specs.append(s)

    if not dedup_specs:
        print("[info] No renderable numeric structures were found in the experiment files.")
        print("[info] No figures were generated.")
        return

    jobs = build_jobs(dedup_specs, scalars)

    created = 0
    for job in jobs:
        if created >= MAX_FIGURES:
            break
        try:
            if job["kind"] == "overview":
                fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)
                plot_data_overview(fig, axes[0], items)
                fig.suptitle(job["title"], y=1.03)
                save_fig(fig, job["filename"])
                created += 1
                continue

            if job["kind"] == "summary":
                vals = [(k, float(v)) for k, v in scalars if isinstance(v, (int, float)) and np.isfinite(v)]
                if vals:
                    vals = sorted(vals, key=lambda kv: abs(kv[1]), reverse=True)[:8]
                    fig, ax = plt.subplots(1, 1, figsize=(8.8, max(4.8, 0.45 * len(vals) + 2.2)))
                    y = np.arange(len(vals))
                    labels = [relevant_label_from_key(k) for k, _ in vals]
                    values = [v for _, v in vals]
                    colors = [infer_color(k, i) for i, (k, _) in enumerate(vals)]
                    ax.barh(y, values, color=colors, alpha=0.92)
                    ax.set_yticks(y)
                    ax.set_yticklabels(labels)
                    ax.invert_yaxis()
                    ax.set_xlabel("value")
                    ax.set_title(job["title"])
                    for i, v in enumerate(values):
                        ax.text(v, i, f"  {v:.3g}", va="center", fontsize=10)
                    style_axes(ax)
                    save_fig(fig, job["filename"])
                    created += 1
                continue

            if not job["specs"]:
                continue
            render_multi_panel_figure(job["specs"], job["filename"], job["title"])
            created += 1
        except Exception as e:
            warnings.warn(f"Failed to create figure '{job['filename']}': {e}")

    print(f"[info] Generated {min(created, MAX_FIGURES)} figure(s) in '{FIG_DIR}/'.")

if __name__ == "__main__":
    main()