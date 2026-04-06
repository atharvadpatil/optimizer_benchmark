"""Generate all figures and tables from results/."""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# ─── Constants ─────────────────────────────────────────────────────────────────

ALL_OPTIMIZERS = ["sgd", "adamw", "lion", "ademamix", "soap", "muon", "schedulefree_adamw"]

OPTIMIZER_COLORS = {
    "sgd": "#1f77b4",
    "adamw": "#ff7f0e",
    "lion": "#2ca02c",
    "ademamix": "#d62728",
    "soap": "#9467bd",
    "muon": "#8c564b",
    "schedulefree_adamw": "#e377c2",
}

OPTIMIZER_LABELS = {
    "sgd": "SGD+M",
    "adamw": "AdamW",
    "lion": "Lion",
    "ademamix": "AdEMAMix",
    "soap": "SOAP",
    "muon": "Muon",
    "schedulefree_adamw": "SF-AdamW",
}


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_results(results_dir, prefix=None):
    """Load all JSON result files. Optionally filter by filename prefix."""
    results = []
    if not os.path.exists(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or "_FAILED" in fname:
            continue
        if prefix and not fname.startswith(prefix):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        data["_filename"] = fname
        results.append(data)
    return results


def group_by_optimizer(results):
    """Group results by optimizer name."""
    groups = defaultdict(list)
    for r in results:
        groups[r["optimizer"]].append(r)
    return dict(groups)


def find_best_run(runs, task):
    """Find the best run from a list (max accuracy for cifar10, min perplexity for ptb)."""
    if task == "cifar10":
        return max(runs, key=lambda r: r.get("final_test_accuracy", 0))
    else:
        return min(runs, key=lambda r: r.get("final_val_perplexity", float("inf")))


# ─── Table 1: Final Metrics ───────────────────────────────────────────────────

def generate_table1(results_dir, figures_dir):
    """Table 1: Final test accuracy (CIFAR-10) / perplexity (PTB) — best, mean, std."""
    print("\n" + "=" * 80)
    print("TABLE 1: Final Metrics Summary")
    print("=" * 80)

    for task in ["cifar10", "ptb"]:
        results = load_results(results_dir, prefix=f"{task}_")
        # Filter out diagnostic results
        results = [r for r in results if "diagnostic" not in r]
        if not results:
            print(f"\n  No results for {task}")
            continue

        groups = group_by_optimizer(results)
        metric_name = "Test Accuracy (%)" if task == "cifar10" else "Val Perplexity"
        metric_key = "final_test_accuracy" if task == "cifar10" else "final_val_perplexity"

        print(f"\n  Task: {task.upper()}")
        print(f"  {'Optimizer':<20s} {'Best':>10s} {'Mean':>10s} {'Std':>10s} {'Wall-time (s)':>14s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

        rows = []
        for opt_name in ALL_OPTIMIZERS:
            if opt_name not in groups:
                continue
            runs = groups[opt_name]
            metrics = [r[metric_key] for r in runs if metric_key in r]
            times = [r.get("total_wall_time", 0) for r in runs]
            if not metrics:
                continue

            best_val = max(metrics) if task == "cifar10" else min(metrics)
            mean_val = np.mean(metrics)
            std_val = np.std(metrics)
            avg_time = np.mean(times)

            print(f"  {OPTIMIZER_LABELS.get(opt_name, opt_name):<20s} "
                  f"{best_val:>10.2f} {mean_val:>10.2f} {std_val:>10.2f} {avg_time:>14.1f}")
            rows.append((opt_name, best_val, mean_val, std_val, avg_time))

        # Save as JSON for later use
        table_path = os.path.join(figures_dir, f"table1_{task}.json")
        with open(table_path, "w") as f:
            json.dump(rows, f, indent=2)


# ─── Figure 1: Convergence Curves ─────────────────────────────────────────────

def generate_figure1(results_dir, figures_dir):
    """Figure 1: Convergence curves (best run per optimizer, both tasks side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task, ylabel, metric_key, history_key) in enumerate([
        ("cifar10", "Test Accuracy (%)", "final_test_accuracy", "test_accuracy"),
        ("ptb", "Val Perplexity", "final_val_perplexity", "val_perplexity"),
    ]):
        ax = axes[idx]
        results = load_results(results_dir, prefix=f"{task}_")
        results = [r for r in results if "diagnostic" not in r]
        groups = group_by_optimizer(results)

        for opt_name in ALL_OPTIMIZERS:
            if opt_name not in groups:
                continue
            best_run = find_best_run(groups[opt_name], task)
            history = best_run.get("history", [])
            if not history:
                continue

            if task == "cifar10":
                x = [h["epoch"] for h in history]
                y = [h.get(history_key, 0) for h in history]
            else:
                x = [h["step"] for h in history]
                y = [h.get(history_key) for h in history]
                y = [v for v in y if v is not None]
                x = x[:len(y)]

            color = OPTIMIZER_COLORS.get(opt_name, None)
            label = OPTIMIZER_LABELS.get(opt_name, opt_name)
            ax.plot(x, y, label=label, color=color, linewidth=1.5)

        ax.set_xlabel("Epoch" if task == "cifar10" else "Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'CIFAR-10' if task == 'cifar10' else 'Penn Treebank'}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "figure1_convergence.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─── Figure 2: Pareto Frontier ────────────────────────────────────────────────

def generate_figure2(results_dir, figures_dir):
    """Figure 2: Pareto frontier — test metric vs total GPU-seconds scatterplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task, ylabel, metric_key) in enumerate([
        ("cifar10", "Test Accuracy (%)", "final_test_accuracy"),
        ("ptb", "Val Perplexity", "final_val_perplexity"),
    ]):
        ax = axes[idx]
        results = load_results(results_dir, prefix=f"{task}_")
        results = [r for r in results if "diagnostic" not in r]

        for r in results:
            opt_name = r["optimizer"]
            metric = r.get(metric_key)
            wall_time = r.get("total_wall_time")
            if metric is None or wall_time is None:
                continue

            color = OPTIMIZER_COLORS.get(opt_name, "gray")
            label = OPTIMIZER_LABELS.get(opt_name, opt_name)
            ax.scatter(wall_time, metric, c=color, label=label, s=50, alpha=0.7, edgecolors="white")

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        ax.set_xlabel("Wall-clock Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'CIFAR-10' if task == 'cifar10' else 'Penn Treebank'} — Pareto Frontier")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "figure2_pareto.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─── Figure 3: HP Sensitivity Heatmaps ────────────────────────────────────────

def generate_figure3(results_dir, figures_dir):
    """Figure 3: Heatmaps of test metric across LR x WD grid for each optimizer."""
    for task, metric_key in [("cifar10", "final_test_accuracy"), ("ptb", "final_val_perplexity")]:
        results = load_results(results_dir, prefix=f"{task}_")
        results = [r for r in results if "diagnostic" not in r]
        groups = group_by_optimizer(results)

        active_opts = [o for o in ALL_OPTIMIZERS if o in groups]
        if not active_opts:
            continue

        n_cols = min(4, len(active_opts))
        n_rows = (len(active_opts) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, opt_name in enumerate(active_opts):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            runs = groups[opt_name]

            lrs = sorted(set(r["lr"] for r in runs))
            wds = sorted(set(r["weight_decay"] for r in runs))

            grid = np.full((len(lrs), len(wds)), np.nan)
            for r in runs:
                li = lrs.index(r["lr"])
                wi = wds.index(r["weight_decay"])
                grid[li, wi] = r.get(metric_key, np.nan)

            sns.heatmap(
                grid, ax=ax, annot=True, fmt=".1f",
                xticklabels=[f"{w:.0e}" for w in wds],
                yticklabels=[f"{l:.0e}" for l in lrs],
                cmap="YlOrRd" if task == "cifar10" else "YlOrRd_r",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_xlabel("Weight Decay")
            ax.set_ylabel("Learning Rate")
            ax.set_title(OPTIMIZER_LABELS.get(opt_name, opt_name))

        # Hide empty axes
        for i in range(len(active_opts), n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        metric_label = "Test Accuracy" if task == "cifar10" else "Val Perplexity"
        fig.suptitle(f"HP Sensitivity — {task.upper()} ({metric_label})", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(figures_dir, f"figure3_heatmaps_{task}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


# ─── Figure 4: Training Horizon Curves ────────────────────────────────────────

def generate_figure4(results_dir, figures_dir):
    """Figure 4: Training horizon curves from diagnostic 1 (RQ2)."""
    results = load_results(results_dir, prefix="diag_horizon_")
    if not results:
        print("  [SKIP] No horizon diagnostic results found")
        return

    groups = defaultdict(list)
    for r in results:
        groups[r["optimizer"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))

    for opt_name in ["ademamix", "adamw", "sgd"]:
        if opt_name not in groups:
            continue
        runs = sorted(groups[opt_name], key=lambda r: r["fraction"])
        fracs = [r["fraction"] for r in runs]
        accs = [r["final_test_accuracy"] for r in runs]

        color = OPTIMIZER_COLORS.get(opt_name, None)
        label = OPTIMIZER_LABELS.get(opt_name, opt_name)
        ax.plot(fracs, accs, marker="o", label=label, color=color, linewidth=2)

    ax.set_xlabel("Fraction of Training Budget")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Training Horizon Sensitivity (RQ2) — CIFAR-10")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "figure4_horizon.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─── Table 2: Practitioner Recommendation ─────────────────────────────────────

def generate_table2(results_dir, figures_dir):
    """Table 2: Practitioner optimizer selection guide."""
    print("\n" + "=" * 80)
    print("TABLE 2: Practitioner Recommendation")
    print("=" * 80)

    rows = []

    for task in ["cifar10", "ptb"]:
        results = load_results(results_dir, prefix=f"{task}_")
        results = [r for r in results if "diagnostic" not in r]
        groups = group_by_optimizer(results)

        if not groups:
            continue

        metric_key = "final_test_accuracy" if task == "cifar10" else "final_val_perplexity"

        print(f"\n  Task: {task.upper()}")
        print(f"  {'Optimizer':<20s} {'Best Metric':>12s} {'SSE':>10s} {'Recommendation':<30s}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*30}")

        for opt_name in ALL_OPTIMIZERS:
            if opt_name not in groups:
                continue
            best_run = find_best_run(groups[opt_name], task)
            metric = best_run.get(metric_key)
            wall_time = best_run.get("total_wall_time", 1)

            if metric is None:
                continue

            # SSE = accuracy / GPU-seconds (cifar10) or (1/perplexity) / GPU-seconds (ptb)
            if task == "cifar10":
                sse = metric / wall_time
            else:
                sse = (1.0 / metric) / wall_time

            # Simple recommendation logic
            if task == "cifar10":
                if sse == max(
                    (find_best_run(groups[o], task).get(metric_key, 0) /
                     max(find_best_run(groups[o], task).get("total_wall_time", 1), 1))
                    for o in groups
                ):
                    rec = "Best SSE"
                elif metric == max(
                    find_best_run(groups[o], task).get(metric_key, 0) for o in groups
                ):
                    rec = "Best accuracy"
                else:
                    rec = ""
            else:
                if metric == min(
                    find_best_run(groups[o], task).get(metric_key, float("inf")) for o in groups
                ):
                    rec = "Best perplexity"
                else:
                    rec = ""

            print(f"  {OPTIMIZER_LABELS.get(opt_name, opt_name):<20s} "
                  f"{metric:>12.2f} {sse:>10.4f} {rec:<30s}")
            rows.append({
                "task": task, "optimizer": opt_name,
                "best_metric": metric, "sse": sse, "recommendation": rec,
            })

    table_path = os.path.join(figures_dir, "table2_recommendations.json")
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)


# ─── SSE Summary ──────────────────────────────────────────────────────────────

def generate_sse_summary(results_dir, figures_dir):
    """Compute and print SSE metric for all optimizers."""
    print("\n" + "=" * 80)
    print("SSE METRIC SUMMARY")
    print("=" * 80)

    for task in ["cifar10", "ptb"]:
        results = load_results(results_dir, prefix=f"{task}_")
        results = [r for r in results if "diagnostic" not in r]
        groups = group_by_optimizer(results)

        if not groups:
            continue

        metric_key = "final_test_accuracy" if task == "cifar10" else "final_val_perplexity"
        print(f"\n  Task: {task.upper()}  (SSE = {'Accuracy / GPU-seconds' if task == 'cifar10' else '(1/Perplexity) / GPU-seconds'})")
        print(f"  {'Optimizer':<20s} {'Best Metric':>12s} {'Time (s)':>10s} {'SSE':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")

        sse_list = []
        for opt_name in ALL_OPTIMIZERS:
            if opt_name not in groups:
                continue
            best_run = find_best_run(groups[opt_name], task)
            metric = best_run.get(metric_key)
            wall_time = best_run.get("total_wall_time", 1)
            if metric is None:
                continue

            if task == "cifar10":
                sse = metric / wall_time
            else:
                sse = (1.0 / metric) / wall_time

            print(f"  {OPTIMIZER_LABELS.get(opt_name, opt_name):<20s} "
                  f"{metric:>12.2f} {wall_time:>10.1f} {sse:>10.4f}")
            sse_list.append({"optimizer": opt_name, "metric": metric,
                             "wall_time": wall_time, "sse": sse})

        sse_path = os.path.join(figures_dir, f"sse_{task}.json")
        with open(sse_path, "w") as f:
            json.dump(sse_list, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all figures and tables from results/")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Use non-interactive backend
    matplotlib.use("Agg")

    print("Generating analysis from", args.results_dir)

    generate_table1(args.results_dir, args.figures_dir)
    generate_figure1(args.results_dir, args.figures_dir)
    generate_figure2(args.results_dir, args.figures_dir)
    generate_figure3(args.results_dir, args.figures_dir)
    generate_figure4(args.results_dir, args.figures_dir)
    generate_table2(args.results_dir, args.figures_dir)
    generate_sse_summary(args.results_dir, args.figures_dir)

    print("\nDone! Check", args.figures_dir)


if __name__ == "__main__":
    main()
