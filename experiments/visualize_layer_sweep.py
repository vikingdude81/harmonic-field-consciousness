"""
Layer Sweep Visualization
=========================
Visualizes how CS-SE correlation, SE magnitude, and category separation
evolve across layers for Mistral-7B and Qwen-7B.
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

CAT_COLORS = {
    "reflective": "#8B5CF6",
    "math": "#3B82F6",
    "factual": "#10B981",
    "creative": "#F59E0B",
    "uncertain": "#EF4444",
}
CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]

DATA_DIR = Path(__file__).parent.parent / "experiment_outputs"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

SWEEP_FILES = {
    "Mistral-7B": DATA_DIR / "layer_sweep_Mistral-7B-Instruct-v0.3_20260206_232751.json",
    "Qwen-7B": DATA_DIR / "layer_sweep_Qwen2.5-7B-Instruct_20260206_231644.json",
}


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE A: CS-SE Correlation Across Layers (both models)
# ═══════════════════════════════════════════════════════════════════════
def fig_correlation_profile(sweeps):
    """CS-SE correlation strength (r) at each layer."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Where Does the Consciousness-Compressibility Link Form?",
                 fontsize=16, fontweight="bold", y=0.98)

    for ax, (name, sweep) in zip([ax1, ax2], sweeps.items()):
        lc = sweep["layer_correlations"]
        layers = [l["layer"] for l in lc]
        depths = [l["depth_pct"] for l in lc]
        rs = [l["cs_se_r"] for l in lc]
        ps = [l["cs_se_p"] for l in lc]

        num_layers = len(layers)

        # Color bars by significance
        colors = []
        for p in ps:
            if p < 0.01:
                colors.append("#10B981")  # green
            elif p < 0.05:
                colors.append("#3B82F6")  # blue
            else:
                colors.append("#D1D5DB")  # gray

        ax.bar(layers, rs, color=colors, edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Highlight the peak zone
        sig_layers = [l for l, p in zip(layers, ps) if p < 0.05]
        if sig_layers:
            ax.axvspan(min(sig_layers) - 0.5, max(sig_layers) + 0.5,
                       alpha=0.1, color="#10B981", label="Significant zone (p<0.05)")

        # Mark the strongest layer
        best_idx = np.argmin(rs)
        ax.annotate(f"Peak: L{layers[best_idx]}\nr={rs[best_idx]:+.3f}",
                    xy=(layers[best_idx], rs[best_idx]),
                    xytext=(layers[best_idx] + 2, rs[best_idx] - 0.1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    fontsize=11, fontweight="bold", color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"))

        # Mark transition point
        r_arr = np.array(rs)
        diffs = np.diff(r_arr)
        steepest = np.argmin(diffs)
        ax.axvline(x=layers[steepest] + 0.5, color="orange", linestyle="--",
                   linewidth=2, alpha=0.7, label=f"Steepest transition: L{layers[steepest]}→L{layers[steepest+1]}")

        ax.set_ylabel("CS-SE Pearson r", fontsize=12)
        ax.set_title(f"{name} ({num_layers} layers)", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(-0.75, 0.55)

    ax2.set_xlabel("Layer", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "fig_layer_correlation_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE B: SE Trajectory by Category Across Layers
# ═══════════════════════════════════════════════════════════════════════
def fig_se_trajectory(sweeps):
    """Mean SE at each layer, colored by category."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spectral Entropy Across Layers by Category",
                 fontsize=14, fontweight="bold")

    for ax, (name, sweep) in zip(axes, sweeps.items()):
        results = sweep["results"]
        num_layers = results[0]["num_layers"]

        for cat in CATEGORIES:
            cat_results = [r for r in results if r["category"] == cat]
            se_by_layer = []
            for layer_idx in range(1, num_layers + 1):
                mean_se = np.mean([r["layers"][str(layer_idx)]["spectral_entropy"]
                                  for r in cat_results])
                se_by_layer.append(mean_se)

            ax.plot(range(1, num_layers + 1), se_by_layer,
                    "o-", color=CAT_COLORS[cat], linewidth=2, markersize=4,
                    label=cat.capitalize(), alpha=0.8)

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Mean Spectral Entropy", fontsize=11)
        ax.set_title(f"{name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig_layer_se_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE C: Category Separation (ANOVA F-stat) Across Layers
# ═══════════════════════════════════════════════════════════════════════
def fig_category_separation(sweeps):
    """ANOVA F-statistic of SE across categories at each layer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Where Do Categories Separate in SE Space?",
                 fontsize=14, fontweight="bold")

    for ax, (name, sweep) in zip(axes, sweeps.items()):
        results = sweep["results"]
        num_layers = results[0]["num_layers"]

        f_stats = []
        p_vals = []
        for layer_idx in range(1, num_layers + 1):
            groups = []
            for cat in CATEGORIES:
                cat_se = [r["layers"][str(layer_idx)]["spectral_entropy"]
                         for r in results if r["category"] == cat]
                groups.append(cat_se)
            try:
                f, p = stats.f_oneway(*groups)
            except Exception:
                f, p = 0, 1
            f_stats.append(f)
            p_vals.append(p)

        colors = ["#10B981" if p < 0.05 else "#D1D5DB" for p in p_vals]
        ax.bar(range(1, num_layers + 1), f_stats, color=colors,
               edgecolor="white", linewidth=0.5, alpha=0.8)

        # Significance line
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("ANOVA F-statistic", fontsize=11)
        ax.set_title(f"{name} (green = p<0.05)", fontsize=12)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = OUT_DIR / "fig_layer_category_separation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE D: Combined Dashboard — The Full Layer Story
# ═══════════════════════════════════════════════════════════════════════
def fig_combined_dashboard(sweeps):
    """4-panel dashboard: correlation, SE mean, SE CoV, category separation."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Layer Sweep Dashboard: Mistral-7B vs Qwen-7B",
                 fontsize=16, fontweight="bold", y=0.98)

    model_colors = {"Mistral-7B": "#EF4444", "Qwen-7B": "#3B82F6"}

    # Row 1: CS-SE correlation and SE mean
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for name, sweep in sweeps.items():
        lc = sweep["layer_correlations"]
        color = model_colors[name]

        # Normalize to depth percentage
        depths = [l["depth_pct"] for l in lc]
        rs = [l["cs_se_r"] for l in lc]
        se_means = [l["se_mean"] for l in lc]
        se_covs = [l["se_cov"] for l in lc]

        ax1.plot(depths, rs, "o-", color=color, linewidth=2.5, markersize=5,
                 label=name, alpha=0.9)

        ax2.plot(depths, se_means, "s-", color=color, linewidth=2, markersize=4,
                 label=name, alpha=0.9)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Depth (%)")
    ax1.set_ylabel("CS-SE Pearson r")
    ax1.set_title("CS-SE Correlation Strength")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Depth (%)")
    ax2.set_ylabel("Mean SE")
    ax2.set_title("Mean Spectral Entropy")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Row 2: SE CoV and ANOVA F-stat
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for name, sweep in sweeps.items():
        lc = sweep["layer_correlations"]
        color = model_colors[name]
        depths = [l["depth_pct"] for l in lc]
        se_covs = [l["se_cov"] for l in lc]

        ax3.plot(depths, se_covs, "^-", color=color, linewidth=2, markersize=4,
                 label=name, alpha=0.9)

        # ANOVA
        results = sweep["results"]
        num_layers = len(lc)
        f_stats = []
        for layer_idx in range(1, num_layers + 1):
            groups = []
            for cat in CATEGORIES:
                cat_se = [r["layers"][str(layer_idx)]["spectral_entropy"]
                         for r in results if r["category"] == cat]
                groups.append(cat_se)
            try:
                f, p = stats.f_oneway(*groups)
            except Exception:
                f = 0
            f_stats.append(f)
        ax4.plot(depths, f_stats, "D-", color=color, linewidth=2, markersize=4,
                 label=name, alpha=0.9)

    ax3.set_xlabel("Depth (%)")
    ax3.set_ylabel("SE Coefficient of Variation")
    ax3.set_title("SE Variation Across Prompts")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("Depth (%)")
    ax4.set_ylabel("ANOVA F-statistic")
    ax4.set_title("Category Separation in SE")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Row 3: SE trajectory by category for each model
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    for ax, (name, sweep) in zip([ax5, ax6], sweeps.items()):
        results = sweep["results"]
        num_layers = results[0]["num_layers"]

        for cat in CATEGORIES:
            cat_results = [r for r in results if r["category"] == cat]
            se_by_layer = []
            for layer_idx in range(1, num_layers + 1):
                mean_se = np.mean([r["layers"][str(layer_idx)]["spectral_entropy"]
                                  for r in cat_results])
                se_by_layer.append(mean_se)
            depths_cat = [(i+1) / num_layers * 100 for i in range(num_layers)]
            ax.plot(depths_cat, se_by_layer, "o-", color=CAT_COLORS[cat],
                    linewidth=2, markersize=3, label=cat.capitalize(), alpha=0.8)

        ax.set_xlabel("Depth (%)")
        ax.set_ylabel("Mean SE")
        ax.set_title(f"{name} — SE by Category")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "fig_layer_sweep_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("LAYER SWEEP VISUALIZATION")
    print("=" * 60)

    sweeps = {}
    for name, path in SWEEP_FILES.items():
        if path.exists():
            sweeps[name] = load_sweep(path)
            n_layers = sweeps[name]["metadata"]["n_layers"]
            print(f"  Loaded {name}: {n_layers} layers")
        else:
            print(f"  Warning: {path} not found")

    if len(sweeps) < 2:
        print("Need both sweep files. Exiting.")
        sys.exit(1)

    print("\nGenerating figures...")
    fig_correlation_profile(sweeps)
    fig_se_trajectory(sweeps)
    fig_category_separation(sweeps)
    fig_combined_dashboard(sweeps)

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done!")
