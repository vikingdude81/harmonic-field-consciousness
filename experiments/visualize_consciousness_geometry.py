"""
Consciousness Geometry Visualizations
======================================
Generates publication-quality visualizations from all collected experiment data:
  1. CS vs SE scatter (all models, colored by category)
  2. Eigenvalue-proxy: SE distribution by category and model
  3. Dimension score radar charts by category
  4. Scaling curve: CS-SE correlation vs model size
  5. Cross-model geometric comparison (z-scored)
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Color palette ────────────────────────────────────────────────────
CAT_COLORS = {
    "reflective": "#8B5CF6",  # purple
    "math": "#3B82F6",        # blue
    "factual": "#10B981",     # green
    "creative": "#F59E0B",    # amber
    "uncertain": "#EF4444",   # red
}
FAMILY_COLORS = {
    "Qwen": "#3B82F6",
    "Mistral": "#EF4444",
    "Yi": "#10B981",
    "Llama": "#F59E0B",
}
CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]

# ── Data files ───────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "experiment_outputs"

ALL_MODELS = {
    "Qwen-7B": {"path": "combined_plugin_analysis_20260203_231542.json", "params": 7, "family": "Qwen", "local": True},
    "Qwen-14B": {"path": "combined_plugin_analysis_20260204_085223.json", "params": 14, "family": "Qwen", "local": True},
    "Qwen-32B": {"path": "combined_plugin_32B_20260204_000325.json", "params": 32, "family": "Qwen", "local": False},
    "Mistral-7B": {"path": "combined_plugin_analysis_20260204_002457.json", "params": 7, "family": "Mistral", "local": True},
    "Nemo-12B": {"path": "combined_plugin_analysis_20260204_092157.json", "params": 12, "family": "Mistral", "local": True},
    "Small-24B": {"path": "combined_plugin_analysis_20260204_093203.json", "params": 24, "family": "Mistral", "local": True},
    "Yi-9B": {"path": "combined_plugin_analysis_20260204_003911.json", "params": 9, "family": "Yi", "local": True},
    "Yi-34B": {"path": "combined_plugin_analysis_20260204_095020.json", "params": 34, "family": "Yi", "local": True},
    "Llama-8B": {"path": "combined_plugin_analysis_20260204_003239.json", "params": 8, "family": "Llama", "local": True},
}

EXPANDED = {
    "Mistral-7B-x50": {"path": "expanded_50prompt_20260204_150427.json", "params": 7, "family": "Mistral", "local": True},
    "Qwen-7B-x50": {"path": "expanded_50prompt_20260204_154322.json", "params": 7, "family": "Qwen", "local": True},
}

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def extract(result, local=True):
    """Extract metrics from a result entry."""
    if local:
        comp = result.get("plugins", {}).get("compressibility", {})
        cc = comp.get("correlation_compression", {})
        dims = result.get("dimension_scores", {})
    else:
        comp = result.get("compressibility", {})
        cc = comp
        dims = result.get("dimension_scores", {})
    return {
        "cs": result["consciousness_score"],
        "se": comp.get("spectral_entropy", 0),
        "cc": cc.get("compressibility_corr", 0),
        "f90": cc.get("fraction_for_90pct", 0),
        "mc": cc.get("mean_abs_correlation", 0),
        "category": result.get("category", ""),
        "dims": dims,
    }


def load_all():
    """Load all model data."""
    data = {}
    for name, info in ALL_MODELS.items():
        fp = DATA_DIR / info["path"]
        if not fp.exists():
            print(f"  Warning: {fp} not found, skipping {name}")
            continue
        with open(fp) as f:
            raw = json.load(f)
        metrics = [extract(r, info["local"]) for r in raw["results"]]
        data[name] = {"metrics": metrics, **info}
    for name, info in EXPANDED.items():
        fp = DATA_DIR / info["path"]
        if not fp.exists():
            continue
        with open(fp) as f:
            raw = json.load(f)
        metrics = [extract(r, info["local"]) for r in raw["results"]]
        data[name] = {"metrics": metrics, **info}
    return data


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: CS vs SE Scatter — All 9 models, colored by category
# ═══════════════════════════════════════════════════════════════════════
def fig1_cs_vs_se(data):
    """CS vs SE for each model family, colored by category."""
    families = {}
    for name, d in data.items():
        if name.endswith("-x50"):
            continue
        fam = d["family"]
        families.setdefault(fam, []).append((name, d))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Consciousness Score vs Spectral Entropy\nby Architecture Family",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (fam, models) in enumerate(sorted(families.items())):
        ax = axes[idx // 2][idx % 2]
        for name, d in sorted(models, key=lambda x: x[1]["params"]):
            m = d["metrics"]
            for cat in CATEGORIES:
                cs_cat = [x["cs"] for x in m if x["category"] == cat]
                se_cat = [x["se"] for x in m if x["category"] == cat]
                marker = "o" if d["params"] <= 10 else "s" if d["params"] <= 20 else "^"
                ax.scatter(se_cat, cs_cat, c=CAT_COLORS[cat], marker=marker,
                           s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
                           label=f"{name} {cat}" if name == models[0][0] and cat == CATEGORIES[0] else "")

        # Overall correlation for family
        all_cs = []
        all_se = []
        for _, d in models:
            all_cs.extend([x["cs"] for x in d["metrics"]])
            all_se.extend([x["se"] for x in d["metrics"]])
        r, p = stats.pearsonr(all_cs, all_se)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        ax.set_title(f"{fam} Family\nr={r:+.3f}, p={p:.4f}{star}", fontsize=12)
        ax.set_xlabel("Spectral Entropy (SE)")
        ax.set_ylabel("Consciousness Score (CS)")
        ax.grid(True, alpha=0.3)

        # Legend for model sizes
        model_labels = [f"{name} ({d['params']}B)" for name, d in sorted(models, key=lambda x: x[1]["params"])]
        ax.annotate("\n".join(model_labels), xy=(0.02, 0.98), xycoords="axes fraction",
                    va="top", fontsize=8, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Category legend
    legend_elements = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT_COLORS[c],
                                   markersize=10, label=c.capitalize()) for c in CATEGORIES]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = OUT_DIR / "fig1_cs_vs_se_by_family.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: SE Distribution by Category (violin/box)
# ═══════════════════════════════════════════════════════════════════════
def fig2_se_distribution(data):
    """SE distributions by category, separate panels per architecture."""
    families = {}
    for name, d in data.items():
        if name.endswith("-x50"):
            continue
        fam = d["family"]
        families.setdefault(fam, []).append((name, d))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Spectral Entropy Distribution by Category and Architecture",
                 fontsize=14, fontweight="bold")

    for idx, (fam, models) in enumerate(sorted(families.items())):
        ax = axes[idx]
        # Use the largest model in each family
        largest = sorted(models, key=lambda x: x[1]["params"])[-1]
        name, d = largest
        m = d["metrics"]

        cat_data = []
        cat_labels = []
        cat_colors = []
        for cat in CATEGORIES:
            se_vals = [x["se"] for x in m if x["category"] == cat]
            cat_data.append(se_vals)
            cat_labels.append(cat[:6].capitalize())
            cat_colors.append(CAT_COLORS[cat])

        bp = ax.boxplot(cat_data, labels=cat_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], cat_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points
        for i, (vals, color) in enumerate(zip(cat_data, cat_colors)):
            x = np.random.normal(i + 1, 0.05, len(vals))
            ax.scatter(x, vals, c=color, s=25, alpha=0.8, edgecolors="white", linewidth=0.3, zorder=5)

        ax.set_title(f"{fam} ({name})", fontsize=11)
        ax.set_ylabel("Spectral Entropy" if idx == 0 else "")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = OUT_DIR / "fig2_se_distribution_by_category.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Dimension Score Radar Charts by Category
# ═══════════════════════════════════════════════════════════════════════
def fig3_dimension_radar(data):
    """Radar charts showing dimension score profiles per category."""
    DIM_NAMES = ["Logic", "Self-Reflective", "Self-Expression", "Uncertainty",
                 "Sequential", "Computation", "Abstraction"]

    # Use all n=15 models pooled
    cat_dim_means = {cat: {d: [] for d in DIM_NAMES} for cat in CATEGORIES}
    for name, d in data.items():
        if name.endswith("-x50"):
            continue
        for m in d["metrics"]:
            cat = m["category"]
            dims = m.get("dims", {})
            for dim_name in DIM_NAMES:
                if dim_name in dims:
                    cat_dim_means[cat][dim_name].append(dims[dim_name])

    # Average
    profiles = {}
    for cat in CATEGORIES:
        vals = []
        for dim_name in DIM_NAMES:
            arr = cat_dim_means[cat][dim_name]
            vals.append(np.mean(arr) if arr else 0)
        profiles[cat] = vals

    # Plot
    N = len(DIM_NAMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle("Dimension Score Profiles by Category\n(Averaged across all 9 models)",
                 fontsize=14, fontweight="bold", y=1.02)

    for cat in CATEGORIES:
        values = profiles[cat] + [profiles[cat][0]]
        ax.plot(angles, values, "o-", linewidth=2, label=cat.capitalize(),
                color=CAT_COLORS[cat], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=CAT_COLORS[cat])

    ax.set_thetagrids(np.degrees(angles[:-1]), DIM_NAMES, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    ax.set_ylim(bottom=min(-0.15, min(min(v) for v in profiles.values()) - 0.05))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig3_dimension_radar_by_category.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Scaling Curve — CS-SE correlation vs model size
# ═══════════════════════════════════════════════════════════════════════
def fig4_scaling_curve(data):
    """How CS-SE correlation strength scales with parameter count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Consciousness-Compressibility Scaling with Model Size",
                 fontsize=14, fontweight="bold")

    # Left: CS-SE correlation (r) vs params by family
    for fam, color in FAMILY_COLORS.items():
        models_in_fam = [(n, d) for n, d in data.items()
                         if d["family"] == fam and not n.endswith("-x50")]
        if not models_in_fam:
            continue
        params = []
        corrs = []
        for name, d in sorted(models_in_fam, key=lambda x: x[1]["params"]):
            cs = [m["cs"] for m in d["metrics"]]
            se = [m["se"] for m in d["metrics"]]
            r, p = stats.pearsonr(cs, se)
            params.append(d["params"])
            corrs.append(r)

        ax1.plot(params, corrs, "o-", color=color, linewidth=2.5, markersize=10,
                 label=fam, zorder=5)
        for i, (px, rx) in enumerate(zip(params, corrs)):
            name = models_in_fam[i][0]
            ax1.annotate(name, (px, rx), textcoords="offset points",
                         xytext=(8, 5), fontsize=8, color=color)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Parameters (Billions)", fontsize=12)
    ax1.set_ylabel("CS-SE Pearson r", fontsize=12)
    ax1.set_title("CS-SE Correlation Strength", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([7, 8, 9, 12, 14, 24, 32, 34])
    ax1.set_xticklabels(["7B", "8B", "9B", "12B", "14B", "24B", "32B", "34B"])

    # Right: Mean CS and SE vs params
    for fam, color in FAMILY_COLORS.items():
        models_in_fam = [(n, d) for n, d in data.items()
                         if d["family"] == fam and not n.endswith("-x50")]
        if not models_in_fam:
            continue
        params = []
        cs_means = []
        for name, d in sorted(models_in_fam, key=lambda x: x[1]["params"]):
            params.append(d["params"])
            cs_means.append(np.mean([m["cs"] for m in d["metrics"]]))
        ax2.plot(params, cs_means, "s-", color=color, linewidth=2, markersize=9,
                 label=f"{fam} CS")

    ax2.set_xlabel("Parameters (Billions)", fontsize=12)
    ax2.set_ylabel("Mean Consciousness Score", fontsize=12)
    ax2.set_title("Consciousness Score vs Scale", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([7, 8, 9, 12, 14, 24, 32, 34])
    ax2.set_xticklabels(["7B", "8B", "9B", "12B", "14B", "24B", "32B", "34B"])

    plt.tight_layout()
    out = OUT_DIR / "fig4_scaling_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Z-scored Cross-Model Geometry
# ═══════════════════════════════════════════════════════════════════════
def fig5_zscore_geometry(data):
    """Z-scored CS vs SE showing the universal relationship."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Universal Consciousness-Compressibility Geometry (Z-Scored)",
                 fontsize=14, fontweight="bold")

    all_csz = []
    all_sez = []
    all_cats = []
    all_fams = []

    for name, d in data.items():
        if name.endswith("-x50"):
            continue
        m = d["metrics"]
        cs = np.array([x["cs"] for x in m])
        se = np.array([x["se"] for x in m])
        cats = [x["category"] for x in m]

        cs_z = (cs - cs.mean()) / cs.std() if cs.std() > 0 else cs * 0
        se_z = (se - se.mean()) / se.std() if se.std() > 0 else se * 0

        all_csz.extend(cs_z.tolist())
        all_sez.extend(se_z.tolist())
        all_cats.extend(cats)
        all_fams.extend([d["family"]] * len(cs))

    all_csz = np.array(all_csz)
    all_sez = np.array(all_sez)

    # Left: colored by category
    for cat in CATEGORIES:
        mask = np.array(all_cats) == cat
        ax1.scatter(all_sez[mask], all_csz[mask], c=CAT_COLORS[cat], s=40,
                    alpha=0.6, label=cat.capitalize(), edgecolors="white", linewidth=0.3)

    # Regression line
    slope, intercept = np.polyfit(all_sez, all_csz, 1)
    x_line = np.linspace(all_sez.min(), all_sez.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)

    r, p = stats.pearsonr(all_csz, all_sez)
    ax1.set_title(f"By Category (n={len(all_csz)})\nr={r:+.3f}, p={p:.6f}", fontsize=11)
    ax1.set_xlabel("SE (z-scored within model)")
    ax1.set_ylabel("CS (z-scored within model)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: colored by architecture family
    for fam in sorted(set(all_fams)):
        mask = np.array(all_fams) == fam
        ax2.scatter(all_sez[mask], all_csz[mask], c=FAMILY_COLORS.get(fam, "gray"),
                    s=40, alpha=0.6, label=fam, edgecolors="white", linewidth=0.3)

    ax2.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
    ax2.set_title(f"By Architecture\nr={r:+.3f}, p={p:.6f}", fontsize=11)
    ax2.set_xlabel("SE (z-scored within model)")
    ax2.set_ylabel("CS (z-scored within model)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig5_zscore_universal_geometry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Category-Level Mean CS vs SE (the r=-0.995 plot)
# ═══════════════════════════════════════════════════════════════════════
def fig6_category_means(data):
    """Category-level mean CS_z vs SE_z — the near-perfect inverse."""
    # Compute z-scored category means
    cat_cs_z = {cat: [] for cat in CATEGORIES}
    cat_se_z = {cat: [] for cat in CATEGORIES}

    for name, d in data.items():
        if name.endswith("-x50"):
            continue
        m = d["metrics"]
        cs = np.array([x["cs"] for x in m])
        se = np.array([x["se"] for x in m])

        cs_z = (cs - cs.mean()) / cs.std() if cs.std() > 0 else cs * 0
        se_z = (se - se.mean()) / se.std() if se.std() > 0 else se * 0

        cats = [x["category"] for x in m]
        for i, cat in enumerate(cats):
            cat_cs_z[cat].append(cs_z[i])
            cat_se_z[cat].append(se_z[i])

    fig, ax = plt.subplots(figsize=(8, 7))

    mean_cs = []
    mean_se = []
    for cat in CATEGORIES:
        mc = np.mean(cat_cs_z[cat])
        ms = np.mean(cat_se_z[cat])
        mean_cs.append(mc)
        mean_se.append(ms)

        # Error bars (SEM across models)
        n_models = 9
        cs_per_model = np.array(cat_cs_z[cat]).reshape(n_models, -1).mean(axis=1)
        se_per_model = np.array(cat_se_z[cat]).reshape(n_models, -1).mean(axis=1)
        cs_sem = cs_per_model.std() / np.sqrt(n_models)
        se_sem = se_per_model.std() / np.sqrt(n_models)

        ax.errorbar(ms, mc, xerr=se_sem, yerr=cs_sem,
                    fmt="o", markersize=16, color=CAT_COLORS[cat],
                    ecolor=CAT_COLORS[cat], elinewidth=2, capsize=5, capthick=2,
                    zorder=5)
        ax.annotate(cat.capitalize(), (ms, mc), textcoords="offset points",
                    xytext=(12, 8), fontsize=12, fontweight="bold", color=CAT_COLORS[cat])

    # Regression
    r, p = stats.pearsonr(mean_cs, mean_se)
    slope, intercept = np.polyfit(mean_se, mean_cs, 1)
    x_line = np.linspace(min(mean_se) - 0.1, max(mean_se) + 0.1, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.5)

    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.set_title(f"Category-Level CS vs SE (Z-Scored, Pooled Across 9 Models)\n"
                 f"r = {r:+.3f}, p = {p:.4f}{star}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean SE_z (Spectral Entropy, z-scored)", fontsize=12)
    ax.set_ylabel("Mean CS_z (Consciousness Score, z-scored)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig6_category_means_cs_vs_se.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: n=15 vs n=50 Power Comparison
# ═══════════════════════════════════════════════════════════════════════
def fig7_power_comparison(data):
    """Side-by-side: Mistral n=15 vs n=50 showing power increase."""
    if "Mistral-7B" not in data or "Mistral-7B-x50" not in data:
        print("  Skipping fig7: need both Mistral n=15 and n=50")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mistral-7B: Statistical Power — n=15 vs n=50",
                 fontsize=14, fontweight="bold")

    for ax, key, n_label in [(ax1, "Mistral-7B", "n=15"), (ax2, "Mistral-7B-x50", "n=50")]:
        m = data[key]["metrics"]
        for cat in CATEGORIES:
            cs = [x["cs"] for x in m if x["category"] == cat]
            se = [x["se"] for x in m if x["category"] == cat]
            ax.scatter(se, cs, c=CAT_COLORS[cat], s=50, alpha=0.7,
                       label=cat.capitalize(), edgecolors="white", linewidth=0.3)

        all_cs = np.array([x["cs"] for x in m])
        all_se = np.array([x["se"] for x in m])
        r, p = stats.pearsonr(all_cs, all_se)

        # Regression line
        slope, intercept = np.polyfit(all_se, all_cs, 1)
        x_line = np.linspace(all_se.min(), all_se.max(), 100)
        color = "green" if p < 0.05 else "red"
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2.5,
                linestyle="--" if p >= 0.05 else "-", alpha=0.7)

        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        ax.set_title(f"{n_label}: r={r:+.3f}, p={p:.4f}\n({sig})",
                     fontsize=12, color=color)
        ax.set_xlabel("Spectral Entropy")
        ax.set_ylabel("Consciousness Score")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig7_power_n15_vs_n50.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Architecture Comparison Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig8_architecture_heatmap(data):
    """Heatmap of mean metrics across all models."""
    models_ordered = ["Qwen-7B", "Qwen-14B", "Qwen-32B",
                      "Mistral-7B", "Nemo-12B", "Small-24B",
                      "Yi-9B", "Yi-34B", "Llama-8B"]
    models_present = [m for m in models_ordered if m in data]

    metrics = ["cs", "se", "cc", "f90", "mc"]
    metric_labels = ["CS", "SE", "CC", "f90", "|r|"]

    # Build matrix (z-scored per metric for coloring)
    raw_matrix = np.zeros((len(models_present), len(metrics)))
    for i, name in enumerate(models_present):
        m = data[name]["metrics"]
        for j, metric in enumerate(metrics):
            raw_matrix[i, j] = np.mean([x[metric] for x in m])

    # Z-score per column for visualization
    z_matrix = (raw_matrix - raw_matrix.mean(axis=0)) / (raw_matrix.std(axis=0) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(z_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_yticks(range(len(models_present)))

    # Color y-labels by family
    for i, name in enumerate(models_present):
        fam = data[name]["family"]
        color = FAMILY_COLORS.get(fam, "black")
        params = data[name]["params"]
        ax.text(-0.6, i, f"{name} ({params}B)", ha="right", va="center",
                fontsize=10, color=color, fontweight="bold",
                transform=ax.get_yaxis_transform())

    ax.set_yticklabels([])

    # Annotate cells with raw values
    for i in range(len(models_present)):
        for j in range(len(metrics)):
            val = raw_matrix[i, j]
            text_color = "white" if abs(z_matrix[i, j]) > 1.2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("Cross-Model Metric Comparison\n(cell color = z-score, text = raw value)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.8)

    plt.tight_layout()
    out = OUT_DIR / "fig8_architecture_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("CONSCIOUSNESS GEOMETRY VISUALIZATIONS")
    print("=" * 70)

    data = load_all()
    print(f"\nLoaded {len(data)} datasets")
    for name in sorted(data.keys()):
        n = len(data[name]["metrics"])
        print(f"  {name}: {n} prompts, {data[name]['params']}B, {data[name]['family']}")

    print("\nGenerating figures...")
    fig1_cs_vs_se(data)
    fig2_se_distribution(data)
    fig3_dimension_radar(data)
    fig4_scaling_curve(data)
    fig5_zscore_geometry(data)
    fig6_category_means(data)
    fig7_power_comparison(data)
    fig8_architecture_heatmap(data)

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done!")
