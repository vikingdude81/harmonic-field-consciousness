"""
Bottleneck Comparison Visualization
====================================
Compares Qwen-7B's extreme compression bottleneck with Mistral-7B's
gradual expansion pattern.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = Path(__file__).parent.parent / "experiment_outputs"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

BOTTLENECK_FILES = {
    "Qwen-7B": DATA_DIR / "bottleneck_Qwen2.5-7B-Instruct_20260206_235049.json",
    "Mistral-7B": DATA_DIR / "bottleneck_Mistral-7B-Instruct-v0.3_20260207_000826.json",
}

MODEL_COLORS = {"Mistral-7B": "#EF4444", "Qwen-7B": "#3B82F6"}
CAT_COLORS = {
    "reflective": "#8B5CF6",
    "math": "#3B82F6",
    "factual": "#10B981",
    "creative": "#F59E0B",
    "uncertain": "#EF4444",
}
CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]


def load_bottleneck(path):
    with open(path) as f:
        return json.load(f)


def fig_main_comparison(data):
    """6-panel comparison dashboard."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Representation Bottleneck: Qwen-7B vs Mistral-7B\n"
                 "How Transformers Organize Information Through Depth",
                 fontsize=16, fontweight="bold", y=0.99)

    # Panel 1: Participation Ratio (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    for name, d in data.items():
        lp = d["layer_profiles"]
        depths = [l["depth_pct"] for l in lp]
        prs = [l["mean_pr"] for l in lp]
        ax1.semilogy(depths, prs, "o-", color=MODEL_COLORS[name], linewidth=2.5,
                     markersize=5, label=name, alpha=0.9)

    ax1.set_xlabel("Depth (%)")
    ax1.set_ylabel("Participation Ratio (log scale)")
    ax1.set_title("Effective Dimensionality")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax1.annotate("PR=1: single dimension", xy=(50, 1.05), fontsize=8, color="gray")

    # Panel 2: Top-1 Eigenvalue Fraction
    ax2 = fig.add_subplot(gs[0, 1])
    for name, d in data.items():
        lp = d["layer_profiles"]
        depths = [l["depth_pct"] for l in lp]
        top1 = [l["mean_top1"] * 100 for l in lp]
        ax2.plot(depths, top1, "s-", color=MODEL_COLORS[name], linewidth=2.5,
                 markersize=5, label=name, alpha=0.9)

    ax2.set_xlabel("Depth (%)")
    ax2.set_ylabel("Top-1 Eigenvalue (%)")
    ax2.set_title("Variance Captured by Dominant Direction")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Dims for 90% variance
    ax3 = fig.add_subplot(gs[1, 0])
    for name, d in data.items():
        lp = d["layer_profiles"]
        depths = [l["depth_pct"] for l in lp]
        d90 = [l["mean_d90"] for l in lp]
        ax3.semilogy(depths, d90, "^-", color=MODEL_COLORS[name], linewidth=2.5,
                     markersize=5, label=name, alpha=0.9)

    ax3.set_xlabel("Depth (%)")
    ax3.set_ylabel("Dimensions for 90% Variance (log)")
    ax3.set_title("Representation Rank")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")

    # Panel 4: Spectral Entropy
    ax4 = fig.add_subplot(gs[1, 1])
    for name, d in data.items():
        lp = d["layer_profiles"]
        depths = [l["depth_pct"] for l in lp]
        se = [l["mean_se"] for l in lp]
        ax4.plot(depths, se, "D-", color=MODEL_COLORS[name], linewidth=2.5,
                 markersize=5, label=name, alpha=0.9)

    ax4.set_xlabel("Depth (%)")
    ax4.set_ylabel("Spectral Entropy")
    ax4.set_title("Spectral Entropy Profile")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Eigenvalue Decay Slope
    ax5 = fig.add_subplot(gs[2, 0])
    for name, d in data.items():
        lp = d["layer_profiles"]
        depths = [l["depth_pct"] for l in lp]
        decay = [l["mean_decay_slope"] for l in lp]
        ax5.plot(depths, decay, "v-", color=MODEL_COLORS[name], linewidth=2.5,
                 markersize=5, label=name, alpha=0.9)

    ax5.set_xlabel("Depth (%)")
    ax5.set_ylabel("Eigenvalue Decay Slope")
    ax5.set_title("How Steeply Eigenvalues Drop Off")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Architecture comparison summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Build summary
    q = data["Qwen-7B"]["layer_profiles"]
    m = data["Mistral-7B"]["layer_profiles"]

    q_min_pr = min(l["mean_pr"] for l in q)
    q_max_pr = max(l["mean_pr"] for l in q)
    m_min_pr = min(l["mean_pr"] for l in m)
    m_max_pr = max(l["mean_pr"] for l in m)

    q_hidden = data["Qwen-7B"]["metadata"]["hidden_dim"]
    m_hidden = data["Mistral-7B"]["metadata"]["hidden_dim"]

    summary_lines = [
        ("Qwen-7B Architecture", "#3B82F6", 14),
        (f"  Hidden dim: {q_hidden}", "#333", 11),
        (f"  Min PR: {q_min_pr:.1f} ({q_min_pr/q_hidden*100:.3f}% of dims)", "#333", 11),
        (f"  Max PR: {q_max_pr:.1f} ({q_max_pr/q_hidden*100:.2f}% of dims)", "#333", 11),
        (f"  Compression: {q_max_pr/q_min_pr:.0f}x", "#333", 11),
        (f"  Pattern: High -> Collapse at L4 -> 1D until L27 -> Expand", "#333", 10),
        ("", "#333", 8),
        ("Mistral-7B Architecture", "#EF4444", 14),
        (f"  Hidden dim: {m_hidden}", "#333", 11),
        (f"  Min PR: {m_min_pr:.1f} ({m_min_pr/m_hidden*100:.3f}% of dims)", "#333", 11),
        (f"  Max PR: {m_max_pr:.1f} ({m_max_pr/m_hidden*100:.2f}% of dims)", "#333", 11),
        (f"  Compression: {m_max_pr/m_min_pr:.0f}x", "#333", 11),
        (f"  Pattern: Low -> Compressed L2-L17 -> Gradual expansion -> Wide L32", "#333", 10),
        ("", "#333", 8),
        ("Key Insight:", "#000", 13),
        ("Both architectures compress to ~1 effective dimension", "#333", 11),
        ("in the middle layers. Qwen does it more abruptly.", "#333", 11),
        ("Mistral expands gradually; Qwen stays flat then decompresses.", "#333", 11),
    ]

    for i, (text, color, size) in enumerate(summary_lines):
        ax6.text(0.05, 0.95 - i * 0.055, text, fontsize=size, color=color,
                 fontweight="bold" if size >= 13 else "normal",
                 transform=ax6.transAxes, va="top", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "fig_bottleneck_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_eigenvalue_spectra(data):
    """Show actual eigenvalue spectra at key layers."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Eigenvalue Spectra at Key Depths\n"
                 "(Top 20 eigenvalues, normalized)",
                 fontsize=14, fontweight="bold")

    for row, (name, d) in enumerate(data.items()):
        results = d["results"]
        num_layers = d["metadata"]["n_layers"]
        key_layers = [1, num_layers // 4, num_layers // 2, num_layers]

        for col, layer_idx in enumerate(key_layers):
            ax = axes[row][col]
            depth_pct = layer_idx / num_layers * 100

            for cat in CATEGORIES:
                cat_results = [r for r in results if r["category"] == cat]
                # Average eigenvalue spectrum across prompts in category
                spectra = []
                for r in cat_results:
                    eigs = r["layers"][str(layer_idx)].get("top20_eigenvalues", [])
                    if eigs:
                        total = sum(eigs)
                        if total > 0:
                            spectra.append([e / total for e in eigs])

                if spectra:
                    mean_spectrum = np.mean(spectra, axis=0)
                    ax.bar(range(len(mean_spectrum)), mean_spectrum,
                           color=CAT_COLORS[cat], alpha=0.4, width=0.8,
                           label=cat.capitalize() if row == 0 and col == 0 else "")

            ax.set_xlabel("Eigenvalue Index")
            if col == 0:
                ax.set_ylabel(f"{name}\nFraction of Variance")
            ax.set_title(f"L{layer_idx} ({depth_pct:.0f}%)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, axis="y")

    axes[0][0].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "fig_eigenvalue_spectra.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_compression_lifecycle(data):
    """Single panel showing the full compression-decompression lifecycle."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, d in data.items():
        lp = d["layer_profiles"]
        hidden_dim = d["metadata"]["hidden_dim"]
        depths = [l["depth_pct"] for l in lp]
        pr_frac = [l["mean_pr"] / hidden_dim * 100 for l in lp]

        ax.plot(depths, pr_frac, "o-", color=MODEL_COLORS[name], linewidth=3,
                markersize=7, label=f"{name} ({hidden_dim}d)", alpha=0.9)

        # Annotate min and max
        min_idx = np.argmin(pr_frac)
        max_idx = np.argmax(pr_frac)
        ax.annotate(f"Min: {pr_frac[min_idx]:.3f}%\n({lp[min_idx]['mean_pr']:.1f} dims)",
                    xy=(depths[min_idx], pr_frac[min_idx]),
                    xytext=(depths[min_idx] + 8, pr_frac[min_idx] + 0.2),
                    arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[name], lw=1.5),
                    fontsize=10, color=MODEL_COLORS[name],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.set_xlabel("Network Depth (%)", fontsize=13)
    ax.set_ylabel("Effective Dimensionality (% of hidden dim)", fontsize=13)
    ax.set_title("The Transformer Information Bottleneck\n"
                 "How many dimensions carry the representation at each layer?",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.05)

    # Add phase annotations
    ax.axvspan(0, 15, alpha=0.05, color="blue", label="_nolegend_")
    ax.axvspan(15, 85, alpha=0.05, color="red", label="_nolegend_")
    ax.axvspan(85, 100, alpha=0.05, color="green", label="_nolegend_")
    ax.text(7, ax.get_ylim()[1] * 0.95, "Encode", fontsize=11, ha="center",
            color="blue", fontstyle="italic", alpha=0.7)
    ax.text(50, ax.get_ylim()[1] * 0.95, "Compress & Compute", fontsize=11, ha="center",
            color="red", fontstyle="italic", alpha=0.7)
    ax.text(92, ax.get_ylim()[1] * 0.95, "Decode", fontsize=11, ha="center",
            color="green", fontstyle="italic", alpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / "fig_compression_lifecycle.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("BOTTLENECK VISUALIZATION")
    print("=" * 60)

    data = {}
    for name, path in BOTTLENECK_FILES.items():
        if path.exists():
            data[name] = load_bottleneck(path)
            print(f"  Loaded {name}: {data[name]['metadata']['n_layers']} layers, "
                  f"{data[name]['metadata']['hidden_dim']}d")
        else:
            print(f"  Warning: {path} not found")

    print("\nGenerating figures...")
    fig_main_comparison(data)
    fig_eigenvalue_spectra(data)
    fig_compression_lifecycle(data)

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done!")
