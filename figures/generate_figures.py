#!/usr/bin/env python3
"""
Figure Generation for:
"A Harmonic Field Model of Consciousness in the Human Brain"

Generates the four canonical figures for the paper:
    fig1_harmonic_modes.png
    fig2_mode_power_states.png
    fig3_consciousness_components.png
    fig4_delta_paradox.png

Run with: python generate_figures.py

Requirements: numpy, matplotlib, networkx
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = Path(__file__).parent

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_toy_graph(n_nodes=24, seed=42):
    """
    Create a 24-node ring graph with additional small-world connections.
    Deterministic and reproducible.
    """
    np.random.seed(seed)

    # Start with a ring lattice
    G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.3, seed=seed)

    # Add weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)

    return G


def compute_laplacian_eigenmodes(G):
    """
    Compute the graph Laplacian L = D - A and its eigenmodes.

    Returns:
        L: Graph Laplacian matrix
        eigenvalues: Sorted eigenvalues
        eigenvectors: Corresponding eigenvectors (columns)
    """
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)

    return L, eigenvalues[idx], eigenvectors[:, idx]


def compute_mode_entropy(power):
    """Compute Shannon entropy of mode power distribution."""
    p = power[power > 0]
    return -np.sum(p * np.log(p))


def compute_participation_ratio(power):
    """Compute participation ratio (effective number of modes)."""
    return 1.0 / np.sum(power ** 2)


# =============================================================================
# FIGURE 1: HARMONIC MODES
# =============================================================================

def figure1_harmonic_modes():
    """
    Figure 1: Harmonic modes on a toy graph.

    Shows:
    - Graph structure
    - Modes psi_1, psi_2, psi_3 (first three nontrivial eigenmodes)
    """
    print("Generating Figure 1: Harmonic modes...")

    G = create_toy_graph(n_nodes=24)
    L, eigenvalues, eigenvectors = compute_laplacian_eigenmodes(G)

    pos = nx.spring_layout(G, seed=42, k=1.5)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Panel A: Graph structure
    ax = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120, node_color='steelblue')
    ax.set_title('(A) Graph $G = (V, E)$')
    ax.axis('off')

    # Panels B-D: Eigenmodes 1, 2, 3
    mode_indices = [1, 2, 3]
    labels = ['B', 'C', 'D']

    for i, (mode_idx, label) in enumerate(zip(mode_indices, labels)):
        ax = axes[i + 1]
        mode = eigenvectors[:, mode_idx]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray')
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=120,
            node_color=mode, cmap='RdBu_r',
            vmin=-np.max(np.abs(mode)),
            vmax=np.max(np.abs(mode))
        )
        ax.set_title(f'({label}) $\\psi_{{{mode_idx}}}$, $\\lambda_{{{mode_idx}}}={eigenvalues[mode_idx]:.2f}$')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_harmonic_modes.png')
    plt.close()
    print("  Saved: fig1_harmonic_modes.png")


# =============================================================================
# FIGURE 2: MODE POWER PATTERNS FOR FOUR STATES
# =============================================================================

def figure2_mode_power_states():
    """
    Figure 2: Synthetic mode power patterns for four brain states.

    States:
    - Wake: broad distribution
    - NREM unconscious: concentrated in low modes
    - NREM dreaming: delta present but broader
    - Anesthesia: very concentrated in lowest modes
    """
    print("Generating Figure 2: Mode power states...")

    np.random.seed(42)
    n_modes = 20
    k = np.arange(n_modes)

    # Define power distributions
    wake = 0.3 + 0.4 * np.exp(-k / 8) + 0.15 * np.random.rand(n_modes)
    wake /= wake.sum()

    nrem_unconscious = np.exp(-k / 2) + 0.03 * np.random.rand(n_modes)
    nrem_unconscious /= nrem_unconscious.sum()

    nrem_dreaming = 0.35 * np.exp(-k / 3) + 0.25 * np.exp(-(k - 5)**2 / 10)
    nrem_dreaming += 0.08 * np.random.rand(n_modes)
    nrem_dreaming /= nrem_dreaming.sum()

    anesthesia = np.exp(-k / 1.5) + 0.02 * np.random.rand(n_modes)
    anesthesia /= anesthesia.sum()

    states = [
        ('(A) Wake', wake, 'forestgreen'),
        ('(B) NREM Unconscious', nrem_unconscious, 'navy'),
        ('(C) NREM Dreaming', nrem_dreaming, 'purple'),
        ('(D) Anesthesia', anesthesia, 'darkred'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, (title, power, color) in zip(axes.flat, states):
        ax.bar(k, power, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Mode index $k$')
        ax.set_ylabel('Normalized power $p_k$')
        ax.set_title(title)
        ax.set_xlim(-0.5, n_modes - 0.5)
        ax.set_ylim(0, max(power) * 1.15)

        H = compute_mode_entropy(power)
        PR = compute_participation_ratio(power)
        ax.text(0.95, 0.95, f'$H_{{mode}} = {H:.2f}$\n$PR = {PR:.1f}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_mode_power_states.png')
    plt.close()
    print("  Saved: fig2_mode_power_states.png")


# =============================================================================
# FIGURE 3: CONSCIOUSNESS FUNCTIONAL COMPONENTS
# =============================================================================

def figure3_consciousness_components():
    """
    Figure 3: Components of the consciousness functional C(t).

    Shows normalized scores for H_mode, PR, R, Sdot, kappa
    across the four states.
    """
    print("Generating Figure 3: Consciousness components...")

    states = ['Wake', 'NREM\nUnconscious', 'NREM\nDreaming', 'Anesthesia']
    colors = ['forestgreen', 'navy', 'purple', 'darkred']

    # Synthetic normalized values (0-1)
    H_mode = np.array([0.85, 0.25, 0.55, 0.20])
    PR = np.array([0.80, 0.20, 0.50, 0.15])
    R = np.array([0.50, 0.80, 0.45, 0.85])
    Sdot = np.array([0.90, 0.30, 0.60, 0.25])
    kappa = np.array([0.85, 0.35, 0.65, 0.30])

    # Compute C(t) with equal weights
    C = 0.2 * (H_mode + PR + R + Sdot + kappa)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    x = np.arange(len(states))

    components = [
        ('(A) Mode Entropy $H_{mode}/H_{max}$', H_mode),
        ('(B) Participation Ratio $PR/PR_{max}$', PR),
        ('(C) Phase Coherence $R$', R),
        ('(D) Entropy Production $\\dot{S}/\\dot{S}_{max}$', Sdot),
        ('(E) Criticality Index $\\kappa$', kappa),
    ]

    for ax, (title, values) in zip(axes.flat[:5], components):
        bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Normalized Value')
        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel F: Combined C(t)
    ax = axes[1, 2]
    bars = ax.bar(x, C, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('$C(t)$')
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=9)
    ax.set_title('(F) Consciousness Functional $C(t)$')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, C):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_consciousness_components.png')
    plt.close()
    print("  Saved: fig3_consciousness_components.png")


# =============================================================================
# FIGURE 4: DELTA PARADOX SCHEMATIC
# =============================================================================

def figure4_delta_paradox():
    """
    Figure 4: The delta paradox.

    Compares NREM "no experience" vs NREM "dreaming":
    - Both have similar low-index mode power (delta)
    - Very different global metric profiles
    """
    print("Generating Figure 4: Delta paradox...")

    np.random.seed(42)
    n_modes = 20
    k = np.arange(n_modes)

    # State A: NREM unconscious (concentrated)
    power_A = np.exp(-k / 1.8) + 0.02 * np.random.rand(n_modes)
    power_A /= power_A.sum()

    # State B: NREM dreaming (broader despite delta)
    power_B = 0.35 * np.exp(-k / 2.5) + 0.25 * np.exp(-(k - 6)**2 / 12)
    power_B += 0.08 * np.random.rand(n_modes)
    power_B /= power_B.sum()

    # Compute metrics
    H_A = compute_mode_entropy(power_A)
    H_B = compute_mode_entropy(power_B)
    PR_A = compute_participation_ratio(power_A)
    PR_B = compute_participation_ratio(power_B)

    H_max = np.log(n_modes)
    PR_max = n_modes

    # Additional synthetic metrics
    R_A, R_B = 0.82, 0.48
    Sdot_A, Sdot_B = 0.28, 0.58
    kappa_A, kappa_B = 0.32, 0.68

    metrics_A = [H_A/H_max, PR_A/PR_max, R_A, Sdot_A, kappa_A]
    metrics_B = [H_B/H_max, PR_B/PR_max, R_B, Sdot_B, kappa_B]

    C_A = 0.2 * sum(metrics_A)
    C_B = 0.2 * sum(metrics_B)

    # Delta power (sum of first 3 modes)
    delta_A = power_A[:3].sum()
    delta_B = power_B[:3].sum()

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    # Row 1: Mode power distributions
    ax = axes[0, 0]
    ax.bar(k, power_A, color='navy', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mode index $k$')
    ax.set_ylabel('Power $p_k$')
    ax.set_title('(A) NREM: No Experience')
    ax.set_xlim(-0.5, n_modes - 0.5)

    ax = axes[0, 1]
    ax.bar(k, power_B, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mode index $k$')
    ax.set_ylabel('Power $p_k$')
    ax.set_title('(B) NREM: Dreaming')
    ax.set_xlim(-0.5, n_modes - 0.5)

    # Row 1, Col 3: Delta power comparison
    ax = axes[0, 2]
    bars = ax.bar(['No Experience', 'Dreaming'], [delta_A, delta_B],
                  color=['navy', 'purple'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Low-Index Mode Power')
    ax.set_title('(C) Delta Power (Similar!)')
    ax.set_ylim(0, 0.8)
    for bar, val in zip(bars, [delta_A, delta_B]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Row 2: Functional component comparison
    ax = axes[1, 0]
    metric_names = ['$H$', '$PR$', '$R$', '$\\dot{S}$', '$\\kappa$']
    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width/2, metrics_A, width, label='No Experience', color='navy', alpha=0.7)
    ax.bar(x + width/2, metrics_B, width, label='Dreaming', color='purple', alpha=0.7)
    ax.set_ylabel('Normalized Value')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title('(D) Functional Components')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.1)

    # Row 2, Col 2: C(t) comparison
    ax = axes[1, 1]
    bars = ax.bar(['No Experience', 'Dreaming'], [C_A, C_B],
                  color=['navy', 'purple'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('$C(t)$')
    ax.set_title('(E) Consciousness Functional\n(Very Different!)')
    ax.set_ylim(0, 0.8)
    for bar, val in zip(bars, [C_A, C_B]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Row 2, Col 3: Summary text
    ax = axes[1, 2]
    ax.text(0.5, 0.7, 'Delta Paradox Resolved:', fontsize=12, ha='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.5, 0.5, 'Both states have high delta power,\nbut very different $C(t)$.',
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Consciousness depends on\n'
            'mode distribution, not band power.',
            fontsize=10, ha='center', transform=ax.transAxes, style='italic')
    ax.axis('off')
    ax.set_title('(F) Conclusion')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_delta_paradox.png')
    plt.close()
    print("  Saved: fig4_delta_paradox.png")


# =============================================================================
# FIGURE 5: OSCILLATORY GATING AS MODE-LEVEL ROUTING
# =============================================================================

def figure5_oscillatory_gating():
    """
    Figure 5: Oscillatory gating as mode-level routing.

    Shows:
    - Mode phases and alignment/misalignment
    - Phase-dependent gating of mode combinations
    - Different phase configurations producing different readout patterns
    """
    print("Generating Figure 5: Oscillatory gating...")

    np.random.seed(42)
    n_modes = 8
    t = np.linspace(0, 4 * np.pi, 200)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Panel A: Phase-aligned modes (high R, rigid gating)
    ax = axes[0, 0]
    phases_aligned = np.zeros(n_modes)  # All phases aligned
    for k in range(4):
        signal = np.cos(t * (k + 1) + phases_aligned[k])
        ax.plot(t / np.pi, signal * (0.8 - k * 0.15), alpha=0.7,
                label=f'$\\psi_{k+1}$')
    ax.set_xlabel('Time ($\\pi$ units)')
    ax.set_ylabel('Mode amplitude')
    ax.set_title('(A) High Coherence ($R \\approx 1$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 4)

    # Panel B: Phase-diverse modes (intermediate R, flexible gating)
    ax = axes[0, 1]
    phases_diverse = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3])
    for k in range(4):
        signal = np.cos(t * (k + 1) + phases_diverse[k])
        ax.plot(t / np.pi, signal * (0.8 - k * 0.15), alpha=0.7,
                label=f'$\\psi_{k+1}$')
    ax.set_xlabel('Time ($\\pi$ units)')
    ax.set_ylabel('Mode amplitude')
    ax.set_title('(B) Intermediate Coherence ($R \\sim 0.5$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 4)

    # Panel C: Phase coherence regimes
    ax = axes[0, 2]
    R_values = np.linspace(0, 1, 100)
    # Inverted U for flexibility/gating capacity
    gating_capacity = 4 * R_values * (1 - R_values)
    ax.plot(R_values, gating_capacity, 'k-', linewidth=2)
    ax.fill_between(R_values, gating_capacity, alpha=0.3, color='steelblue')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Optimal zone')
    ax.set_xlabel('Phase Coherence $R$')
    ax.set_ylabel('Gating Flexibility')
    ax.set_title('(C) Gating Capacity vs Coherence')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    # Panel D: Mode coupling matrix (mixing rules)
    ax = axes[1, 0]
    coupling_matrix = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            # Stronger coupling for nearby modes
            coupling_matrix[i, j] = np.exp(-abs(i - j) / 2) * np.random.uniform(0.5, 1.0)
    np.fill_diagonal(coupling_matrix, 0)
    im = ax.imshow(coupling_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Mode $j$')
    ax.set_ylabel('Mode $k$')
    ax.set_title('(D) Coupling Strength $\\nu_{jk}$')
    ax.set_xticks(range(n_modes))
    ax.set_yticks(range(n_modes))
    ax.set_xticklabels([f'$\\psi_{k+1}$' for k in range(n_modes)], fontsize=8)
    ax.set_yticklabels([f'$\\psi_{k+1}$' for k in range(n_modes)], fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel E: Effective dimensionality under different gating
    ax = axes[1, 1]
    states = ['Low $R$\n(Fragmented)', 'Mid $R$\n(Flexible)', 'High $R$\n(Locked)']
    dimensionality = [3.5, 12.0, 2.0]
    colors = ['coral', 'forestgreen', 'steelblue']
    bars = ax.bar(states, dimensionality, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('(E) Representational Capacity')
    ax.set_ylim(0, 15)
    for bar, val in zip(bars, dimensionality):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # Panel F: Summary schematic
    ax = axes[1, 2]
    ax.text(0.5, 0.85, 'Oscillatory Gating Framework:', fontsize=11, ha='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.5, 0.68, '$\\psi_k$ = spatial carrier pattern', fontsize=10, ha='center',
            transform=ax.transAxes, family='serif')
    ax.text(0.5, 0.53, '$\\theta_k(t)$ = phase determines routing', fontsize=10, ha='center',
            transform=ax.transAxes, family='serif')
    ax.text(0.5, 0.38, '$U(a_1, ..., a_N)$ = mixing rules', fontsize=10, ha='center',
            transform=ax.transAxes, family='serif')
    ax.text(0.5, 0.18, 'Phase alignment $\\rightarrow$ variable binding\n'
            'Phase diversity $\\rightarrow$ flexible gating',
            fontsize=9, ha='center', transform=ax.transAxes, style='italic')
    ax.axis('off')
    ax.set_title('(F) Key Principles')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_oscillatory_gating.png')
    plt.close()
    print("  Saved: fig5_oscillatory_gating.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all five canonical figures."""
    print("=" * 60)
    print("Generating Figures for Harmonic Field Model Paper")
    print("=" * 60)

    figure1_harmonic_modes()
    figure2_mode_power_states()
    figure3_consciousness_components()
    figure4_delta_paradox()
    figure5_oscillatory_gating()

    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
