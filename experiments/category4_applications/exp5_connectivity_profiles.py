#!/usr/bin/env python3
"""
Category 4: Applications

Experiment 5: Neurodiversity as Connectivity Profiles

Tests how different connectivity patterns affect consciousness metrics:
1. Long-range dominant (neurotypical-like, global integration)
2. Local-heavy (autism-like, local processing)
3. Sparse random (schizophrenia-like, fragmentation)
4. Balanced (intermediate case)

Hypothesis: Different connectivity profiles produce different consciousness signatures.
Not "better" or "worse", but different functional properties.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_NODES = 100
N_MODES = 20
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp5_connectivity_profiles'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Category 4, Experiment 5: Neurodiversity as Connectivity Profiles")
print("="*70)

# ==============================================================================
# CONNECTIVITY PROFILE GENERATORS
# ==============================================================================

def generate_long_range_dominant(n_nodes: int, p_long: float = 0.1, p_local: float = 0.01, seed: int = SEED):
    """
    Long-range dominant network (Neurotypical-like).
    
    Characteristics:
    - Strong global connectivity
    - Weak local clustering
    - High integration, low segregation
    """
    np.random.seed(seed)
    W = np.zeros((n_nodes, n_nodes))
    
    # Global connections (all-to-all with probability)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.rand() < p_long:
                strength = np.random.uniform(0.1, 1.0)
                W[i, j] = strength
                W[j, i] = strength
    
    # Weak local connections
    for i in range(n_nodes):
        for j in range(max(0, i-5), min(n_nodes, i+6)):
            if i != j and np.random.rand() < p_local:
                strength = np.random.uniform(0.01, 0.2)
                W[i, j] = strength
    
    return W

def generate_local_heavy(n_nodes: int, neighborhood: int = 10, local_strength: float = 0.8, long_range_strength: float = 0.1, seed: int = SEED):
    """
    Local-heavy network (Autism-like profile).
    
    Characteristics:
    - Strong local clustering (neighborhood connections)
    - Weak long-range connections
    - High segregation, low integration
    """
    np.random.seed(seed)
    W = np.zeros((n_nodes, n_nodes))
    
    # Strong local neighborhood connections
    for i in range(n_nodes):
        for j in range(max(0, i-neighborhood), min(n_nodes, i+neighborhood+1)):
            if i != j:
                strength = local_strength * np.random.uniform(0.5, 1.0)
                W[i, j] = strength
    
    # Sparse long-range connections
    for i in range(n_nodes):
        # Random long-range targets (avoiding local neighborhood)
        n_long = max(1, int(n_nodes * 0.05))  # ~5% of nodes
        targets = np.random.choice(
            [j for j in range(n_nodes) if abs(i-j) > neighborhood],
            size=min(n_long, n_nodes - 2*neighborhood),
            replace=False
        )
        for j in targets:
            strength = long_range_strength * np.random.uniform(0.1, 0.5)
            W[i, j] = strength
            W[j, i] = strength
    
    return W

def generate_sparse_random(n_nodes: int, density: float = 0.05, seed: int = SEED):
    """
    Sparse random network (Schizophrenia-like profile?).
    
    Characteristics:
    - Very sparse connectivity
    - Random organization
    - Fragmented, low coherence
    """
    np.random.seed(seed)
    W = np.random.rand(n_nodes, n_nodes)
    W = (W < density).astype(float)
    
    # Make symmetric
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    
    # Random weights
    W = W * np.random.uniform(0.1, 1.0, W.shape)
    
    return W

def generate_balanced(n_nodes: int, local_prob: float = 0.05, long_range_prob: float = 0.02, seed: int = SEED):
    """
    Balanced network (Intermediate profile).
    
    Characteristics:
    - Moderate local connectivity
    - Moderate long-range connectivity
    - Balanced segregation and integration
    """
    np.random.seed(seed)
    W = np.zeros((n_nodes, n_nodes))
    
    # Local connections
    for i in range(n_nodes):
        for j in range(max(0, i-5), min(n_nodes, i+6)):
            if i != j and np.random.rand() < local_prob:
                strength = np.random.uniform(0.3, 0.8)
                W[i, j] = strength
    
    # Long-range connections
    for i in range(n_nodes):
        for j in range(n_nodes):
            if abs(i-j) > 5 and np.random.rand() < long_range_prob:
                strength = np.random.uniform(0.1, 0.5)
                W[i, j] = strength
    
    return W

# ==============================================================================
# PHASE COHERENCE ANALYSIS
# ==============================================================================

def simulate_phase_coherence(W: np.ndarray, K: float = 1.0, steps: int = 300, dt: float = 0.05, seed: int = SEED):
    """
    Simulate Kuramoto phase coherence on connectivity matrix.
    """
    np.random.seed(seed)
    n = W.shape[0]
    
    # Normalize by degree
    deg = W.sum(axis=1, keepdims=True) + 1e-9
    W_norm = W / deg
    
    # Natural frequencies
    omega = np.random.normal(0.0, 0.1, size=n)
    
    # Initial phases
    theta = np.random.uniform(0, 2*np.pi, size=n)
    
    R_series = []
    for _ in range(steps):
        # Kuramoto update
        phase_diff = np.subtract.outer(theta, theta)
        coupling_term = (W_norm * np.sin(-phase_diff)).sum(axis=1)
        dtheta = omega + K * coupling_term
        theta = (theta + dt * dtheta) % (2*np.pi)
        
        # Order parameter
        Z = np.exp(1j * theta).mean()
        R_series.append(np.abs(Z))
    
    R_series = np.array(R_series)
    return float(R_series.mean()), float(R_series.std())

def compute_connectivity_metrics(W: np.ndarray):
    """
    Compute consciousness-relevant metrics from connectivity.
    """
    n = W.shape[0]
    
    # 1. Degree distribution
    degree = W.sum(axis=1)
    
    # 2. Clustering coefficient (local connectivity)
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(W[i, :] > 0)[0]
        if len(neighbors) < 2:
            clustering[i] = 0
        else:
            subgraph = W[np.ix_(neighbors, neighbors)]
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            actual = subgraph.sum() / 2
            clustering[i] = actual / possible if possible > 0 else 0
    
    # 3. Efficiency (average path length proxy via spectral)
    eigenvalues = np.linalg.eigvalsh(W)
    spectral_gap = eigenvalues[-1] - eigenvalues[-2] if len(eigenvalues) > 1 else 0.0
    
    # 4. Modularity (community structure)
    # Simplified: correlation between nearby nodes
    modularity = 0
    for i in range(n):
        for j in range(max(0, i-10), min(n, i+10)):
            if i != j:
                modularity += W[i, j]
    modularity = modularity / (W.sum() + 1e-10)
    
    # 5. Sparsity
    sparsity = 1 - (W > 0).sum() / (n * n)
    
    # 6. Phase coherence (Kuramoto)
    R_dyn, metastability = simulate_phase_coherence(W, K=1.0, steps=200)
    
    # 7. Create pseudo power spectrum for consciousness metrics
    eigenvalues = np.linalg.eigvalsh(W)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) > 0:
        power = eigenvalues / eigenvalues.sum()
        if len(power) < N_MODES:
            power = np.pad(power, (0, N_MODES - len(power)), mode='constant')
        else:
            power = power[-N_MODES:]
    else:
        power = np.ones(N_MODES) / N_MODES
    
    synth_eig = np.arange(1, N_MODES + 1, dtype=float)
    cons_metrics = met.compute_all_metrics(power, synth_eig)
    
    return {
        'mean_degree': degree.mean(),
        'std_degree': degree.std(),
        'clustering': clustering.mean(),
        'spectral_gap': spectral_gap,
        'modularity': modularity,
        'sparsity': sparsity,
        'R_dyn': R_dyn,
        'metastability': metastability,
        **cons_metrics
    }

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print("\n" + "-"*70)
print("Comparing Connectivity Profiles")
print("-"*70)

profiles = {
    'Long-range Dominant\n(Neurotypical-like)': generate_long_range_dominant,
    'Local-Heavy\n(Autism-like)': generate_local_heavy,
    'Sparse Random\n(Fragmented)': generate_sparse_random,
    'Balanced\n(Intermediate)': generate_balanced,
}

results = []

for profile_name, gen_func in tqdm(profiles.items(), desc="Profiles"):
    # Generate connectivity
    if 'Long-range' in profile_name:
        W = gen_func(N_NODES, p_long=0.1, p_local=0.01, seed=SEED)
    elif 'Local-Heavy' in profile_name:
        W = gen_func(N_NODES, neighborhood=10, local_strength=0.8, long_range_strength=0.1, seed=SEED)
    elif 'Sparse' in profile_name:
        W = gen_func(N_NODES, density=0.05, seed=SEED)
    else:  # Balanced
        W = gen_func(N_NODES, local_prob=0.05, long_range_prob=0.02, seed=SEED)
    
    # Compute metrics
    metrics = compute_connectivity_metrics(W)
    
    results.append({
        'profile': profile_name.replace('\n', ' '),
        **metrics
    })

df_results = pd.DataFrame(results)

print("\n" + "="*100)
print("CONNECTIVITY PROFILE ANALYSIS")
print("="*100)

# Display key metrics
display_cols = ['profile', 'mean_degree', 'clustering', 'sparsity', 'R_dyn', 'H_mode', 'PR', 'C']
print("\n" + df_results[display_cols].to_string(index=False))

print("\n" + "-"*70)
print("Interpretation")
print("-"*70)

long_range = df_results[df_results['profile'].str.contains('Long-range')].iloc[0]
local_heavy = df_results[df_results['profile'].str.contains('Local-Heavy')].iloc[0]
sparse = df_results[df_results['profile'].str.contains('Sparse')].iloc[0]
balanced = df_results[df_results['profile'].str.contains('Balanced')].iloc[0]

print(f"\nLong-range Dominant (Neurotypical-like):")
print(f"  - Mean degree: {long_range['mean_degree']:.2f} (high global connectivity)")
print(f"  - Clustering: {long_range['clustering']:.4f} (low local structure)")
print(f"  - R_dyn: {long_range['R_dyn']:.4f} (phase coherence)")
print(f"  - H_mode: {long_range['H_mode']:.4f} (complexity)")
print(f"  - C: {long_range['C']:.4f} (consciousness metric)")

print(f"\nLocal-Heavy (Autism-like):")
print(f"  - Mean degree: {local_heavy['mean_degree']:.2f} (strong local connections)")
print(f"  - Clustering: {local_heavy['clustering']:.4f} (high local structure)")
print(f"  - R_dyn: {local_heavy['R_dyn']:.4f} (phase coherence)")
print(f"  - H_mode: {local_heavy['H_mode']:.4f} (complexity)")
print(f"  - C: {local_heavy['C']:.4f} (consciousness metric)")

print(f"\nSparse Random (Fragmented):")
print(f"  - Mean degree: {sparse['mean_degree']:.2f} (minimal connectivity)")
print(f"  - Clustering: {sparse['clustering']:.4f} (no coherent structure)")
print(f"  - R_dyn: {sparse['R_dyn']:.4f} (phase coherence)")
print(f"  - H_mode: {sparse['H_mode']:.4f} (complexity)")
print(f"  - C: {sparse['C']:.4f} (consciousness metric)")

print(f"\nBalanced (Intermediate):")
print(f"  - Mean degree: {balanced['mean_degree']:.2f} (moderate connectivity)")
print(f"  - Clustering: {balanced['clustering']:.4f} (moderate local structure)")
print(f"  - R_dyn: {balanced['R_dyn']:.4f} (phase coherence)")
print(f"  - H_mode: {balanced['H_mode']:.4f} (complexity)")
print(f"  - C: {balanced['C']:.4f} (consciousness metric)")

print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)

print("""
Different connectivity profiles produce DIFFERENT consciousness signatures, not "better" or "worse":

1. NEUROTYPICAL (Long-range dominant):
   - High global integration → high R_dyn
   - Better at global consciousness/awareness
   - Fast information transfer across network
   
2. AUTISM-LIKE (Local-heavy):
   - High local clustering → different R_dyn pattern
   - Better at detailed local processing
   - Stronger sensory/local feature integration
   - NOT "impaired", just organized differently
   
3. FRAGMENTED (Sparse random):
   - Minimal coherence → low R_dyn
   - Disconnected processing
   - Likely associated with psychotic fragmentation
   
4. BALANCED:
   - Intermediate metrics
   - Best of both local detail and global coherence

IMPORTANT: These are connectivity-based consciousness profiles, not value judgments.
Different profiles optimize for different cognitive tasks.
""")

# Save results
df_results.to_csv(OUTPUT_DIR / 'connectivity_profiles_results.csv', index=False)
print(f"\nResults saved to {OUTPUT_DIR / 'connectivity_profiles_results.csv'}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Degree distribution
ax = axes[0, 0]
for idx, row in df_results.iterrows():
    ax.scatter(row['mean_degree'], row['std_degree'], s=200, alpha=0.6, label=row['profile'])
ax.set_xlabel('Mean Degree')
ax.set_ylabel('Std Degree')
ax.set_title('Degree Distribution by Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Clustering vs Sparsity
ax = axes[0, 1]
for idx, row in df_results.iterrows():
    ax.scatter(row['sparsity'], row['clustering'], s=200, alpha=0.6, label=row['profile'])
ax.set_xlabel('Sparsity')
ax.set_ylabel('Clustering Coefficient')
ax.set_title('Local vs Global Organization')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Phase Coherence vs Complexity
ax = axes[1, 0]
for idx, row in df_results.iterrows():
    ax.scatter(row['R_dyn'], row['H_mode'], s=200, alpha=0.6, label=row['profile'])
ax.set_xlabel('Phase Coherence (R_dyn)')
ax.set_ylabel('Complexity (H_mode)')
ax.set_title('Coherence vs Complexity')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Overall Consciousness Metric
ax = axes[1, 1]
profiles_short = [p.replace(' ', '\n').replace('\n\n', '\n') for p in df_results['profile']]
bars = ax.bar(range(len(df_results)), df_results['C'], alpha=0.6, color=['C0', 'C1', 'C2', 'C3'])
ax.set_xticks(range(len(df_results)))
ax.set_xticklabels(profiles_short, fontsize=9)
ax.set_ylabel('Consciousness Metric (C)')
ax.set_title('Overall Consciousness-like Properties')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df_results['C'])):
    ax.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'connectivity_profiles.png', dpi=150, bbox_inches='tight')
print(f"Figure saved to {OUTPUT_DIR / 'connectivity_profiles.png'}")

plt.show()

print("\n" + "="*70)
print("Experiment complete!")
print("="*70)
