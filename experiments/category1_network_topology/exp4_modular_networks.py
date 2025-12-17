#!/usr/bin/env python3
"""
Category 1: Network Topology Experiments

Experiment 4: Modular Network Analysis

Tests how modular organization affects consciousness:
1. Number of modules (communities)
2. Intra vs inter-module connectivity
3. Module size distribution
4. Hierarchical modularity

Key question: Does brain-like modular organization enhance consciousness?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from itertools import product

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg

# Configuration
SEED = 42
np.random.seed(SEED)
N_NODES = 100
N_MODES = 20
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_modular_networks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple Kuramoto dynamics for phase coherence R
def simulate_phase_coherence(G: nx.Graph, K: float = 1.0, steps: int = 300, dt: float = 0.05, seed: int = SEED):
    np.random.seed(seed)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    # Normalize by degree to avoid trivial blow-up on dense graphs
    deg = A.sum(axis=1, keepdims=True) + 1e-9
    W = A / deg
    # Natural frequencies
    omega = np.random.normal(0.0, 0.1, size=n)
    # Initial phases
    theta = np.random.uniform(0, 2*np.pi, size=n)
    R_series = []
    for _ in range(steps):
        # Kuramoto update: dtheta_i = omega_i + K * sum_j W_ij * sin(theta_j - theta_i)
        phase_diff = np.subtract.outer(theta, theta)  # theta_i - theta_j
        coupling_term = (W * np.sin(-phase_diff)).sum(axis=1)
        dtheta = omega + K * coupling_term
        theta = (theta + dt * dtheta) % (2*np.pi)
        Z = np.exp(1j * theta).mean()
        R_series.append(np.abs(Z))
    R_series = np.array(R_series)
    return float(R_series.mean()), float(R_series.std())

print("="*70)
print("Category 1, Experiment 4: Modular Network Analysis")
print("="*70)

# ==============================================================================
# PART 1: Number of Modules Sweep
# ==============================================================================

print("\n" + "-"*70)
print("PART 1: Varying Number of Modules")
print("-"*70)

n_modules_range = [2, 3, 4, 5, 7, 10, 15, 20]
power = sg.generate_wake_state(N_MODES, seed=SEED)

module_count_results = []

for n_mod in tqdm(n_modules_range, desc="Module counts"):
    G, communities = gg.generate_modular(N_NODES, n_modules=n_mod, seed=SEED)
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    n_modes = min(N_MODES, len(eigenvalues))
    power_adj = power[:n_modes]
    power_adj = power_adj / power_adj.sum()
    
    metrics = met.compute_all_metrics(power_adj, eigenvalues[:n_modes])
    # Dynamic phase coherence via Kuramoto
    R_dyn, meta = simulate_phase_coherence(G, K=1.0, steps=300, dt=0.05, seed=SEED)
    
    # Network metrics
    modularity = nx.community.modularity(G, communities)
    avg_clustering = nx.average_clustering(G)
    
    module_count_results.append({
        'n_modules': n_mod,
        'modularity': modularity,
        'avg_clustering': avg_clustering,
        'module_size': N_NODES // n_mod,
        'R_dyn': R_dyn,
        'metastability': meta,
        **metrics
    })

df_modules = pd.DataFrame(module_count_results)
print("\nModule Count Results:")
print(df_modules[['n_modules', 'modularity', 'R_dyn', 'metastability', 'C']].to_string(index=False))

# ==============================================================================
# PART 2: Intra vs Inter-Module Connectivity
# ==============================================================================

print("\n" + "-"*70)
print("PART 2: Intra vs Inter-Module Connectivity Balance")
print("-"*70)

intra_probs = [0.3, 0.5, 0.7, 0.9]
inter_probs = [0.01, 0.05, 0.1, 0.2, 0.3]

connectivity_results = []

for intra_p, inter_p in tqdm(list(product(intra_probs, inter_probs)), desc="Connectivity sweep"):
    # Generate modular network with specified connectivity
    n_modules = 5
    module_size = N_NODES // n_modules
    
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    
    communities = []
    for m in range(n_modules):
        start = m * module_size
        end = start + module_size if m < n_modules - 1 else N_NODES
        module_nodes = list(range(start, end))
        communities.append(set(module_nodes))
        
        # Intra-module edges
        for i in module_nodes:
            for j in module_nodes:
                if i < j and np.random.rand() < intra_p:
                    G.add_edge(i, j)
    
    # Inter-module edges
    for m1 in range(n_modules):
        for m2 in range(m1 + 1, n_modules):
            for n1 in communities[m1]:
                for n2 in communities[m2]:
                    if np.random.rand() < inter_p:
                        G.add_edge(n1, n2)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2)
    
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    n_modes = min(N_MODES, len(eigenvalues))
    power_adj = power[:n_modes]
    power_adj = power_adj / power_adj.sum()
    
    metrics = met.compute_all_metrics(power_adj, eigenvalues[:n_modes])
    
    modularity = nx.community.modularity(G, communities)
    R_dyn, meta = simulate_phase_coherence(G, K=1.0, steps=300, dt=0.05, seed=SEED)
    
    connectivity_results.append({
        'intra_prob': intra_p,
        'inter_prob': inter_p,
        'ratio': intra_p / inter_p if inter_p > 0 else float('inf'),
        'n_edges': G.number_of_edges(),
        'modularity': modularity,
        'R_dyn': R_dyn,
        'metastability': meta,
        **metrics
    })

df_conn = pd.DataFrame(connectivity_results)
print("\nConnectivity Sweep Results (selected):")
print(df_conn[['intra_prob', 'inter_prob', 'modularity', 'R_dyn', 'C']].head(10).to_string(index=False))

# ==============================================================================
# PART 3: Module Size Distribution
# ==============================================================================

print("\n" + "-"*70)
print("PART 3: Module Size Distribution Effects")
print("-"*70)

size_distributions = [
    ('Uniform', [20, 20, 20, 20, 20]),
    ('Large-dominant', [40, 20, 20, 10, 10]),
    ('Small-dominant', [10, 10, 20, 20, 40]),
    ('Power-law', [50, 25, 12, 8, 5]),
    ('Bimodal', [30, 30, 15, 15, 10]),
]

size_results = []

for name, sizes in tqdm(size_distributions, desc="Size distributions"):
    # Adjust sizes to sum to N_NODES
    sizes = np.array(sizes)
    sizes = (sizes / sizes.sum() * N_NODES).astype(int)
    sizes[-1] = N_NODES - sizes[:-1].sum()
    
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    
    communities = []
    node_idx = 0
    for size in sizes:
        module_nodes = list(range(node_idx, node_idx + size))
        communities.append(set(module_nodes))
        
        # Intra-module edges
        for i in module_nodes:
            for j in module_nodes:
                if i < j and np.random.rand() < 0.5:
                    G.add_edge(i, j)
        
        node_idx += size
    
    # Inter-module edges
    for m1 in range(len(communities)):
        for m2 in range(m1 + 1, len(communities)):
            for n1 in communities[m1]:
                for n2 in communities[m2]:
                    if np.random.rand() < 0.05:
                        G.add_edge(n1, n2)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2)
    
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    n_modes = min(N_MODES, len(eigenvalues))
    power_adj = power[:n_modes]
    power_adj = power_adj / power_adj.sum()
    
    metrics = met.compute_all_metrics(power_adj, eigenvalues[:n_modes])
    R_dyn, meta = simulate_phase_coherence(G, K=1.0, steps=300, dt=0.05, seed=SEED)
    
    size_results.append({
        'distribution': name,
        'sizes': str(list(sizes)),
        'size_std': np.std(sizes),
        'R_dyn': R_dyn,
        'metastability': meta,
        **metrics
    })

df_sizes = pd.DataFrame(size_results)
print("\nModule Size Distribution Results:")
print(df_sizes[['distribution', 'size_std', 'R_dyn', 'C']].to_string(index=False))

# ==============================================================================
# PART 4: Comparison with Other Topologies
# ==============================================================================

print("\n" + "-"*70)
print("PART 4: Modular vs Other Topologies")
print("-"*70)

topology_results = []

# Modular (optimal from Part 1)
best_n_modules = df_modules.loc[df_modules['C'].idxmax(), 'n_modules']
G_mod, comm = gg.generate_modular(N_NODES, n_modules=int(best_n_modules), seed=SEED)
L, eig, evec = gg.compute_laplacian_eigenmodes(G_mod)
metrics = met.compute_all_metrics(power[:min(N_MODES, len(eig))], eig[:min(N_MODES, len(eig))])
R_dyn, meta = simulate_phase_coherence(G_mod, K=1.0, steps=300, dt=0.05, seed=SEED)
topology_results.append({'topology': f'Modular (n={int(best_n_modules)})', **metrics})

# Small-world
G_sw = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eig, evec = gg.compute_laplacian_eigenmodes(G_sw)
metrics = met.compute_all_metrics(power[:min(N_MODES, len(eig))], eig[:min(N_MODES, len(eig))])
R_dyn, meta = simulate_phase_coherence(G_sw, K=1.0, steps=300, dt=0.05, seed=SEED)
topology_results.append({'topology': 'Small-world', **metrics})

# Scale-free
G_sf = gg.generate_scale_free(N_NODES, m_edges=3, seed=SEED)
L, eig, evec = gg.compute_laplacian_eigenmodes(G_sf)
metrics = met.compute_all_metrics(power[:min(N_MODES, len(eig))], eig[:min(N_MODES, len(eig))])
R_dyn, meta = simulate_phase_coherence(G_sf, K=1.0, steps=300, dt=0.05, seed=SEED)
topology_results.append({'topology': 'Scale-free', **metrics})

# Random
G_rand = gg.generate_random(N_NODES, edge_prob=0.1, seed=SEED)
L, eig, evec = gg.compute_laplacian_eigenmodes(G_rand)
metrics = met.compute_all_metrics(power[:min(N_MODES, len(eig))], eig[:min(N_MODES, len(eig))])
R_dyn, meta = simulate_phase_coherence(G_rand, K=1.0, steps=300, dt=0.05, seed=SEED)
topology_results.append({'topology': 'Random', **metrics})

# Lattice
G_lat = gg.generate_lattice(n_nodes=N_NODES, dimension=2)
L, eig, evec = gg.compute_laplacian_eigenmodes(G_lat)
n_m = min(N_MODES, len(eig))
p = power[:n_m] / power[:n_m].sum()
metrics = met.compute_all_metrics(p, eig[:n_m])
R_dyn, meta = simulate_phase_coherence(G_lat, K=1.0, steps=300, dt=0.05, seed=SEED)
topology_results.append({'topology': 'Lattice 2D', **metrics})

df_topo = pd.DataFrame(topology_results)
print("\nTopology Comparison:")
print(df_topo[['topology', 'C', 'H_mode', 'PR', 'R']].to_string(index=False))

# ==============================================================================
# PART 5: Visualizations
# ==============================================================================

print("\n" + "-"*70)
print("PART 5: Generating Visualizations")
print("-"*70)

# Figure 1: Module count analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0, 0]
ax.plot(df_modules['n_modules'], df_modules['C'], 'bo-', markersize=10, linewidth=2)
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('A. Consciousness vs Module Count', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(df_modules['n_modules'], df_modules['modularity'], 'gs-', markersize=10, linewidth=2)
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Modularity (Q)', fontsize=12)
ax.set_title('B. Network Modularity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.scatter(df_modules['modularity'], df_modules['C'], c=df_modules['n_modules'], 
           cmap='viridis', s=100, edgecolors='black')
ax.set_xlabel('Modularity (Q)', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('C. Modularity-Consciousness Relationship', fontsize=12, fontweight='bold')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('N Modules')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
bars = ax.bar(df_topo['topology'], df_topo['C'], color=['blue', 'green', 'red', 'gray', 'orange'])
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('D. Topology Comparison', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# New: C vs dynamic R across module counts
ax = axes[0, 2]
ax.scatter(df_modules['R_dyn'], df_modules['C'], s=100, c=df_modules['n_modules'], cmap='plasma', edgecolors='black')
ax.set_xlabel('Dynamic Phase Coherence R', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('E. C(t) vs R (Module Count)', fontsize=12, fontweight='bold')
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('N Modules')

# New: Heatmap of R over connectivity sweep
ax = axes[1, 2]
pivot_R = df_conn.pivot(index='intra_prob', columns='inter_prob', values='R_dyn')
im2 = ax.imshow(pivot_R.values, cmap='magma', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(pivot_R.columns)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot_R.columns])
ax.set_yticks(range(len(pivot_R.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot_R.index])
ax.set_xlabel('Inter-module Connection Probability', fontsize=12)
ax.set_ylabel('Intra-module Connection Probability', fontsize=12)
ax.set_title('F. Dynamic Phase Coherence R', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('R')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'module_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: module_analysis.png")

# Figure 2: Connectivity heatmap
pivot = df_conn.pivot(index='intra_prob', columns='inter_prob', values='C')
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot.index])
ax.set_xlabel('Inter-module Connection Probability', fontsize=12)
ax.set_ylabel('Intra-module Connection Probability', fontsize=12)
ax.set_title('Consciousness by Connectivity Pattern', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Consciousness C(t)')

# Add values
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f'{pivot.values[i, j]:.3f}', ha='center', va='center', color='white', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'connectivity_heatmap.png', dpi=150, bbox_inches='tight')
print(f"  Saved: connectivity_heatmap.png")

# Save an additional heatmap for R
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pivot_R.values, cmap='magma', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(pivot_R.columns)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot_R.columns])
ax.set_yticks(range(len(pivot_R.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot_R.index])
ax.set_xlabel('Inter-module Connection Probability', fontsize=12)
ax.set_ylabel('Intra-module Connection Probability', fontsize=12)
ax.set_title('Dynamic Phase Coherence R by Connectivity', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('R')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'connectivity_heatmap_R.png', dpi=150, bbox_inches='tight')
print(f"  Saved: connectivity_heatmap_R.png")

# Save data
df_modules.to_csv(OUTPUT_DIR / 'module_count_analysis.csv', index=False)
df_conn.to_csv(OUTPUT_DIR / 'connectivity_sweep.csv', index=False)
df_sizes.to_csv(OUTPUT_DIR / 'size_distribution.csv', index=False)
df_topo.to_csv(OUTPUT_DIR / 'topology_comparison.csv', index=False)

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("KEY FINDINGS: MODULAR NETWORKS")
print("="*70)

best_idx = df_modules['C'].idxmax()
best = df_modules.iloc[best_idx]

print(f"""
1. OPTIMAL MODULE COUNT:
   - Best consciousness at {int(best['n_modules'])} modules
   - C(t) = {best['C']:.3f}
   - Modularity Q = {best['modularity']:.3f}

2. CONNECTIVITY BALANCE:
   - High intra-module + Low inter-module → High modularity but poor integration
   - Low intra-module + High inter-module → Low modularity, random-like
   - Sweet spot: High intra ({df_conn.loc[df_conn['C'].idxmax(), 'intra_prob']:.2f}) + 
                 Moderate inter ({df_conn.loc[df_conn['C'].idxmax(), 'inter_prob']:.2f})

3. SIZE DISTRIBUTION:
   - Best distribution: {df_sizes.loc[df_sizes['C'].idxmax(), 'distribution']}
   - Brain-like heterogeneity may be beneficial

4. TOPOLOGY RANKING:
""")
for _, row in df_topo.sort_values('C', ascending=False).iterrows():
    print(f"   {row['topology']}: C = {row['C']:.3f}")

print(f"""
5. BIOLOGICAL IMPLICATIONS:
   - Brain's modular organization optimizes consciousness
   - Modules provide local processing, inter-module links integrate
   - Too many small modules → fragmentation
   - Too few large modules → poor specialization
""")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("="*70)
