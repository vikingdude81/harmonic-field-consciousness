#!/usr/bin/env python3
"""
Category 1, Experiment 1: Topology Comparison

Compare consciousness metrics across different network architectures:
- Small-world, scale-free, random, lattice, modular networks
- Fixed node count (100 nodes) for fair comparison
- Apply all 4 brain states to each topology
- Generate comprehensive comparison visualizations
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

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz

# Configuration
SEED = 42
N_NODES = 100
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp1_topology_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 1, Experiment 1: Topology Comparison")
print("="*60)

# Define topologies to test
topologies = {
    'Small-world': lambda: gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED),
    'Scale-free': lambda: gg.generate_scale_free(N_NODES, m_edges=3, seed=SEED),
    'Random': lambda: gg.generate_random(N_NODES, edge_prob=0.06, seed=SEED),
    'Lattice-2D': lambda: gg.generate_lattice(N_NODES, dimension=2, seed=SEED),
    'Modular': lambda: gg.generate_modular(N_NODES, n_modules=4, intra_prob=0.3, inter_prob=0.05, seed=SEED)[0],
}

# Define brain states
states = {
    'Wake': sg.generate_wake_state(n_modes=30, seed=SEED),
    'NREM': sg.generate_nrem_unconscious(n_modes=30, seed=SEED),
    'Dream': sg.generate_nrem_dreaming(n_modes=30, seed=SEED),
    'Anesthesia': sg.generate_anesthesia_state(n_modes=30, seed=SEED),
}

# Store results
results = []

print("\nComputing metrics for all topology-state combinations...")
for topo_name, topo_func in tqdm(topologies.items(), desc="Topologies"):
    # Generate network
    G = topo_func()
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    # Truncate eigenvalues to match state dimensions
    n_modes = min(30, len(eigenvalues))
    eigenvalues_trunc = eigenvalues[:n_modes]
    
    for state_name, power in states.items():
        # Truncate power if needed
        power_trunc = power[:n_modes]
        power_trunc = power_trunc / power_trunc.sum()
        
        # Compute all metrics
        metrics = met.compute_all_metrics(
            power_trunc,
            eigenvalues_trunc,
            phases=None,
            power_previous=None
        )
        
        # Store result
        result = {
            'topology': topo_name,
            'state': state_name,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            **metrics
        }
        results.append(result)

# Convert to DataFrame
df = pd.DataFrame(results)

print("\n" + "="*60)
print("Results Summary:")
print("="*60)
print(df.groupby(['topology', 'state'])[['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']].mean())

# Save results
csv_path = OUTPUT_DIR / 'topology_comparison_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Bar chart of C(t) across topologies and states
fig, ax = plt.subplots(figsize=(12, 6))
states_list = list(states.keys())
x = np.arange(len(states_list))
width = 0.15

for i, topo in enumerate(topologies.keys()):
    c_values = [df[(df['topology'] == topo) & (df['state'] == s)]['C'].values[0] 
                for s in states_list]
    offset = (i - len(topologies)/2) * width
    ax.bar(x + offset, c_values, width, label=topo, alpha=0.8)

ax.set_xlabel('Brain State')
ax.set_ylabel('Consciousness Functional C(t)')
ax.set_title('Consciousness Functional Across Network Topologies and Brain States')
ax.set_xticks(x)
ax.set_xticklabels(states_list)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'c_comparison_bar.png')
print(f"  Saved: c_comparison_bar.png")

# 2. Heatmap of all 5 components
components = ['H_mode', 'PR', 'R', 'S_dot', 'kappa']
for component in components:
    heatmap_data = df.pivot(index='state', columns='topology', values=component)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(heatmap_data.index)
    
    # Annotate cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = ax.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'Heatmap: {component}')
    plt.colorbar(im, ax=ax, label=component)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'heatmap_{component}.png')
    print(f"  Saved: heatmap_{component}.png")

# 3. Network visualizations side-by-side
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (topo_name, topo_func) in enumerate(topologies.items()):
    G = topo_func()
    ax = axes[idx]
    
    # Use spring layout
    pos = None
    if topo_name == 'Lattice-2D':
        # Special layout for lattice
        pos = nx.spring_layout(G, seed=SEED, k=0.5)
    else:
        pos = nx.spring_layout(G, seed=SEED, k=1.0)
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20, node_color='steelblue', alpha=0.8)
    
    ax.set_title(f'{topo_name}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    ax.axis('off')

# Hide extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'network_topologies.png')
print(f"  Saved: network_topologies.png")

# 4. Statistical analysis
print("\n" + "="*60)
print("Statistical Analysis:")
print("="*60)

# Compute mean and std for each topology
stats = df.groupby('topology')[['C']].agg(['mean', 'std'])
print("\nConsciousness Functional C(t) by Topology:")
print(stats)

# Best topology for consciousness
best_topo = df.groupby('topology')['C'].mean().idxmax()
print(f"\nBest topology for consciousness: {best_topo}")

# Topology sensitivity to state
for topo in topologies.keys():
    topo_data = df[df['topology'] == topo]
    c_range = topo_data['C'].max() - topo_data['C'].min()
    print(f"{topo:15s}: C range = {c_range:.3f}")

plt.close('all')

print("\n" + "="*60)
print("Experiment completed successfully!")
print(f"All results saved to: {OUTPUT_DIR}")
print("="*60)
