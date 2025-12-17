#!/usr/bin/env python3
"""
Category 1, Experiment 3: Hub Disruption

Model lesions by removing hub nodes:
- Create scale-free network with clear hubs
- Identify top hubs by degree, betweenness, eigenvector centrality
- Progressively remove hubs (1%, 5%, 10%, 20%)
- Measure C(t) degradation after each removal
- Compare to random node removal
- Visualize network fragmentation
- Model stroke/TBI effects

Uses GPU acceleration when available for large-scale computations.
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
from utils.gpu_utils import get_device_info, gpu_eigendecomposition, print_gpu_status
from utils.chaos_metrics import estimate_lyapunov_exponent, compute_branching_ratio, compute_all_chaos_metrics
from utils.category_theory_metrics import compute_sheaf_consistency, compute_integration_phi

# Configuration
SEED = 42
N_NODES = 500  # Larger network for comprehensive hub analysis
N_MODES = 100  # More modes for finer resolution
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_hub_disruption'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Category 1, Experiment 3: Hub Disruption")
print("=" * 60)

# Check GPU availability
print_gpu_status()
gpu_info = get_device_info()
USE_GPU = gpu_info['cupy_available']

np.random.seed(SEED)

# Generate scale-free network (has clear hub structure)
print("\nGenerating scale-free network...")
G = gg.generate_scale_free(N_NODES, m_edges=3, seed=SEED)

print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Compute centrality measures
print("\nComputing centrality measures...")
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Rank nodes by each centrality
degree_ranked = sorted(degree_centrality.keys(), key=lambda x: degree_centrality[x], reverse=True)
betweenness_ranked = sorted(betweenness_centrality.keys(), key=lambda x: betweenness_centrality[x], reverse=True)
eigenvector_ranked = sorted(eigenvector_centrality.keys(), key=lambda x: eigenvector_centrality[x], reverse=True)

# Random ranking for comparison
random_ranked = list(G.nodes())
np.random.shuffle(random_ranked)

# Define removal strategies
strategies = {
    'degree': degree_ranked,
    'betweenness': betweenness_ranked,
    'eigenvector': eigenvector_ranked,
    'random': random_ranked
}

# Define removal percentages - denser sweep for higher resolution
removal_percentages = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

# Multiple brain states for comprehensive testing
brain_states = {
    'wake': sg.generate_wake_state(n_modes=N_MODES, seed=SEED),
    'nrem': sg.generate_nrem_unconscious(n_modes=N_MODES, seed=SEED),
    'dream': sg.generate_nrem_dreaming(n_modes=N_MODES, seed=SEED),
    'anesthesia': sg.generate_anesthesia_state(n_modes=N_MODES, seed=SEED),
    'psychedelic': sg.generate_psychedelic_state(n_modes=N_MODES, intensity=0.6, seed=SEED),
}

# Store results
results = []

print("\nTesting hub disruption strategies across brain states...")
for state_name, state_power in tqdm(brain_states.items(), desc="Brain States"):
    for strategy_name, ranked_nodes in tqdm(strategies.items(), desc=f"  {state_name} strategies", leave=False):
        for pct in removal_percentages:
            # Copy network
            G_damaged = G.copy()
            
            # Calculate number of nodes to remove
            n_remove = int(N_NODES * pct / 100)
            
            # Remove nodes
            nodes_to_remove = ranked_nodes[:n_remove]
            G_damaged.remove_nodes_from(nodes_to_remove)
            
            # Check if network is still connected
            if G_damaged.number_of_nodes() < 10:
                # Network too small
                results.append({
                    'brain_state': state_name,
                    'strategy': strategy_name,
                    'removal_pct': pct,
                    'n_removed': n_remove,
                    'n_nodes_remaining': G_damaged.number_of_nodes(),
                    'is_connected': False,
                    'largest_component_size': 0,
                    'n_components': 0,
                    'C': 0,
                    'H_mode': 0,
                    'PR': 0,
                    'R': 0,
                    'S_dot': 0,
                    'kappa': 0
                })
                continue
            
            # Get largest connected component
            if not nx.is_connected(G_damaged):
                largest_cc = max(nx.connected_components(G_damaged), key=len)
                G_component = G_damaged.subgraph(largest_cc).copy()
            else:
                G_component = G_damaged
                largest_cc = set(G_damaged.nodes())
            
            n_components = nx.number_connected_components(G_damaged)
            
            # Compute eigenmodes on the remaining network
            try:
                if USE_GPU and G_component.number_of_nodes() > 50:
                    # Use GPU for larger matrices
                    L = nx.laplacian_matrix(G_component).toarray()
                    eigenvalues, eigenvectors = gpu_eigendecomposition(L.astype(np.float64), use_gpu=True)
                    idx = np.argsort(eigenvalues)
                    eigenvalues = eigenvalues[idx]
                else:
                    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G_component)
                
                # Truncate to available modes
                n_modes = min(N_MODES, len(eigenvalues))
                eigenvalues_trunc = eigenvalues[:n_modes]
                power = state_power[:n_modes]
                power = power / power.sum()
                
                # Compute metrics
                metrics = met.compute_all_metrics(power, eigenvalues_trunc)
                
                results.append({
                    'brain_state': state_name,
                    'strategy': strategy_name,
                    'removal_pct': pct,
                    'n_removed': n_remove,
                    'n_nodes_remaining': G_damaged.number_of_nodes(),
                    'is_connected': nx.is_connected(G_damaged),
                    'largest_component_size': len(largest_cc),
                    'n_components': n_components,
                    **metrics
                })
            except Exception as e:
                print(f"  Warning: Error computing metrics for {state_name}/{strategy_name} at {pct}%: {e}")
                results.append({
                    'brain_state': state_name,
                    'strategy': strategy_name,
                    'removal_pct': pct,
                    'n_removed': n_remove,
                    'n_nodes_remaining': G_damaged.number_of_nodes(),
                    'is_connected': False,
                    'largest_component_size': len(largest_cc) if 'largest_cc' in dir() else 0,
                    'n_components': n_components if 'n_components' in dir() else 0,
                    'C': 0, 'H_mode': 0, 'PR': 0, 'R': 0, 'S_dot': 0, 'kappa': 0
                })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save results
csv_path = OUTPUT_DIR / 'hub_disruption_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. C(t) degradation curves for each strategy
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot C(t) vs removal percentage
ax = axes[0, 0]
for strategy in strategies.keys():
    subset = df[df['strategy'] == strategy]
    ax.plot(subset['removal_pct'], subset['C'], 'o-', linewidth=2, markersize=6, label=strategy.capitalize())
ax.set_xlabel('Nodes Removed (%)', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('C(t) Degradation Under Hub Disruption', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot largest component size
ax = axes[0, 1]
for strategy in strategies.keys():
    subset = df[df['strategy'] == strategy]
    ax.plot(subset['removal_pct'], subset['largest_component_size'], 'o-', linewidth=2, markersize=6, label=strategy.capitalize())
ax.set_xlabel('Nodes Removed (%)', fontsize=12)
ax.set_ylabel('Largest Component Size', fontsize=12)
ax.set_title('Network Fragmentation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot number of components
ax = axes[1, 0]
for strategy in strategies.keys():
    subset = df[df['strategy'] == strategy]
    ax.plot(subset['removal_pct'], subset['n_components'], 'o-', linewidth=2, markersize=6, label=strategy.capitalize())
ax.set_xlabel('Nodes Removed (%)', fontsize=12)
ax.set_ylabel('Number of Components', fontsize=12)
ax.set_title('Network Disconnection', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot H_mode and PR
ax = axes[1, 1]
for strategy in ['degree', 'random']:
    subset = df[df['strategy'] == strategy]
    ax.plot(subset['removal_pct'], subset['H_mode'], 'o-', linewidth=2, label=f'H_mode ({strategy})')
    ax.plot(subset['removal_pct'], subset['PR'], 's--', linewidth=2, label=f'PR ({strategy})')
ax.set_xlabel('Nodes Removed (%)', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Entropy and Participation Ratio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hub_disruption_analysis.png', dpi=300)
print("  Saved: hub_disruption_analysis.png")

# 2. Network visualization at different damage levels
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
damage_levels = [0, 5, 10, 20]

for idx, pct in enumerate(damage_levels):
    # Degree-based removal
    n_remove = int(N_NODES * pct / 100)
    G_damaged = G.copy()
    G_damaged.remove_nodes_from(degree_ranked[:n_remove])
    
    ax = axes[0, idx]
    if G_damaged.number_of_nodes() > 0:
        pos = nx.spring_layout(G_damaged, seed=SEED, k=2)
        nx.draw_networkx_edges(G_damaged, pos, ax=ax, alpha=0.2, edge_color='gray')
        
        # Color by degree in damaged network
        if G_damaged.number_of_nodes() > 0:
            degrees = [G_damaged.degree(n) for n in G_damaged.nodes()]
            nx.draw_networkx_nodes(G_damaged, pos, ax=ax, node_size=50,
                                  node_color=degrees, cmap='YlOrRd')
    ax.set_title(f'Degree: {pct}% removed', fontsize=12)
    ax.axis('off')
    
    # Random removal
    G_random = G.copy()
    G_random.remove_nodes_from(random_ranked[:n_remove])
    
    ax = axes[1, idx]
    if G_random.number_of_nodes() > 0:
        pos = nx.spring_layout(G_random, seed=SEED, k=2)
        nx.draw_networkx_edges(G_random, pos, ax=ax, alpha=0.2, edge_color='gray')
        
        if G_random.number_of_nodes() > 0:
            degrees = [G_random.degree(n) for n in G_random.nodes()]
            nx.draw_networkx_nodes(G_random, pos, ax=ax, node_size=50,
                                  node_color=degrees, cmap='YlOrRd')
    ax.set_title(f'Random: {pct}% removed', fontsize=12)
    ax.axis('off')

axes[0, 0].set_ylabel('Hub Removal', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Random Removal', fontsize=14, fontweight='bold')

plt.suptitle('Network Visualization Under Disruption', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'network_disruption_visualization.png', dpi=300, bbox_inches='tight')
print("  Saved: network_disruption_visualization.png")

# 3. Summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

# Critical threshold analysis
print("\nCritical Removal Thresholds (C(t) drops below 0.5):")
for strategy in strategies.keys():
    subset = df[df['strategy'] == strategy]
    threshold_rows = subset[subset['C'] < 0.5]
    if len(threshold_rows) > 0:
        threshold = threshold_rows['removal_pct'].min()
        print(f"  {strategy.capitalize():12}: {threshold}%")
    else:
        print(f"  {strategy.capitalize():12}: >30% (robust)")

# Vulnerability index
print("\nVulnerability Index (C drop per 10% removal):")
for strategy in strategies.keys():
    subset = df[df['strategy'] == strategy]
    c_0 = subset[subset['removal_pct'] == 0]['C'].values[0]
    c_10 = subset[subset['removal_pct'] == 10]['C'].values
    if len(c_10) > 0:
        vulnerability = (c_0 - c_10[0]) / c_0 * 100
        print(f"  {strategy.capitalize():12}: {vulnerability:.1f}% C reduction")

plt.close('all')

print("\n" + "=" * 60)
print(f"Experiment completed! Results saved to: {OUTPUT_DIR}")
print("=" * 60)
