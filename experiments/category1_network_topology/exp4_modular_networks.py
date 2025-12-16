#!/usr/bin/env python3
"""
Category 1, Experiment 4: Modular Networks

Test consciousness in modular architectures:
- Generate networks with 2, 4, 6, 8 modules
- Vary inter-module vs intra-module connectivity
- Test integration across modules
- Measure how modularity affects PR and H_mode
- Model hemisphere disconnection (split-brain)
- Visualize community structure and eigenmodes

Uses GPU acceleration for eigendecomposition when available.
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
import seaborn as sns

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz
from utils.gpu_utils import get_device_info, gpu_eigendecomposition, print_gpu_status

# Configuration
SEED = 42
N_NODES = 120  # Divisible by 2, 3, 4, 6, 8
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_modular_networks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Category 1, Experiment 4: Modular Networks")
print("=" * 60)

# Check GPU availability
print_gpu_status()
gpu_info = get_device_info()
USE_GPU = gpu_info['cupy_available']

np.random.seed(SEED)

# Brain states for testing
states = {
    'Wake': sg.generate_wake_state(n_modes=40, seed=SEED),
    'NREM': sg.generate_nrem_unconscious(n_modes=40, seed=SEED),
    'Dream': sg.generate_nrem_dreaming(n_modes=40, seed=SEED),
    'Anesthesia': sg.generate_anesthesia_state(n_modes=40, seed=SEED),
}

# ============================================================================
# EXPERIMENT 1: Vary number of modules
# ============================================================================

print("\n1. Testing different numbers of modules...")

n_modules_list = [2, 3, 4, 6, 8]
intra_prob = 0.3  # Fixed intra-module connectivity
inter_prob = 0.05  # Fixed inter-module connectivity

module_results = []

for n_modules in tqdm(n_modules_list, desc="Module counts"):
    # Generate modular network
    G, communities = gg.generate_modular(
        N_NODES, 
        n_modules=n_modules,
        intra_prob=intra_prob,
        inter_prob=inter_prob,
        seed=SEED
    )
    
    # Compute modularity score
    modularity = nx.algorithms.community.modularity(G, communities)
    
    # Compute eigenmodes
    if USE_GPU:
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = gpu_eigendecomposition(L.astype(np.float64), use_gpu=True)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
    else:
        L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    n_modes = min(40, len(eigenvalues))
    eigenvalues_trunc = eigenvalues[:n_modes]
    
    # Test all brain states
    for state_name, power in states.items():
        power_trunc = power[:n_modes]
        power_trunc = power_trunc / power_trunc.sum()
        
        metrics = met.compute_all_metrics(power_trunc, eigenvalues_trunc)
        
        module_results.append({
            'n_modules': n_modules,
            'modularity': modularity,
            'state': state_name,
            'n_edges': G.number_of_edges(),
            **metrics
        })

df_modules = pd.DataFrame(module_results)

# ============================================================================
# EXPERIMENT 2: Vary inter-module connectivity (integration)
# ============================================================================

print("\n2. Testing inter-module connectivity (integration)...")

n_modules = 4  # Fixed number of modules
inter_probs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

integration_results = []

for inter_prob in tqdm(inter_probs, desc="Inter-module prob"):
    # Generate modular network
    G, communities = gg.generate_modular(
        N_NODES,
        n_modules=n_modules,
        intra_prob=0.3,
        inter_prob=inter_prob,
        seed=SEED
    )
    
    # Check connectivity
    is_connected = nx.is_connected(G)
    if not is_connected:
        # Get largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Compute modularity
    try:
        modularity = nx.algorithms.community.modularity(G, communities)
    except:
        modularity = 0
    
    # Compute eigenmodes
    if USE_GPU and G.number_of_nodes() > 50:
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = gpu_eigendecomposition(L.astype(np.float64), use_gpu=True)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
    else:
        L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    n_modes = min(40, len(eigenvalues))
    eigenvalues_trunc = eigenvalues[:n_modes]
    
    # Test wake state
    power = states['Wake'][:n_modes]
    power = power / power.sum()
    
    metrics = met.compute_all_metrics(power, eigenvalues_trunc)
    
    integration_results.append({
        'inter_prob': inter_prob,
        'modularity': modularity,
        'is_connected': is_connected,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        **metrics
    })

df_integration = pd.DataFrame(integration_results)

# ============================================================================
# EXPERIMENT 3: Split-brain simulation
# ============================================================================

print("\n3. Simulating split-brain (hemisphere disconnection)...")

# Create two-module network
G_split, communities = gg.generate_modular(
    N_NODES,
    n_modules=2,
    intra_prob=0.4,
    inter_prob=0.15,  # Some inter-hemisphere connections
    seed=SEED
)

# Identify inter-hemisphere edges
inter_edges = []
for u, v in G_split.edges():
    for i, comm in enumerate(communities):
        if u in comm:
            u_comm = i
        if v in comm:
            v_comm = i
    if u_comm != v_comm:
        inter_edges.append((u, v))

print(f"  Inter-hemisphere edges: {len(inter_edges)}")

# Progressively disconnect hemispheres
disconnect_pcts = [0, 25, 50, 75, 100]
split_results = []

for pct in tqdm(disconnect_pcts, desc="Disconnection"):
    G_test = G_split.copy()
    
    # Remove percentage of inter-hemisphere edges
    n_remove = int(len(inter_edges) * pct / 100)
    edges_to_remove = inter_edges[:n_remove]
    G_test.remove_edges_from(edges_to_remove)
    
    # Compute metrics
    if USE_GPU:
        L = nx.laplacian_matrix(G_test).toarray()
        eigenvalues, eigenvectors = gpu_eigendecomposition(L.astype(np.float64), use_gpu=True)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
    else:
        L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G_test)
    
    n_modes = min(40, len(eigenvalues))
    eigenvalues_trunc = eigenvalues[:n_modes]
    
    # Test all states
    for state_name, power in states.items():
        power_trunc = power[:n_modes]
        power_trunc = power_trunc / power_trunc.sum()
        
        metrics = met.compute_all_metrics(power_trunc, eigenvalues_trunc)
        
        split_results.append({
            'disconnection_pct': pct,
            'state': state_name,
            'n_components': nx.number_connected_components(G_test),
            'remaining_inter_edges': len(inter_edges) - n_remove,
            **metrics
        })

df_split = pd.DataFrame(split_results)

# Save all results
df_modules.to_csv(OUTPUT_DIR / 'modular_networks_results.csv', index=False)
df_integration.to_csv(OUTPUT_DIR / 'integration_results.csv', index=False)
df_split.to_csv(OUTPUT_DIR / 'split_brain_results.csv', index=False)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Effect of module count
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# C(t) by module count and state
ax = axes[0, 0]
for state in states.keys():
    subset = df_modules[df_modules['state'] == state]
    ax.plot(subset['n_modules'], subset['C'], 'o-', linewidth=2, markersize=8, label=state)
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('C(t) vs Network Modularity', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Modularity score
ax = axes[0, 1]
subset = df_modules[df_modules['state'] == 'Wake']
ax.bar(subset['n_modules'], subset['modularity'], color='steelblue', alpha=0.7)
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Newman Modularity Q', fontsize=12)
ax.set_title('Network Modularity Score', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# H_mode and PR by module count
ax = axes[1, 0]
for state in ['Wake', 'Anesthesia']:
    subset = df_modules[df_modules['state'] == state]
    ax.plot(subset['n_modules'], subset['H_mode'], 'o-', linewidth=2, label=f'H_mode ({state})')
    ax.plot(subset['n_modules'], subset['PR'], 's--', linewidth=2, label=f'PR ({state})')
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Entropy and Participation Ratio', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Criticality by module count
ax = axes[1, 1]
for state in states.keys():
    subset = df_modules[df_modules['state'] == state]
    ax.plot(subset['n_modules'], subset['kappa'], 'o-', linewidth=2, markersize=8, label=state)
ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Criticality Index κ', fontsize=12)
ax.set_title('Criticality vs Modularity', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'modular_networks_analysis.png', dpi=300)
print("  Saved: modular_networks_analysis.png")

# 2. Integration analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(df_integration['inter_prob'], df_integration['C'], 'o-', linewidth=2, markersize=8, color='darkblue')
ax.set_xlabel('Inter-module Connection Probability', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('C(t) vs Integration', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df_integration['inter_prob'], df_integration['modularity'], 'o-', linewidth=2, markersize=8, color='darkgreen')
ax.set_xlabel('Inter-module Connection Probability', fontsize=12)
ax.set_ylabel('Modularity Q', fontsize=12)
ax.set_title('Modularity vs Integration', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(df_integration['modularity'], df_integration['C'], c=df_integration['inter_prob'], 
          cmap='viridis', s=100, edgecolors='black')
ax.set_xlabel('Modularity Q', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('C(t) vs Modularity Trade-off', fontsize=14, fontweight='bold')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Inter-prob', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'integration_analysis.png', dpi=300)
print("  Saved: integration_analysis.png")

# 3. Split-brain results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for state in states.keys():
    subset = df_split[df_split['state'] == state]
    ax.plot(subset['disconnection_pct'], subset['C'], 'o-', linewidth=2, markersize=8, label=state)
ax.set_xlabel('Hemisphere Disconnection (%)', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('Split-Brain Simulation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
subset = df_split[df_split['state'] == 'Wake']
ax2 = ax.twinx()
ax.plot(subset['disconnection_pct'], subset['kappa'], 'b-o', linewidth=2, markersize=8, label='Criticality κ')
ax2.plot(subset['disconnection_pct'], subset['n_components'], 'r--s', linewidth=2, markersize=8, label='Components')
ax.set_xlabel('Hemisphere Disconnection (%)', fontsize=12)
ax.set_ylabel('Criticality Index κ', fontsize=12, color='blue')
ax2.set_ylabel('Number of Components', fontsize=12, color='red')
ax.set_title('Criticality and Fragmentation', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'split_brain_analysis.png', dpi=300)
print("  Saved: split_brain_analysis.png")

# 4. Network visualization for different module counts
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, n_mod in enumerate([2, 4, 6, 8]):
    G, communities = gg.generate_modular(N_NODES, n_modules=n_mod, intra_prob=0.3, inter_prob=0.05, seed=SEED)
    
    ax = axes[idx]
    pos = nx.spring_layout(G, seed=SEED, k=2)
    
    # Color nodes by community
    colors = []
    for node in G.nodes():
        for i, comm in enumerate(communities):
            if node in comm:
                colors.append(i)
                break
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color=colors, cmap='Set3')
    ax.set_title(f'{n_mod} Modules', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle('Modular Network Architectures', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'modular_network_visualization.png', dpi=300, bbox_inches='tight')
print("  Saved: modular_network_visualization.png")

plt.close('all')

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\nOptimal Number of Modules (for Wake state):")
wake_data = df_modules[df_modules['state'] == 'Wake']
best_idx = wake_data['C'].idxmax()
print(f"  Best C(t) = {wake_data.loc[best_idx, 'C']:.4f} at {wake_data.loc[best_idx, 'n_modules']} modules")

print("\nIntegration-Segregation Trade-off:")
print(f"  Highest integration (inter_prob=0.3): C(t) = {df_integration[df_integration['inter_prob'] == 0.3]['C'].values[0]:.4f}")
print(f"  Highest segregation (inter_prob=0.01): C(t) = {df_integration[df_integration['inter_prob'] == 0.01]['C'].values[0]:.4f}")

print("\nSplit-Brain Effect (Wake state):")
wake_split = df_split[df_split['state'] == 'Wake']
c_intact = wake_split[wake_split['disconnection_pct'] == 0]['C'].values[0]
c_split = wake_split[wake_split['disconnection_pct'] == 100]['C'].values[0]
print(f"  Intact: C(t) = {c_intact:.4f}")
print(f"  Complete split: C(t) = {c_split:.4f}")
print(f"  Reduction: {(c_intact - c_split) / c_intact * 100:.1f}%")

print("\n" + "=" * 60)
print(f"Experiment completed! Results saved to: {OUTPUT_DIR}")
print("=" * 60)
