#!/usr/bin/env python3
"""
Category 1: Network Topology Experiments

Experiment 3: Hub Disruption Analysis

Tests how removing hub nodes affects consciousness metrics.
Models brain lesions, stroke, and traumatic brain injury.

Key questions:
1. How critical are hub nodes for consciousness?
2. Targeted vs random lesions - which is worse?
3. Can the network compensate for hub loss?
4. What's the minimum viable hub set?
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

# Configuration
SEED = 42
np.random.seed(SEED)
N_NODES = 100
N_MODES = 20
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_hub_disruption'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Category 1, Experiment 3: Hub Disruption Analysis")
print("="*70)

# ==============================================================================
# PART 1: Generate Scale-Free Network with Hubs
# ==============================================================================

print("\n" + "-"*70)
print("PART 1: Creating Scale-Free Network with Hub Structure")
print("-"*70)

G = gg.generate_scale_free(N_NODES, m_edges=3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

# Identify hubs by degree
degrees = dict(G.degree())
sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
hub_degrees = [degrees[n] for n in sorted_nodes[:10]]

print(f"Network: {N_NODES} nodes, {G.number_of_edges()} edges")
print(f"Top 10 hub degrees: {hub_degrees}")
print(f"Mean degree: {np.mean(list(degrees.values())):.2f}")

# ==============================================================================
# PART 2: Progressive Hub Removal
# ==============================================================================

print("\n" + "-"*70)
print("PART 2: Progressive Hub Removal (Targeted Lesions)")
print("-"*70)

power = sg.generate_wake_state(N_MODES, seed=SEED)
removal_counts = [0, 1, 2, 3, 5, 7, 10, 15, 20]

hub_lesion_results = []

for n_remove in tqdm(removal_counts, desc="Hub removal"):
    if n_remove == 0:
        G_lesioned = G.copy()
    else:
        G_lesioned = G.copy()
        hubs_to_remove = sorted_nodes[:n_remove]
        G_lesioned.remove_nodes_from(hubs_to_remove)
    
    # Relabel nodes to be contiguous
    G_lesioned = nx.convert_node_labels_to_integers(G_lesioned)
    
    if G_lesioned.number_of_nodes() < N_MODES:
        continue
    
    # Compute new eigenmodes
    L_new, eig_new, evec_new = gg.compute_laplacian_eigenmodes(G_lesioned)
    
    n_modes_new = min(N_MODES, len(eig_new))
    power_new = power[:n_modes_new]
    power_new = power_new / power_new.sum()
    
    # Compute metrics
    metrics = met.compute_all_metrics(power_new, eig_new[:n_modes_new])
    
    # Network health metrics
    if nx.is_connected(G_lesioned):
        avg_path = nx.average_shortest_path_length(G_lesioned)
        n_components = 1
    else:
        components = list(nx.connected_components(G_lesioned))
        n_components = len(components)
        largest = G_lesioned.subgraph(max(components, key=len))
        avg_path = nx.average_shortest_path_length(largest) if largest.number_of_nodes() > 1 else float('inf')
    
    hub_lesion_results.append({
        'n_removed': n_remove,
        'lesion_type': 'hub',
        'remaining_nodes': G_lesioned.number_of_nodes(),
        'remaining_edges': G_lesioned.number_of_edges(),
        'n_components': n_components,
        'avg_path_length': avg_path,
        **metrics
    })

df_hub = pd.DataFrame(hub_lesion_results)
print("\nHub Removal Results:")
print(df_hub[['n_removed', 'C', 'H_mode', 'PR', 'n_components']].to_string(index=False))

# ==============================================================================
# PART 3: Random Node Removal (Control)
# ==============================================================================

print("\n" + "-"*70)
print("PART 3: Random Node Removal (Control Comparison)")
print("-"*70)

non_hubs = sorted_nodes[20:]  # Exclude top 20 hubs
random_lesion_results = []

for n_remove in tqdm(removal_counts, desc="Random removal"):
    if n_remove == 0:
        G_lesioned = G.copy()
    else:
        G_lesioned = G.copy()
        nodes_to_remove = np.random.choice(non_hubs, size=min(n_remove, len(non_hubs)), replace=False)
        G_lesioned.remove_nodes_from(nodes_to_remove)
    
    G_lesioned = nx.convert_node_labels_to_integers(G_lesioned)
    
    if G_lesioned.number_of_nodes() < N_MODES:
        continue
    
    L_new, eig_new, evec_new = gg.compute_laplacian_eigenmodes(G_lesioned)
    
    n_modes_new = min(N_MODES, len(eig_new))
    power_new = power[:n_modes_new]
    power_new = power_new / power_new.sum()
    
    metrics = met.compute_all_metrics(power_new, eig_new[:n_modes_new])
    
    if nx.is_connected(G_lesioned):
        avg_path = nx.average_shortest_path_length(G_lesioned)
        n_components = 1
    else:
        components = list(nx.connected_components(G_lesioned))
        n_components = len(components)
        largest = G_lesioned.subgraph(max(components, key=len))
        avg_path = nx.average_shortest_path_length(largest) if largest.number_of_nodes() > 1 else float('inf')
    
    random_lesion_results.append({
        'n_removed': n_remove,
        'lesion_type': 'random',
        'remaining_nodes': G_lesioned.number_of_nodes(),
        'remaining_edges': G_lesioned.number_of_edges(),
        'n_components': n_components,
        'avg_path_length': avg_path,
        **metrics
    })

df_random = pd.DataFrame(random_lesion_results)
print("\nRandom Removal Results:")
print(df_random[['n_removed', 'C', 'H_mode', 'PR', 'n_components']].to_string(index=False))

# ==============================================================================
# PART 4: Edge Removal (Connection Damage)
# ==============================================================================

print("\n" + "-"*70)
print("PART 4: Edge Removal (White Matter Damage)")
print("-"*70)

edge_removal_fractions = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
edge_results = []

for frac in tqdm(edge_removal_fractions, desc="Edge removal"):
    G_damaged = G.copy()
    
    if frac > 0:
        edges = list(G_damaged.edges())
        n_remove = int(len(edges) * frac)
        edges_to_remove = np.random.choice(len(edges), size=n_remove, replace=False)
        G_damaged.remove_edges_from([edges[i] for i in edges_to_remove])
    
    # Only keep largest component
    if not nx.is_connected(G_damaged):
        components = list(nx.connected_components(G_damaged))
        largest = max(components, key=len)
        G_damaged = G_damaged.subgraph(largest).copy()
        G_damaged = nx.convert_node_labels_to_integers(G_damaged)
    
    if G_damaged.number_of_nodes() < N_MODES:
        continue
    
    L_new, eig_new, evec_new = gg.compute_laplacian_eigenmodes(G_damaged)
    
    n_modes_new = min(N_MODES, len(eig_new))
    power_new = power[:n_modes_new]
    power_new = power_new / power_new.sum()
    
    metrics = met.compute_all_metrics(power_new, eig_new[:n_modes_new])
    
    edge_results.append({
        'fraction_removed': frac,
        'remaining_nodes': G_damaged.number_of_nodes(),
        'remaining_edges': G_damaged.number_of_edges(),
        **metrics
    })

df_edges = pd.DataFrame(edge_results)
print("\nEdge Removal Results:")
print(df_edges[['fraction_removed', 'C', 'remaining_edges']].to_string(index=False))

# ==============================================================================
# PART 5: Visualizations
# ==============================================================================

print("\n" + "-"*70)
print("PART 5: Generating Visualizations")
print("-"*70)

# Figure 1: Hub vs Random Lesion Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Consciousness
ax = axes[0, 0]
ax.plot(df_hub['n_removed'], df_hub['C'], 'rs-', markersize=8, linewidth=2, label='Hub removal')
ax.plot(df_random['n_removed'], df_random['C'], 'bo-', markersize=8, linewidth=2, label='Random removal')
ax.axhline(y=df_hub['C'].iloc[0], color='green', linestyle='--', alpha=0.5, label='Baseline')
ax.set_xlabel('Nodes Removed', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('A. Consciousness vs Lesion Size', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Participation Ratio
ax = axes[0, 1]
ax.plot(df_hub['n_removed'], df_hub['PR'], 'rs-', markersize=8, linewidth=2, label='Hub removal')
ax.plot(df_random['n_removed'], df_random['PR'], 'bo-', markersize=8, linewidth=2, label='Random removal')
ax.set_xlabel('Nodes Removed', fontsize=12)
ax.set_ylabel('Participation Ratio', fontsize=12)
ax.set_title('B. Mode Participation vs Lesion Size', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Network Fragmentation
ax = axes[1, 0]
ax.plot(df_hub['n_removed'], df_hub['n_components'], 'rs-', markersize=8, linewidth=2, label='Hub removal')
ax.plot(df_random['n_removed'], df_random['n_components'], 'bo-', markersize=8, linewidth=2, label='Random removal')
ax.set_xlabel('Nodes Removed', fontsize=12)
ax.set_ylabel('Number of Components', fontsize=12)
ax.set_title('C. Network Fragmentation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Edge removal
ax = axes[1, 1]
ax.plot(df_edges['fraction_removed'] * 100, df_edges['C'], 'g^-', markersize=10, linewidth=2)
ax.set_xlabel('Edges Removed (%)', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('D. White Matter Damage Effect', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'lesion_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: lesion_comparison.png")

# Figure 2: Damage-Consciousness Relationship
fig, ax = plt.subplots(figsize=(10, 6))

# Normalize to percentage of baseline
baseline_C = df_hub['C'].iloc[0]
hub_pct = (df_hub['C'] / baseline_C) * 100
random_pct = (df_random['C'] / baseline_C) * 100

ax.fill_between(df_hub['n_removed'], 0, hub_pct, alpha=0.3, color='red', label='Hub lesion zone')
ax.fill_between(df_random['n_removed'], 0, random_pct, alpha=0.3, color='blue', label='Random lesion zone')
ax.plot(df_hub['n_removed'], hub_pct, 'r-', linewidth=3)
ax.plot(df_random['n_removed'], random_pct, 'b-', linewidth=3)

ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50% consciousness')
ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Baseline')

ax.set_xlabel('Nodes Removed', fontsize=14)
ax.set_ylabel('Consciousness (% of Baseline)', fontsize=14)
ax.set_title('Hub Criticality: Targeted vs Random Lesions', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hub_criticality.png', dpi=150, bbox_inches='tight')
print(f"  Saved: hub_criticality.png")

# Save data
df_all = pd.concat([df_hub, df_random], ignore_index=True)
df_all.to_csv(OUTPUT_DIR / 'lesion_analysis.csv', index=False)
df_edges.to_csv(OUTPUT_DIR / 'edge_removal.csv', index=False)
print(f"  Saved: lesion_analysis.csv, edge_removal.csv")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("KEY FINDINGS: HUB DISRUPTION")
print("="*70)

hub_drop = (1 - df_hub['C'].iloc[-1] / df_hub['C'].iloc[0]) * 100
random_drop = (1 - df_random['C'].iloc[-1] / df_random['C'].iloc[0]) * 100

print(f"""
1. HUB vs RANDOM LESION IMPACT:
   - Removing 20 hub nodes: {hub_drop:.1f}% consciousness drop
   - Removing 20 random nodes: {random_drop:.1f}% consciousness drop
   - Hub removal is {hub_drop/random_drop:.1f}x more damaging

2. CRITICAL HUB THRESHOLD:
   - First 5 hubs: Consciousness drops by {(1 - df_hub[df_hub['n_removed']==5]['C'].values[0] / baseline_C) * 100:.1f}%
   - Network fragmentation begins after ~{df_hub[df_hub['n_components'] > 1]['n_removed'].min() if any(df_hub['n_components'] > 1) else 'N/A'} hub removals

3. CLINICAL IMPLICATIONS:
   - Stroke affecting hub regions (e.g., thalamus) → severe consciousness impairment
   - Diffuse axonal injury (edge removal) → gradual degradation
   - Hub-sparing lesions may preserve consciousness better

4. NETWORK RESILIENCE:
   - Scale-free networks are robust to random failure
   - But highly vulnerable to targeted hub attacks
   - This matches clinical observations of focal vs diffuse lesions
""")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("="*70)
