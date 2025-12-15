#!/usr/bin/env python3
"""
Category 1, Experiment 2: Network Scaling

Test how network size affects consciousness metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg

SEED = 42
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp2_network_scaling'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 1, Experiment 2: Network Scaling")
print("="*60)

# Test different network sizes
network_sizes = [50, 100, 200, 500, 1000]
wake_power_base = sg.generate_wake_state(n_modes=50, seed=SEED)

results = []
for n_nodes in tqdm(network_sizes, desc="Network sizes"):
    # Generate network
    G = gg.generate_small_world(n_nodes, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    
    # Use appropriate number of modes
    n_modes = min(50, len(eigenvalues))
    eigenvalues_trunc = eigenvalues[:n_modes]
    power = wake_power_base[:n_modes]
    power = power / power.sum()
    
    # Compute metrics
    metrics = met.compute_all_metrics(power, eigenvalues_trunc)
    
    results.append({
        'n_nodes': n_nodes,
        'n_edges': G.number_of_edges(),
        **metrics
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'scaling_results.csv', index=False)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics_to_plot = ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']

for ax, metric in zip(axes.flat, metrics_to_plot):
    ax.plot(df['n_nodes'], df[metric], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Network Size')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'scaling_analysis.png')
print(f"\nExperiment completed! Results saved to: {OUTPUT_DIR}")
