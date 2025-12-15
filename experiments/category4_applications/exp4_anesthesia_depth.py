#!/usr/bin/env python3
"""
Category 4, Experiment 4: Anesthesia Depth

Model anesthesia depth monitoring with continuous spectrum.
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
N_NODES = 100
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_anesthesia_depth'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 4, Experiment 4: Anesthesia Depth")
print("="*60)

# Generate network
G = gg.generate_small_world(N_NODES, k_neighbors=6, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
eigenvalues = eigenvalues[:30]

# Generate anesthesia depth spectrum
depth_levels = np.linspace(0, 1, 21)  # 0 = awake, 1 = deep anesthesia

results = []
for depth in tqdm(depth_levels, desc="Anesthesia depths"):
    # Interpolate between wake and deep anesthesia
    wake_power = sg.generate_wake_state(n_modes=30, seed=SEED)
    anes_power = sg.generate_anesthesia_state(n_modes=30, depth=1.0, seed=SEED)
    
    power = sg.interpolate_states(wake_power, anes_power, depth)
    
    # Compute metrics
    metrics = met.compute_all_metrics(power, eigenvalues)
    
    # Add clinical scales (simulated correlation)
    # BIS scale: 100 = awake, 0 = deep anesthesia
    bis_score = 100 * (1 - depth) + np.random.normal(0, 5)
    bis_score = np.clip(bis_score, 0, 100)
    
    # Ramsay scale: 1 = awake, 6 = deep
    ramsay_score = 1 + 5 * depth + np.random.normal(0, 0.3)
    ramsay_score = np.clip(ramsay_score, 1, 6)
    
    results.append({
        'depth': depth,
        'bis_score': bis_score,
        'ramsay_score': ramsay_score,
        **metrics
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'anesthesia_depth_results.csv', index=False)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Metrics vs depth
for ax, metric in zip(axes.flat[:5], ['H_mode', 'PR', 'R', 'S_dot', 'kappa']):
    ax.plot(df['depth'], df[metric], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Anesthesia Depth')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Depth')
    ax.grid(True, alpha=0.3)

# C(t) vs clinical scales
ax = axes[1, 2]
ax2 = ax.twinx()

ax.plot(df['depth'], df['C'], 'b-o', linewidth=2, markersize=6, label='C(t)')
ax2.plot(df['depth'], df['bis_score'], 'r--s', linewidth=2, markersize=4, label='BIS')

ax.set_xlabel('Anesthesia Depth')
ax.set_ylabel('C(t)', color='b')
ax2.set_ylabel('BIS Score', color='r')
ax.set_title('C(t) vs BIS Scale')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anesthesia_monitoring.png')

# Correlation with clinical scales
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['C'], df['bis_score'], alpha=0.6, s=50)
axes[0].set_xlabel('C(t)')
axes[0].set_ylabel('BIS Score')
axes[0].set_title('C(t) vs BIS Correlation')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['C'], df['ramsay_score'], alpha=0.6, s=50, color='orange')
axes[1].set_xlabel('C(t)')
axes[1].set_ylabel('Ramsay Score')
axes[1].set_title('C(t) vs Ramsay Correlation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'clinical_correlation.png')

print(f"\nCorrelations with clinical scales:")
print(f"  C(t) vs BIS:    r = {df[['C', 'bis_score']].corr().iloc[0, 1]:.3f}")
print(f"  C(t) vs Ramsay: r = {df[['C', 'ramsay_score']].corr().iloc[0, 1]:.3f}")
print(f"\nExperiment completed! Results saved to: {OUTPUT_DIR}")
