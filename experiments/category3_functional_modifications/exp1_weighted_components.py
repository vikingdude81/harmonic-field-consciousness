#!/usr/bin/env python3
"""
Category 3, Experiment 1: Weighted Components

Optimize component weightings:
- Test combinations of weights for 5 components
- Grid search approach
- Maximize separation between Wake and Anesthesia
- Generate weight sensitivity heatmaps
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz

# Configuration
SEED = 42
N_NODES = 100
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp1_weighted_components'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 3, Experiment 1: Weighted Components")
print("="*60)

# Generate network
print("\nGenerating network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

n_modes = 30
eigenvalues = eigenvalues[:n_modes]

# Generate wake and anesthesia states
print("Generating brain states...")
wake_power = sg.generate_wake_state(n_modes=n_modes, seed=SEED)
anesthesia_power = sg.generate_anesthesia_state(n_modes=n_modes, seed=SEED)

# Compute base metrics
print("Computing base metrics...")
wake_metrics = met.compute_all_metrics(wake_power, eigenvalues)
anesthesia_metrics = met.compute_all_metrics(anesthesia_power, eigenvalues)

print("\nBase metrics (equal weights):")
print(f"  Wake C(t):       {wake_metrics['C']:.3f}")
print(f"  Anesthesia C(t): {anesthesia_metrics['C']:.3f}")
print(f"  Separation:      {wake_metrics['C'] - anesthesia_metrics['C']:.3f}")

# ============================================================================
# WEIGHT OPTIMIZATION
# ============================================================================

print("\nTesting different weight combinations...")

# Grid search over weights
# Test 6 values for each weight, but ensure they sum to 1
weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

results = []

# Generate all combinations (this is a lot, so we'll sample)
np.random.seed(SEED)
n_samples = 1000

for _ in tqdm(range(n_samples), desc="Weight combinations"):
    # Generate random weights
    weights = np.random.dirichlet(np.ones(5))
    
    # Compute C(t) with these weights
    C_wake = met.compute_consciousness_functional(
        wake_metrics['H_mode'],
        wake_metrics['PR'],
        wake_metrics['R'],
        wake_metrics['S_dot'],
        wake_metrics['kappa'],
        weights=tuple(weights)
    )
    
    C_anesthesia = met.compute_consciousness_functional(
        anesthesia_metrics['H_mode'],
        anesthesia_metrics['PR'],
        anesthesia_metrics['R'],
        anesthesia_metrics['S_dot'],
        anesthesia_metrics['kappa'],
        weights=tuple(weights)
    )
    
    separation = C_wake - C_anesthesia
    
    results.append({
        'w_H': weights[0],
        'w_PR': weights[1],
        'w_R': weights[2],
        'w_Sdot': weights[3],
        'w_kappa': weights[4],
        'C_wake': C_wake,
        'C_anesthesia': C_anesthesia,
        'separation': separation,
    })

df = pd.DataFrame(results)

# Find best weights
best_idx = df['separation'].idxmax()
best_weights = df.loc[best_idx, ['w_H', 'w_PR', 'w_R', 'w_Sdot', 'w_kappa']].values
best_separation = df.loc[best_idx, 'separation']

print("\n" + "="*60)
print("Optimization Results:")
print("="*60)
print(f"\nBest weights:")
print(f"  w_H:     {best_weights[0]:.3f}")
print(f"  w_PR:    {best_weights[1]:.3f}")
print(f"  w_R:     {best_weights[2]:.3f}")
print(f"  w_Sdot:  {best_weights[3]:.3f}")
print(f"  w_kappa: {best_weights[4]:.3f}")
print(f"\nBest separation: {best_separation:.3f}")

# Save results
csv_path = OUTPUT_DIR / 'weight_optimization_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Separation distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['separation'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(best_separation, color='red', linestyle='--', linewidth=2, 
          label=f'Best: {best_separation:.3f}')
ax.axvline(wake_metrics['C'] - anesthesia_metrics['C'], color='green', 
          linestyle='--', linewidth=2, label='Equal weights')
ax.set_xlabel('Separation (C_wake - C_anesthesia)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Wake-Anesthesia Separation', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'separation_distribution.png')
print(f"  Saved: separation_distribution.png")

# 2. Weight importance
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

weight_names = ['w_H', 'w_PR', 'w_R', 'w_Sdot', 'w_kappa']
for idx, (ax, weight_name) in enumerate(zip(axes[:5], weight_names)):
    ax.scatter(df[weight_name], df['separation'], alpha=0.3, s=10, color='steelblue')
    ax.set_xlabel(weight_name, fontsize=11)
    ax.set_ylabel('Separation', fontsize=11)
    ax.set_title(f'Separation vs {weight_name}', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df[[weight_name, 'separation']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
           va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'weight_vs_separation.png')
print(f"  Saved: weight_vs_separation.png")

# 3. Pairwise weight interactions (top 3 most important)
# Find most important weights by correlation
correlations = [df[[wn, 'separation']].corr().iloc[0, 1] for wn in weight_names]
top_weights_idx = np.argsort(np.abs(correlations))[::-1][:3]
top_weights = [weight_names[i] for i in top_weights_idx]

print(f"\nMost important weights: {top_weights}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (w1, w2) in enumerate(itertools.combinations(top_weights, 2)):
    ax = axes[idx]
    scatter = ax.scatter(df[w1], df[w2], c=df['separation'], 
                        cmap='RdYlGn', alpha=0.5, s=20)
    ax.set_xlabel(w1, fontsize=11)
    ax.set_ylabel(w2, fontsize=11)
    ax.set_title(f'{w1} vs {w2}', fontsize=11)
    plt.colorbar(scatter, ax=ax, label='Separation')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'weight_interactions.png')
print(f"  Saved: weight_interactions.png")

# 4. Comparison of best vs equal weights
fig, ax = plt.subplots(figsize=(10, 6))

weights_comparison = {
    'Equal weights': (0.2, 0.2, 0.2, 0.2, 0.2),
    'Optimized weights': tuple(best_weights),
}

x = np.arange(5)
width = 0.35
component_names = ['$H_{mode}$', '$PR$', '$R$', '$\\dot{S}$', '$\\kappa$']

for i, (label, weights) in enumerate(weights_comparison.items()):
    offset = (i - 0.5) * width
    ax.bar(x + offset, weights, width, label=label, alpha=0.7)

ax.set_xlabel('Component', fontsize=11)
ax.set_ylabel('Weight', fontsize=11)
ax.set_title('Comparison of Weight Schemes', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(component_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'weight_comparison.png')
print(f"  Saved: weight_comparison.png")

# 5. Feature importance ranking
print("\nFeature importance (by correlation with separation):")
for weight_name, corr in zip(weight_names, correlations):
    print(f"  {weight_name:10s}: r = {corr:+.3f}")

plt.close('all')

print("\n" + "="*60)
print("Experiment completed successfully!")
print(f"All results saved to: {OUTPUT_DIR}")
print("="*60)
