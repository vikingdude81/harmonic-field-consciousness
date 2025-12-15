#!/usr/bin/env python3
"""
Category 2, Experiment 2: Perturbation Recovery

Test resilience of conscious states to perturbations.
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
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp2_perturbation_recovery'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 2, Experiment 2: Perturbation Recovery")
print("="*60)

# Generate network
G = gg.generate_small_world(N_NODES, k_neighbors=6, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
eigenvalues = eigenvalues[:30]

# Generate baseline wake state
baseline = sg.generate_wake_state(n_modes=30, seed=SEED)

# Test different perturbation magnitudes
perturbation_levels = [0.1, 0.2, 0.3, 0.5, 0.7]
recovery_steps = 50

all_results = []

for pert_level in tqdm(perturbation_levels, desc="Perturbation levels"):
    # Generate recovery sequence
    recovery = sg.generate_recovery_dynamics(
        baseline,
        perturbation_magnitude=pert_level,
        recovery_steps=recovery_steps,
        recovery_rate=0.1,
        seed=SEED
    )
    
    # Compute metrics over time
    for t in range(recovery_steps):
        power = recovery[t]
        metrics = met.compute_all_metrics(power, eigenvalues)
        all_results.append({
            'perturbation_level': pert_level,
            'time': t,
            **metrics
        })

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / 'recovery_results.csv', index=False)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for pert_level in perturbation_levels:
    subset = df[df['perturbation_level'] == pert_level]
    for ax, metric in zip(axes.flat, ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']):
        ax.plot(subset['time'], subset[metric], label=f'pert={pert_level:.1f}', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Recovery')
        ax.grid(True, alpha=0.3)
        if ax == axes.flat[0]:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'recovery_dynamics.png')
print(f"\nExperiment completed! Results saved to: {OUTPUT_DIR}")
