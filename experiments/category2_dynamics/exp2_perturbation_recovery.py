#!/usr/bin/env python3
"""
Category 2, Experiment 2: Perturbation Recovery

Test resilience of conscious states to perturbations.

RTX 5090 Enhanced: Supports environment variable configuration for scaled experiments.
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

# Configuration - supports environment variable overrides for RTX 5090 scaling
SEED = 42
N_NODES = int(os.environ.get('EXP_N_NODES', 100))
N_MODES = int(os.environ.get('EXP_N_MODES', 30))
RECOVERY_STEPS = int(os.environ.get('EXP_RECOVERY_STEPS', 50))
N_TRIALS = int(os.environ.get('EXP_N_TRIALS', 1))
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp2_perturbation_recovery'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check for PyTorch GPU support (RTX 5090)
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        USE_GPU = True
        print(f"[GPU] Using {gpu_name} ({gpu_mem:.1f} GB)")
except ImportError:
    pass

print("="*60)
print("Category 2, Experiment 2: Perturbation Recovery")
print(f"Configuration: {N_NODES} nodes, {N_MODES} modes, {RECOVERY_STEPS} recovery steps")
if USE_GPU:
    print(f"Acceleration: PyTorch CUDA (RTX 5090)")
print("="*60)

# Generate network
print("\nGenerating network...")
if USE_GPU and N_NODES > 500:
    import torch
    import networkx as nx
    G = gg.generate_small_world(N_NODES, k_neighbors=6, seed=SEED)
    L_sparse = nx.laplacian_matrix(G).toarray().astype(np.float32)
    L_torch = torch.from_numpy(L_sparse).to(device)
    eigenvalues_torch, eigenvectors_torch = torch.linalg.eigh(L_torch)
    eigenvalues = eigenvalues_torch.cpu().numpy()[:N_MODES]
    eigenvectors = eigenvectors_torch.cpu().numpy()
    L = L_sparse
    del L_torch, eigenvalues_torch, eigenvectors_torch
    torch.cuda.empty_cache()
    print(f"  Eigendecomposition on GPU complete")
else:
    G = gg.generate_small_world(N_NODES, k_neighbors=6, seed=SEED)
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    eigenvalues = eigenvalues[:N_MODES]

# Generate baseline wake state
baseline = sg.generate_wake_state(n_modes=N_MODES, seed=SEED)

# Test different perturbation magnitudes
perturbation_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
recovery_steps = RECOVERY_STEPS

all_results = []

for pert_level in tqdm(perturbation_levels, desc="Perturbation levels"):
    for trial in range(N_TRIALS):
        # Generate recovery sequence
        recovery = sg.generate_recovery_dynamics(
            baseline,
            perturbation_magnitude=pert_level,
            recovery_steps=recovery_steps,
            recovery_rate=0.1,
            seed=SEED + trial
        )
        
        # Compute metrics over time
        for t in range(recovery_steps):
            power = recovery[t]
            metrics = met.compute_all_metrics(power, eigenvalues[:len(power)])
            all_results.append({
                'perturbation_level': pert_level,
                'trial': trial,
                'time': t,
                **metrics
            })

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / 'recovery_results.csv', index=False)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for pert_level in perturbation_levels:
    subset = df[df['perturbation_level'] == pert_level]
    # Average across trials
    subset_mean = subset.groupby('time').mean().reset_index()
    for ax, metric in zip(axes.flat, ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']):
        ax.plot(subset_mean['time'], subset_mean[metric], label=f'pert={pert_level:.1f}', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Recovery')
        ax.grid(True, alpha=0.3)
        if ax == axes.flat[0]:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'recovery_dynamics.png')
print(f"\nExperiment completed! Results saved to: {OUTPUT_DIR}")
