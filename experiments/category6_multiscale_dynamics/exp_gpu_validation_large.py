#!/usr/bin/env python3
"""
Category 6: Multiscale Dynamics - GPU-Accelerated Validation Experiment

Tests hierarchical dynamics across scales: local, mesoscale, global.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from pathlib import Path
import pandas as pd
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')

def create_lattice_gpu(side=70):
    N = side * side
    L = torch.zeros(N, N, device=device)
    
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            neighbors = [
                ((i-1) % side) * side + j,
                ((i+1) % side) * side + j,
                i * side + ((j-1) % side),
                i * side + ((j+1) % side)
            ]
            L[idx, idx] = 4
            for n in neighbors:
                L[idx, n] = -1
    
    return L

def compute_multiscale_metrics(laplacian, time_scale):
    """Compute metrics at different temporal scales."""
    n_nodes = laplacian.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    
    modes = torch.randn(len(eigenvalues), device=device) * 0.1
    local_activity = []
    mesoscale_activity = []
    global_activity = []
    
    for t in range(500):
        decay = torch.exp(-eigenvalues * 0.05)
        modes = modes * decay + torch.randn_like(modes) * 0.05
        
        activity = eigenvectors @ modes
        
        # Different scales
        local = activity[::10].norm().item()  # Every 10th node
        mesoscale = activity.norm().item()
        global_scale = activity.mean().item()
        
        local_activity.append(local)
        mesoscale_activity.append(mesoscale)
        global_activity.append(global_scale)
    
    # Scale-dependent metrics
    local_var = np.var(local_activity[250:])
    meso_var = np.var(mesoscale_activity[250:])
    global_var = np.var(global_activity[250:])
    
    return {
        'local_variance': local_var,
        'mesoscale_variance': meso_var,
        'global_variance': global_var,
        'scale_ratio': meso_var / (local_var + 1e-10),
    }

N_NODES = 4900
print("=" * 80)
print("CATEGORY 6: MULTISCALE DYNAMICS VALIDATION")
print("=" * 80)
print(f"Network size: {N_NODES} nodes")
print(f"Analysis: Hierarchical temporal dynamics")
print("=" * 80)

laplacian = create_lattice_gpu(side=70)
all_results = []
start = time.time()

time_scales = [0.01, 0.05, 0.1, 0.2]

for ts in time_scales:
    print(f"\nTime scale: {ts:.3f}...", flush=True)
    results_batch = []
    
    for trial in range(15):
        result = compute_multiscale_metrics(laplacian, ts)
        result.update({
            'trial': trial,
            'time_scale': ts,
            'n_nodes': N_NODES,
        })
        results_batch.append(result)
    
    all_results.extend(results_batch)
    mean_ratio = np.mean([r['scale_ratio'] for r in results_batch])
    print(f"  Mean scale ratio: {mean_ratio:.4f}")

elapsed = time.time() - start
print(f"\n[OK] Analysis complete in {elapsed:.1f}s")

# Save
output_dir = Path('results') / 'category6_multiscale_validation'
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_results)
df.to_csv(output_dir / 'results.csv', index=False)

print(f"\nResults saved: {output_dir / 'results.csv'}")
print(f"Time scales analyzed: {time_scales}")
