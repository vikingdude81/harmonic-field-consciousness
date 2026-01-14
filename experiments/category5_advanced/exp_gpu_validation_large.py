#!/usr/bin/env python3
"""
Category 5: Advanced - GPU-Accelerated Validation Experiment

Tests advanced phenomena: bifurcations, chaos, critical transitions.
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
    """Create lattice Laplacian."""
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

def analyze_bifurcation(laplacian, coupling_strength):
    """Analyze critical behavior as coupling varies."""
    n_nodes = laplacian.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    
    modes = torch.randn(len(eigenvalues), device=device) * 0.1
    activity_history = []
    
    for t in range(500):
        # Coupled dynamics with bifurcation parameter
        decay = torch.exp(-eigenvalues * coupling_strength * 0.01)
        modes = modes * decay + torch.randn_like(modes) * 0.05
        
        # Nonlinearity (cubic)
        modes = modes - coupling_strength * 0.001 * modes**3
        
        activity = eigenvectors @ modes
        activity_history.append(activity.norm().item())
    
    # Metrics
    transient = np.mean(activity_history[:100])
    steady = np.mean(activity_history[250:])
    variance = np.var(activity_history[250:])
    
    return {
        'transient': transient,
        'steady': steady,
        'variance': variance,
        'lyapunov_like': variance * steady,
    }

N_NODES = 4900
print("=" * 80)
print("CATEGORY 5: ADVANCED PHENOMENA VALIDATION")
print("=" * 80)
print(f"Network size: {N_NODES} nodes")
print(f"Analysis: Bifurcations and critical transitions")
print("=" * 80)

laplacian = create_lattice_gpu(side=70)
all_results = []
start = time.time()

coupling_strengths = np.linspace(0.5, 5.0, 10)

for coupling in coupling_strengths:
    print(f"\nCoupling strength: {coupling:.2f}...", flush=True)
    results_batch = []
    
    for trial in range(10):
        result = analyze_bifurcation(laplacian, coupling)
        result.update({
            'trial': trial,
            'coupling_strength': coupling,
            'n_nodes': N_NODES,
        })
        results_batch.append(result)
    
    all_results.extend(results_batch)
    mean_var = np.mean([r['variance'] for r in results_batch])
    print(f"  Mean variance: {mean_var:.6f}")

elapsed = time.time() - start
print(f"\n[OK] Analysis complete in {elapsed:.1f}s")

# Save
output_dir = Path('results') / 'category5_advanced_validation'
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_results)
df.to_csv(output_dir / 'results.csv', index=False)

print(f"\nResults saved: {output_dir / 'results.csv'}")
print(f"Coupling range: {coupling_strengths.min():.2f} to {coupling_strengths.max():.2f}")
