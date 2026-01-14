#!/usr/bin/env python3
"""
Category 4: Applications - GPU-Accelerated Validation Experiment

Tests real-world applications: brain stimulation, drug effects, cognitive tasks.
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
    print("WARNING: CUDA not available; using CPU")

def create_lattice_gpu(side=70):
    """Create 2D lattice Laplacian."""
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

def simulate_with_intervention(laplacian, intervention_type, intensity=0.1):
    """Simulate dynamics with different interventions."""
    n_nodes = laplacian.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    
    # Initial state (resting)
    activity = torch.randn(n_nodes, device=device) * 0.1
    modes = eigenvectors.t() @ activity
    
    trajectory = []
    
    for t in range(500):
        # Baseline dynamics
        decay = torch.exp(-eigenvalues * 0.05)
        modes = modes * decay + torch.randn_like(modes) * 0.05
        
        # Apply intervention
        if intervention_type == 'stimulation' and t > 100:
            # Focal stimulation
            focal_idx = 50
            modes += torch.randn(len(modes), device=device) * intensity * (1 if t < 200 else 0)
        
        elif intervention_type == 'pharmacological' and t > 50:
            # Reduce variability (like anesthesia)
            modes = modes * (1 - intensity * 0.01)
        
        elif intervention_type == 'cognitive' and t > 100:
            # Increase complexity
            modes = modes * (1 + intensity * 0.001)
        
        # Project back to spatial
        activity = eigenvectors @ modes
        trajectory.append(activity.norm().item())
    
    # Compute metrics
    baseline = np.mean(trajectory[:100])
    response = np.mean(trajectory[150:250])
    recovery = np.mean(trajectory[400:])
    
    return {
        'baseline': baseline,
        'response': response,
        'recovery': recovery,
        'change': response - baseline,
    }

# Validation scale
N_NODES = 4900
print("=" * 80)
print("CATEGORY 4: APPLICATIONS VALIDATION")
print("=" * 80)
print(f"Network size: {N_NODES} nodes")
print(f"Interventions: stimulation, pharmacological, cognitive")
print("=" * 80)

laplacian = create_lattice_gpu(side=70)
all_results = []
start = time.time()

interventions = ['stimulation', 'pharmacological', 'cognitive']
intensities = [0.05, 0.1, 0.2]

for inter in interventions:
    for intensity in intensities:
        print(f"\n{inter.upper()} (intensity={intensity})...")
        results_batch = []
        
        for trial in range(20):
            result = simulate_with_intervention(laplacian, inter, intensity)
            result.update({
                'trial': trial,
                'intervention': inter,
                'intensity': intensity,
                'n_nodes': N_NODES,
            })
            results_batch.append(result)
        
        all_results.extend(results_batch)
        mean_change = np.mean([r['change'] for r in results_batch])
        print(f"  Mean response: {mean_change:.6f}")

elapsed = time.time() - start
print(f"\n[OK] All interventions completed in {elapsed:.1f}s")

# Save
output_dir = Path('results') / 'category4_applications_validation'
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_results)
df.to_csv(output_dir / 'results.csv', index=False)

print(f"\nResults saved: {output_dir / 'results.csv'}")
print(f"\nSummary:")
print(f"  Total trials: {len(df)}")
print(f"  Interventions: {df['intervention'].unique()}")
print(f"  Mean baseline: {df['baseline'].mean():.6f}")
print(f"  Mean response: {df['response'].mean():.6f}")
