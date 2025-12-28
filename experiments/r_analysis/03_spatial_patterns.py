#!/usr/bin/env python3
"""
R Analysis 3: Spatial Patterns
Compute local R for network communities and test if heterogeneity matters.
"""

import numpy as np
import networkx as nx
from scipy import stats
from pathlib import Path

print("=" * 70)
print("SPATIAL R PATTERNS: LOCAL VS GLOBAL SYNCHRONIZATION")
print("=" * 70)

np.random.seed(42)

def create_modular_network(n_modules=5, nodes_per_module=20, p_in=0.3, p_out=0.05):
    """Create network with community structure."""
    sizes = [nodes_per_module] * n_modules
    probs = [[p_in if i == j else p_out for j in range(n_modules)] for i in range(n_modules)]
    G = nx.stochastic_block_model(sizes, probs)
    return G, [list(range(i*nodes_per_module, (i+1)*nodes_per_module)) for i in range(n_modules)]

def compute_local_R(phases, community):
    """Compute synchronization within a community."""
    comm_phases = phases[community]
    return np.abs(np.mean(np.exp(1j * comm_phases)))

def compute_global_R(phases):
    """Compute global synchronization."""
    return np.abs(np.mean(np.exp(1j * phases)))

def simulate_dynamics(G, communities, coupling=0.5, steps=50, state='wake'):
    """Simulate Kuramoto dynamics on network (vectorized)."""
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G)
    
    # State-dependent natural frequencies
    omega_base = {'wake': 10, 'nrem': 2, 'rem': 6, 'anesthesia': 1}
    omega = omega_base.get(state, 5) + np.random.randn(n) * 2
    
    # Initialize phases
    theta = np.random.rand(n) * 2 * np.pi
    dt = 0.05
    
    local_R_history = {i: [] for i in range(len(communities))}
    global_R_history = []
    
    for _ in range(steps):
        # Vectorized Kuramoto update
        sin_diff = np.sin(theta[None, :] - theta[:, None])  # n x n
        coupling_term = coupling * np.sum(adj * sin_diff, axis=1)
        theta = theta + dt * (omega + coupling_term)
        
        for i, comm in enumerate(communities):
            local_R_history[i].append(compute_local_R(theta, comm))
        global_R_history.append(compute_global_R(theta))
    
    return {
        'local_R': {i: np.mean(v[-10:]) for i, v in local_R_history.items()},
        'global_R': np.mean(global_R_history[-10:]),
        'R_heterogeneity': np.std([np.mean(v[-10:]) for v in local_R_history.values()])
    }

# Run simulations
print("\n### LOCAL vs GLOBAL SYNCHRONIZATION BY STATE ###\n")

n_trials = 10
states = ['wake', 'nrem', 'rem', 'anesthesia']
consciousness = {'wake': 0.75, 'nrem': 0.40, 'rem': 0.60, 'anesthesia': 0.25}

all_results = []

for state in states:
    for trial in range(n_trials):
        G, communities = create_modular_network()
        result = simulate_dynamics(G, communities, state=state)
        result['state'] = state
        result['C'] = consciousness[state]
        result['trial'] = trial
        all_results.append(result)

import pandas as pd
df = pd.DataFrame(all_results)

print(f"{'State':<12} {'Global R':>10} {'R_hetero':>10} {'C':>8}")
print("-" * 50)
for state in states:
    subset = df[df['state'] == state]
    print(f"{state:<12} {subset['global_R'].mean():>10.3f} "
          f"{subset['R_heterogeneity'].mean():>10.3f} {consciousness[state]:>8.2f}")

# Correlations
print("\n### SPATIAL R FEATURES vs CONSCIOUSNESS ###")
r1, p1 = stats.pearsonr(df['global_R'], df['C'])
r2, p2 = stats.pearsonr(df['R_heterogeneity'], df['C'])
print(f"  Global R vs C:        r = {r1:+.3f} (p = {p1:.3f})")
print(f"  R heterogeneity vs C: r = {r2:+.3f} (p = {p2:.3f})")

# Multiple regression
print("\n### MULTIVARIATE: C ~ Global_R + Heterogeneity ###")
X = np.column_stack([np.ones(len(df)), df['global_R'], df['R_heterogeneity']])
y = df['C'].values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"  C = {beta[0]:.3f} + {beta[1]:.3f}*Global_R + {beta[2]:.3f}*Heterogeneity")

y_pred = X @ beta
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
print(f"  R^2 = {r2:.3f}")

print("\n### KEY INSIGHT ###")
print("  R heterogeneity (variance across modules) may be more")
print("  informative than global R alone - suggests differentiated")
print("  integration as consciousness marker.")
print("\n" + "=" * 70)
