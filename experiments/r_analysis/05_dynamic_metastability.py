#!/usr/bin/env python3
"""
R Analysis 5: Dynamic R & Metastability
Analyze R variance over time and state transition patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal as sig

print("=" * 70)
print("DYNAMIC R ANALYSIS: METASTABILITY & STATE TRANSITIONS")
print("=" * 70)

np.random.seed(42)

def simulate_kuramoto_dynamics(n_nodes, coupling, steps, natural_freq_std=1.0):
    """Simulate Kuramoto oscillators and return R time series."""
    omega = np.random.randn(n_nodes) * natural_freq_std
    theta = np.random.rand(n_nodes) * 2 * np.pi
    dt = 0.1
    
    R_series = []
    for _ in range(steps):
        # Kuramoto update
        mean_field = np.mean(np.exp(1j * theta))
        R = np.abs(mean_field)
        psi = np.angle(mean_field)
        
        dtheta = omega + coupling * R * np.sin(psi - theta)
        theta = (theta + dt * dtheta) % (2 * np.pi)
        R_series.append(R)
    
    return np.array(R_series)

def compute_metastability(R_series, window=50):
    """Metastability = variance of R over time."""
    return np.var(R_series)

def detect_transitions(R_series, threshold=0.1):
    """Detect rapid transitions in R."""
    dR = np.abs(np.diff(R_series))
    transitions = np.sum(dR > threshold)
    return transitions

def compute_dwell_times(R_series, R_threshold=0.5):
    """Compute dwell times in high/low R states."""
    high_state = R_series > R_threshold
    changes = np.diff(high_state.astype(int))
    
    # Find dwell times
    high_dwells = []
    low_dwells = []
    current_dwell = 1
    in_high = high_state[0]
    
    for i in range(1, len(high_state)):
        if high_state[i] == high_state[i-1]:
            current_dwell += 1
        else:
            if in_high:
                high_dwells.append(current_dwell)
            else:
                low_dwells.append(current_dwell)
            current_dwell = 1
            in_high = high_state[i]
    
    return np.mean(high_dwells) if high_dwells else 0, np.mean(low_dwells) if low_dwells else 0

# Simulate different brain states
print("\n### METASTABILITY BY BRAIN STATE ###\n")

states_config = {
    'wake': {'coupling': 0.5, 'freq_std': 2.0, 'C': 0.75},
    'nrem': {'coupling': 1.5, 'freq_std': 0.5, 'C': 0.40},
    'rem': {'coupling': 0.8, 'freq_std': 1.5, 'C': 0.65},
    'anesthesia': {'coupling': 2.0, 'freq_std': 0.3, 'C': 0.25},
    'psychedelic': {'coupling': 0.3, 'freq_std': 3.0, 'C': 0.80}
}

results = []
n_nodes = 50
n_steps = 500

print(f"{'State':<12} {'Mean R':>8} {'Meta':>8} {'Trans':>8} {'High_dwell':>10} {'C':>6}")
print("-" * 60)

for state, config in states_config.items():
    R_series = simulate_kuramoto_dynamics(
        n_nodes, config['coupling'], n_steps, config['freq_std']
    )
    
    meta = compute_metastability(R_series)
    trans = detect_transitions(R_series)
    high_dwell, low_dwell = compute_dwell_times(R_series)
    
    results.append({
        'state': state,
        'mean_R': np.mean(R_series),
        'metastability': meta,
        'transitions': trans,
        'high_dwell': high_dwell,
        'low_dwell': low_dwell,
        'C': config['C']
    })
    
    print(f"{state:<12} {np.mean(R_series):>8.3f} {meta:>8.4f} {trans:>8d} "
          f"{high_dwell:>10.1f} {config['C']:>6.2f}")

df = pd.DataFrame(results)

# Correlations with consciousness
print("\n### DYNAMIC R METRICS vs CONSCIOUSNESS ###")
metrics = ['mean_R', 'metastability', 'transitions', 'high_dwell']
for metric in metrics:
    r, p = stats.pearsonr(df[metric], df['C'])
    sig = "***" if p < 0.1 else ""
    print(f"  {metric:<15} vs C: r = {r:+.3f} (p = {p:.3f}) {sig}")

# Optimal metastability
print("\n### METASTABILITY SWEET SPOT ###")
print("  Theory: Consciousness maximized at intermediate metastability")
print("  - Too low: Stuck in fixed state (anesthesia)")
print("  - Too high: Chaotic, no stable patterns (noise)")
print("  - Optimal: Dynamic but structured (wake, psychedelic)")

# Transition rate analysis
print("\n### TRANSITION RATE ANALYSIS ###")
df_sorted = df.sort_values('C', ascending=False)
print("  States ordered by consciousness:")
for _, row in df_sorted.iterrows():
    print(f"    {row['state']:<12}: C={row['C']:.2f}, transitions={row['transitions']}, "
          f"meta={row['metastability']:.4f}")

print("\n### KEY INSIGHT ###")
print("  Metastability (R variance) may be optimal consciousness marker:")
print("  - Captures dynamic repertoire")
print("  - Reflects integration-differentiation balance")
print("  - Predicts state transitions")
print("\n" + "=" * 70)
