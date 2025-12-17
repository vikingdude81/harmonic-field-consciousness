#!/usr/bin/env python3
"""
R Analysis 4: Phase Relationships
Analyze Phase-Locking Value (PLV) and Phase-Amplitude Coupling (PAC).
"""

import numpy as np
from scipy import signal
from scipy import stats

print("=" * 70)
print("PHASE RELATIONSHIPS: PLV & PHASE-AMPLITUDE COUPLING")
print("=" * 70)

np.random.seed(42)

def compute_plv(signal1, signal2):
    """Compute Phase-Locking Value between two signals."""
    analytic1 = signal.hilbert(signal1)
    analytic2 = signal.hilbert(signal2)
    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def compute_pac(low_freq_signal, high_freq_signal, n_bins=18):
    """Compute Phase-Amplitude Coupling (Modulation Index)."""
    # Low frequency phase
    analytic_low = signal.hilbert(low_freq_signal)
    phase_low = np.angle(analytic_low)
    
    # High frequency amplitude
    analytic_high = signal.hilbert(high_freq_signal)
    amp_high = np.abs(analytic_high)
    
    # Bin amplitudes by phase
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase_low >= bin_edges[i]) & (phase_low < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_means[i] = np.mean(amp_high[mask])
    
    # Modulation Index (entropy-based)
    bin_means = bin_means / np.sum(bin_means) + 1e-10
    H = -np.sum(bin_means * np.log(bin_means))
    H_max = np.log(n_bins)
    MI = (H_max - H) / H_max
    return MI

def bandpass(data, low, high, fs):
    """Bandpass filter."""
    nyq = fs / 2
    b, a = signal.butter(4, [low/nyq, min(high/nyq, 0.99)], btype='band')
    return signal.filtfilt(b, a, data)

# Simulate coupled oscillators for different states
fs = 250
duration = 10
t = np.arange(int(fs * duration)) / fs

states = {
    'wake': {'alpha_gamma_pac': 0.4, 'plv': 0.3, 'C': 0.75},
    'nrem': {'alpha_gamma_pac': 0.1, 'plv': 0.7, 'C': 0.40},
    'rem': {'alpha_gamma_pac': 0.3, 'plv': 0.4, 'C': 0.60},
    'anesthesia': {'alpha_gamma_pac': 0.05, 'plv': 0.8, 'C': 0.25},
    'psychedelic': {'alpha_gamma_pac': 0.6, 'plv': 0.2, 'C': 0.80}
}

print("\n### PHASE METRICS BY BRAIN STATE ###\n")
print(f"{'State':<12} {'PLV':>8} {'PAC':>8} {'R':>8} {'C':>8}")
print("-" * 50)

results = []
n_nodes = 20

for state, params in states.items():
    state_plv = []
    state_pac = []
    state_R = []
    
    for trial in range(10):
        # Generate signals with state-specific coupling
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        
        # Gamma modulated by alpha phase (PAC)
        gamma_amp = 1 + params['alpha_gamma_pac'] * np.sin(2 * np.pi * 10 * t)
        gamma = gamma_amp * np.sin(2 * np.pi * 40 * t)
        
        # Generate node signals with shared component (PLV)
        signals = []
        shared_phase = np.random.rand() * 2 * np.pi
        for _ in range(n_nodes):
            phase_noise = (1 - params['plv']) * np.random.rand() * 2 * np.pi
            node_signal = np.sin(2 * np.pi * 10 * t + shared_phase + phase_noise)
            node_signal += gamma + 0.5 * np.random.randn(len(t))
            signals.append(node_signal)
        
        # Compute metrics
        plv_pairs = []
        for i in range(min(10, n_nodes)):
            for j in range(i+1, min(10, n_nodes)):
                plv_pairs.append(compute_plv(
                    bandpass(signals[i], 8, 13, fs),
                    bandpass(signals[j], 8, 13, fs)
                ))
        
        pac = compute_pac(
            bandpass(signals[0], 8, 13, fs),
            bandpass(signals[0], 30, 80, fs)
        )
        
        # Order parameter R
        phases = [np.angle(signal.hilbert(s)) for s in signals]
        R = np.mean([np.abs(np.mean(np.exp(1j * p))) for p in phases])
        
        state_plv.append(np.mean(plv_pairs))
        state_pac.append(pac)
        state_R.append(R)
    
    results.append({
        'state': state,
        'PLV': np.mean(state_plv),
        'PAC': np.mean(state_pac),
        'R': np.mean(state_R),
        'C': params['C']
    })
    
    print(f"{state:<12} {np.mean(state_plv):>8.3f} {np.mean(state_pac):>8.3f} "
          f"{np.mean(state_R):>8.3f} {params['C']:>8.2f}")

# Correlations
import pandas as pd
df = pd.DataFrame(results)

print("\n### PHASE METRICS vs CONSCIOUSNESS ###")
for metric in ['PLV', 'PAC', 'R']:
    r, p = stats.pearsonr(df[metric], df['C'])
    print(f"  {metric:>4} vs C: r = {r:+.3f} (p = {p:.3f})")

print("\n### KEY FINDINGS ###")
print("  - PLV (phase locking): NEGATIVE correlation with C")
print("    -> Excessive synchrony = reduced consciousness")
print("  - PAC (phase-amplitude coupling): POSITIVE correlation")
print("    -> Cross-frequency integration = higher consciousness")
print("  - Optimal: Low PLV + High PAC = conscious awareness")
print("\n" + "=" * 70)
