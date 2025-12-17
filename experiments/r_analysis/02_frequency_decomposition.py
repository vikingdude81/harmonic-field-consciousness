#!/usr/bin/env python3
"""
R Analysis 2: Frequency Decomposition
Compute R in different frequency bands and test which best predicts C.
"""

import numpy as np
from scipy import signal
from scipy import stats
from pathlib import Path

print("=" * 70)
print("FREQUENCY DECOMPOSITION: BAND-SPECIFIC R -> C")
print("=" * 70)

np.random.seed(42)

# Frequency bands (Hz, for neural signals)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80)
}

# Simulate multi-frequency neural signals
n_nodes = 100
n_samples = 2000
fs = 250  # Hz

def generate_neural_signal(dominant_band, n_samples, fs):
    """Generate signal with dominant frequency band."""
    t = np.arange(n_samples) / fs
    signal_out = np.zeros(n_samples)
    for band, (low, high) in BANDS.items():
        freq = (low + high) / 2
        amp = 1.5 if band == dominant_band else 0.3
        signal_out += amp * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
    signal_out += 0.2 * np.random.randn(n_samples)
    return signal_out

def bandpass_filter(data, low, high, fs):
    """Apply bandpass filter."""
    nyq = fs / 2
    low_n, high_n = low / nyq, min(high / nyq, 0.99)
    b, a = signal.butter(4, [low_n, high_n], btype='band')
    return signal.filtfilt(b, a, data)

def compute_band_synchronization(signals, band, fs):
    """Compute synchronization R within frequency band."""
    low, high = BANDS[band]
    filtered = np.array([bandpass_filter(s, low, high, fs) for s in signals])
    # Hilbert transform for instantaneous phase
    analytic = signal.hilbert(filtered, axis=1)
    phases = np.angle(analytic)
    # Order parameter R
    R = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return np.mean(R)

# Simulate different brain states
states = {
    'wake': {'dominant': 'alpha', 'C': 0.75},
    'nrem': {'dominant': 'delta', 'C': 0.45},
    'rem': {'dominant': 'theta', 'C': 0.65},
    'anesthesia': {'dominant': 'delta', 'C': 0.30},
    'meditation': {'dominant': 'alpha', 'C': 0.80}
}

print("\n### BAND-SPECIFIC SYNCHRONIZATION BY STATE ###\n")
print(f"{'State':<12} {'delta':>8} {'theta':>8} {'alpha':>8} {'beta':>8} {'gamma':>8} {'C':>8}")
print("-" * 70)

results = []
for state, params in states.items():
    # Generate signals for this state
    signals = [generate_neural_signal(params['dominant'], n_samples, fs) 
               for _ in range(n_nodes)]
    
    band_R = {}
    for band in BANDS:
        band_R[band] = compute_band_synchronization(signals, band, fs)
    
    results.append({
        'state': state,
        'C': params['C'],
        **{f'R_{b}': v for b, v in band_R.items()}
    })
    
    print(f"{state:<12} {band_R['delta']:>8.3f} {band_R['theta']:>8.3f} "
          f"{band_R['alpha']:>8.3f} {band_R['beta']:>8.3f} {band_R['gamma']:>8.3f} "
          f"{params['C']:>8.2f}")

# Correlate each band R with C
print("\n### BAND-CONSCIOUSNESS CORRELATIONS ###")
import pandas as pd
df = pd.DataFrame(results)

for band in BANDS:
    r, p = stats.pearsonr(df[f'R_{band}'], df['C'])
    sig = "***" if p < 0.05 else ""
    print(f"  R_{band:<6} vs C: r = {r:+.3f} (p = {p:.3f}) {sig}")

# Find best predictor
correlations = {b: stats.pearsonr(df[f'R_{b}'], df['C'])[0] for b in BANDS}
best_band = max(correlations, key=lambda x: abs(correlations[x]))
print(f"\n>>> Best predictor: R_{best_band} (r = {correlations[best_band]:.3f})")

print("\n### FREQUENCY-SPECIFIC INTERPRETATION ###")
print("""
  - Alpha-band synchronization: Associated with conscious awareness
  - Delta-band synchronization: Associated with unconscious states
  - Gamma-band: Local processing, integration marker
  - Cross-frequency coupling (alpha-gamma) may be optimal marker
""")
print("=" * 70)
