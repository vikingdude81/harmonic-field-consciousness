#!/usr/bin/env python3
"""
Category 2, Experiment 4: Criticality Tuning

Find the edge-of-chaos optimal point for consciousness:
- Vary system parameters to control criticality index κ
- Test C(t) as function of κ
- Identify critical regime (κ ≈ 1)
- Compare subcritical, critical, supercritical regimes
- Generate phase transition plots
- Relate to empirical brain data (power-law distributions)

Uses GPU acceleration for batch simulations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz
from utils.gpu_utils import get_device_info, get_array_module, batch_compute_metrics_gpu, print_gpu_status
from utils.chaos_metrics import estimate_lyapunov_exponent, detect_avalanches, fit_power_law
from utils.category_theory_metrics import compute_sheaf_consistency

# Configuration - Enhanced for criticality analysis
SEED = 42
N_NODES = 300  # Larger network for better statistics
N_MODES = 80   # More modes for phase transition detection
N_ALPHA_STEPS = 60  # Denser alpha sweep for critical point detection
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_criticality_tuning'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Category 2, Experiment 4: Criticality Tuning")
print("=" * 60)

# Check GPU availability
print_gpu_status()
gpu_info = get_device_info()
USE_GPU = gpu_info['cupy_available']
xp = get_array_module(USE_GPU)

np.random.seed(SEED)

# Generate network
print("\nGenerating network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
eigenvalues = eigenvalues[:N_MODES]


def generate_critical_power(n_modes: int, alpha: float, seed: int = None) -> np.ndarray:
    """
    Generate power distribution with tunable criticality.
    
    Uses power-law distribution P(k) ~ k^(-alpha)
    - alpha < 2: subcritical (dominated by few modes)
    - alpha ≈ 2: critical (scale-free)
    - alpha > 3: supercritical (uniform-like)
    
    Args:
        n_modes: Number of modes
        alpha: Power-law exponent
        seed: Random seed
        
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(1, n_modes + 1)
    
    # Power-law base
    power = k ** (-alpha)
    
    # Add small noise for variation
    power += 0.01 * np.random.rand(n_modes)
    
    # Normalize
    power = np.maximum(power, 0)
    power = power / power.sum()
    
    return power


def compute_power_law_exponent(power: np.ndarray) -> float:
    """
    Estimate power-law exponent from distribution.
    
    Args:
        power: Power distribution
        
    Returns:
        Estimated exponent alpha
    """
    k = np.arange(1, len(power) + 1)
    
    # Fit log-log linear regression
    log_k = np.log(k)
    log_p = np.log(power + 1e-12)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)
    
    return -slope, r_value ** 2


def compute_branching_ratio(power: np.ndarray, eigenvalues: np.ndarray) -> float:
    """
    Compute branching ratio sigma as proxy for criticality.
    
    sigma < 1: subcritical (activity dies out)
    sigma = 1: critical (balanced)
    sigma > 1: supercritical (activity explodes)
    
    This is a simplified model based on spectral properties.
    """
    # Weighted sum of eigenvalues by power
    weighted_lambda = np.sum(power * eigenvalues)
    
    # Normalize to get branching ratio
    sigma = 1.0 / (1.0 + weighted_lambda)
    
    return sigma


def avalanche_size_distribution(n_samples: int = 1000, branching_ratio: float = 1.0, seed: int = None):
    """
    Simulate avalanche sizes for a given branching ratio.
    
    At criticality (sigma=1), should follow power law with exponent -3/2.
    
    Args:
        n_samples: Number of avalanches to simulate
        branching_ratio: Mean offspring per event
        seed: Random seed
        
    Returns:
        Array of avalanche sizes
    """
    if seed is not None:
        np.random.seed(seed)
    
    sizes = []
    for _ in range(n_samples):
        size = 0
        active = 1  # Start with one event
        
        while active > 0 and size < 10000:  # Prevent infinite loops
            size += active
            # Each active event spawns Poisson(branching_ratio) offspring
            active = np.random.poisson(branching_ratio * active)
        
        sizes.append(size)
    
    return np.array(sizes)


# ============================================================================
# EXPERIMENT 1: Vary power-law exponent
# ============================================================================

print("\n1. Tuning power-law exponent...")

alphas = np.linspace(0.5, 4.0, 30)
exponent_results = []

for alpha in tqdm(alphas, desc="Alpha values"):
    power = generate_critical_power(N_MODES, alpha, seed=SEED)
    
    # Compute metrics
    metrics = met.compute_all_metrics(power, eigenvalues)
    
    # Compute additional criticality measures
    estimated_alpha, r_squared = compute_power_law_exponent(power)
    branching = compute_branching_ratio(power, eigenvalues)
    
    exponent_results.append({
        'alpha': alpha,
        'estimated_alpha': estimated_alpha,
        'r_squared': r_squared,
        'branching_ratio': branching,
        **metrics
    })

df_exponent = pd.DataFrame(exponent_results)

# ============================================================================
# EXPERIMENT 2: Detailed criticality analysis
# ============================================================================

print("\n2. Analyzing critical regime...")

# Focus on alpha range around criticality (1.5 to 3.0)
critical_alphas = np.linspace(1.5, 3.0, 50)
critical_results = []

for alpha in tqdm(critical_alphas, desc="Critical range"):
    # Generate multiple samples for statistics
    c_values = []
    kappa_values = []
    
    for sample in range(20):
        power = generate_critical_power(N_MODES, alpha, seed=SEED + sample)
        metrics = met.compute_all_metrics(power, eigenvalues)
        c_values.append(metrics['C'])
        kappa_values.append(metrics['kappa'])
    
    critical_results.append({
        'alpha': alpha,
        'C_mean': np.mean(c_values),
        'C_std': np.std(c_values),
        'kappa_mean': np.mean(kappa_values),
        'kappa_std': np.std(kappa_values),
    })

df_critical = pd.DataFrame(critical_results)

# ============================================================================
# EXPERIMENT 3: Avalanche analysis
# ============================================================================

print("\n3. Avalanche size distributions...")

branching_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
avalanche_results = {}

for sigma in tqdm(branching_ratios, desc="Branching ratios"):
    sizes = avalanche_size_distribution(n_samples=5000, branching_ratio=sigma, seed=SEED)
    avalanche_results[sigma] = sizes

# ============================================================================
# EXPERIMENT 4: Brain state criticality comparison
# ============================================================================

print("\n4. Comparing brain state criticality...")

brain_states = {
    'Wake': sg.generate_wake_state(N_MODES, seed=SEED),
    'NREM': sg.generate_nrem_unconscious(N_MODES, seed=SEED),
    'Dream': sg.generate_nrem_dreaming(N_MODES, seed=SEED),
    'Anesthesia': sg.generate_anesthesia_state(N_MODES, seed=SEED),
    'Psychedelic': sg.generate_psychedelic_state(N_MODES, intensity=0.7, seed=SEED),
}

state_criticality = []
for state_name, power in brain_states.items():
    metrics = met.compute_all_metrics(power, eigenvalues)
    alpha_est, r2 = compute_power_law_exponent(power)
    branching = compute_branching_ratio(power, eigenvalues)
    
    state_criticality.append({
        'state': state_name,
        'alpha': alpha_est,
        'r_squared': r2,
        'branching_ratio': branching,
        **metrics
    })

df_states = pd.DataFrame(state_criticality)

# Save results
df_exponent.to_csv(OUTPUT_DIR / 'exponent_sweep_results.csv', index=False)
df_critical.to_csv(OUTPUT_DIR / 'critical_range_results.csv', index=False)
df_states.to_csv(OUTPUT_DIR / 'state_criticality_results.csv', index=False)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Main criticality tuning results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# C(t) vs alpha
ax = axes[0, 0]
ax.plot(df_exponent['alpha'], df_exponent['C'], 'o-', linewidth=2, markersize=6, color='darkblue')
ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='α=2 (critical)')
ax.set_xlabel('Power-law exponent α', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('C(t) vs Criticality Parameter', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Criticality index κ vs alpha
ax = axes[0, 1]
ax.plot(df_exponent['alpha'], df_exponent['kappa'], 'o-', linewidth=2, markersize=6, color='darkgreen')
ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Power-law exponent α', fontsize=12)
ax.set_ylabel('Criticality Index κ', fontsize=12)
ax.set_title('Criticality Index vs α', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Branching ratio vs alpha
ax = axes[0, 2]
ax.plot(df_exponent['alpha'], df_exponent['branching_ratio'], 'o-', linewidth=2, markersize=6, color='darkred')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='σ=1 (critical)')
ax.set_xlabel('Power-law exponent α', fontsize=12)
ax.set_ylabel('Branching Ratio σ', fontsize=12)
ax.set_title('Branching Ratio vs α', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# H_mode and PR vs alpha
ax = axes[1, 0]
ax.plot(df_exponent['alpha'], df_exponent['H_mode'], 'o-', linewidth=2, label='H_mode')
ax.plot(df_exponent['alpha'], df_exponent['PR'], 's--', linewidth=2, label='PR')
ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Power-law exponent α', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Entropy and Participation Ratio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Phase diagram: C vs κ
ax = axes[1, 1]
scatter = ax.scatter(df_exponent['kappa'], df_exponent['C'], c=df_exponent['alpha'],
                    cmap='RdYlBu_r', s=80, edgecolors='black')
ax.set_xlabel('Criticality Index κ', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('Phase Space: C vs κ', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('α', fontsize=10)
ax.grid(True, alpha=0.3)

# Critical regime zoom
ax = axes[1, 2]
ax.fill_between(df_critical['alpha'], 
                df_critical['C_mean'] - df_critical['C_std'],
                df_critical['C_mean'] + df_critical['C_std'],
                alpha=0.3, color='blue')
ax.plot(df_critical['alpha'], df_critical['C_mean'], 'b-', linewidth=2)
ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Critical point')

# Find and mark maximum
max_idx = df_critical['C_mean'].idxmax()
ax.scatter([df_critical.loc[max_idx, 'alpha']], [df_critical.loc[max_idx, 'C_mean']], 
          color='red', s=150, marker='*', zorder=5, label=f'Max at α={df_critical.loc[max_idx, "alpha"]:.2f}')

ax.set_xlabel('Power-law exponent α', fontsize=12)
ax.set_ylabel('C(t) mean ± std', fontsize=12)
ax.set_title('Critical Regime Detail', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'criticality_tuning.png', dpi=300)
print("  Saved: criticality_tuning.png")

# 2. Avalanche size distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histograms
ax = axes[0]
for sigma in branching_ratios:
    sizes = avalanche_results[sigma]
    sizes = sizes[sizes < 1000]  # Truncate for visualization
    ax.hist(sizes, bins=50, alpha=0.5, label=f'σ={sigma}', density=True)
ax.set_xlabel('Avalanche Size', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Avalanche Size Distributions', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xlim(0, 500)

# Log-log plot (power-law check)
ax = axes[1]
for sigma in branching_ratios:
    sizes = avalanche_results[sigma]
    sizes = sizes[sizes > 0]
    
    # Compute histogram
    bins = np.logspace(0, 4, 30)
    hist, bin_edges = np.histogram(sizes, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter zeros
    mask = hist > 0
    ax.loglog(bin_centers[mask], hist[mask], 'o-', alpha=0.7, label=f'σ={sigma}')

# Add reference power-law (slope -3/2 for critical)
x_ref = np.logspace(0, 3, 100)
y_ref = 0.5 * x_ref ** (-1.5)
ax.loglog(x_ref, y_ref, 'k--', linewidth=2, label='Power-law (α=-3/2)')

ax.set_xlabel('Avalanche Size (log)', fontsize=12)
ax.set_ylabel('Probability (log)', fontsize=12)
ax.set_title('Power-law Analysis', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'avalanche_analysis.png', dpi=300)
print("  Saved: avalanche_analysis.png")

# 3. Brain state criticality comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart of criticality index
ax = axes[0]
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df_states)))
bars = ax.bar(df_states['state'], df_states['kappa'], color=colors, edgecolor='black')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Optimal κ')
ax.set_ylabel('Criticality Index κ', fontsize=12)
ax.set_title('Criticality by Brain State', fontsize=14, fontweight='bold')
ax.set_xticklabels(df_states['state'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Scatter: C vs κ for brain states
ax = axes[1]
for idx, row in df_states.iterrows():
    ax.scatter(row['kappa'], row['C'], s=200, label=row['state'], edgecolors='black')
ax.set_xlabel('Criticality Index κ', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('Brain States in C-κ Space', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Power-law fit quality
ax = axes[2]
bars = ax.bar(df_states['state'], df_states['r_squared'], color='steelblue', edgecolor='black')
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Good fit (R²=0.9)')
ax.set_ylabel('Power-law Fit R²', fontsize=12)
ax.set_title('Power-law Quality by State', fontsize=14, fontweight='bold')
ax.set_xticklabels(df_states['state'], rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'brain_state_criticality.png', dpi=300)
print("  Saved: brain_state_criticality.png")

# 4. Power distributions at different alpha values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
alpha_examples = [1.0, 2.0, 3.0, 4.0]
titles = ['Subcritical (α=1)', 'Critical (α=2)', 'Near-uniform (α=3)', 'Uniform (α=4)']

for idx, (alpha, title) in enumerate(zip(alpha_examples, titles)):
    ax = axes[idx]
    power = generate_critical_power(N_MODES, alpha, seed=SEED)
    
    ax.bar(range(N_MODES), power, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mode k', fontsize=10)
    ax.set_ylabel('Power P(k)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3)

plt.suptitle('Mode Power Distributions at Different Criticality Levels', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'power_distributions.png', dpi=300, bbox_inches='tight')
print("  Saved: power_distributions.png")

plt.close('all')

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

# Find optimal criticality
optimal_idx = df_exponent['C'].idxmax()
optimal_alpha = df_exponent.loc[optimal_idx, 'alpha']
optimal_C = df_exponent.loc[optimal_idx, 'C']
optimal_kappa = df_exponent.loc[optimal_idx, 'kappa']

print(f"\nOptimal Criticality:")
print(f"  α* = {optimal_alpha:.3f}")
print(f"  C* = {optimal_C:.4f}")
print(f"  κ* = {optimal_kappa:.4f}")

print("\nRegime Classification:")
subcrit = df_exponent[df_exponent['alpha'] < 1.5]['C'].mean()
crit = df_exponent[(df_exponent['alpha'] >= 1.5) & (df_exponent['alpha'] <= 2.5)]['C'].mean()
supercrit = df_exponent[df_exponent['alpha'] > 2.5]['C'].mean()

print(f"  Subcritical (α<1.5):     C = {subcrit:.4f}")
print(f"  Critical (1.5≤α≤2.5):   C = {crit:.4f}")
print(f"  Supercritical (α>2.5):  C = {supercrit:.4f}")

print("\nBrain State Criticality Rankings:")
df_sorted = df_states.sort_values('kappa', ascending=False)
for _, row in df_sorted.iterrows():
    print(f"  {row['state']:12}: κ = {row['kappa']:.4f}, C = {row['C']:.4f}")

print("\n" + "=" * 60)
print(f"Experiment completed! Results saved to: {OUTPUT_DIR}")
print("=" * 60)
