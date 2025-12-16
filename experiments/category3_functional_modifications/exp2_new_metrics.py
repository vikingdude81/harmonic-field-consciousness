#!/usr/bin/env python3
"""
Category 3, Experiment 2: New Metrics

Add novel complexity measures to the consciousness functional:
- Lempel-Ziv complexity (LZC)
- Multiscale entropy (MSE)
- Sample entropy (SampEn)
- Permutation entropy (PE)
- Neural complexity (NC)
- Integrated information proxy (Φ*)

Compare to existing 5 components, analyze correlations,
and rank feature importance for consciousness detection.

Uses GPU acceleration for batch entropy computations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz
from utils.gpu_utils import get_device_info, get_array_module, print_gpu_status

# Configuration
SEED = 42
N_NODES = 100
N_MODES = 30
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp2_new_metrics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Category 3, Experiment 2: New Metrics")
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


# ============================================================================
# NEW METRIC IMPLEMENTATIONS
# ============================================================================

def lempel_ziv_complexity(signal: np.ndarray, threshold: str = 'median') -> float:
    """
    Compute Lempel-Ziv complexity of a signal.
    
    LZC measures the rate of new pattern generation in a binary sequence.
    Higher LZC = more complex/random, Lower LZC = more regular/compressible.
    
    Args:
        signal: 1D signal array
        threshold: 'median', 'mean', or float value for binarization
        
    Returns:
        Normalized LZC value in [0, 1]
    """
    # Binarize the signal
    if threshold == 'median':
        thresh = np.median(signal)
    elif threshold == 'mean':
        thresh = np.mean(signal)
    else:
        thresh = threshold
    
    binary = ''.join(['1' if x > thresh else '0' for x in signal])
    
    # Lempel-Ziv 76 algorithm
    n = len(binary)
    i = 0
    c = 1  # complexity counter
    k = 1
    l = 1
    
    while k + l <= n:
        if binary[k + l - 1] != binary[i + l - 1]:
            i += 1
            if i == k:
                c += 1
                k += l
                l = 1
                i = 0
            else:
                l = 1
        else:
            l += 1
    
    # Normalize by theoretical maximum
    b = n / np.log2(n) if n > 1 else 1
    lzc = c / b
    
    return min(lzc, 1.0)


def sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy of a signal.
    
    SampEn measures the irregularity/unpredictability of a time series.
    Lower SampEn = more regular, Higher SampEn = more irregular.
    
    Args:
        signal: 1D signal array
        m: Embedding dimension
        r: Tolerance (as fraction of signal std)
        
    Returns:
        Sample entropy value
    """
    N = len(signal)
    if N < m + 2:
        return 0.0
    
    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    r_abs = r * np.std(signal)
    
    def count_matches(template_length):
        count = 0
        for i in range(N - template_length):
            for j in range(i + 1, N - template_length):
                # Check if templates match within tolerance
                diff = np.abs(signal[i:i+template_length] - signal[j:j+template_length])
                if np.max(diff) < r_abs:
                    count += 1
        return count
    
    A = count_matches(m + 1)
    B = count_matches(m)
    
    if B == 0 or A == 0:
        return 0.0
    
    return -np.log(A / B)


def permutation_entropy(signal: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """
    Compute Permutation Entropy of a signal.
    
    PE quantifies the complexity based on ordinal patterns.
    Higher PE = more complex/random patterns.
    
    Args:
        signal: 1D signal array
        order: Embedding dimension (pattern length)
        delay: Time delay between elements
        normalize: Whether to normalize by maximum entropy
        
    Returns:
        Permutation entropy value
    """
    N = len(signal)
    if N < order:
        return 0.0
    
    # Generate all possible permutations
    perm_list = list(permutations(range(order)))
    perm_dict = {p: i for i, p in enumerate(perm_list)}
    
    # Count ordinal patterns
    counts = np.zeros(len(perm_list))
    
    for i in range(N - (order - 1) * delay):
        # Extract subsequence
        indices = [i + j * delay for j in range(order)]
        subseq = signal[indices]
        
        # Get permutation pattern
        pattern = tuple(np.argsort(subseq))
        counts[perm_dict[pattern]] += 1
    
    # Compute entropy
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs))
    
    if normalize:
        pe = pe / np.log2(np.math.factorial(order))
    
    return pe


def multiscale_entropy(signal: np.ndarray, scales: list = None, m: int = 2, r: float = 0.2) -> np.ndarray:
    """
    Compute Multiscale Entropy of a signal.
    
    MSE applies sample entropy at multiple time scales (coarse-grained).
    
    Args:
        signal: 1D signal array
        scales: List of scale factors
        m: Embedding dimension for SampEn
        r: Tolerance for SampEn
        
    Returns:
        Array of entropy values at each scale
    """
    if scales is None:
        scales = [1, 2, 3, 4, 5]
    
    mse = []
    for scale in scales:
        # Coarse-grain the signal
        n = len(signal) // scale
        coarse = np.array([signal[i*scale:(i+1)*scale].mean() for i in range(n)])
        
        # Compute sample entropy
        if len(coarse) > m + 2:
            se = sample_entropy(coarse, m, r)
        else:
            se = 0.0
        mse.append(se)
    
    return np.array(mse)


def neural_complexity(covariance_matrix: np.ndarray) -> float:
    """
    Compute Neural Complexity (Tononi-Sporns-Edelman).
    
    NC = sum over k of [H(X_k) - H(X_k | X_rest)]
    
    Measures the degree to which the system is both integrated and segregated.
    
    Args:
        covariance_matrix: Covariance matrix of the system
        
    Returns:
        Neural complexity value
    """
    n = covariance_matrix.shape[0]
    
    # Total entropy of the system (multivariate Gaussian)
    sign, logdet = np.linalg.slogdet(covariance_matrix + 1e-6 * np.eye(n))
    H_total = 0.5 * logdet + 0.5 * n * np.log(2 * np.pi * np.e)
    
    # Sum of individual entropies
    H_sum = 0
    for i in range(n):
        H_sum += 0.5 * np.log(2 * np.pi * np.e * (covariance_matrix[i, i] + 1e-12))
    
    # Integration = H_sum - H_total
    integration = H_sum - H_total
    
    # Complexity: integration across all scales (simplified version)
    # Here we use a single-scale approximation
    nc = integration / n if n > 0 else 0
    
    return max(nc, 0)


def phi_star(covariance_matrix: np.ndarray, partition: list = None) -> float:
    """
    Compute Φ* (integrated information proxy).
    
    Φ* measures how much the whole is greater than the sum of parts.
    Uses mutual information between partitions.
    
    Args:
        covariance_matrix: Covariance matrix of the system
        partition: Optional list of two index sets for bipartition
        
    Returns:
        Φ* value
    """
    n = covariance_matrix.shape[0]
    
    if partition is None:
        # Default: split in half
        partition = [list(range(n//2)), list(range(n//2, n))]
    
    # Entropy of whole system
    sign, logdet_whole = np.linalg.slogdet(covariance_matrix + 1e-6 * np.eye(n))
    H_whole = 0.5 * logdet_whole
    
    # Entropy of parts
    H_parts = 0
    for part in partition:
        if len(part) > 0:
            sub_cov = covariance_matrix[np.ix_(part, part)]
            sign, logdet_part = np.linalg.slogdet(sub_cov + 1e-6 * np.eye(len(part)))
            H_parts += 0.5 * logdet_part
    
    # Φ* = H_parts - H_whole (how much information is lost by partitioning)
    phi = H_parts - H_whole
    
    return max(phi, 0)


def compute_all_new_metrics(power: np.ndarray, time_series: np.ndarray = None) -> dict:
    """
    Compute all new complexity metrics.
    
    Args:
        power: Mode power distribution
        time_series: Optional time series data (if None, generates from power)
        
    Returns:
        Dictionary of metric values
    """
    if time_series is None:
        # Generate synthetic time series from power distribution
        n_points = 100
        phases = np.random.uniform(0, 2*np.pi, len(power))
        t = np.linspace(0, 10, n_points)
        
        time_series = np.zeros(n_points)
        for k, (p, phi) in enumerate(zip(power, phases)):
            time_series += np.sqrt(p) * np.sin(2 * np.pi * (k+1) * t / 10 + phi)
    
    # Compute metrics
    lzc = lempel_ziv_complexity(time_series)
    pe = permutation_entropy(time_series, order=3)
    se = sample_entropy(time_series, m=2, r=0.2)
    mse = multiscale_entropy(time_series, scales=[1, 2, 3, 4, 5])
    mse_mean = np.mean(mse)
    mse_slope = np.polyfit(range(len(mse)), mse, 1)[0] if len(mse) > 1 else 0
    
    # Covariance-based metrics (using power as proxy)
    cov_matrix = np.outer(power, power) + 0.1 * np.diag(power)
    nc = neural_complexity(cov_matrix)
    phi = phi_star(cov_matrix)
    
    return {
        'LZC': lzc,
        'PE': pe,
        'SampEn': se,
        'MSE_mean': mse_mean,
        'MSE_slope': mse_slope,
        'NC': nc,
        'Phi_star': phi
    }


# ============================================================================
# EXPERIMENT 1: Compute all metrics for all brain states
# ============================================================================

print("\n1. Computing new metrics for brain states...")

brain_states = {
    'Wake': sg.generate_wake_state(N_MODES, seed=SEED),
    'NREM': sg.generate_nrem_unconscious(N_MODES, seed=SEED),
    'Dream': sg.generate_nrem_dreaming(N_MODES, seed=SEED),
    'Anesthesia': sg.generate_anesthesia_state(N_MODES, seed=SEED),
    'Psychedelic': sg.generate_psychedelic_state(N_MODES, intensity=0.7, seed=SEED),
}

state_results = []

for state_name, power in tqdm(brain_states.items(), desc="States"):
    # Original metrics
    orig_metrics = met.compute_all_metrics(power, eigenvalues)
    
    # New metrics
    new_metrics = compute_all_new_metrics(power)
    
    state_results.append({
        'state': state_name,
        **orig_metrics,
        **new_metrics
    })

df_states = pd.DataFrame(state_results)

# ============================================================================
# EXPERIMENT 2: Generate many random samples for correlation analysis
# ============================================================================

print("\n2. Generating samples for correlation analysis...")

n_samples = 500
sample_results = []

for i in tqdm(range(n_samples), desc="Samples"):
    # Random power distribution
    power = np.random.dirichlet(np.ones(N_MODES) * 0.5)
    
    # Original metrics
    orig_metrics = met.compute_all_metrics(power, eigenvalues)
    
    # New metrics
    new_metrics = compute_all_new_metrics(power)
    
    sample_results.append({
        **orig_metrics,
        **new_metrics
    })

df_samples = pd.DataFrame(sample_results)

# ============================================================================
# EXPERIMENT 3: Feature importance for consciousness detection
# ============================================================================

print("\n3. Analyzing feature importance...")

# Use C(t) as target, compute correlations with all metrics
feature_cols = ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'LZC', 'PE', 'SampEn', 'MSE_mean', 'NC', 'Phi_star']
correlations = {}

for col in feature_cols:
    corr, p_value = stats.pearsonr(df_samples[col], df_samples['C'])
    correlations[col] = {'correlation': corr, 'p_value': p_value, 'abs_corr': abs(corr)}

df_importance = pd.DataFrame(correlations).T
df_importance = df_importance.sort_values('abs_corr', ascending=False)

# ============================================================================
# EXPERIMENT 4: Redundancy analysis
# ============================================================================

print("\n4. Analyzing metric redundancy...")

# Compute full correlation matrix
all_metrics = ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'LZC', 'PE', 'SampEn', 'MSE_mean', 'NC', 'Phi_star', 'C']
corr_matrix = df_samples[all_metrics].corr()

# PCA for dimensionality analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_samples[feature_cols])

pca = PCA()
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Save results
df_states.to_csv(OUTPUT_DIR / 'state_metrics_results.csv', index=False)
df_samples.to_csv(OUTPUT_DIR / 'sample_metrics_results.csv', index=False)
df_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv')
corr_matrix.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. State comparison for new metrics
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

new_metric_names = ['LZC', 'PE', 'SampEn', 'MSE_mean', 'NC', 'Phi_star', 'C']

for idx, metric in enumerate(new_metric_names):
    ax = axes.flat[idx]
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_states)))
    bars = ax.bar(df_states['state'], df_states[metric], color=colors, edgecolor='black')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Brain State', fontsize=12, fontweight='bold')
    ax.set_xticklabels(df_states['state'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

# Remove last empty subplot
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'state_comparison.png', dpi=300)
print("  Saved: state_comparison.png")

# 2. Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in df_importance['correlation']]
bars = ax.barh(df_importance.index, df_importance['correlation'], color=colors, edgecolor='black')
ax.set_xlabel('Correlation with C(t)', fontsize=12)
ax.set_title('Feature Importance for Consciousness', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Add significance markers
for i, (idx, row) in enumerate(df_importance.iterrows()):
    if row['p_value'] < 0.001:
        ax.text(row['correlation'] + 0.02, i, '***', fontsize=12, va='center')
    elif row['p_value'] < 0.01:
        ax.text(row['correlation'] + 0.02, i, '**', fontsize=12, va='center')
    elif row['p_value'] < 0.05:
        ax.text(row['correlation'] + 0.02, i, '*', fontsize=12, va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300)
print("  Saved: feature_importance.png")

# 3. Correlation matrix heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True, ax=ax,
           cbar_kws={'shrink': 0.8})
ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300)
print("  Saved: correlation_heatmap.png")

# 4. PCA explained variance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(range(1, len(explained_variance) + 1), explained_variance, color='steelblue', edgecolor='black')
ax.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'ro-', linewidth=2)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% threshold')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# PCA scatter (first 2 components)
ax = axes[1]
X_pca = pca.transform(X_scaled)

# Color by C value
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_samples['C'], cmap='viridis', 
                     alpha=0.5, s=20, edgecolors='none')
ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} var)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} var)', fontsize=12)
ax.set_title('PCA of All Metrics', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('C(t)', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=300)
print("  Saved: pca_analysis.png")

# 5. Scatter plots of new metrics vs C
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
new_features = ['LZC', 'PE', 'SampEn', 'MSE_mean', 'NC', 'Phi_star']

for ax, feat in zip(axes.flat, new_features):
    ax.scatter(df_samples[feat], df_samples['C'], alpha=0.3, s=10, color='steelblue')
    
    # Add regression line
    z = np.polyfit(df_samples[feat], df_samples['C'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_samples[feat].min(), df_samples[feat].max(), 100)
    ax.plot(x_range, p(x_range), 'r-', linewidth=2)
    
    corr = df_importance.loc[feat, 'correlation']
    ax.set_xlabel(feat, fontsize=12)
    ax.set_ylabel('C(t)', fontsize=12)
    ax.set_title(f'{feat} vs C(t) (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'new_metrics_scatter.png', dpi=300)
print("  Saved: new_metrics_scatter.png")

plt.close('all')

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\nNew Metrics by Brain State:")
print(df_states[['state', 'C', 'LZC', 'PE', 'NC', 'Phi_star']].to_string(index=False))

print("\nFeature Importance Ranking (|correlation with C|):")
for idx, row in df_importance.iterrows():
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
    print(f"  {idx:10}: r = {row['correlation']:+.4f} {sig}")

print("\nPCA Dimensionality:")
cumsum = np.cumsum(explained_variance)
n_components_90 = np.argmax(cumsum >= 0.90) + 1
print(f"  Components for 90% variance: {n_components_90}")
print(f"  Effective dimensionality: ~{n_components_90} (out of {len(feature_cols)})")

print("\nMetric Redundancy (high correlations |r| > 0.7):")
for i in range(len(all_metrics)):
    for j in range(i+1, len(all_metrics)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"  {all_metrics[i]} <-> {all_metrics[j]}: r = {corr_matrix.iloc[i, j]:.3f}")

print("\n" + "=" * 60)
print(f"Experiment completed! Results saved to: {OUTPUT_DIR}")
print("=" * 60)
