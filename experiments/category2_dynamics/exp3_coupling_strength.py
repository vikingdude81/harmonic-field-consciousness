#!/usr/bin/env python3
"""
Category 2, Experiment 3: Coupling Strength

Explore mode coupling effects on consciousness:
- Implement nonlinear coupling between modes
- Vary coupling strength parameter
- Test effect on synchronization (R)
- Find optimal coupling for consciousness
- Generate bifurcation diagrams
- Model pharmacological effects (e.g., ketamine, propofol)

RTX 5090 Enhanced: Uses PyTorch for GPU-accelerated eigendecomposition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz
from utils.gpu_utils import get_device_info, get_array_module, to_cpu, print_gpu_status
from utils.chaos_metrics import estimate_lyapunov_exponent, compute_branching_ratio
from utils.category_theory_metrics import compute_integration_phi

# Configuration - supports environment variable overrides for RTX 5090 scaling
SEED = 42
N_NODES = int(os.environ.get('EXP_N_NODES', 300))
N_MODES = int(os.environ.get('EXP_N_MODES', 80))
N_COUPLING_STEPS = int(os.environ.get('EXP_N_COUPLING_STEPS', 50))
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_coupling_strength'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check for PyTorch GPU support (RTX 5090)
USE_PYTORCH_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        USE_PYTORCH_GPU = True
        print(f"[GPU] Using {gpu_name} ({gpu_mem:.1f} GB)")
except ImportError:
    pass

print("=" * 60)
print("Category 2, Experiment 3: Coupling Strength")
print(f"Configuration: {N_NODES} nodes, {N_MODES} modes, {N_COUPLING_STEPS} coupling steps")
if USE_PYTORCH_GPU:
    print(f"Acceleration: PyTorch CUDA (RTX 5090)")
print("=" * 60)

# Check GPU availability
print_gpu_status()
gpu_info = get_device_info()
USE_GPU = gpu_info['cupy_available'] or USE_PYTORCH_GPU

np.random.seed(SEED)

# Generate network and eigenmodes
print("\nGenerating network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)

if USE_PYTORCH_GPU and N_NODES > 500:
    print(f"Computing Laplacian eigenmodes on GPU ({N_NODES}x{N_NODES} matrix)...")
    import torch
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
    L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
    eigenvalues = eigenvalues[:N_MODES]


def kuramoto_dynamics(phases, t, omega, K, adjacency):
    """
    Kuramoto oscillator dynamics for phase synchronization.
    
    dθ_i/dt = ω_i + (K/N) * Σ_j A_ij * sin(θ_j - θ_i)
    
    Args:
        phases: Phase angles for each oscillator
        t: Time (unused, required by odeint)
        omega: Natural frequencies
        K: Global coupling strength
        adjacency: Adjacency matrix
        
    Returns:
        Phase derivatives
    """
    N = len(phases)
    dphases = omega.copy()
    
    for i in range(N):
        coupling_sum = 0
        for j in range(N):
            if adjacency[i, j] > 0:
                coupling_sum += adjacency[i, j] * np.sin(phases[j] - phases[i])
        dphases[i] += (K / N) * coupling_sum
    
    return dphases


def simulate_coupled_dynamics(
    n_oscillators: int,
    K: float,
    adjacency: np.ndarray,
    t_max: float = 50.0,
    dt: float = 0.1,
    omega_spread: float = 0.5,
    seed: int = None
):
    """
    Simulate coupled oscillator dynamics.
    
    Args:
        n_oscillators: Number of oscillators
        K: Coupling strength
        adjacency: Adjacency/weight matrix
        t_max: Maximum simulation time
        dt: Time step
        omega_spread: Spread of natural frequencies
        seed: Random seed
        
    Returns:
        Dictionary with simulation results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Natural frequencies (centered around 1 Hz)
    omega = 1.0 + omega_spread * (np.random.rand(n_oscillators) - 0.5)
    
    # Initial phases (random)
    phases_init = 2 * np.pi * np.random.rand(n_oscillators)
    
    # Time points
    t = np.arange(0, t_max, dt)
    
    # Integrate
    solution = odeint(kuramoto_dynamics, phases_init, t, args=(omega, K, adjacency))
    
    # Compute order parameter R(t) = |<e^{iθ}>|
    complex_order = np.exp(1j * solution)
    R_t = np.abs(complex_order.mean(axis=1))
    
    # Final state metrics
    final_phases = solution[-1] % (2 * np.pi)
    R_final = R_t[-1]
    
    # Compute phase coherence variance (metastability)
    R_mean = R_t[len(t)//2:].mean()  # Steady-state mean
    R_std = R_t[len(t)//2:].std()    # Fluctuations
    
    return {
        'phases': solution,
        'R_t': R_t,
        'R_final': R_final,
        'R_mean': R_mean,
        'R_std': R_std,
        'metastability': R_std,  # Higher = more dynamic
        'final_phases': final_phases,
        't': t
    }


# ============================================================================
# EXPERIMENT 1: Coupling strength sweep
# ============================================================================

print("\n1. Sweeping coupling strength...")

# Get adjacency matrix
adjacency = nx.adjacency_matrix(G).toarray().astype(float)

coupling_strengths = np.linspace(0, 5, 25)
sweep_results = []

for K in tqdm(coupling_strengths, desc="Coupling K"):
    # Run simulation
    sim = simulate_coupled_dynamics(
        n_oscillators=N_MODES,
        K=K,
        adjacency=adjacency[:N_MODES, :N_MODES],
        t_max=50,
        dt=0.1,
        seed=SEED
    )
    
    # Generate power distribution based on phase coherence
    # Higher coherence -> more low-mode concentration
    power = sg.generate_wake_state(n_modes=N_MODES, seed=SEED)
    
    # Modulate by coupling (higher coupling = more coherent = more low-mode)
    if K > 0:
        mode_weights = np.exp(-np.arange(N_MODES) * K / 10)
        power = power * mode_weights
        power = power / power.sum()
    
    # Compute consciousness metrics
    metrics = met.compute_all_metrics(
        power, 
        eigenvalues,
        phases=sim['final_phases'],
        power_previous=None
    )
    
    sweep_results.append({
        'K': K,
        'R_final': sim['R_final'],
        'R_mean': sim['R_mean'],
        'metastability': sim['metastability'],
        **metrics
    })

df_sweep = pd.DataFrame(sweep_results)

# ============================================================================
# EXPERIMENT 2: Pharmacological modeling
# ============================================================================

print("\n2. Modeling pharmacological effects...")

# Different drugs affect coupling differently
drug_profiles = {
    'Baseline': {'K': 1.0, 'omega_spread': 0.5},
    'Propofol_low': {'K': 2.0, 'omega_spread': 0.3},      # Increased coupling, less diversity
    'Propofol_high': {'K': 4.0, 'omega_spread': 0.1},     # Very high coupling, uniform
    'Ketamine_low': {'K': 0.5, 'omega_spread': 0.8},      # Reduced coupling, more diversity
    'Ketamine_high': {'K': 0.2, 'omega_spread': 1.2},     # Very low coupling, high diversity
    'Psychedelic': {'K': 0.3, 'omega_spread': 1.5},       # Low coupling, very diverse
}

drug_results = []

for drug_name, params in tqdm(drug_profiles.items(), desc="Drugs"):
    sim = simulate_coupled_dynamics(
        n_oscillators=N_MODES,
        K=params['K'],
        adjacency=adjacency[:N_MODES, :N_MODES],
        t_max=100,
        dt=0.1,
        omega_spread=params['omega_spread'],
        seed=SEED
    )
    
    # Generate appropriate power distribution
    if 'Propofol' in drug_name:
        power = sg.generate_anesthesia_state(n_modes=N_MODES, seed=SEED)
    elif 'Ketamine' in drug_name:
        # Ketamine: dissociative, mixed state
        power = sg.generate_nrem_dreaming(n_modes=N_MODES, seed=SEED)
    elif 'Psychedelic' in drug_name:
        power = sg.generate_psychedelic_state(n_modes=N_MODES, intensity=0.7, seed=SEED)
    else:
        power = sg.generate_wake_state(n_modes=N_MODES, seed=SEED)
    
    metrics = met.compute_all_metrics(
        power,
        eigenvalues,
        phases=sim['final_phases']
    )
    
    drug_results.append({
        'drug': drug_name,
        'K': params['K'],
        'omega_spread': params['omega_spread'],
        'R_final': sim['R_final'],
        'R_mean': sim['R_mean'],
        'metastability': sim['metastability'],
        **metrics
    })

df_drugs = pd.DataFrame(drug_results)

# ============================================================================
# EXPERIMENT 3: Time evolution at different coupling strengths
# ============================================================================

print("\n3. Analyzing time evolution...")

coupling_examples = [0.0, 0.5, 1.0, 2.0, 4.0]
time_results = {}

for K in tqdm(coupling_examples, desc="Time evolution"):
    sim = simulate_coupled_dynamics(
        n_oscillators=N_MODES,
        K=K,
        adjacency=adjacency[:N_MODES, :N_MODES],
        t_max=100,
        dt=0.1,
        seed=SEED
    )
    time_results[K] = sim

# Save results
df_sweep.to_csv(OUTPUT_DIR / 'coupling_sweep_results.csv', index=False)
df_drugs.to_csv(OUTPUT_DIR / 'drug_effects_results.csv', index=False)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# Need networkx import for adjacency matrix
import networkx as nx

# 1. Coupling strength sweep
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# R vs K (synchronization curve)
ax = axes[0, 0]
ax.plot(df_sweep['K'], df_sweep['R_final'], 'o-', linewidth=2, markersize=6, color='darkblue')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Critical threshold')
ax.set_xlabel('Coupling Strength K', fontsize=12)
ax.set_ylabel('Phase Coherence R', fontsize=12)
ax.set_title('Synchronization Transition', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C(t) vs K
ax = axes[0, 1]
ax.plot(df_sweep['K'], df_sweep['C'], 'o-', linewidth=2, markersize=6, color='darkgreen')
ax.set_xlabel('Coupling Strength K', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('Consciousness vs Coupling', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Metastability vs K
ax = axes[0, 2]
ax.plot(df_sweep['K'], df_sweep['metastability'], 'o-', linewidth=2, markersize=6, color='darkred')
ax.set_xlabel('Coupling Strength K', fontsize=12)
ax.set_ylabel('Metastability (R fluctuations)', fontsize=12)
ax.set_title('Dynamic Flexibility', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# C vs R (phase coherence trade-off)
ax = axes[1, 0]
scatter = ax.scatter(df_sweep['R_final'], df_sweep['C'], c=df_sweep['K'], 
                     cmap='viridis', s=80, edgecolors='black')
ax.set_xlabel('Phase Coherence R', fontsize=12)
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('Consciousness-Synchronization Trade-off', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Coupling K', fontsize=10)
ax.grid(True, alpha=0.3)

# H_mode and PR vs K
ax = axes[1, 1]
ax.plot(df_sweep['K'], df_sweep['H_mode'], 'o-', linewidth=2, label='H_mode')
ax.plot(df_sweep['K'], df_sweep['PR'], 's--', linewidth=2, label='PR')
ax.plot(df_sweep['K'], df_sweep['kappa'], '^:', linewidth=2, label='κ')
ax.set_xlabel('Coupling Strength K', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Component Metrics', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Optimal coupling
ax = axes[1, 2]
optimal_idx = df_sweep['C'].idxmax()
optimal_K = df_sweep.loc[optimal_idx, 'K']
optimal_C = df_sweep.loc[optimal_idx, 'C']

ax.plot(df_sweep['K'], df_sweep['C'], 'b-', linewidth=2)
ax.axvline(x=optimal_K, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_K:.2f}')
ax.scatter([optimal_K], [optimal_C], color='red', s=150, zorder=5, marker='*')
ax.fill_between(df_sweep['K'], df_sweep['C'], alpha=0.2)
ax.set_xlabel('Coupling Strength K', fontsize=12)
ax.set_ylabel('C(t)', fontsize=12)
ax.set_title(f'Optimal Coupling (K*={optimal_K:.2f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'coupling_sweep_analysis.png', dpi=300)
print("  Saved: coupling_sweep_analysis.png")

# 2. Drug effects
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart of C(t) by drug
ax = axes[0]
colors = ['steelblue', 'lightcoral', 'darkred', 'lightgreen', 'darkgreen', 'purple']
bars = ax.bar(df_drugs['drug'], df_drugs['C'], color=colors, edgecolor='black')
ax.set_ylabel('Consciousness Functional C(t)', fontsize=12)
ax.set_title('Drug Effects on Consciousness', fontsize=14, fontweight='bold')
ax.set_xticklabels(df_drugs['drug'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# R vs C for drugs
ax = axes[1]
for idx, row in df_drugs.iterrows():
    ax.scatter(row['R_final'], row['C'], s=150, label=row['drug'], edgecolors='black')
ax.set_xlabel('Phase Coherence R', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('Coherence-Consciousness Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Radar chart of metrics
ax = axes[2]
categories = ['H_mode', 'PR', 'R', 'kappa']
n_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

for idx, row in df_drugs.iterrows():
    values = [row[cat] for cat in categories]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['drug'], alpha=0.7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Metric Profiles by Drug', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1))
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'drug_effects_analysis.png', dpi=300, bbox_inches='tight')
print("  Saved: drug_effects_analysis.png")

# 3. Time evolution plots
fig, axes = plt.subplots(2, len(coupling_examples), figsize=(20, 8))

for idx, K in enumerate(coupling_examples):
    sim = time_results[K]
    
    # R(t) time series
    ax = axes[0, idx]
    ax.plot(sim['t'], sim['R_t'], linewidth=1, color='darkblue')
    ax.axhline(y=sim['R_mean'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('R(t)', fontsize=10)
    ax.set_title(f'K = {K}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Phase distribution (final)
    ax = axes[1, idx]
    ax.hist(sim['final_phases'], bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Phase (rad)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Final Phase Distribution', fontsize=10)
    ax.set_xlim(0, 2*np.pi)

plt.suptitle('Time Evolution at Different Coupling Strengths', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'time_evolution.png', dpi=300, bbox_inches='tight')
print("  Saved: time_evolution.png")

plt.close('all')

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print(f"\nOptimal Coupling Strength: K* = {optimal_K:.3f}")
print(f"  Maximum C(t) = {optimal_C:.4f}")
print(f"  Phase coherence R = {df_sweep.loc[optimal_idx, 'R_final']:.4f}")

print("\nCoupling Regimes:")
low_K = df_sweep[df_sweep['K'] <= 0.5]
mid_K = df_sweep[(df_sweep['K'] > 0.5) & (df_sweep['K'] <= 2.0)]
high_K = df_sweep[df_sweep['K'] > 2.0]

print(f"  Low coupling (K<=0.5):   C = {low_K['C'].mean():.4f} +/- {low_K['C'].std():.4f}")
print(f"  Medium coupling (0.5<K<=2): C = {mid_K['C'].mean():.4f} +/- {mid_K['C'].std():.4f}")
print(f"  High coupling (K>2):     C = {high_K['C'].mean():.4f} +/- {high_K['C'].std():.4f}")

print("\nDrug Effects Summary:")
for _, row in df_drugs.iterrows():
    print(f"  {row['drug']:15}: C = {row['C']:.4f}, R = {row['R_final']:.4f}")

print("\n" + "=" * 60)
print(f"Experiment completed! Results saved to: {OUTPUT_DIR}")
print("=" * 60)
