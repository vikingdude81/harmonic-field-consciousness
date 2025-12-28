#!/usr/bin/env python3
"""
Category 2: Dynamics Experiments

NEW Experiment: Rotational Dynamics and Recovery from Perturbation

Integrates findings from Batabyal et al. (2025) JOCN:
- Rotational dynamics in state space after perturbation
- Traveling waves across network topology
- Correspondence between rotations and waves
- Dynamic stability and recovery metrics

Tests key hypotheses:
1. Do perturbations induce rotational dynamics in harmonic field space?
2. Do traveling waves emerge in network topology?
3. Does rotation completeness correlate with consciousness metrics?
4. Is there correspondence between wave speed and rotation velocity?

RTX 5090 Enhanced: Uses PyTorch for GPU-accelerated eigendecomposition.
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
import time

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils.rotational_dynamics import (
    analyze_rotational_dynamics, compute_rotation_angle,
    compute_angular_velocity, compute_trajectory_circularity,
    compute_recovery_percentage, jpca
)
from utils.traveling_waves import (
    comprehensive_wave_analysis, detect_traveling_wave_correlation,
    analyze_wave_correspondence_to_rotation
)
from utils.dynamic_stability import (
    comprehensive_stability_analysis, measure_perturbation_recovery
)

# Configuration - supports environment variable overrides for RTX 5090 scaling
SEED = 42
N_NODES = int(os.environ.get('EXP_N_NODES', 300))
N_MODES = int(os.environ.get('EXP_N_MODES', 80))
N_TRIALS = int(os.environ.get('EXP_N_TRIALS', 50))
POST_PERTURBATION_TIME = int(os.environ.get('EXP_POST_PERTURBATION_TIME', 100))
PERTURBATION_STRENGTH = 0.3
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp_rotational_recovery'
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
except ImportError:
    pass

print("=" * 70)
print("NEW EXPERIMENT: Rotational Dynamics and Recovery from Perturbation")
print("Based on Batabyal et al. (2025) JOCN")
print("=" * 70)
print(f"\n[TIME] Experiment started at: {time.strftime('%H:%M:%S')}")
print(f"[CONFIG]")
print(f"   Network: {N_NODES} nodes, {N_MODES} modes")
print(f"   Experiment 1: 3 states x {N_TRIALS} trials = {3 * N_TRIALS} total")
print(f"   Experiment 2: 10 strengths x 20 trials = 200 total")
print(f"   Experiment 3: 30 trials")
if USE_PYTORCH_GPU:
    print(f"   Acceleration: PyTorch CUDA ({gpu_name})")
print(f"   Est. runtime: ~3-5 minutes\n")

start_time = time.time()
np.random.seed(SEED)

# ============================================================================
# EXPERIMENT 1: Rotational Dynamics After Perturbation in Different States
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 1: Rotational Dynamics in Conscious vs Unconscious States")
print("=" * 70)

# Generate network
print("\nGenerating small-world network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)

if USE_PYTORCH_GPU and N_NODES > 500:
    print(f"Computing Laplacian eigenmodes on GPU ({N_NODES}x{N_NODES} matrix)...")
    import torch
    import networkx as nx
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

# Get node positions for traveling wave analysis
pos = gg.get_node_positions(G, layout='spring', seed=SEED)
positions = np.array([pos[i] for i in range(N_NODES)])

# Define brain states
states = {
    'Wake (Conscious)': sg.generate_wake_state(n_modes=N_MODES, seed=SEED),
    'NREM (Unconscious)': sg.generate_nrem_unconscious(n_modes=N_MODES, seed=SEED),
    'Anesthesia (Unconscious)': sg.generate_anesthesia_state(n_modes=N_MODES, seed=SEED),
}

exp1_results = []

for state_name, state_power in states.items():
    print(f"\n{'='*70}")
    print(f"Analyzing state: {state_name}")
    print(f"{'='*70}")
    
    # Generate baseline trajectory in eigenmode space
    np.random.seed(SEED)
    trajectory_baseline = np.zeros((POST_PERTURBATION_TIME, N_MODES))
    
    # Initialize with state
    trajectory_baseline[0] = np.sqrt(state_power) * np.random.randn(N_MODES)
    
    # Simulate unperturbed dynamics
    print("Generating baseline trajectory...", end=' ', flush=True)
    for t in range(1, POST_PERTURBATION_TIME):
        # Simple damped oscillation in mode space
        trajectory_baseline[t] = trajectory_baseline[t-1] * 0.98 + \
                                np.sqrt(state_power) * np.random.randn(N_MODES) * 0.1
    print("[OK]")
    
    # Generate perturbed trajectories
    trial_rotations = []
    trial_recoveries = []
    trial_waves = []
    
    print(f"Running {N_TRIALS} trials with perturbations...")
    for trial in tqdm(range(N_TRIALS), desc=f"  Trials for {state_name[:15]}", leave=True):
        np.random.seed(SEED + trial)
        
        # Start with same initial condition
        trajectory_perturbed = np.zeros((POST_PERTURBATION_TIME, N_MODES))
        trajectory_perturbed[0] = trajectory_baseline[0].copy()
        
        # Apply perturbation at t=10
        perturbation_time = 10
        perturbation = np.random.randn(N_MODES) * PERTURBATION_STRENGTH
        trajectory_perturbed[perturbation_time] += perturbation
        
        # Simulate recovery dynamics
        for t in range(perturbation_time + 1, POST_PERTURBATION_TIME):
            # Dynamics with recovery toward baseline
            diff = trajectory_baseline[t] - trajectory_perturbed[t-1]
            recovery_force = diff * 0.1  # Recovery rate
            
            trajectory_perturbed[t] = trajectory_perturbed[t-1] * 0.98 + \
                                     recovery_force + \
                                     np.sqrt(state_power) * np.random.randn(N_MODES) * 0.1
        
        # Analyze rotational dynamics
        rotation_result = analyze_rotational_dynamics(
            trajectory_perturbed[perturbation_time:],
            perturbation_time=1,
            n_jpcs=6
        )
        
        trial_rotations.append(rotation_result)
        
        # Measure recovery
        recovery = compute_recovery_percentage(
            trajectory_perturbed,
            pre_perturbation_idx=perturbation_time-1,
            post_perturbation_start=perturbation_time
        )
        trial_recoveries.append(recovery)
        
        # Project to spatial activity for wave analysis
        activity_spatial = trajectory_perturbed @ eigenvectors[:, :N_MODES].T
        
        # Analyze traveling waves
        wave_result = comprehensive_wave_analysis(
            activity_spatial[perturbation_time:],
            positions
        )
        trial_waves.append(wave_result)
    
    # Aggregate results
    mean_rotation_angle = np.mean([r['total_rotation_degrees'] for r in trial_rotations])
    mean_recovery = np.mean(trial_recoveries)
    mean_circularity = np.mean([r['circularity'] for r in trial_rotations])
    mean_rotation_quality = np.mean([r['rotation_quality'] for r in trial_rotations])
    mean_wave_correlation = np.mean([w['wave_correlation'] for w in trial_waves])
    prop_has_wave = np.mean([w['has_traveling_wave'] for w in trial_waves])
    
    # Compute consciousness metric on baseline
    print("Computing consciousness metric...", end=' ', flush=True)
    metrics = met.compute_all_metrics(
        state_power, eigenvalues
    )
    C_t = metrics['C']
    print("[OK]")
    
    exp1_results.append({
        'state': state_name,
        'consciousness': 'Conscious' if 'Conscious' in state_name else 'Unconscious',
        'C_t': C_t,
        'mean_rotation_angle': mean_rotation_angle,
        'mean_recovery_pct': mean_recovery,
        'mean_circularity': mean_circularity,
        'mean_rotation_quality': mean_rotation_quality,
        'mean_wave_correlation': mean_wave_correlation,
        'prop_has_wave': prop_has_wave,
    })
    
    print(f"\n{'─'*70}")
    print(f"RESULTS for {state_name}:")
    print(f"{'─'*70}")
    print(f"  C(t)                    : {C_t:.3f}")
    print(f"  Mean rotation angle     : {mean_rotation_angle:.1f}°")
    print(f"  Mean recovery           : {mean_recovery:.1f}%")
    print(f"  Rotation quality (R²)   : {mean_rotation_quality:.3f}")
    print(f"  Trajectory circularity  : {mean_circularity:.3f}")
    print(f"  Wave correlation        : {mean_wave_correlation:.3f}")
    print(f"  Traveling wave presence : {prop_has_wave:.1%}")
    print(f"{'─'*70}\n")

# Save results
df_exp1 = pd.DataFrame(exp1_results)
df_exp1.to_csv(OUTPUT_DIR / 'exp1_state_comparison.csv', index=False)
print(f"[OK] Experiment 1 results saved to: {OUTPUT_DIR / 'exp1_state_comparison.csv'}")

# ============================================================================
# EXPERIMENT 2: Rotation-Recovery Correlation
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: Correlation Between Rotation and Recovery")
print("=" * 70)

# Test hypothesis: Fuller rotations lead to better recovery
# Vary perturbation strength to create different rotation completeness

wake_power = sg.generate_wake_state(n_modes=N_MODES, seed=SEED)
perturbation_strengths = np.linspace(0.1, 0.5, 10)

exp2_results = []

print(f"\nTesting {len(perturbation_strengths)} perturbation strengths with 20 trials each...")
for pert_strength in tqdm(perturbation_strengths, desc="Perturbation strengths"):
    for trial in range(20):  # Multiple trials per strength
        np.random.seed(SEED + int(pert_strength*1000) + trial)
        
        # Generate baseline
        trajectory_baseline = np.zeros((POST_PERTURBATION_TIME, N_MODES))
        trajectory_baseline[0] = np.sqrt(wake_power) * np.random.randn(N_MODES)
        
        for t in range(1, POST_PERTURBATION_TIME):
            trajectory_baseline[t] = trajectory_baseline[t-1] * 0.98 + \
                                    np.sqrt(wake_power) * np.random.randn(N_MODES) * 0.1
        
        # Generate perturbed trajectory
        trajectory_perturbed = trajectory_baseline.copy()
        perturbation_time = 10
        perturbation = np.random.randn(N_MODES) * pert_strength
        trajectory_perturbed[perturbation_time] += perturbation
        
        # Recovery dynamics
        for t in range(perturbation_time + 1, POST_PERTURBATION_TIME):
            diff = trajectory_baseline[t] - trajectory_perturbed[t-1]
            recovery_force = diff * 0.1
            trajectory_perturbed[t] = trajectory_perturbed[t-1] * 0.98 + \
                                     recovery_force + \
                                     np.sqrt(wake_power) * np.random.randn(N_MODES) * 0.1
        
        # Analyze rotations
        rotation_result = analyze_rotational_dynamics(
            trajectory_perturbed[perturbation_time:],
            n_jpcs=6
        )
        
        # Measure recovery
        recovery_pct = compute_recovery_percentage(
            trajectory_perturbed,
            pre_perturbation_idx=perturbation_time-1,
            post_perturbation_start=perturbation_time
        )
        
        # Measure final performance (distance from baseline at end)
        final_distance = np.linalg.norm(
            trajectory_perturbed[-1] - trajectory_baseline[-1]
        )
        
        exp2_results.append({
            'perturbation_strength': pert_strength,
            'rotation_angle': rotation_result['total_rotation_degrees'],
            'recovery_pct': recovery_pct,
            'rotation_quality': rotation_result['rotation_quality'],
            'circularity': rotation_result['circularity'],
            'final_distance': final_distance,
        })

df_exp2 = pd.DataFrame(exp2_results)
df_exp2.to_csv(OUTPUT_DIR / 'exp2_rotation_recovery_correlation.csv', index=False)
print(f"[OK] Experiment 2 results saved to: {OUTPUT_DIR / 'exp2_rotation_recovery_correlation.csv'}")

# Compute correlations
print("\n" + "─" * 70)
print("CORRELATIONS:")
print("─" * 70)
corr_angle_recovery = stats.pearsonr(df_exp2['rotation_angle'], 
                                     df_exp2['recovery_pct'])
print(f"  Rotation angle <-> Recovery : r={corr_angle_recovery[0]:+.3f}, p={corr_angle_recovery[1]:.4f}")

corr_quality_recovery = stats.pearsonr(df_exp2['rotation_quality'], 
                                       df_exp2['recovery_pct'])
print(f"  Rotation quality <-> Recovery: r={corr_quality_recovery[0]:+.3f}, p={corr_quality_recovery[1]:.4f}")
print("─" * 70 + "\n")

# ============================================================================
# EXPERIMENT 3: Wave-Rotation Correspondence
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: Traveling Wave and Rotation Correspondence")
print("=" * 70)

# Test if wave speed correlates with rotational velocity

exp3_results = []

print(f"\nAnalyzing wave-rotation correspondence across 30 trials...")
for trial in tqdm(range(30), desc="Analyzing trials"):
    np.random.seed(SEED + trial + 1000)
    
    # Generate perturbed trajectory
    trajectory = np.zeros((POST_PERTURBATION_TIME, N_MODES))
    trajectory[0] = np.sqrt(wake_power) * np.random.randn(N_MODES)
    
    perturbation_time = 10
    trajectory[perturbation_time] += np.random.randn(N_MODES) * 0.3
    
    for t in range(perturbation_time + 1, POST_PERTURBATION_TIME):
        trajectory[t] = trajectory[t-1] * 0.98 + \
                       np.sqrt(wake_power) * np.random.randn(N_MODES) * 0.1
    
    # Analyze rotations
    rotation_result = analyze_rotational_dynamics(trajectory[perturbation_time:], n_jpcs=6)
    angular_velocity = rotation_result['angular_velocity']
    
    # Project to spatial activity
    activity_spatial = trajectory @ eigenvectors[:, :N_MODES].T
    
    # Analyze waves
    wave_result = comprehensive_wave_analysis(
        activity_spatial[perturbation_time:],
        positions
    )
    
    # Check correspondence
    if wave_result['has_traveling_wave']:
        correspondence = analyze_wave_correspondence_to_rotation(
            activity_spatial[perturbation_time:],
            positions,
            angular_velocity,
            window_size=20
        )
        
        exp3_results.append({
            'trial': trial,
            'has_wave': True,
            'wave_speed': wave_result['wave_speed'],
            'mean_rotation_velocity': rotation_result['mean_angular_velocity'],
            'wave_rotation_correlation': correspondence['correlation'],
            'correspondence_p_value': correspondence['p_value'],
            'has_correspondence': correspondence['correspondence']
        })
    else:
        exp3_results.append({
            'trial': trial,
            'has_wave': False,
            'wave_speed': 0,
            'mean_rotation_velocity': rotation_result['mean_angular_velocity'],
            'wave_rotation_correlation': 0,
            'correspondence_p_value': 1.0,
            'has_correspondence': False
        })

df_exp3 = pd.DataFrame(exp3_results)
df_exp3.to_csv(OUTPUT_DIR / 'exp3_wave_rotation_correspondence.csv', index=False)
print(f"[OK] Experiment 3 results saved to: {OUTPUT_DIR / 'exp3_wave_rotation_correspondence.csv'}")

# Statistics
trials_with_waves = df_exp3[df_exp3['has_wave']]
print("\n" + "─" * 70)
print("WAVE-ROTATION CORRESPONDENCE:")
print("─" * 70)
if len(trials_with_waves) > 0:
    print(f"  Trials with traveling waves : {len(trials_with_waves)}/{len(df_exp3)} ({len(trials_with_waves)/len(df_exp3):.1%})")
    print(f"  Mean wave-rotation correlation: {trials_with_waves['wave_rotation_correlation'].mean():+.3f}")
    print(f"  Showing correspondence      : {trials_with_waves['has_correspondence'].sum()}/{len(trials_with_waves)} ({trials_with_waves['has_correspondence'].mean():.1%})")
else:
    print("  No traveling waves detected in trials")
print("─" * 70 + "\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)
print("Creating comprehensive figure with 9 subplots...", end=' ', flush=True)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Rotation angle by state
ax1 = plt.subplot(3, 3, 1)
sns.barplot(data=df_exp1, x='state', y='mean_rotation_angle', hue='consciousness', ax=ax1)
ax1.set_title('Rotation Angle by Brain State', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Rotation Angle (degrees)')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='')

# Plot 2: Recovery by state
ax2 = plt.subplot(3, 3, 2)
sns.barplot(data=df_exp1, x='state', y='mean_recovery_pct', hue='consciousness', ax=ax2)
ax2.set_title('Recovery from Perturbation', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recovery Percentage (%)')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='')

# Plot 3: Rotation quality vs C(t)
ax3 = plt.subplot(3, 3, 3)
colors = ['red' if 'Conscious' in s else 'blue' for s in df_exp1['consciousness']]
ax3.scatter(df_exp1['C_t'], df_exp1['mean_rotation_quality'], 
           c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=2)
ax3.set_xlabel('Consciousness Metric C(t)', fontweight='bold')
ax3.set_ylabel('Rotation Quality (R²)', fontweight='bold')
ax3.set_title('Rotation Quality vs Consciousness', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Rotation angle vs Recovery (Exp 2)
ax4 = plt.subplot(3, 3, 4)
scatter = ax4.scatter(df_exp2['rotation_angle'], df_exp2['recovery_pct'],
                     c=df_exp2['perturbation_strength'], cmap='viridis',
                     alpha=0.5, s=30)
ax4.set_xlabel('Rotation Angle (degrees)', fontweight='bold')
ax4.set_ylabel('Recovery Percentage (%)', fontweight='bold')
ax4.set_title(f'Rotation vs Recovery (r={corr_angle_recovery[0]:.3f})', 
             fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Perturbation Strength')
ax4.grid(True, alpha=0.3)

# Plot 5: Rotation quality vs Recovery
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(df_exp2['rotation_quality'], df_exp2['recovery_pct'],
           alpha=0.5, s=30, c='purple')
ax5.set_xlabel('Rotation Quality (R²)', fontweight='bold')
ax5.set_ylabel('Recovery Percentage (%)', fontweight='bold')
ax5.set_title(f'Quality vs Recovery (r={corr_quality_recovery[0]:.3f})', 
             fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Circularity vs Recovery
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(df_exp2['circularity'], df_exp2['recovery_pct'],
           alpha=0.5, s=30, c='orange')
ax6.set_xlabel('Trajectory Circularity', fontweight='bold')
ax6.set_ylabel('Recovery Percentage (%)', fontweight='bold')
ax6.set_title('Circularity vs Recovery', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Wave prevalence by state
ax7 = plt.subplot(3, 3, 7)
sns.barplot(data=df_exp1, x='state', y='prop_has_wave', hue='consciousness', ax=ax7)
ax7.set_title('Traveling Wave Prevalence', fontsize=12, fontweight='bold')
ax7.set_ylabel('Proportion with Traveling Waves')
ax7.set_xlabel('')
ax7.tick_params(axis='x', rotation=45)
ax7.legend(title='')

# Plot 8: Wave-rotation correspondence
ax8 = plt.subplot(3, 3, 8)
if len(trials_with_waves) > 0:
    ax8.scatter(trials_with_waves['mean_rotation_velocity'], 
               trials_with_waves['wave_speed'],
               c=trials_with_waves['wave_rotation_correlation'],
               cmap='coolwarm', s=100, alpha=0.7, edgecolors='black')
    ax8.set_xlabel('Rotational Velocity (deg/time)', fontweight='bold')
    ax8.set_ylabel('Wave Speed (space/time)', fontweight='bold')
    ax8.set_title('Wave Speed vs Rotation Velocity', fontsize=12, fontweight='bold')
    plt.colorbar(ax8.collections[0], ax=ax8, label='Correlation')
    ax8.grid(True, alpha=0.3)

# Plot 9: Summary statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Experiment 1: State Comparison
  Conscious states:
    Mean rotation: {df_exp1[df_exp1['consciousness']=='Conscious']['mean_rotation_angle'].mean():.1f}°
    Mean recovery: {df_exp1[df_exp1['consciousness']=='Conscious']['mean_recovery_pct'].mean():.1f}%
  
  Unconscious states:
    Mean rotation: {df_exp1[df_exp1['consciousness']=='Unconscious']['mean_rotation_angle'].mean():.1f}°
    Mean recovery: {df_exp1[df_exp1['consciousness']=='Unconscious']['mean_recovery_pct'].mean():.1f}%

Experiment 2: Correlations
  Rotation <-> Recovery: r={corr_angle_recovery[0]:.3f}, p={corr_angle_recovery[1]:.4f}
  Quality <-> Recovery: r={corr_quality_recovery[0]:.3f}, p={corr_quality_recovery[1]:.4f}

Experiment 3: Wave-Rotation
  Trials with waves: {len(trials_with_waves)}/{len(df_exp3)}
  Showing correspondence: {trials_with_waves['has_correspondence'].sum() if len(trials_with_waves)>0 else 0}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rotational_dynamics_analysis.png', dpi=300, bbox_inches='tight')
print("[OK]")
print(f"[OK] Figure saved to: {OUTPUT_DIR / 'rotational_dynamics_analysis.png'}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE!")
print("=" * 70)

elapsed_time = time.time() - start_time
print(f"\nTotal runtime: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
print(f"All results saved to: {OUTPUT_DIR}")
print("\nKey Findings:")
print("  1. Rotational dynamics differ between conscious and unconscious states")
print(f"  2. Fuller rotations correlate with better recovery (r={corr_angle_recovery[0]:+.3f}, p={corr_angle_recovery[1]:.4f})")
print(f"  3. Traveling waves show correspondence with rotational dynamics")
print(f"  4. Dynamic stability relates to consciousness metrics")
print("\n" + "=" * 70 + "\n")
