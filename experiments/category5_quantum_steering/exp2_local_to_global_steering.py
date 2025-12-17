#!/usr/bin/env python3
"""
Category 5: Quantum Reality Steering
Experiment 2: Local-to-Global Steering Effects

Demonstrates how local quantum operations on small brain regions can steer
the global consciousness state. This models the key insight from quantum
reality steering that local memory operations can affect the entire system.

This experiment:
1. Applies steering to progressively larger local regions
2. Measures global state changes from local perturbations
3. Analyzes non-local correlation effects
4. Identifies critical region size for global steering
5. Tracks entanglement growth during local steering
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import quantum modules
from src.quantum import (
    RealityRegister,
    SteeringProtocol,
    compute_entanglement_entropy,
    compute_mutual_information,
    model_nonlocal_effects
)

# Import existing utilities
from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_MODES = 40
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp2_local_to_global_steering'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Quantum Reality Steering - Experiment 2")
print("Local-to-Global Steering Effects")
print("="*80)

# ==============================================================================
# PART 1: Progressive Region Expansion
# ==============================================================================

print("\n" + "-"*80)
print("PART 1: Progressive Local Region Expansion")
print("-"*80)

# Create reality register
register = RealityRegister(n_modes=N_MODES, seed=SEED)
print(f"Created reality register with {N_MODES} harmonic modes")

# Initialize in wake state
register.set_state(register.get_basis_state('wake'))
print("Initialized to WAKE state")

# Create steering protocol
protocol = SteeringProtocol(register)

# Perform local-to-global steering
print("\nPerforming local-to-global steering...")
result = protocol.local_to_global_steering(
    'nrem_sleep',
    local_region_size=5,
    n_steps=15
)

states = result['states']
region_sizes = result['region_sizes']

# Track metrics for each state
local_global_metrics = {
    'H_mode': [],
    'PR': [],
    'overlap_sleep': [],
    'entanglement': [],
    'region_sizes': region_sizes
}

print("\nAnalyzing state evolution...")
for i, state in tqdm(enumerate(states), total=len(states), desc="Computing metrics"):
    local_global_metrics['H_mode'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )
    local_global_metrics['PR'].append(
        met.compute_participation_ratio(state.power, normalize=True)
    )
    local_global_metrics['overlap_sleep'].append(
        state.overlap_probability(register.get_basis_state('nrem_sleep'))
    )
    
    # Compute entanglement of local region with rest
    if region_sizes[i] > 0 and region_sizes[i] < N_MODES:
        local_modes = np.arange(region_sizes[i])
        ent = compute_entanglement_entropy(state, local_modes)
        local_global_metrics['entanglement'].append(ent)
    else:
        local_global_metrics['entanglement'].append(0.0)

for key in ['H_mode', 'PR', 'overlap_sleep', 'entanglement']:
    local_global_metrics[key] = np.array(local_global_metrics[key])

# ==============================================================================
# PART 2: Fixed Local Region Steering
# ==============================================================================

print("\n" + "-"*80)
print("PART 2: Steering with Fixed Local Regions")
print("-"*80)

# Test different fixed local region sizes
region_sizes_to_test = [3, 5, 10, 15, 20]
n_steps = 30

fixed_region_results = {}

for region_size in region_sizes_to_test:
    print(f"\nTesting local region of size {region_size} modes...")
    
    # Reset to wake state
    register.set_state(register.get_basis_state('wake'))
    
    local_modes = np.arange(region_size)
    states_local = [register.get_state()]
    
    # Apply steering steps
    for _ in range(n_steps):
        new_state = protocol.steer_to_state(
            'nrem_sleep',
            strength=0.05,
            local_modes=local_modes,
            update_register=True
        )
        states_local.append(new_state)
    
    # Track final overlap with target
    final_overlap = states_local[-1].overlap_probability(
        register.get_basis_state('nrem_sleep')
    )
    
    fixed_region_results[region_size] = {
        'states': states_local,
        'final_overlap': final_overlap
    }
    
    print(f"  Final overlap with sleep: {final_overlap:.3f}")

# ==============================================================================
# PART 3: Non-local Effects from Local Perturbations
# ==============================================================================

print("\n" + "-"*80)
print("PART 3: Non-local Effects from Local Perturbations")
print("-"*80)

# Reset to wake state
register.set_state(register.get_basis_state('wake'))
initial_state = register.get_state()

# Apply local perturbation to first 10 modes
local_perturb_modes = np.arange(10)
print(f"\nApplying local perturbation to modes {local_perturb_modes[0]}-{local_perturb_modes[-1]}...")

perturbed_state, nonlocal_effects = model_nonlocal_effects(
    initial_state,
    local_perturb_modes,
    perturbation_strength=0.2
)

print("\nNon-local effects:")
print(f"  Distant power change:      {nonlocal_effects['distant_power_change']:.4f}")
print(f"  Entanglement change:       {nonlocal_effects['entropy_change']:.4f}")
print(f"  Local-distant correlation: {nonlocal_effects['local_distant_correlation']:.4f}")

# ==============================================================================
# PART 4: Mutual Information Between Regions
# ==============================================================================

print("\n" + "-"*80)
print("PART 4: Mutual Information Between Brain Regions")
print("-"*80)

# Reset to wake state
register.set_state(register.get_basis_state('wake'))

# Define multiple brain regions
n_regions = 4
region_size = N_MODES // n_regions
regions = [
    np.arange(i * region_size, (i + 1) * region_size)
    for i in range(n_regions)
]

print(f"\nDivided {N_MODES} modes into {n_regions} regions of size {region_size}")

# Compute mutual information matrix before steering
print("\nComputing mutual information (before steering)...")
mi_matrix_before = np.zeros((n_regions, n_regions))

for i in range(n_regions):
    for j in range(i+1, n_regions):
        mi = compute_mutual_information(
            register.get_state(), regions[i], regions[j]
        )
        mi_matrix_before[i, j] = mi
        mi_matrix_before[j, i] = mi

# Perform global steering
print("\nPerforming global steering to sleep state...")
for _ in range(20):
    protocol.steer_to_state('nrem_sleep', strength=0.05, update_register=True)

# Compute mutual information after steering
print("Computing mutual information (after steering)...")
mi_matrix_after = np.zeros((n_regions, n_regions))

for i in range(n_regions):
    for j in range(i+1, n_regions):
        mi = compute_mutual_information(
            register.get_state(), regions[i], regions[j]
        )
        mi_matrix_after[i, j] = mi
        mi_matrix_after[j, i] = mi

# ==============================================================================
# PART 5: Critical Region Size Analysis
# ==============================================================================

print("\n" + "-"*80)
print("PART 5: Critical Region Size for Global Steering")
print("-"*80)

# Sweep through region sizes and measure effectiveness
region_sizes_sweep = np.arange(1, N_MODES, 2)
effectiveness = []

print("\nSweeping region sizes...")
for region_size in tqdm(region_sizes_sweep, desc="Region size sweep"):
    # Reset to wake state
    register.set_state(register.get_basis_state('wake'))
    
    # Apply local steering
    local_modes = np.arange(region_size)
    for _ in range(20):
        protocol.steer_to_state(
            'nrem_sleep',
            strength=0.05,
            local_modes=local_modes,
            update_register=True
        )
    
    # Measure overlap with target
    overlap = register.get_state().overlap_probability(
        register.get_basis_state('nrem_sleep')
    )
    effectiveness.append(overlap)

effectiveness = np.array(effectiveness)

# Find critical region size (50% effectiveness threshold)
critical_idx = np.argmin(np.abs(effectiveness - 0.5))
critical_size = region_sizes_sweep[critical_idx]
print(f"\nCritical region size: {critical_size} modes ({critical_size/N_MODES*100:.1f}% of total)")

# ==============================================================================
# PART 6: Visualization
# ==============================================================================

print("\n" + "-"*80)
print("PART 6: Generating Visualizations")
print("-"*80)

# Figure 1: Progressive region expansion
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Local-to-Global Steering: Progressive Region Expansion', 
             fontsize=14, fontweight='bold')

steps = np.arange(len(local_global_metrics['H_mode']))

axes[0, 0].plot(local_global_metrics['region_sizes'], 
                local_global_metrics['H_mode'], 'b-o', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Local Region Size (modes)')
axes[0, 0].set_ylabel('Mode Entropy (normalized)')
axes[0, 0].set_title('Global H_mode vs. Local Region Size')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(local_global_metrics['region_sizes'], 
                local_global_metrics['PR'], 'r-o', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('Local Region Size (modes)')
axes[0, 1].set_ylabel('Participation Ratio (normalized)')
axes[0, 1].set_title('Global PR vs. Local Region Size')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(local_global_metrics['region_sizes'], 
                local_global_metrics['overlap_sleep'], 'g-o', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Local Region Size (modes)')
axes[1, 0].set_ylabel('Overlap with Sleep State')
axes[1, 0].set_title('Steering Effectiveness vs. Region Size')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(local_global_metrics['region_sizes'], 
                local_global_metrics['entanglement'], 'm-o', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Local Region Size (modes)')
axes[1, 1].set_ylabel('Entanglement Entropy')
axes[1, 1].set_title('Entanglement Growth')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'progressive_region_expansion.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'progressive_region_expansion.png'}")

# Figure 2: Fixed region sizes comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fixed Local Region Steering Comparison', fontsize=14, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0, 1, len(region_sizes_to_test)))

for i, region_size in enumerate(region_sizes_to_test):
    states_local = fixed_region_results[region_size]['states']
    overlaps = [s.overlap_probability(register.get_basis_state('nrem_sleep')) 
                for s in states_local]
    
    steps = np.arange(len(overlaps))
    axes[0].plot(steps, overlaps, color=colors[i], linewidth=2, 
                 label=f'{region_size} modes', alpha=0.8)

axes[0].set_xlabel('Steering Step')
axes[0].set_ylabel('Overlap with Sleep State')
axes[0].set_title('Steering Effectiveness by Region Size')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Final overlaps bar chart
final_overlaps = [fixed_region_results[rs]['final_overlap'] 
                  for rs in region_sizes_to_test]
axes[1].bar(range(len(region_sizes_to_test)), final_overlaps, 
            color=colors, alpha=0.8)
axes[1].set_xticks(range(len(region_sizes_to_test)))
axes[1].set_xticklabels([f'{rs}' for rs in region_sizes_to_test])
axes[1].set_xlabel('Local Region Size (modes)')
axes[1].set_ylabel('Final Overlap with Sleep')
axes[1].set_title('Final Steering Effectiveness')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fixed_region_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'fixed_region_comparison.png'}")

# Figure 3: Mutual information matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Inter-Regional Mutual Information', fontsize=14, fontweight='bold')

im1 = axes[0].imshow(mi_matrix_before, cmap='Blues', vmin=0)
axes[0].set_title('Before Steering (Wake State)')
axes[0].set_xlabel('Region Index')
axes[0].set_ylabel('Region Index')
axes[0].set_xticks(range(n_regions))
axes[0].set_yticks(range(n_regions))
plt.colorbar(im1, ax=axes[0], label='Mutual Information')

# Add values as text
for i in range(n_regions):
    for j in range(n_regions):
        if i != j:
            text = axes[0].text(j, i, f'{mi_matrix_before[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)

im2 = axes[1].imshow(mi_matrix_after, cmap='Blues', vmin=0)
axes[1].set_title('After Steering (Sleep State)')
axes[1].set_xlabel('Region Index')
axes[1].set_ylabel('Region Index')
axes[1].set_xticks(range(n_regions))
axes[1].set_yticks(range(n_regions))
plt.colorbar(im2, ax=axes[1], label='Mutual Information')

# Add values as text
for i in range(n_regions):
    for j in range(n_regions):
        if i != j:
            text = axes[1].text(j, i, f'{mi_matrix_after[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mutual_information_matrices.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'mutual_information_matrices.png'}")

# Figure 4: Critical region size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Critical Region Size for Global Steering', fontsize=14, fontweight='bold')

ax.plot(region_sizes_sweep, effectiveness, 'b-', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='50% effectiveness')
ax.axvline(x=critical_size, color='g', linestyle='--', linewidth=2, 
           label=f'Critical size = {critical_size} modes')
ax.fill_between(region_sizes_sweep, 0, effectiveness, alpha=0.3)

ax.set_xlabel('Local Region Size (modes)')
ax.set_ylabel('Steering Effectiveness (overlap with target)')
ax.set_title('Emergence of Global Effect from Local Operations')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'critical_region_size.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'critical_region_size.png'}")

plt.close('all')

# ==============================================================================
# PART 7: Summary Statistics
# ==============================================================================

print("\n" + "-"*80)
print("PART 7: Summary Statistics")
print("-"*80)

print("\nProgressive Region Expansion:")
print(f"  Initial region size: {region_sizes[0]} modes")
print(f"  Final region size: {region_sizes[-1]} modes")
print(f"  Initial overlap with sleep: {local_global_metrics['overlap_sleep'][0]:.3f}")
print(f"  Final overlap with sleep: {local_global_metrics['overlap_sleep'][-1]:.3f}")
print(f"  Final entanglement: {local_global_metrics['entanglement'][-1]:.3f}")

print("\nFixed Region Steering:")
for region_size in region_sizes_to_test:
    print(f"  Region size {region_size:2d}: final overlap = {fixed_region_results[region_size]['final_overlap']:.3f}")

print("\nNon-local Effects:")
print(f"  Local perturbation strength: {nonlocal_effects['perturbation_strength']:.2f}")
print(f"  Distant power change: {nonlocal_effects['distant_power_change']:.4f}")
print(f"  Entropy change: {nonlocal_effects['entropy_change']:.4f}")

print("\nCritical Region Size:")
print(f"  Critical size: {critical_size} modes ({critical_size/N_MODES*100:.1f}% of total)")
print(f"  Effectiveness at critical size: {effectiveness[critical_idx]:.3f}")

print("\n" + "="*80)
print("Experiment 2 Complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)
