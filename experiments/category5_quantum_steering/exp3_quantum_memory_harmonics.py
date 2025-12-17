#!/usr/bin/env python3
"""
Category 5: Quantum Reality Steering
Experiment 3: Quantum Memory and Harmonics

Explores the quantum register as a memory system for consciousness states.
Demonstrates how harmonic modes serve as quantum basis states and how
the system maintains coherent superpositions.

This experiment:
1. Creates superposition states mixing multiple consciousness types
2. Analyzes harmonic mode decomposition
3. Studies quantum memory capacity
4. Tests superposition stability over time
5. Examines interference patterns between consciousness states
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.quantum import RealityRegister, SteeringProtocol
from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_MODES = 30
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_quantum_memory_harmonics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Quantum Reality Steering - Experiment 3")
print("Quantum Memory and Harmonics")
print("="*80)

# ==============================================================================
# PART 1: Superposition States
# ==============================================================================

print("\n" + "-"*80)
print("PART 1: Creating Superposition States")
print("-"*80)

register = RealityRegister(n_modes=N_MODES, seed=SEED)
print(f"Created reality register with {N_MODES} harmonic modes")

# Create equal superposition of wake and sleep
print("\nCreating wake-sleep superposition...")
superpos_wake_sleep = register.create_superposition(['wake', 'nrem_sleep'])
register.set_state(superpos_wake_sleep)

decomp = register.get_state_decomposition()
print("State decomposition:")
for state_name, overlap in decomp.items():
    print(f"  {state_name:15s}: {overlap:.3f}")

# Create three-state superposition
print("\nCreating wake-sleep-rem superposition...")
coeffs = np.array([0.5, 0.5, 0.5], dtype=complex)
coeffs /= np.linalg.norm(coeffs)
superpos_three = register.create_superposition(
    ['wake', 'nrem_sleep', 'rem_sleep'],
    coefficients=coeffs
)
register.set_state(superpos_three)

decomp = register.get_state_decomposition()
print("State decomposition:")
for state_name, overlap in decomp.items():
    print(f"  {state_name:15s}: {overlap:.3f}")

# ==============================================================================
# PART 2: Harmonic Mode Analysis
# ==============================================================================

print("\n" + "-"*80)
print("PART 2: Harmonic Mode Analysis")
print("-"*80)

# Get all basis states
basis_states = ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia', 'meditation', 'psychedelic']
mode_profiles = {}

print("\nAnalyzing mode profiles for each consciousness state...")
for state_name in basis_states:
    state = register.get_basis_state(state_name)
    mode_profiles[state_name] = state.power

# Compute mode-wise variance across states
mode_variance = np.var([mode_profiles[s] for s in basis_states], axis=0)
high_variance_modes = np.argsort(mode_variance)[-10:]  # Top 10 discriminative modes

print(f"\nTop 10 discriminative modes (indices): {high_variance_modes}")
print(f"Their variances: {mode_variance[high_variance_modes]}")

# ==============================================================================
# PART 3: Quantum Memory Capacity
# ==============================================================================

print("\n" + "-"*80)
print("PART 3: Quantum Memory Capacity")
print("-"*80)

# Test how many distinct states can be reliably stored and retrieved
n_test_states = 20
memory_states = []
retrieval_accuracy = []

print(f"\nGenerating {n_test_states} random superposition states...")
for i in range(n_test_states):
    # Random superposition
    n_basis = np.random.randint(2, 4)
    states = np.random.choice(basis_states, size=n_basis, replace=False)
    coeffs = np.random.randn(n_basis) + 1j * np.random.randn(n_basis)
    coeffs /= np.linalg.norm(coeffs)
    
    state = register.create_superposition(list(states), coefficients=coeffs)
    memory_states.append((state, list(states)))

# Test retrieval by measuring overlap
print("\nTesting retrieval accuracy...")
for i, (state, original_basis) in enumerate(memory_states):
    register.set_state(state)
    measured_state, prob = register.measure_consciousness_type()
    
    # Check if measured state was in original superposition
    accuracy = 1.0 if measured_state in original_basis else 0.0
    retrieval_accuracy.append(accuracy)

avg_accuracy = np.mean(retrieval_accuracy)
print(f"Average retrieval accuracy: {avg_accuracy:.2%}")

# ==============================================================================
# PART 4: Superposition Stability
# ==============================================================================

print("\n" + "-"*80)
print("PART 4: Superposition Stability Over Time")
print("-"*80)

# Create superposition and track its evolution
register.set_state(register.create_superposition(['wake', 'nrem_sleep']))
protocol = SteeringProtocol(register)

print("\nTracking superposition stability over weak perturbations...")
n_steps = 50
stability_metrics = {
    'overlap_wake': [],
    'overlap_sleep': [],
    'entropy': [],
    'norm': []
}

for step in range(n_steps):
    state = register.get_state()
    
    stability_metrics['overlap_wake'].append(
        state.overlap_probability(register.get_basis_state('wake'))
    )
    stability_metrics['overlap_sleep'].append(
        state.overlap_probability(register.get_basis_state('nrem_sleep'))
    )
    stability_metrics['entropy'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )
    stability_metrics['norm'].append(state.norm)
    
    # Small random perturbation
    if step % 10 == 0 and step > 0:
        target = np.random.choice(['wake', 'nrem_sleep'])
        protocol.steer_to_state(target, strength=0.02, update_register=True)

for key in stability_metrics:
    stability_metrics[key] = np.array(stability_metrics[key])

# ==============================================================================
# PART 5: Interference Patterns
# ==============================================================================

print("\n" + "-"*80)
print("PART 5: Quantum Interference Patterns")
print("-"*80)

# Create interference by combining two paths to same final state
print("\nCreating quantum interference pattern...")

# Path 1: wake -> sleep directly
register.set_state(register.get_basis_state('wake'))
path1_states = protocol.gradual_steering('nrem_sleep', n_steps=20)

# Path 2: wake -> meditation -> sleep
register.set_state(register.get_basis_state('wake'))
path2a_states = protocol.gradual_steering('meditation', n_steps=10)
path2b_states = protocol.gradual_steering('nrem_sleep', n_steps=10)
path2_states = path2a_states + path2b_states

# Compare final states
final_path1 = path1_states[-1]
final_path2 = path2_states[-1]

interference = final_path1.inner_product(final_path2)
print(f"Path 1 final state: overlap with sleep = {final_path1.overlap_probability(register.get_basis_state('nrem_sleep')):.3f}")
print(f"Path 2 final state: overlap with sleep = {final_path2.overlap_probability(register.get_basis_state('nrem_sleep')):.3f}")
print(f"Interference between paths: {np.abs(interference):.3f}")
print(f"Relative phase: {np.angle(interference):.3f} rad")

# ==============================================================================
# PART 6: Visualization
# ==============================================================================

print("\n" + "-"*80)
print("PART 6: Generating Visualizations")
print("-"*80)

# Figure 1: Mode profiles for different consciousness states
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Harmonic Mode Profiles for Consciousness States', fontsize=14, fontweight='bold')

for idx, state_name in enumerate(basis_states):
    ax = axes[idx // 3, idx % 3]
    modes = np.arange(N_MODES)
    power = mode_profiles[state_name]
    
    ax.bar(modes, power, alpha=0.7, color=f'C{idx}')
    ax.set_xlabel('Harmonic Mode Index')
    ax.set_ylabel('Power')
    ax.set_title(f'{state_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mode_profiles.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'mode_profiles.png'}")

# Figure 2: Superposition stability
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Superposition Stability Over Time', fontsize=14, fontweight='bold')

steps = np.arange(n_steps)

axes[0, 0].plot(steps, stability_metrics['overlap_wake'], 'b-', linewidth=2, label='Wake')
axes[0, 0].plot(steps, stability_metrics['overlap_sleep'], 'r-', linewidth=2, label='Sleep')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Overlap Probability')
axes[0, 0].set_title('Superposition Component Overlaps')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(steps, stability_metrics['entropy'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Mode Entropy')
axes[0, 1].set_title('Entropy Evolution')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(steps, stability_metrics['norm'], 'm-', linewidth=2)
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('State Norm')
axes[1, 0].set_title('Normalization (should be ≈ 1)')
axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].grid(True, alpha=0.3)

# Total superposition (wake + sleep overlap)
total_superpos = stability_metrics['overlap_wake'] + stability_metrics['overlap_sleep']
axes[1, 1].plot(steps, total_superpos, 'k-', linewidth=2)
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Total Overlap')
axes[1, 1].set_title('Superposition Stability (sum of overlaps)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'superposition_stability.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'superposition_stability.png'}")

# Figure 3: Quantum interference
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Quantum Interference Between Paths', fontsize=14, fontweight='bold')

# Path comparison
path1_overlaps = [s.overlap_probability(register.get_basis_state('nrem_sleep')) 
                  for s in path1_states]
path2_overlaps = [s.overlap_probability(register.get_basis_state('nrem_sleep')) 
                  for s in path2_states]

axes[0].plot(range(len(path1_overlaps)), path1_overlaps, 'b-', linewidth=2, label='Direct path')
axes[0].plot(range(len(path2_overlaps)), path2_overlaps, 'r-', linewidth=2, label='Via meditation')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Overlap with Target (Sleep)')
axes[0].set_title('Alternative Paths to Same State')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Power distribution comparison
modes = np.arange(N_MODES)
axes[1].plot(modes, path1_states[-1].power, 'b-', linewidth=2, alpha=0.7, label='Direct path')
axes[1].plot(modes, path2_states[-1].power, 'r-', linewidth=2, alpha=0.7, label='Via meditation')
axes[1].set_xlabel('Harmonic Mode Index')
axes[1].set_ylabel('Power')
axes[1].set_title('Final State Mode Distributions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'quantum_interference.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'quantum_interference.png'}")

plt.close('all')

# ==============================================================================
# PART 7: Summary
# ==============================================================================

print("\n" + "-"*80)
print("PART 7: Summary Statistics")
print("-"*80)

print("\nQuantum Memory:")
print(f"  Number of test states: {n_test_states}")
print(f"  Average retrieval accuracy: {avg_accuracy:.2%}")

print("\nSuperposition Stability:")
print(f"  Initial wake overlap: {stability_metrics['overlap_wake'][0]:.3f}")
print(f"  Final wake overlap: {stability_metrics['overlap_wake'][-1]:.3f}")
print(f"  Initial sleep overlap: {stability_metrics['overlap_sleep'][0]:.3f}")
print(f"  Final sleep overlap: {stability_metrics['overlap_sleep'][-1]:.3f}")

print("\nQuantum Interference:")
print(f"  Path 1 (direct) final overlap: {path1_overlaps[-1]:.3f}")
print(f"  Path 2 (via meditation) final overlap: {path2_overlaps[-1]:.3f}")
print(f"  Interference magnitude: {np.abs(interference):.3f}")

print("\n" + "="*80)
print("Experiment 3 Complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)
