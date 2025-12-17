#!/usr/bin/env python3
"""
Category 5: Quantum Reality Steering
Experiment 4: Measurement and Collapse

Studies quantum measurement effects on consciousness states including:
- Projective measurements in different bases
- State collapse dynamics
- Weak measurements and continuous monitoring
- Measurement back-action on consciousness metrics

This experiment:
1. Performs projective measurements and observes state collapse
2. Implements weak measurements with partial collapse
3. Models continuous monitoring of consciousness
4. Analyzes measurement back-action effects
5. Studies the quantum Zeno effect in consciousness
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.quantum import (
    RealityRegister,
    QuantumMeasurement,
    SteeringProtocol,
    measure_consciousness_state
)
from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_MODES = 30
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_measurement_collapse'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Quantum Reality Steering - Experiment 4")
print("Measurement and Collapse")
print("="*80)

# ==============================================================================
# PART 1: Projective Measurement and Collapse
# ==============================================================================

print("\n" + "-"*80)
print("PART 1: Projective Measurement and State Collapse")
print("-"*80)

register = RealityRegister(n_modes=N_MODES, seed=SEED)
measurement = QuantumMeasurement(register)

# Create superposition state
print("Creating wake-sleep superposition...")
superpos_state = register.create_superposition(['wake', 'nrem_sleep'])
register.set_state(superpos_state)

print("\nBefore measurement:")
decomp = register.get_state_decomposition()
for state_name, overlap in decomp.items():
    if overlap > 0.01:
        print(f"  {state_name:15s}: {overlap:.3f}")

# Perform measurement
print("\nPerforming projective measurement...")
measured_state, prob, collapsed_state = measurement.measure_consciousness_state(collapse=True)
print(f"Measured state: {measured_state} (probability: {prob:.3f})")

print("\nAfter measurement (collapsed):")
decomp = register.get_state_decomposition()
for state_name, overlap in decomp.items():
    if overlap > 0.01:
        print(f"  {state_name:15s}: {overlap:.3f}")

# Repeat measurement multiple times
print("\n" + "-"*40)
print("Repeated measurements (30 trials):")
register.set_state(superpos_state)  # Reset

measurement_results = []
for i in range(30):
    register.set_state(superpos_state)  # Reset for each trial
    measured, prob, _ = measurement.measure_consciousness_state(collapse=True)
    measurement_results.append(measured)

from collections import Counter
counts = Counter(measurement_results)
print("\nMeasurement statistics:")
for state, count in counts.most_common():
    print(f"  {state:15s}: {count}/30 ({count/30:.1%})")

# ==============================================================================
# PART 2: Weak Measurements
# ==============================================================================

print("\n" + "-"*80)
print("PART 2: Weak Measurements (Partial Collapse)")
print("-"*80)

# Create measurement observable (mode energy)
energy_operator = np.diag(np.arange(N_MODES, dtype=float))

# Test different measurement strengths
measurement_strengths = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
weak_measurement_results = {}

print("\nTesting weak measurements with different strengths...")
for strength in measurement_strengths:
    register.set_state(register.create_superposition(['wake', 'nrem_sleep']))
    
    states_before_after = {
        'before': register.get_state(),
        'entropy_before': met.compute_mode_entropy(register.get_state().power, normalize=True)
    }
    
    # Perform weak measurement
    expectation, new_state = measurement.weak_measurement(energy_operator, strength=strength)
    
    states_before_after['after'] = new_state
    states_before_after['entropy_after'] = met.compute_mode_entropy(new_state.power, normalize=True)
    states_before_after['expectation'] = expectation
    
    weak_measurement_results[strength] = states_before_after
    
    print(f"  Strength {strength:.1f}: E = {expectation:.2f}, "
          f"ΔH = {states_before_after['entropy_after'] - states_before_after['entropy_before']:.3f}")

# ==============================================================================
# PART 3: Continuous Monitoring
# ==============================================================================

print("\n" + "-"*80)
print("PART 3: Continuous Monitoring")
print("-"*80)

# Reset and create dynamic state
register.set_state(register.get_basis_state('wake'))
protocol = SteeringProtocol(register)

# Start steering while continuously monitoring
print("\nSteering to sleep while continuously monitoring...")
n_timesteps = 100
monitor_data = measurement.continuous_monitoring(
    n_timesteps=n_timesteps,
    measurement_rate=0.3,
    weak_measurement_strength=0.05
)

print(f"Completed {n_timesteps} timesteps with continuous monitoring")
print(f"Number of entropy measurements: {len(monitor_data['entropies'])}")

# ==============================================================================
# PART 4: Quantum Zeno Effect
# ==============================================================================

print("\n" + "-"*80)
print("PART 4: Quantum Zeno Effect")
print("-"*80)

# Demonstrate Zeno effect: frequent measurements slow evolution
print("\nComparing evolution with different measurement frequencies...")

# No measurements (free evolution)
register.set_state(register.get_basis_state('wake'))
no_measure_overlaps = []
for _ in range(50):
    no_measure_overlaps.append(
        register.get_state().overlap_probability(register.get_basis_state('wake'))
    )
    protocol.steer_to_state('nrem_sleep', strength=0.05, update_register=True)

# Moderate measurements
register.set_state(register.get_basis_state('wake'))
moderate_measure_overlaps = []
for i in range(50):
    moderate_measure_overlaps.append(
        register.get_state().overlap_probability(register.get_basis_state('wake'))
    )
    if i % 5 == 0:  # Measure every 5 steps
        _, _ = measurement.weak_measurement(energy_operator, strength=0.3)
    protocol.steer_to_state('nrem_sleep', strength=0.05, update_register=True)

# Frequent measurements
register.set_state(register.get_basis_state('wake'))
frequent_measure_overlaps = []
for i in range(50):
    frequent_measure_overlaps.append(
        register.get_state().overlap_probability(register.get_basis_state('wake'))
    )
    # Measure every step
    _, _ = measurement.weak_measurement(energy_operator, strength=0.3)
    protocol.steer_to_state('nrem_sleep', strength=0.05, update_register=True)

no_measure_overlaps = np.array(no_measure_overlaps)
moderate_measure_overlaps = np.array(moderate_measure_overlaps)
frequent_measure_overlaps = np.array(frequent_measure_overlaps)

print(f"\nFinal overlap with wake state:")
print(f"  No measurements:      {no_measure_overlaps[-1]:.3f}")
print(f"  Moderate measurements: {moderate_measure_overlaps[-1]:.3f}")
print(f"  Frequent measurements: {frequent_measure_overlaps[-1]:.3f}")
print(f"\nZeno effect: {frequent_measure_overlaps[-1] - no_measure_overlaps[-1]:.3f} increase")

# ==============================================================================
# PART 5: Mode-Specific Measurements
# ==============================================================================

print("\n" + "-"*80)
print("PART 5: Mode-Specific Measurements")
print("-"*80)

# Measure individual modes and observe collapse
register.set_state(register.create_superposition(['wake', 'nrem_sleep', 'rem_sleep']))
print("\nInitial state: 3-state superposition")
print(f"Initial entropy: {met.compute_mode_entropy(register.get_state().power, normalize=True):.3f}")

mode_measurement_sequence = []
for mode_idx in [0, 5, 10, 15, 20]:
    power, collapsed = measurement.measure_mode_occupation(mode_idx, collapse=False)
    mode_measurement_sequence.append({
        'mode': mode_idx,
        'power': power,
        'entropy': met.compute_mode_entropy(register.get_state().power, normalize=True)
    })
    print(f"  Mode {mode_idx:2d}: power = {power:.4f}")

# ==============================================================================
# PART 6: Visualization
# ==============================================================================

print("\n" + "-"*80)
print("PART 6: Generating Visualizations")
print("-"*80)

# Figure 1: Weak measurement effects
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Weak Measurement Effects', fontsize=14, fontweight='bold')

strengths = list(weak_measurement_results.keys())
entropy_changes = [weak_measurement_results[s]['entropy_after'] - weak_measurement_results[s]['entropy_before'] 
                   for s in strengths]

axes[0, 0].plot(strengths, entropy_changes, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Measurement Strength')
axes[0, 0].set_ylabel('Entropy Change')
axes[0, 0].set_title('Measurement-Induced Entropy Change')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Power distributions before/after strong measurement
modes = np.arange(N_MODES)
before_power = weak_measurement_results[1.0]['before'].power
after_power = weak_measurement_results[1.0]['after'].power

axes[0, 1].plot(modes, before_power, 'b-', linewidth=2, alpha=0.7, label='Before')
axes[0, 1].plot(modes, after_power, 'r-', linewidth=2, alpha=0.7, label='After (strength=1.0)')
axes[0, 1].set_xlabel('Harmonic Mode')
axes[0, 1].set_ylabel('Power')
axes[0, 1].set_title('Power Distribution: Strong Measurement')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Multiple weak measurements
for strength in [0.1, 0.3, 0.5, 1.0]:
    after = weak_measurement_results[strength]['after']
    axes[1, 0].plot(modes, after.power, linewidth=2, alpha=0.7, label=f'Strength {strength}')

axes[1, 0].set_xlabel('Harmonic Mode')
axes[1, 0].set_ylabel('Power')
axes[1, 0].set_title('Power Distribution: Various Measurement Strengths')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Expectation values
expectations = [weak_measurement_results[s]['expectation'] for s in strengths]
axes[1, 1].plot(strengths, expectations, 'go-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Measurement Strength')
axes[1, 1].set_ylabel('Energy Expectation Value')
axes[1, 1].set_title('Measured Energy vs. Measurement Strength')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'weak_measurements.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'weak_measurements.png'}")

# Figure 2: Continuous monitoring
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Continuous Monitoring of Consciousness', fontsize=14, fontweight='bold')

time = monitor_data['time']

axes[0, 0].plot(time, monitor_data['entropies'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Mode Entropy')
axes[0, 0].set_title('Entropy Evolution Under Monitoring')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time, monitor_data['participation_ratios'], 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Participation Ratio')
axes[0, 1].set_title('PR Evolution Under Monitoring')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(time, monitor_data['coherences'], 'g-', linewidth=2)
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Phase Coherence')
axes[1, 0].set_title('Coherence Evolution Under Monitoring')
axes[1, 0].grid(True, alpha=0.3)

# State probabilities over time
state_names = [s[0] for s in monitor_data['measured_states']]
unique_states = list(set(state_names))
state_counts = {s: [] for s in unique_states}

window = 10
for i in range(len(state_names) - window):
    window_states = state_names[i:i+window]
    for s in unique_states:
        state_counts[s].append(window_states.count(s) / window)

for state in unique_states:
    if len(state_counts[state]) > 0:
        axes[1, 1].plot(range(len(state_counts[state])), state_counts[state], 
                       linewidth=2, label=state, alpha=0.7)

axes[1, 1].set_xlabel('Time Window')
axes[1, 1].set_ylabel('State Probability')
axes[1, 1].set_title('Measured State Distribution (rolling window)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'continuous_monitoring.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'continuous_monitoring.png'}")

# Figure 3: Quantum Zeno effect
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Quantum Zeno Effect in Consciousness State Evolution', 
             fontsize=14, fontweight='bold')

steps = np.arange(50)
ax.plot(steps, no_measure_overlaps, 'b-', linewidth=2, label='No measurements')
ax.plot(steps, moderate_measure_overlaps, 'g-', linewidth=2, label='Moderate measurements')
ax.plot(steps, frequent_measure_overlaps, 'r-', linewidth=2, label='Frequent measurements')

ax.set_xlabel('Steering Step')
ax.set_ylabel('Overlap with Wake State')
ax.set_title('Measurement Frequency Slows State Evolution (Zeno Effect)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'quantum_zeno.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'quantum_zeno.png'}")

plt.close('all')

# ==============================================================================
# PART 7: Summary
# ==============================================================================

print("\n" + "-"*80)
print("PART 7: Summary Statistics")
print("-"*80)

print("\nProjective Measurement:")
print(f"  Superposition collapsed to: {measured_state}")
print(f"  Measurement statistics over 30 trials:")
for state, count in counts.most_common():
    print(f"    {state:15s}: {count/30:.1%}")

print("\nWeak Measurements:")
print(f"  Entropy change at strength 0.0: {weak_measurement_results[0.0]['entropy_after'] - weak_measurement_results[0.0]['entropy_before']:.3f}")
print(f"  Entropy change at strength 1.0: {weak_measurement_results[1.0]['entropy_after'] - weak_measurement_results[1.0]['entropy_before']:.3f}")

print("\nQuantum Zeno Effect:")
print(f"  No measurements - final wake overlap: {no_measure_overlaps[-1]:.3f}")
print(f"  Frequent measurements - final wake overlap: {frequent_measure_overlaps[-1]:.3f}")
print(f"  Zeno effect magnitude: {frequent_measure_overlaps[-1] - no_measure_overlaps[-1]:.3f}")

print("\n" + "="*80)
print("Experiment 4 Complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)
