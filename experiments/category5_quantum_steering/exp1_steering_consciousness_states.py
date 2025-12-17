#!/usr/bin/env python3
"""
Category 5: Quantum Reality Steering
Experiment 1: Steering Between Consciousness States

Demonstrates quantum steering operations for transitioning between different
consciousness states (wake, NREM sleep, REM sleep, anesthesia, meditation, psychedelic).

This experiment:
1. Creates a quantum reality register with harmonic modes
2. Initializes in wake state
3. Performs gradual steering to various target states
4. Measures transition probabilities and state overlaps
5. Visualizes trajectories in quantum state space
6. Tracks evolution of consciousness metrics during steering
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
    compute_steering_probability
)

# Import existing utilities for consciousness metrics
from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_MODES = 30
N_STEERING_STEPS = 50
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp1_steering_consciousness_states'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Quantum Reality Steering - Experiment 1")
print("Steering Between Consciousness States")
print("="*80)

# ==============================================================================
# PART 1: Initialize Quantum Reality Register
# ==============================================================================

print("\n" + "-"*80)
print("PART 1: Initialize Quantum Reality Register")
print("-"*80)

# Create reality register
register = RealityRegister(n_modes=N_MODES, seed=SEED)
print(f"Created reality register with {N_MODES} harmonic modes")

# Display available basis states
print("\nAvailable consciousness basis states:")
basis_states = ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia', 'meditation', 'psychedelic']
for state_name in basis_states:
    state = register.get_basis_state(state_name)
    H_mode = met.compute_mode_entropy(state.power, normalize=True)
    PR = met.compute_participation_ratio(state.power, normalize=True)
    print(f"  {state_name:15s}: H_mode={H_mode:.3f}, PR={PR:.3f}")

# ==============================================================================
# PART 2: Steering from Wake to Sleep
# ==============================================================================

print("\n" + "-"*80)
print("PART 2: Steering from Wake to NREM Sleep")
print("-"*80)

# Initialize in wake state
register.set_state(register.get_basis_state('wake'))
print("Initialized to WAKE state")

# Create steering protocol
protocol = SteeringProtocol(register)

# Perform gradual steering
print(f"\nPerforming gradual steering over {N_STEERING_STEPS} steps...")
states_wake_to_sleep = protocol.gradual_steering(
    'nrem_sleep',
    n_steps=N_STEERING_STEPS,
    total_strength=1.0
)

# Measure final state
final_state, final_prob = register.measure_consciousness_type()
print(f"Final measured state: {final_state} (probability: {final_prob:.3f})")

# Track metrics during steering
print("\nTracking consciousness metrics during steering...")
wake_to_sleep_metrics = {
    'H_mode': [],
    'PR': [],
    'R': [],
    'overlap_wake': [],
    'overlap_sleep': []
}

for state in states_wake_to_sleep:
    wake_to_sleep_metrics['H_mode'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )
    wake_to_sleep_metrics['PR'].append(
        met.compute_participation_ratio(state.power, normalize=True)
    )
    wake_to_sleep_metrics['R'].append(
        met.compute_phase_coherence(state.phases, state.power)
    )
    wake_to_sleep_metrics['overlap_wake'].append(
        state.overlap_probability(register.get_basis_state('wake'))
    )
    wake_to_sleep_metrics['overlap_sleep'].append(
        state.overlap_probability(register.get_basis_state('nrem_sleep'))
    )

# Convert to arrays
for key in wake_to_sleep_metrics:
    wake_to_sleep_metrics[key] = np.array(wake_to_sleep_metrics[key])

# ==============================================================================
# PART 3: Steering from Sleep to Wake (Awakening)
# ==============================================================================

print("\n" + "-"*80)
print("PART 3: Steering from NREM Sleep to Wake (Awakening)")
print("-"*80)

# Reset to sleep state
register.set_state(register.get_basis_state('nrem_sleep'))
print("Reset to NREM SLEEP state")

# Perform gradual steering back to wake
print(f"\nPerforming awakening over {N_STEERING_STEPS} steps...")
states_sleep_to_wake = protocol.gradual_steering(
    'wake',
    n_steps=N_STEERING_STEPS,
    total_strength=1.0
)

# Track metrics
sleep_to_wake_metrics = {
    'H_mode': [],
    'PR': [],
    'R': [],
    'overlap_wake': [],
    'overlap_sleep': []
}

for state in states_sleep_to_wake:
    sleep_to_wake_metrics['H_mode'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )
    sleep_to_wake_metrics['PR'].append(
        met.compute_participation_ratio(state.power, normalize=True)
    )
    sleep_to_wake_metrics['R'].append(
        met.compute_phase_coherence(state.phases, state.power)
    )
    sleep_to_wake_metrics['overlap_wake'].append(
        state.overlap_probability(register.get_basis_state('wake'))
    )
    sleep_to_wake_metrics['overlap_sleep'].append(
        state.overlap_probability(register.get_basis_state('nrem_sleep'))
    )

for key in sleep_to_wake_metrics:
    sleep_to_wake_metrics[key] = np.array(sleep_to_wake_metrics[key])

# ==============================================================================
# PART 4: Anesthesia Induction
# ==============================================================================

print("\n" + "-"*80)
print("PART 4: Anesthesia Induction (Wake → Anesthesia)")
print("-"*80)

# Reset to wake state
register.set_state(register.get_basis_state('wake'))
print("Reset to WAKE state")

# Steer to anesthesia
print(f"\nInducing anesthesia over {N_STEERING_STEPS} steps...")
states_wake_to_anesthesia = protocol.gradual_steering(
    'anesthesia',
    n_steps=N_STEERING_STEPS,
    total_strength=1.2  # Stronger steering for anesthesia
)

# Track metrics
anesthesia_metrics = {
    'H_mode': [],
    'PR': [],
    'R': []
}

for state in states_wake_to_anesthesia:
    anesthesia_metrics['H_mode'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )
    anesthesia_metrics['PR'].append(
        met.compute_participation_ratio(state.power, normalize=True)
    )
    anesthesia_metrics['R'].append(
        met.compute_phase_coherence(state.phases, state.power)
    )

for key in anesthesia_metrics:
    anesthesia_metrics[key] = np.array(anesthesia_metrics[key])

# ==============================================================================
# PART 5: Oscillatory Transitions (Wake/Sleep Cycles)
# ==============================================================================

print("\n" + "-"*80)
print("PART 5: Oscillatory Wake/Sleep Cycles")
print("-"*80)

# Reset to wake state
register.set_state(register.get_basis_state('wake'))
print("Reset to WAKE state")

# Create oscillatory transitions
print("\nSimulating 3 wake/sleep cycles...")
oscillatory_states = protocol.oscillatory_steering(
    'wake',
    'nrem_sleep',
    n_cycles=3,
    steps_per_cycle=30
)

# Track which state dominates at each point
oscillatory_metrics = {
    'overlap_wake': [],
    'overlap_sleep': [],
    'H_mode': []
}

for state in oscillatory_states:
    oscillatory_metrics['overlap_wake'].append(
        state.overlap_probability(register.get_basis_state('wake'))
    )
    oscillatory_metrics['overlap_sleep'].append(
        state.overlap_probability(register.get_basis_state('nrem_sleep'))
    )
    oscillatory_metrics['H_mode'].append(
        met.compute_mode_entropy(state.power, normalize=True)
    )

for key in oscillatory_metrics:
    oscillatory_metrics[key] = np.array(oscillatory_metrics[key])

# ==============================================================================
# PART 6: Visualization
# ==============================================================================

print("\n" + "-"*80)
print("PART 6: Generating Visualizations")
print("-"*80)

# Figure 1: Wake → Sleep transition
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quantum Steering: Wake → NREM Sleep', fontsize=14, fontweight='bold')

steps = np.arange(len(wake_to_sleep_metrics['H_mode']))

# Consciousness metrics
axes[0, 0].plot(steps, wake_to_sleep_metrics['H_mode'], 'b-', linewidth=2, label='H_mode')
axes[0, 0].plot(steps, wake_to_sleep_metrics['PR'], 'r-', linewidth=2, label='PR')
axes[0, 0].set_xlabel('Steering Step')
axes[0, 0].set_ylabel('Normalized Value')
axes[0, 0].set_title('Mode Entropy & Participation Ratio')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase coherence
axes[0, 1].plot(steps, wake_to_sleep_metrics['R'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Steering Step')
axes[0, 1].set_ylabel('Phase Coherence R')
axes[0, 1].set_title('Phase Coherence Evolution')
axes[0, 1].grid(True, alpha=0.3)

# State overlaps
axes[1, 0].plot(steps, wake_to_sleep_metrics['overlap_wake'], 'b-', 
                linewidth=2, label='Overlap with Wake')
axes[1, 0].plot(steps, wake_to_sleep_metrics['overlap_sleep'], 'r-', 
                linewidth=2, label='Overlap with Sleep')
axes[1, 0].set_xlabel('Steering Step')
axes[1, 0].set_ylabel('Overlap Probability')
axes[1, 0].set_title('Quantum State Overlaps')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Power distribution evolution (show first, middle, last)
indices = [0, len(states_wake_to_sleep)//2, len(states_wake_to_sleep)-1]
colors = ['blue', 'purple', 'red']
labels = ['Initial (Wake)', 'Mid-transition', 'Final (Sleep)']
modes = np.arange(N_MODES)

for idx, color, label in zip(indices, colors, labels):
    axes[1, 1].plot(modes, states_wake_to_sleep[idx].power, 
                    color=color, linewidth=2, alpha=0.7, label=label)

axes[1, 1].set_xlabel('Harmonic Mode Index')
axes[1, 1].set_ylabel('Power')
axes[1, 1].set_title('Mode Power Distribution Evolution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'wake_to_sleep_steering.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'wake_to_sleep_steering.png'}")

# Figure 2: Sleep → Wake transition
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quantum Steering: NREM Sleep → Wake (Awakening)', fontsize=14, fontweight='bold')

steps = np.arange(len(sleep_to_wake_metrics['H_mode']))

axes[0, 0].plot(steps, sleep_to_wake_metrics['H_mode'], 'b-', linewidth=2, label='H_mode')
axes[0, 0].plot(steps, sleep_to_wake_metrics['PR'], 'r-', linewidth=2, label='PR')
axes[0, 0].set_xlabel('Steering Step')
axes[0, 0].set_ylabel('Normalized Value')
axes[0, 0].set_title('Mode Entropy & Participation Ratio')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(steps, sleep_to_wake_metrics['R'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Steering Step')
axes[0, 1].set_ylabel('Phase Coherence R')
axes[0, 1].set_title('Phase Coherence Evolution')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(steps, sleep_to_wake_metrics['overlap_sleep'], 'r-', 
                linewidth=2, label='Overlap with Sleep')
axes[1, 0].plot(steps, sleep_to_wake_metrics['overlap_wake'], 'b-', 
                linewidth=2, label='Overlap with Wake')
axes[1, 0].set_xlabel('Steering Step')
axes[1, 0].set_ylabel('Overlap Probability')
axes[1, 0].set_title('Quantum State Overlaps')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

indices = [0, len(states_sleep_to_wake)//2, len(states_sleep_to_wake)-1]
colors = ['red', 'purple', 'blue']
labels = ['Initial (Sleep)', 'Mid-transition', 'Final (Wake)']

for idx, color, label in zip(indices, colors, labels):
    axes[1, 1].plot(modes, states_sleep_to_wake[idx].power, 
                    color=color, linewidth=2, alpha=0.7, label=label)

axes[1, 1].set_xlabel('Harmonic Mode Index')
axes[1, 1].set_ylabel('Power')
axes[1, 1].set_title('Mode Power Distribution Evolution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sleep_to_wake_steering.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'sleep_to_wake_steering.png'}")

# Figure 3: Anesthesia induction
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quantum Steering: Wake → Anesthesia Induction', fontsize=14, fontweight='bold')

steps = np.arange(len(anesthesia_metrics['H_mode']))

axes[0, 0].plot(steps, anesthesia_metrics['H_mode'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Steering Step')
axes[0, 0].set_ylabel('Mode Entropy (normalized)')
axes[0, 0].set_title('Mode Entropy Collapse')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(steps, anesthesia_metrics['PR'], 'r-', linewidth=2)
axes[0, 1].set_xlabel('Steering Step')
axes[0, 1].set_ylabel('Participation Ratio (normalized)')
axes[0, 1].set_title('Participation Ratio Collapse')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(steps, anesthesia_metrics['R'], 'g-', linewidth=2)
axes[1, 0].set_xlabel('Steering Step')
axes[1, 0].set_ylabel('Phase Coherence')
axes[1, 0].set_title('Phase Coherence Increase')
axes[1, 0].grid(True, alpha=0.3)

indices = [0, len(states_wake_to_anesthesia)//2, len(states_wake_to_anesthesia)-1]
colors = ['blue', 'orange', 'darkred']
labels = ['Initial (Wake)', 'Mid-induction', 'Final (Anesthesia)']

for idx, color, label in zip(indices, colors, labels):
    axes[1, 1].plot(modes, states_wake_to_anesthesia[idx].power, 
                    color=color, linewidth=2, alpha=0.7, label=label)

axes[1, 1].set_xlabel('Harmonic Mode Index')
axes[1, 1].set_ylabel('Power')
axes[1, 1].set_title('Extreme Low-Mode Concentration')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anesthesia_induction.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'anesthesia_induction.png'}")

# Figure 4: Oscillatory cycles
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Oscillatory Wake/Sleep Cycles', fontsize=14, fontweight='bold')

steps = np.arange(len(oscillatory_metrics['overlap_wake']))

axes[0].plot(steps, oscillatory_metrics['overlap_wake'], 'b-', 
             linewidth=2, label='Overlap with Wake')
axes[0].plot(steps, oscillatory_metrics['overlap_sleep'], 'r-', 
             linewidth=2, label='Overlap with Sleep')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Overlap Probability')
axes[0].set_title('Oscillatory State Transitions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(steps, oscillatory_metrics['H_mode'], 'g-', linewidth=2)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Mode Entropy')
axes[1].set_title('Mode Entropy Oscillations')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'oscillatory_cycles.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'oscillatory_cycles.png'}")

plt.close('all')

# ==============================================================================
# PART 7: Summary Statistics
# ==============================================================================

print("\n" + "-"*80)
print("PART 7: Summary Statistics")
print("-"*80)

print("\nWake → Sleep Transition:")
print(f"  Initial H_mode: {wake_to_sleep_metrics['H_mode'][0]:.3f}")
print(f"  Final H_mode:   {wake_to_sleep_metrics['H_mode'][-1]:.3f}")
print(f"  Initial PR:     {wake_to_sleep_metrics['PR'][0]:.3f}")
print(f"  Final PR:       {wake_to_sleep_metrics['PR'][-1]:.3f}")
print(f"  Final overlap with sleep: {wake_to_sleep_metrics['overlap_sleep'][-1]:.3f}")

print("\nSleep → Wake Transition:")
print(f"  Initial H_mode: {sleep_to_wake_metrics['H_mode'][0]:.3f}")
print(f"  Final H_mode:   {sleep_to_wake_metrics['H_mode'][-1]:.3f}")
print(f"  Initial PR:     {sleep_to_wake_metrics['PR'][0]:.3f}")
print(f"  Final PR:       {sleep_to_wake_metrics['PR'][-1]:.3f}")
print(f"  Final overlap with wake: {sleep_to_wake_metrics['overlap_wake'][-1]:.3f}")

print("\nAnesthesia Induction:")
print(f"  Initial H_mode: {anesthesia_metrics['H_mode'][0]:.3f}")
print(f"  Final H_mode:   {anesthesia_metrics['H_mode'][-1]:.3f}")
print(f"  Initial PR:     {anesthesia_metrics['PR'][0]:.3f}")
print(f"  Final PR:       {anesthesia_metrics['PR'][-1]:.3f}")
print(f"  Final R:        {anesthesia_metrics['R'][-1]:.3f}")

print("\n" + "="*80)
print("Experiment 1 Complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)
