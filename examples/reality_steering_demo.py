#!/usr/bin/env python3
"""
Reality Steering Demo

A comprehensive demonstration of quantum reality steering for consciousness states.
This script shows the key features of the quantum steering framework:
1. Creating and manipulating quantum consciousness states
2. Steering between wake, sleep, and other states
3. Measuring and observing state collapse
4. Exploring entanglement and non-local effects
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.quantum import (
    RealityRegister,
    SteeringProtocol,
    QuantumMeasurement,
    compute_entanglement_entropy,
    compute_mutual_information
)

print("="*80)
print("Quantum Reality Steering - Interactive Demo")
print("="*80)

# ==============================================================================
# Demo 1: Create Quantum Reality Register
# ==============================================================================

print("\n" + "="*80)
print("Demo 1: Creating a Quantum Reality Register")
print("="*80)

print("\nThe reality register models consciousness as a quantum state in")
print("the space of harmonic field modes...")

n_modes = 20
register = RealityRegister(n_modes=n_modes, seed=42)

print(f"\n✓ Created register with {n_modes} harmonic modes")
print(f"  Current state: {register.current_state.label}")
print(f"  State norm: {register.current_state.norm:.6f}")

# Show available basis states
print("\nAvailable consciousness basis states:")
basis_states = ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia', 'meditation', 'psychedelic']
for state_name in basis_states:
    state = register.get_basis_state(state_name)
    print(f"  • {state_name:15s} - {state.n_modes} modes")

# ==============================================================================
# Demo 2: Quantum Superposition
# ==============================================================================

print("\n" + "="*80)
print("Demo 2: Quantum Superposition of Consciousness States")
print("="*80)

print("\nCreating a superposition of wake and sleep states...")
print("This represents a consciousness state that is simultaneously")
print("wake-like and sleep-like (quantum indeterminacy).")

superpos = register.create_superposition(['wake', 'nrem_sleep'])
register.set_state(superpos)

print(f"\n✓ Created wake-sleep superposition")
print(f"  State norm: {superpos.norm:.6f}")

# Show decomposition
decomp = register.get_state_decomposition()
print("\nState decomposition (overlap with basis states):")
for state_name, overlap in sorted(decomp.items(), key=lambda x: -x[1])[:4]:
    print(f"  {state_name:15s}: {overlap:.3f}")

# ==============================================================================
# Demo 3: Quantum Steering
# ==============================================================================

print("\n" + "="*80)
print("Demo 3: Steering Between Consciousness States")
print("="*80)

print("\nInitializing to wake state and steering to sleep...")
register.set_state(register.get_basis_state('wake'))
protocol = SteeringProtocol(register)

print("\nSteering parameters:")
print(f"  Target state: NREM Sleep")
print(f"  Number of steps: 30")
print(f"  Steering strength: 1.0 (total)")

states = protocol.gradual_steering('nrem_sleep', n_steps=30, total_strength=1.0)

print(f"\n✓ Steering complete")
print(f"  Initial overlap with sleep: {states[0].overlap_probability(register.get_basis_state('nrem_sleep')):.3f}")
print(f"  Final overlap with sleep: {states[-1].overlap_probability(register.get_basis_state('nrem_sleep')):.3f}")

# ==============================================================================
# Demo 4: Quantum Measurement
# ==============================================================================

print("\n" + "="*80)
print("Demo 4: Quantum Measurement and Collapse")
print("="*80)

print("\nCreating superposition and performing measurement...")
superpos = register.create_superposition(['wake', 'nrem_sleep', 'rem_sleep'])
register.set_state(superpos)

measurement = QuantumMeasurement(register)

print("\nBefore measurement:")
decomp = register.get_state_decomposition()
for state_name in ['wake', 'nrem_sleep', 'rem_sleep']:
    print(f"  {state_name:15s}: {decomp[state_name]:.3f}")

# Perform measurement
measured_state, prob, collapsed = measurement.measure_consciousness_state(collapse=True)

print(f"\n✓ Measurement performed")
print(f"  Measured state: {measured_state}")
print(f"  Measurement probability: {prob:.3f}")

print("\nAfter measurement (collapsed):")
decomp = register.get_state_decomposition()
for state_name in ['wake', 'nrem_sleep', 'rem_sleep']:
    print(f"  {state_name:15s}: {decomp[state_name]:.3f}")

# ==============================================================================
# Demo 5: Local-to-Global Steering
# ==============================================================================

print("\n" + "="*80)
print("Demo 5: Local Operations Affecting Global State")
print("="*80)

print("\nDemonstrating how local operations on a few modes can")
print("affect the global consciousness state...")

register.set_state(register.get_basis_state('wake'))

# Apply local steering to first 5 modes
local_modes = np.arange(5)
print(f"\nApplying steering to local region: modes {local_modes[0]}-{local_modes[-1]}")
print(f"(Only {len(local_modes)} out of {n_modes} modes)")

initial_global_overlap = register.current_state.overlap_probability(
    register.get_basis_state('nrem_sleep')
)

for _ in range(20):
    protocol.steer_to_state('nrem_sleep', strength=0.05, local_modes=local_modes, 
                           update_register=True)

final_global_overlap = register.current_state.overlap_probability(
    register.get_basis_state('nrem_sleep')
)

print(f"\n✓ Local steering complete")
print(f"  Initial global overlap with sleep: {initial_global_overlap:.3f}")
print(f"  Final global overlap with sleep: {final_global_overlap:.3f}")
print(f"  Change: {final_global_overlap - initial_global_overlap:+.3f}")

# ==============================================================================
# Demo 6: Entanglement and Non-local Correlations
# ==============================================================================

print("\n" + "="*80)
print("Demo 6: Quantum Entanglement Between Brain Regions")
print("="*80)

print("\nComputing entanglement between different brain regions...")

# Define regions
n_regions = 4
region_size = n_modes // n_regions
regions = [np.arange(i * region_size, (i + 1) * region_size) for i in range(n_regions)]

print(f"\nDivided {n_modes} modes into {n_regions} regions")

# Set to wake state
register.set_state(register.get_basis_state('wake'))

print("\nEntanglement entropy for each region (wake state):")
for i, region in enumerate(regions):
    entropy = compute_entanglement_entropy(register.current_state, region)
    print(f"  Region {i}: {entropy:.3f}")

print("\nMutual information between regions:")
for i in range(n_regions):
    for j in range(i+1, n_regions):
        mi = compute_mutual_information(register.current_state, regions[i], regions[j])
        print(f"  Region {i} ↔ Region {j}: {mi:.3f}")

# ==============================================================================
# Demo 7: Visualization
# ==============================================================================

print("\n" + "="*80)
print("Demo 7: Visualizing Quantum State Evolution")
print("="*80)

print("\nGenerating visualization of steering trajectory...")

# Reset and steer
register.set_state(register.get_basis_state('wake'))
states = protocol.gradual_steering('nrem_sleep', n_steps=40, total_strength=1.0)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Quantum Reality Steering: Wake → Sleep Transition', 
             fontsize=14, fontweight='bold')

# Plot 1: Overlap evolution
steps = np.arange(len(states))
overlaps_wake = [s.overlap_probability(register.get_basis_state('wake')) for s in states]
overlaps_sleep = [s.overlap_probability(register.get_basis_state('nrem_sleep')) for s in states]

axes[0, 0].plot(steps, overlaps_wake, 'b-', linewidth=2, label='Wake')
axes[0, 0].plot(steps, overlaps_sleep, 'r-', linewidth=2, label='Sleep')
axes[0, 0].set_xlabel('Steering Step')
axes[0, 0].set_ylabel('Overlap Probability')
axes[0, 0].set_title('State Overlap Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Power distributions at key points
modes = np.arange(n_modes)
indices = [0, len(states)//2, len(states)-1]
colors = ['blue', 'purple', 'red']
labels = ['Initial (Wake)', 'Mid-transition', 'Final (Sleep)']

for idx, color, label in zip(indices, colors, labels):
    axes[0, 1].plot(modes, states[idx].power, color=color, linewidth=2, 
                    alpha=0.7, label=label)

axes[0, 1].set_xlabel('Harmonic Mode Index')
axes[0, 1].set_ylabel('Power')
axes[0, 1].set_title('Mode Power Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Phase evolution
phases_over_time = np.array([s.phases for s in states])

im = axes[1, 0].imshow(phases_over_time.T, aspect='auto', cmap='twilight', 
                       vmin=-np.pi, vmax=np.pi)
axes[1, 0].set_xlabel('Steering Step')
axes[1, 0].set_ylabel('Harmonic Mode')
axes[1, 0].set_title('Phase Evolution')
plt.colorbar(im, ax=axes[1, 0], label='Phase (rad)')

# Plot 4: Mode entropy evolution
entropies = []
for state in states:
    power = state.power / (state.power.sum() + 1e-12)
    p = power[power > 1e-12]
    entropy = -np.sum(p * np.log(p + 1e-12))
    entropies.append(entropy)

axes[1, 1].plot(steps, entropies, 'g-', linewidth=2)
axes[1, 1].set_xlabel('Steering Step')
axes[1, 1].set_ylabel('Mode Entropy')
axes[1, 1].set_title('Entropy Evolution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save or display
output_path = 'reality_steering_demo.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("Demo Complete!")
print("="*80)

print("\nKey Concepts Demonstrated:")
print("  1. Quantum consciousness states in harmonic mode space")
print("  2. Superposition of multiple consciousness types")
print("  3. Steering operations for state transitions")
print("  4. Quantum measurement and state collapse")
print("  5. Local operations affecting global state")
print("  6. Entanglement between brain regions")
print("  7. Visualization of quantum trajectories")

print("\nFor more details, see:")
print("  • experiments/category5_quantum_steering/")
print("  • papers/quantum/2512.14377_steering_realities.md")
print("  • tests/test_quantum_reality.py")

print("\n" + "="*80)
