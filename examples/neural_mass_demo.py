#!/usr/bin/env python3
"""
Neural Mass Model Demo

Demonstrates the push-pull oscillator framework and its integration with
harmonic field theory for consciousness modeling.

This demo shows:
1. Single push-pull oscillator dynamics
2. Multi-scale hierarchical oscillators
3. Conversion to harmonic modes
4. Consciousness state prediction
5. Visualization of dynamics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.neural_mass import (
    PushPullOscillator,
    MultiScalePushPull,
    oscillation_to_harmonic_mode,
    HarmonicNeuralMassModel
)

print("="*80)
print("Neural Mass Model Demo - Push-Pull Oscillators")
print("="*80)

# ==============================================================================
# Demo 1: Single Push-Pull Oscillator
# ==============================================================================

print("\n" + "="*80)
print("Demo 1: Single Push-Pull Oscillator Dynamics")
print("="*80)

print("\nCreating a push-pull oscillator with E-I populations...")

oscillator = PushPullOscillator(
    tau_e=10.0,      # Excitatory time constant (ms)
    tau_i=5.0,       # Inhibitory time constant (ms)
    w_ee=1.5,        # E→E coupling
    w_ie=2.0,        # I→E coupling
    w_ei=2.5,        # E→I coupling
    w_ii=0.5,        # I→I coupling
    dt=0.1,
    seed=42
)

print("\n✓ Oscillator created with parameters:")
print(f"  τ_E = {oscillator.tau_e} ms")
print(f"  τ_I = {oscillator.tau_i} ms")
print(f"  w_EE = {oscillator.w_ee}, w_IE = {oscillator.w_ie}")
print(f"  w_EI = {oscillator.w_ei}, w_II = {oscillator.w_ii}")

# Simulate with constant input
duration = 1000  # ms
external_input = 0.5 * np.ones(int(duration / oscillator.dt))

print(f"\nSimulating for {duration} ms with constant input...")
result = oscillator.simulate(duration, external_input=external_input)

print(f"✓ Simulation complete")
print(f"  Final E activity: {result['e_activity'][-1]:.3f}")
print(f"  Final I activity: {result['i_activity'][-1]:.3f}")

# Compute oscillation frequency
freq = oscillator.compute_oscillation_frequency()
print(f"  Dominant frequency: {freq:.2f} Hz")

# ==============================================================================
# Demo 2: Multi-Scale Push-Pull Oscillators
# ==============================================================================

print("\n" + "="*80)
print("Demo 2: Multi-Scale Hierarchical Oscillators")
print("="*80)

print("\nCreating multi-scale oscillator hierarchy...")

multi_osc = MultiScalePushPull(
    n_scales=4,
    base_tau_e=8.0,
    base_tau_i=4.0,
    tau_scale_factor=2.5,
    cross_scale_coupling=0.3,
    dt=0.1,
    seed=42
)

print(f"\n✓ Created {multi_osc.n_scales} scales with increasing time constants")

# Simulate multi-scale dynamics
duration = 2000  # ms
print(f"\nSimulating multi-scale dynamics for {duration} ms...")

multi_result = multi_osc.simulate(duration)

print("✓ Multi-scale simulation complete")

# Get frequencies at each scale
scale_freqs = multi_osc.get_scale_frequencies()
print("\nDominant frequencies at each scale:")
for i, freq in enumerate(scale_freqs):
    print(f"  Scale {i}: {freq:.2f} Hz")

# Get cross-frequency coupling
coupling_matrix = multi_osc.get_cross_frequency_coupling()
print(f"\nCross-scale coupling matrix:")
print(f"  Shape: {coupling_matrix.shape}")
print(f"  Mean off-diagonal coupling: {np.mean(coupling_matrix[~np.eye(multi_osc.n_scales, dtype=bool)]):.3f}")

# ==============================================================================
# Demo 3: Conversion to Harmonic Modes
# ==============================================================================

print("\n" + "="*80)
print("Demo 3: Converting Oscillations to Harmonic Modes")
print("="*80)

print("\nConverting E-I dynamics to harmonic field representation...")

# Use the single oscillator result
harmonic_result = oscillation_to_harmonic_mode(
    result['e_activity'],
    dt=oscillator.dt,
    n_modes=20
)

print(f"\n✓ Converted to {len(harmonic_result['modes'])} harmonic modes")
print(f"  Mode frequencies range: {harmonic_result['frequencies'][0]:.2f} - {harmonic_result['frequencies'][-1]:.2f} Hz")

# Find most active modes
power = harmonic_result['modes'] ** 2
top_modes = np.argsort(power)[-5:][::-1]

print("\nTop 5 most active harmonic modes:")
for rank, mode_idx in enumerate(top_modes):
    freq = harmonic_result['frequencies'][mode_idx]
    amplitude = harmonic_result['modes'][mode_idx]
    print(f"  {rank+1}. Mode {mode_idx}: {freq:.2f} Hz (amplitude: {amplitude:.3f})")

# ==============================================================================
# Demo 4: Integrated Harmonic-Neural Mass Model
# ==============================================================================

print("\n" + "="*80)
print("Demo 4: Integrated Harmonic-Neural Mass Model")
print("="*80)

print("\nCreating integrated model...")

integrated_model = HarmonicNeuralMassModel(
    n_modes=20,
    n_scales=3,
    dt=0.1,
    seed=42
)

print("✓ Integrated model initialized")

# Simulate and convert
duration = 1500  # ms
print(f"\nSimulating for {duration} ms and converting to harmonic modes...")

integrated_result = integrated_model.simulate_and_convert(duration)

print("✓ Simulation and conversion complete")

# Predict consciousness state
print("\nPredicting consciousness state from harmonic modes...")
consciousness_metrics = integrated_model.predict_consciousness_state()

print("\nConsciousness metrics:")
print(f"  Harmonic richness: {consciousness_metrics['harmonic_richness']:.3f}")
print(f"  Participation ratio: {consciousness_metrics['participation_ratio']:.3f}")
print(f"  Dominant frequency: {consciousness_metrics['dominant_frequency']:.2f} Hz")
print(f"  Consciousness score: {consciousness_metrics['consciousness_score']:.3f}")

# Classify state
state_label = integrated_model.classify_consciousness_state()
print(f"\n✓ Predicted consciousness state: {state_label.upper()}")

# ==============================================================================
# Demo 5: Consciousness State Trajectory
# ==============================================================================

print("\n" + "="*80)
print("Demo 5: Consciousness State Trajectory")
print("="*80)

print("\nComputing trajectory through consciousness state space...")

integrated_model.reset()
trajectory = integrated_model.compute_consciousness_trajectory(
    duration=3000,
    n_samples=15
)

print(f"\n✓ Computed trajectory with {len(trajectory['state_labels'])} time points")

print("\nState evolution:")
for i, (time, state, score) in enumerate(zip(
    trajectory['time_points'],
    trajectory['state_labels'],
    trajectory['consciousness_score']
)):
    print(f"  t={time:6.0f}ms: {state:12s} (score: {score:.3f})")

# ==============================================================================
# Demo 6: Visualization
# ==============================================================================

print("\n" + "="*80)
print("Demo 6: Visualizing Neural Mass Dynamics")
print("="*80)

print("\nGenerating comprehensive visualization...")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Single oscillator E-I dynamics
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(result['time'], result['e_activity'], 'b-', linewidth=1.5, label='Excitatory')
ax1.plot(result['time'], result['i_activity'], 'r-', linewidth=1.5, label='Inhibitory')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Activity')
ax1.set_title('Single Oscillator E-I Dynamics')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Phase space
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(result['e_activity'], result['i_activity'], 'k-', linewidth=0.5, alpha=0.6)
ax2.plot(result['e_activity'][0], result['i_activity'][0], 'go', markersize=8, label='Start')
ax2.plot(result['e_activity'][-1], result['i_activity'][-1], 'ro', markersize=8, label='End')
ax2.set_xlabel('E Activity')
ax2.set_ylabel('I Activity')
ax2.set_title('Phase Space')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Power spectrum
ax3 = fig.add_subplot(gs[0, 2])
ax3.semilogy(harmonic_result['freq_axis'], harmonic_result['power_spectrum'], 'b-', linewidth=1.5)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power')
ax3.set_title('Power Spectrum')
ax3.set_xlim([0, 50])
ax3.grid(True, alpha=0.3)

# Plot 4: Multi-scale dynamics
ax4 = fig.add_subplot(gs[1, :])
for scale in range(multi_osc.n_scales):
    offset = scale * 0.5
    signal = multi_result[f'scale_{scale}_e'] + offset
    ax4.plot(multi_result['time'], signal, linewidth=1.5, label=f'Scale {scale}')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Activity (offset)')
ax4.set_title('Multi-Scale Oscillator Hierarchy')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Harmonic modes
ax5 = fig.add_subplot(gs[2, 0])
mode_indices = np.arange(len(harmonic_result['modes']))
ax5.bar(mode_indices, harmonic_result['modes'], color='steelblue', alpha=0.7)
ax5.set_xlabel('Mode Index')
ax5.set_ylabel('Amplitude')
ax5.set_title('Harmonic Mode Amplitudes')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Consciousness trajectory
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(trajectory['time_points'], trajectory['harmonic_richness'], 'b-', 
         linewidth=2, label='Richness', marker='o')
ax6.plot(trajectory['time_points'], trajectory['consciousness_score'], 'r-', 
         linewidth=2, label='Consciousness', marker='s')
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Metric Value')
ax6.set_title('Consciousness Metrics Over Time')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Cross-scale coupling
ax7 = fig.add_subplot(gs[2, 2])
im = ax7.imshow(coupling_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax7.set_xlabel('Scale')
ax7.set_ylabel('Scale')
ax7.set_title('Cross-Scale Coupling Matrix')
plt.colorbar(im, ax=ax7, label='Correlation')

plt.suptitle('Neural Mass Model: Push-Pull Oscillators and Harmonic Field Theory', 
             fontsize=14, fontweight='bold')

# Save figure
output_path = 'neural_mass_demo.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("Demo Complete!")
print("="*80)

print("\nKey Concepts Demonstrated:")
print("  1. Push-pull E-I oscillator dynamics")
print("  2. Multi-scale hierarchical oscillations")
print("  3. Spectral decomposition into harmonic modes")
print("  4. Consciousness state prediction from neural dynamics")
print("  5. Integration of microscopic and macroscopic theories")
print("  6. Cross-scale coupling and frequency interactions")

print("\nTheoretical Foundations:")
print("  • arXiv:2512.10982 - Rosetta Stone of Neural Mass Models")
print("  • Push-pull oscillators generate brain rhythms from E-I balance")
print("  • Harmonic modes emerge from collective oscillatory patterns")
print("  • Multi-scale dynamics explain cross-frequency coupling")

print("\nFor more details, see:")
print("  • docs/neural_mass_integration.md")
print("  • experiments/validate_nmm_consciousness.py")
print("  • tests/test_neural_mass.py")

print("\n" + "="*80)
