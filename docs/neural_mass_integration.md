# Neural Mass Model Integration

## Overview

This document describes the integration of neural mass models (NMM) with the harmonic field consciousness framework, based on insights from **"A Rosetta Stone of Neural Mass Models" (arXiv:2512.10982)**.

## Theoretical Foundation

### Push-Pull Oscillator Framework

The push-pull oscillator model provides a unified framework for understanding brain oscillations through excitatory-inhibitory (E-I) population dynamics.

#### Key Concepts

1. **Excitatory Population (E)**: Provides "push" drive, increasing network activity
2. **Inhibitory Population (I)**: Provides "pull" feedback, modulating and regulating activity
3. **Coupling Dynamics**: Interactions between E and I populations generate oscillations

#### Mathematical Formulation

The dynamics are governed by coupled differential equations:

```
τ_E dE/dt = -E + S(w_EE*E - w_IE*I + I_ext)
τ_I dI/dt = -I + S(w_EI*E - w_II*I)
```

Where:
- `E`, `I`: Excitatory and inhibitory population activities
- `τ_E`, `τ_I`: Time constants (ms)
- `w_XY`: Synaptic weight from population Y to population X
- `S(x)`: Sigmoid activation function
- `I_ext`: External input current

#### Parameter Ranges

Physiologically realistic parameter ranges:
- Time constants: τ_E ≈ 5-20 ms, τ_I ≈ 2-10 ms
- Coupling weights: w_EE ≈ 1-2, w_IE ≈ 1.5-3, w_EI ≈ 2-4, w_II ≈ 0.2-1
- These produce oscillations in the 0.5-100 Hz range (delta through gamma)

### Multi-Scale Hierarchical Organization

Brain oscillations exist at multiple spatial and temporal scales:

1. **Local circuits** (microseconds to milliseconds, sub-mm scale)
2. **Cortical columns** (milliseconds to tens of ms, mm scale)
3. **Brain regions** (tens to hundreds of ms, cm scale)
4. **Whole brain** (hundreds of ms to seconds, global scale)

The `MultiScalePushPull` class implements this hierarchy with:
- Scale-dependent time constants (larger scales = slower dynamics)
- Cross-scale coupling (bottom-up drive, top-down modulation)
- Emergent cross-frequency coupling between scales

## Connection to Harmonic Field Theory

### Spectral Decomposition

The `oscillation_to_harmonic_mode()` function converts E-I oscillations to harmonic field modes:

1. **FFT Analysis**: Compute frequency spectrum of neural activity
2. **Mode Extraction**: Divide spectrum into frequency bands, each corresponding to a harmonic mode
3. **Amplitude & Phase**: Extract mode amplitudes and phases from spectral power and phase

### Harmonic Mode Interpretation

| Frequency Band | Harmonic Modes | Consciousness Role |
|---------------|----------------|-------------------|
| Delta (0.5-4 Hz) | Low modes (k=0-2) | Sleep, unconsciousness |
| Theta (4-8 Hz) | Low-mid modes (k=3-5) | Memory, drowsiness |
| Alpha (8-13 Hz) | Mid modes (k=6-9) | Relaxed wakefulness |
| Beta (13-30 Hz) | Mid-high modes (k=10-15) | Active thinking |
| Gamma (30-100 Hz) | High modes (k=16-20) | Binding, awareness |

### Consciousness Metrics

The integrated model computes consciousness metrics from harmonic modes:

#### 1. Harmonic Richness (H)
```python
H = -Σ p_k log(p_k) / log(N)
```
- Measures diversity of active modes
- Range: [0, 1]
- Higher values → more conscious states

#### 2. Participation Ratio (PR)
```python
PR = (Σ |a_k|²)² / Σ |a_k|⁴
```
- Quantifies number of contributing modes
- Range: [1, N]
- Higher values → distributed, wake-like activity

#### 3. Consciousness Score
```python
Score = 0.6 * H + 0.4 * (PR / N)
```
- Combined metric for state classification
- Range: [0, 1]
- Thresholds:
  - > 0.7: Wake
  - 0.5-0.7: REM sleep
  - 0.3-0.5: NREM sleep
  - < 0.3: Anesthesia

## Biological Relevance

### E-I Balance and Consciousness

The push-pull oscillator model captures key aspects of consciousness neuroscience:

1. **Wake State**: Balanced E-I with diverse frequency content
   - High harmonic richness
   - Distributed mode participation
   - Active gamma and beta rhythms

2. **Sleep States**:
   - **NREM**: Dominant slow oscillations (delta), low-mode concentration
   - **REM**: Mixed frequencies, partial mode distribution
   
3. **Anesthesia**: Extreme E-I imbalance
   - Ultra-low modes dominate
   - Minimal harmonic richness
   - Loss of high-frequency content

### Cross-Frequency Coupling

Multi-scale oscillators naturally produce cross-frequency coupling:
- **Phase-amplitude coupling**: Low-frequency phase modulates high-frequency amplitude
- **Phase-phase coupling**: Synchronization across frequencies
- **Amplitude-amplitude coupling**: Co-modulation of different bands

These phenomena are hallmarks of conscious information integration.

## Implementation Details

### Class Hierarchy

```
PushPullOscillator
├── Single E-I oscillator
├── Configurable parameters
└── Time evolution via ODE integration

MultiScalePushPull
├── Hierarchy of PushPullOscillators
├── Cross-scale coupling
└── Frequency analysis per scale

HarmonicNeuralMassModel
├── Integrates MultiScalePushPull
├── Spectral conversion to modes
├── Consciousness prediction
└── State classification
```

### Usage Example

```python
from src.neural_mass import HarmonicNeuralMassModel

# Create integrated model
model = HarmonicNeuralMassModel(
    n_modes=20,
    n_scales=3,
    dt=0.1
)

# Simulate and convert to harmonic modes
result = model.simulate_and_convert(duration=1000)

# Predict consciousness state
metrics = model.predict_consciousness_state()
state = model.classify_consciousness_state()

print(f"State: {state}")
print(f"Consciousness score: {metrics['consciousness_score']:.3f}")
print(f"Harmonic richness: {metrics['harmonic_richness']:.3f}")
```

## Validation and Testing

### Unit Tests
- 22 comprehensive tests in `tests/test_neural_mass.py`
- Coverage includes:
  - Oscillator dynamics
  - Multi-scale coupling
  - Harmonic conversion
  - Consciousness prediction
  - End-to-end workflows

### Physiological Validation
- Generated frequencies match known brain rhythms
- E-I balance produces stable oscillations
- Multi-scale coupling exhibits realistic cross-frequency interactions
- Consciousness metrics correlate with expected state properties

## Future Directions

1. **Coupling to Real Data**: Integration with EEG/MEG/fMRI measurements
2. **Spatial Extension**: Incorporate spatial connectivity (connectome)
3. **Learning Dynamics**: Plasticity and adaptation in E-I weights
4. **Detailed Biophysics**: Ion channels, synaptic dynamics, neurotransmitters
5. **Clinical Applications**: Anesthesia depth monitoring, sleep disorders, disorders of consciousness

## References

1. **arXiv:2512.10982** - "A Rosetta Stone of Neural Mass Models"
   - Unified framework for brain oscillations
   - Push-pull E-I dynamics
   - Cross-frequency coupling mechanisms

2. Smart, L. (2025). "A Harmonic Field Model of Consciousness in the Human Brain"
   - Connectome harmonics
   - Mode participation and richness
   - Consciousness functional metrics

3. Related neural mass model literature:
   - Wilson-Cowan models
   - Jansen-Rit models
   - Neural field theory
   - Mean field approximations

## Code Structure

```
src/neural_mass/
├── __init__.py                  # Module exports
├── push_pull_oscillator.py      # E-I oscillator classes
└── harmonic_bridge.py           # Conversion and integration

examples/
└── neural_mass_demo.py          # Comprehensive demonstration

tests/
└── test_neural_mass.py          # Unit tests

experiments/
└── validate_nmm_consciousness.py # Validation experiments
```

## Mathematical Appendix

### Sigmoid Activation Function

```python
S(x) = 1 / (1 + exp(-β(x - θ)))
```
- `β`: Slope parameter (controls steepness)
- `θ`: Threshold parameter (controls offset)

### Frequency of Oscillation (Approximate)

For simple E-I oscillator:
```
f ≈ 1 / (2π√(τ_E * τ_I))
```

This provides rough estimate; actual frequency depends on coupling weights and nonlinearities.

### Spectral Power Distribution

Given time series x(t), power spectrum:
```
P(f) = |FFT{x(t)}|²
```

Harmonic mode k amplitude:
```
a_k = √(∫_{f_k - Δf/2}^{f_k + Δf/2} P(f) df)
```

Where f_k is center frequency of mode k, Δf is bandwidth.
