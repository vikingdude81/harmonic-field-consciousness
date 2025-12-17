# Category 5: Quantum Reality Steering

This directory contains experiments implementing quantum-inspired formalism for consciousness state transitions, based on the paper "Steering Alternative Realities through Local Quantum Memory Operations" (arXiv:2512.14377) by Xiongfeng Ma.

## Overview

The experiments in this category demonstrate how quantum reality steering concepts can be integrated with the harmonic field consciousness model. Key innovations include:

1. **Quantum Reality Register**: Models consciousness states as quantum superpositions of harmonic field modes
2. **Reality Steering Protocol**: Implements probabilistic transitions between wake, sleep, and anesthesia states
3. **Local-to-Global Effects**: Shows how local operations on brain regions affect global conscious state
4. **Quantum Measurement**: Models observation and state collapse in consciousness dynamics

## Theoretical Foundation

The quantum reality steering framework treats consciousness states as:
- **Quantum state vectors** in the space of harmonic modes
- **Basis states** corresponding to different consciousness types (wake, sleep, anesthesia, etc.)
- **Superposition states** representing mixed or transitional consciousness
- **Entanglement** modeling non-local correlations between brain regions

### Connection to Existing Framework

The quantum steering approach integrates seamlessly with the existing consciousness metrics:
- **H_mode (Mode Entropy)**: Quantum entropy of the consciousness state
- **PR (Participation Ratio)**: Effective dimension of the quantum state
- **R(t) (Phase Coherence)**: Quantum coherence between modes
- **Ṡ (Entropy Production)**: Rate of quantum state evolution
- **κ (Criticality)**: Balance between quantum order and disorder

## Experiments

### Experiment 1: Steering Consciousness States
**File**: `exp1_steering_consciousness_states.py`

Demonstrates quantum steering between different consciousness states:
- Wake → NREM Sleep transitions
- Sleep → Wake transitions  
- Anesthesia induction/emergence
- Meditation state transitions
- Psychedelic state modeling

**Key Results**:
- Visualization of state trajectories in quantum Hilbert space
- Evolution of consciousness metrics during steering
- Measurement of transition probabilities
- Quantum state overlap analysis

### Experiment 2: Local-to-Global Steering
**File**: `exp2_local_to_global_steering.py`

Shows how local operations on small brain regions can steer global consciousness:
- Progressive expansion of local steering regions
- Emergence of global state change from local perturbations
- Non-local correlation effects
- Critical region size for global steering

**Key Results**:
- Local vs. global steering effectiveness
- Entanglement growth during steering
- Minimal local region for consciousness state change
- Regional influence on global metrics

### Experiment 3: Quantum Memory and Harmonics
**File**: `exp3_quantum_memory_harmonics.py`

Explores the quantum register as a memory system for consciousness:
- Quantum superposition of multiple states
- Harmonic modes as quantum basis states
- State history and memory dynamics
- Interference patterns between consciousness states

**Key Results**:
- Quantum memory capacity analysis
- Harmonic mode decomposition
- Superposition stability over time
- Basis state overlap patterns

### Experiment 4: Measurement and Collapse
**File**: `exp4_measurement_collapse.py`

Studies quantum measurement effects on consciousness:
- Projective measurements in different bases
- State collapse dynamics
- Weak measurements and continuous monitoring
- Measurement back-action on consciousness

**Key Results**:
- Measurement-induced state changes
- Collapse time scales
- Weak measurement trajectories
- Observer effects on consciousness metrics

## Integration with Existing Framework

### Compatibility

All quantum steering experiments are compatible with:
- Existing graph generators (`utils/graph_generators.py`)
- Consciousness metrics (`utils/metrics.py`)
- State generators (`utils/state_generators.py`)
- Visualization tools (`utils/visualization.py`)

### Extended Metrics

New quantum-specific metrics introduced:
- **Quantum state overlap**: Transition probabilities between states
- **Entanglement entropy**: Non-local correlations
- **Quantum coherence**: Superposition stability
- **Measurement probabilities**: Observable outcomes

## Usage

### Run All Quantum Steering Experiments

```bash
cd experiments/category5_quantum_steering
python exp1_steering_consciousness_states.py
python exp2_local_to_global_steering.py
python exp3_quantum_memory_harmonics.py
python exp4_measurement_collapse.py
```

### Run via Master Script

```bash
cd experiments
python run_all.py --category category5_quantum_steering
```

### Interactive Exploration

```python
import sys
sys.path.insert(0, '../../src')

from quantum import RealityRegister, SteeringProtocol

# Create quantum reality register
register = RealityRegister(n_modes=20, seed=42)

# Set initial state
register.set_state(register.get_basis_state('wake'))

# Create steering protocol
protocol = SteeringProtocol(register)

# Steer to sleep state
states = protocol.gradual_steering('nrem_sleep', n_steps=50)

# Analyze transition
print(f"Final state: {register.measure_consciousness_type()}")
```

## Mathematical Framework

### Quantum State Representation

Consciousness state as quantum vector:
```
|ψ⟩ = Σₖ cₖ|φₖ⟩
```
where:
- |φₖ⟩ are harmonic mode basis states
- cₖ are complex amplitudes (satisfying Σₖ|cₖ|² = 1)

### Steering Operator

Transition between states via unitary evolution:
```
U_steering = exp(-i H_steering Δt)
```
where H_steering guides current state toward target.

### Measurement

Projective measurement in consciousness basis:
```
P(state) = |⟨state|ψ⟩|²
```

### Entanglement

Non-local correlations via entropy:
```
S_ent = -Tr(ρ_A log ρ_A)
```
where ρ_A is reduced density matrix of region A.

## Results Directory

Experimental results are saved to `results/` subdirectory:
- **Plots**: PNG/PDF visualizations
- **Data**: JSON/CSV data files
- **Animations**: GIF files of state evolution
- **Analysis**: Summary statistics

## Dependencies

Core dependencies:
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `matplotlib` - Visualization
- `networkx` - Graph structures (for brain connectivity)

Optional:
- `pandas` - Data analysis
- `seaborn` - Enhanced visualization

## References

1. **Xiongfeng Ma (2024)**. "Steering Alternative Realities through Local Quantum Memory Operations". arXiv:2512.14377

2. **Smart, L. (2025)**. "A Harmonic Field Model of Consciousness in the Human Brain". Vibrational Field Dynamics Project.

3. **Integration Document**: See `papers/quantum/2512.14377_steering_realities.md` for detailed paper summary and integration notes.

## Citation

If you use these quantum steering experiments, please cite:

```bibtex
@article{smart2025harmonic,
  title={A Harmonic Field Model of Consciousness in the Human Brain},
  author={Smart, Lee},
  year={2025},
  note={Quantum Reality Steering Extension}
}

@article{ma2024steering,
  title={Steering Alternative Realities through Local Quantum Memory Operations},
  author={Ma, Xiongfeng},
  journal={arXiv preprint arXiv:2512.14377},
  year={2024}
}
```

## Contributing

To add new quantum steering experiments:
1. Follow the existing experiment structure
2. Use the quantum modules from `src/quantum/`
3. Integrate with existing consciousness metrics
4. Add visualization and results saving
5. Update this README with experiment description

## License

MIT License - open for academic and scientific use.
