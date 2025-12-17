🚀 A Harmonic Field Model of Consciousness in the Human Brain
Code, Figures, and Reproducible Materials

This repository accompanies the paper:

Lee Smart (2025).
A Harmonic Field Model of Consciousness in the Human Brain.
Independent Researcher, Vibrational Field Dynamics Project.

The paper presents a unified, mathematically grounded account of consciousness based on connectome harmonics, oscillatory gating, and mode-wide integration, resolving the long-standing delta paradox and integrating recent advances in population coding and mixed selectivity (MillerLab, 2024–2025).

This repository contains the full reproducible workflow:

Python scripts for generating all five figures

Synthetic harmonic-mode simulations

Example “brain graph” Laplacians

Consciousness functional evaluation (H_mode, PR, R(t), Ṡ, κ)

LaTeX source for the full manuscript

Optional extensions for real-data examples (EEG/MEG using MNE)

🔬 Reproducibility

All figures in the paper can be reproduced by running:

python code/generate_fig1_modes.py
python code/generate_fig2_states.py
python code/generate_fig3_functional.py
python code/generate_fig4_delta_paradox.py
python code/generate_fig5_gating.py


The scripts require only standard Python scientific libraries:

NumPy

SciPy

Matplotlib

NetworkX

(Optional real-data examples require mne).

## 🧬 Neural Mass Models Integration

This repository now includes integration with neural mass models based on **"A Rosetta Stone of Neural Mass Models" (arXiv:2512.10982)**:

### Push-Pull Oscillator Framework
- **E-I Dynamics**: Excitatory-inhibitory population interactions generate brain oscillations
- **Multi-Scale Hierarchy**: Hierarchical oscillators at different temporal scales
- **Harmonic Bridge**: Conversion between neural mass dynamics and harmonic field modes
- **Consciousness Prediction**: Direct prediction of consciousness states from E-I oscillations

### Demo and Examples
```bash
python examples/neural_mass_demo.py
```

### Features
- Single push-pull oscillators with configurable E-I coupling
- Multi-scale hierarchical oscillators with cross-frequency coupling
- Spectral decomposition to harmonic modes
- Consciousness state classification (wake, sleep, anesthesia, etc.)
- Integrated harmonic-neural mass model

See `docs/neural_mass_integration.md` for detailed documentation.

## 🌌 Quantum Reality Steering

Integration with quantum consciousness framework based on **"Steering Alternative Realities through Local Quantum Memory Operations" (arXiv:2512.14377)**:

### Reality Register Framework
- **Quantum States**: Consciousness as superposition of harmonic modes
- **State Collapse**: Measurement-induced transitions between states
- **Steering Protocol**: Local operations affecting global consciousness
- **Entanglement**: Non-local correlations between brain regions

### Demo and Examples
```bash
python examples/reality_steering_demo.py
```

### Features
- Quantum consciousness state representation
- Superposition of multiple consciousness types
- Steering operations between states (wake ↔ sleep ↔ meditation, etc.)
- Measurement and collapse dynamics
- Entanglement analysis between brain regions
- Reality branching and landscape visualization

See `docs/reality_steering_theory.md` for detailed documentation.

## 🧪 Validation Experiments

Run comprehensive validation experiments:

```bash
# Validate neural mass model predictions
python experiments/validate_nmm_consciousness.py

# Analyze reality branching structure
python experiments/reality_branching_analysis.py
```

🧠 About the Paper

The model formalizes consciousness as a global field configuration across the connectome:

The connectome Laplacian eigenmodes (ψₖ) form the natural harmonic basis.

Dynamics follow modewise second-order oscillators with nonlinear coupling.

Consciousness corresponds to a state with:

high harmonic richness

high mode participation ratio

high phase coherence

positive entropy production

a stable but metastable criticality index

This approach resolves the Delta Paradox by showing that frequency bands do not determine conscious state —
global field configuration and oscillatory gating do.

The framework is substrate-agnostic (biological, artificial, hybrid) and geometry-agnostic (any Laplace-type operator).

📬 Contact

Author: Lee Smart
Independent Researcher
Vibrational Field Dynamics Project

Email:
📧 contact@vibrationalfielddynamics.org

Twitter/X:
🔗 @vfd_org

📄 Citation

If you use this work, please cite:

Smart, L. (2025).
A Harmonic Field Model of Consciousness in the Human Brain.
Vibrational Field Dynamics Project.
https://github.com/vfd-org/harmonic-field-consciousness


(Once an arXiv DOI is available, we can update this block.)

🎯 Goals of This Repository

Enable transparent reproducibility of all figures

Provide a clean scientific baseline for further extensions

Support researchers studying:

connectome harmonics

population coding

mixed selectivity

oscillatory gating

consciousness metrics

large-scale neural dynamics

neural mass models and E-I balance

quantum-inspired consciousness frameworks

reality steering and state transitions

Offer a foundation for future publications and more advanced models

## 📚 Repository Structure

```
src/
├── neural_mass/           # Neural mass models (NEW)
│   ├── push_pull_oscillator.py
│   ├── harmonic_bridge.py
│   └── __init__.py
├── quantum/               # Quantum reality steering
│   ├── reality_register.py
│   ├── steering_protocol.py
│   ├── quantum_measurement.py
│   └── entanglement.py
├── multiscale/            # Multi-scale dynamics
├── snn/                   # Spiking neural networks
├── transformers/          # Transformer architectures
└── memcomputing/          # Memory computing

examples/
├── neural_mass_demo.py           # Neural mass model demo (NEW)
└── reality_steering_demo.py      # Reality steering demo

experiments/
├── validate_nmm_consciousness.py       # NMM validation (NEW)
├── reality_branching_analysis.py       # Reality landscape (NEW)
└── category*/                          # Organized experiments

docs/
├── neural_mass_integration.md     # NMM theory & usage (NEW)
└── reality_steering_theory.md     # Quantum steering theory (NEW)

tests/
├── test_neural_mass.py       # NMM unit tests (NEW)
├── test_quantum_reality.py   # Quantum tests
└── test_*.py                 # Other module tests
```

## 🔬 Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific module tests:
```bash
python -m pytest tests/test_neural_mass.py -v
python -m pytest tests/test_quantum_reality.py -v
```

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Neural Mass Integration** (`docs/neural_mass_integration.md`): Theory, implementation, and usage of push-pull oscillators
- **Reality Steering Theory** (`docs/reality_steering_theory.md`): Quantum consciousness framework and steering protocols

## 🔗 Related Papers

1. **arXiv:2512.10982** - "A Rosetta Stone of Neural Mass Models"
   - Push-pull oscillator framework
   - E-I balance and brain rhythms
   - Cross-frequency coupling

2. **arXiv:2512.14377** - "Steering Alternative Realities through Local Quantum Memory Operations"
   - Quantum reality registers
   - Local-to-global steering
   - Consciousness state transitions

🔗 License

MIT License — open for academic and scientific use.

⭐ Final Note

This repository represents the “public scientific layer” of a larger ongoing research program exploring harmonic field dynamics and large-scale integrative neuroscience. Contributions, discussions, and collaborations are welcome.
