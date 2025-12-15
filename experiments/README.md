# Experimental Framework for Harmonic Field Consciousness Model

This directory contains a comprehensive experimental framework for systematically testing the harmonic field model of consciousness across multiple dimensions.

## Overview

The framework is organized into four main categories of experiments, each exploring different aspects of the consciousness model:

1. **Network Topology** - How network structure affects consciousness metrics
2. **Dynamics** - Temporal evolution and state transitions
3. **Functional Modifications** - Optimizing and extending consciousness metrics
4. **Applications** - Real-world applications to neural networks, social systems, and clinical scenarios

## Directory Structure

```
experiments/
├── README.md                           # This file
├── requirements.txt                    # Additional dependencies
├── run_all.py                         # Master experiment runner
├── utils/                             # Core utilities
│   ├── __init__.py
│   ├── graph_generators.py            # Network topology generators
│   ├── metrics.py                     # Consciousness metrics calculations
│   ├── visualization.py               # Plotting utilities
│   └── state_generators.py            # Brain state power distributions
├── category1_network_topology/        # Network structure experiments
│   ├── README.md
│   ├── exp1_topology_comparison.py
│   ├── exp2_network_scaling.py
│   ├── exp3_hub_disruption.py
│   └── exp4_modular_networks.py
├── category2_dynamics/                # Dynamical systems experiments
│   ├── README.md
│   ├── exp1_state_transitions.py
│   ├── exp2_perturbation_recovery.py
│   ├── exp3_coupling_strength.py
│   └── exp4_criticality_tuning.py
├── category3_functional_modifications/ # Consciousness metric modifications
│   ├── README.md
│   ├── exp1_weighted_components.py
│   ├── exp2_new_metrics.py
│   ├── exp3_threshold_detection.py
│   └── exp4_component_correlation.py
├── category4_applications/            # Real-world applications
│   ├── README.md
│   ├── exp1_neural_networks.py
│   ├── exp2_social_networks.py
│   ├── exp3_psychedelic_states.py
│   └── exp4_anesthesia_depth.py
└── notebooks/                         # Interactive analysis
    ├── interactive_explorer.ipynb
    ├── results_analysis.ipynb
    └── visualization_gallery.ipynb
```

## Installation

Install the required dependencies:

```bash
cd experiments
pip install -r requirements.txt
```

## Quick Start

### Installation

First, install the required dependencies:

```bash
cd experiments
pip install -r requirements.txt
```

### Run All Experiments

Execute all experiments with a single command:

```bash
python run_all.py
```

This will discover and run all experiments with progress tracking.

### Run Specific Experiments

Filter by category:
```bash
python run_all.py --category category1_network_topology
```

Run a specific experiment:
```bash
python run_all.py --experiment exp1_topology_comparison
```

List all available experiments:
```bash
python run_all.py --list
```

### Run Individual Experiments

Each experiment can be run independently:

```bash
python category1_network_topology/exp1_topology_comparison.py
python category2_dynamics/exp1_state_transitions.py
python category3_functional_modifications/exp1_weighted_components.py
python category4_applications/exp3_psychedelic_states.py
```

### Interactive Exploration

Launch the interactive Jupyter notebooks:

```bash
jupyter notebook notebooks/interactive_explorer.ipynb
jupyter notebook notebooks/results_analysis.ipynb
```

### Verify Installation

Test that everything is working:

```bash
python -c "from utils import *; print('Framework operational!')"
```

## Core Utilities

### Graph Generators (`utils/graph_generators.py`)

Generate various network topologies:
- Small-world networks (Watts-Strogatz)
- Scale-free networks (Barabási-Albert)
- Random graphs (Erdős-Rényi)
- Lattice graphs (2D/3D grids)
- Modular networks with communities
- Hub-based networks

### Consciousness Metrics (`utils/metrics.py`)

Calculate key consciousness indicators:
- Mode entropy (H_mode)
- Participation ratio (PR)
- Phase coherence (R)
- Entropy production rate (Ṡ)
- Criticality index (κ)
- Combined consciousness functional C(t)
- Additional complexity metrics

### State Generators (`utils/state_generators.py`)

Generate realistic brain state power distributions:
- Wake state (broad distribution)
- NREM unconscious (low-mode concentration)
- NREM dreaming (mixed distribution)
- Anesthesia (extreme low-mode concentration)
- Psychedelic states (enhanced high-mode activity)

### Visualization (`utils/visualization.py`)

Comprehensive plotting utilities for all experimental results.

## Experiment Categories

### Category 1: Network Topology

Explore how network structure affects consciousness metrics:
- **exp1**: Compare different network architectures
- **exp2**: Test network scaling behavior
- **exp3**: Model lesions through hub disruption
- **exp4**: Analyze modular network properties

### Category 2: Dynamics

Study temporal evolution and state transitions:
- **exp1**: Animate transitions between conscious states
- **exp2**: Test resilience and recovery from perturbations
- **exp3**: Explore mode coupling effects
- **exp4**: Find optimal criticality regime

### Category 3: Functional Modifications

Optimize and extend consciousness metrics:
- **exp1**: Optimize component weightings
- **exp2**: Add novel complexity measures
- **exp3**: Find clinical consciousness thresholds
- **exp4**: Analyze metric correlations

### Category 4: Applications

Apply to real-world scenarios:
- **exp1**: Artificial neural networks
- **exp2**: Social network analysis
- **exp3**: Psychedelic state modeling
- **exp4**: Clinical anesthesia monitoring

## Results

All experiments save results in structured formats:
- JSON/CSV for tabular data
- PNG/PDF for visualizations
- HDF5 for large datasets

Results are saved in subdirectories within each category folder.

## Reproducibility

All experiments use fixed random seeds for reproducibility. Running the same experiment multiple times will produce identical results.

## Contributing

To add new experiments:
1. Follow the existing experiment structure
2. Use the utilities in `utils/` for consistency
3. Save results in a structured format
4. Add documentation to the category README

## Citation

If you use this experimental framework, please cite:

```
Smart, L. (2025).
A Harmonic Field Model of Consciousness in the Human Brain.
Vibrational Field Dynamics Project.
https://github.com/vfd-org/harmonic-field-consciousness
```

## License

MIT License - open for academic and scientific use.
