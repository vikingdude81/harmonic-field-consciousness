# Category 8: Spiking Consciousness

Integration of spiking neural networks for biologically realistic consciousness modeling based on arXiv:2512.11743.

## Overview

This category implements consciousness analysis using spiking neurons (LIF models) with dynamic network growth and pathway reusability. Enables neuromorphic hardware deployment.

## Core Concepts

1. **LIF Neurons**: Leaky integrate-and-fire spiking neurons
2. **Spike Encoding**: Convert harmonics to/from spike trains
3. **Dynamic Growth**: Network expands during learning
4. **Pathway Reuse**: Efficient learning without forgetting
5. **Neuromorphic Hardware**: Deploy to SpiNNaker/Loihi

## Experiments

### exp1_snn_harmonic_oscillators.py
- **Goal**: Replace classical oscillators with SNNs
- **Tests**: Compute all consciousness metrics from spikes
- **Target**: Match classical metrics within 10%

### exp2_pathway_reuse_transitions.py
- **Goal**: Efficient state transitions via pathway reuse
- **Tests**: Learn wake ↔ sleep, apply to other transitions
- **Target**: 50%+ faster learning

### exp3_dynamic_network_growth.py
- **Goal**: Consciousness emergence during growth
- **Tests**: Start with 100 neurons, grow to 500
- **Target**: 20%+ performance improvement

### exp4_neuromorphic_deployment.py
- **Goal**: Hardware export and benchmarking
- **Tests**: Convert to SpiNNaker/Loihi format
- **Target**: <10W power, <10ms latency

## Running Experiments

```bash
# Run individual experiment
cd experiments/category8_spiking_consciousness
python exp1_snn_harmonic_oscillators.py

# View results
ls results/exp1_snn_harmonic_oscillators/
```

## Key Features

- ✅ Biologically realistic spiking dynamics
- ✅ Event-driven computation
- ✅ Dynamic network growth
- ✅ Neuromorphic hardware compatible

## Success Criteria

- [ ] SNN metrics match classical within 10%
- [ ] Dynamic growth improves performance by 20%+
- [ ] Pathway reuse accelerates learning by 50%+
- [ ] Neuromorphic deployment <10W, <10ms

## References

- Paper: arXiv:2512.11743 "Spiking Neural Networks with Random Graph Architectures"
- Documentation: `papers/neuroscience/2512.11743_cognisnn.md`
- Core modules: `src/snn/`
