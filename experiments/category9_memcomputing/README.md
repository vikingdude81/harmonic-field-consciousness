# Category 9: Memcomputing

Integration of meminductor-based RLC circuits for memory-encoded consciousness dynamics based on arXiv:2512.11002.

## Overview

This category implements consciousness modeling using meminductors (inductors with memory) and RLC oscillators. Enables physical circuit implementation and anticipatory behavior.

## Core Concepts

1. **Meminductors**: Inductors with memory-dependent inductance
2. **RLC Oscillators**: Resonance and richer dynamics than RC
3. **Memory Encoding**: Store amplitudes in magnetic flux
4. **Anticipation**: Predict future states via LC resonance
5. **Physical Implementation**: Design actual circuits

## Experiments

### exp1_rlc_oscillators.py
- **Goal**: Replace RC with RLC oscillators
- **Tests**: Compute consciousness metrics from RLC dynamics
- **Target**: Valid metrics, richer dynamics

### exp2_memory_encoding.py
- **Goal**: Store/retrieve harmonic amplitudes
- **Tests**: Encode in meminductors, track persistence
- **Target**: >100 timesteps with <20% degradation

### exp3_timing_anticipation.py
- **Goal**: Predict future consciousness states
- **Tests**: LC resonance for anticipation
- **Target**: Better-than-baseline up to 50 steps ahead

### exp4_physical_consciousness.py
- **Goal**: Design physical circuit
- **Tests**: SPICE simulation of consciousness circuit
- **Target**: Scientifically plausible design

## Running Experiments

```bash
# Run individual experiment
cd experiments/category9_memcomputing
python exp1_rlc_oscillators.py

# View results
ls results/exp1_rlc_oscillators/
```

## Key Features

- ✅ Memory-dependent inductance
- ✅ Resonance and oscillations
- ✅ Biological timing mechanisms
- ✅ Physical hardware substrate

## Success Criteria

- [ ] RLC oscillators produce valid metrics
- [ ] Memory encoding >100 timesteps
- [ ] Anticipation better than baseline
- [ ] Physical circuit design is plausible

## References

- Paper: arXiv:2512.11002 "Beyond Memristor: Neuromorphic Computing Using Meminductor"
- Documentation: `papers/neuroscience/2512.11002_meminductors.md`
- Core modules: `src/memcomputing/`
