# Category 6: Multiscale Dynamics

Integration of multiscale temporal dynamics for consciousness modeling based on arXiv:2512.12462.

## Overview

This category implements multiscale encoding and real-time decoding for harmonic field consciousness analysis. The key innovation is handling different temporal resolutions simultaneously, enabling robust analysis even with missing data.

## Core Concepts

1. **Multiscale Encoder**: Processes harmonic modes at multiple temporal scales (1x, 2x, 4x, 8x downsampling)
2. **Real-time Decoder**: Streaming consciousness state prediction with <100ms latency
3. **Missing Data Handling**: Robust reconstruction from incomplete observations
4. **Nonlinear Dynamics**: Beyond linear oscillator coupling

## Experiments

### exp1_multiscale_encoder.py
- **Goal**: Compare multiscale vs single-scale encoding
- **Tests**: Wake, NREM, anesthesia state classification
- **Metrics**: Mode entropy, participation ratio, reconstruction accuracy

### exp2_realtime_decoding.py
- **Goal**: Real-time C(t) prediction from streaming data
- **Tests**: Latency benchmarks, prediction accuracy
- **Target**: <100ms latency with >80% accuracy

### exp3_missing_data_robustness.py
- **Goal**: Degradation analysis with incomplete data
- **Tests**: 10%, 20%, 30% missing harmonic amplitudes
- **Target**: <10% accuracy loss at 30% missing

### exp4_nonlinear_dynamics.py
- **Goal**: Nonlinear oscillator coupling
- **Tests**: Compare linear vs nonlinear dynamics
- **Target**: Better state transition modeling

## Running Experiments

```bash
# Run individual experiment
cd experiments/category6_multiscale_dynamics
python exp1_multiscale_encoder.py

# View results
ls results/exp1_multiscale_encoder/
```

## Key Features

- ✅ Multiple temporal scales for robust analysis
- ✅ Real-time streaming capability
- ✅ Graceful degradation with missing data
- ✅ Integration with existing consciousness metrics

## Success Criteria

- [ ] Multiscale encoding outperforms single-scale by 15%+
- [ ] Real-time latency <100ms
- [ ] <10% accuracy loss at 30% missing data
- [ ] Nonlinear dynamics improve state transition modeling

## References

- Paper: arXiv:2512.12462 "Dynamical modeling of nonlinear latent factors in multiscale neural activity"
- Documentation: `papers/neuroscience/2512.12462_multiscale.md`
- Core modules: `src/multiscale/`
