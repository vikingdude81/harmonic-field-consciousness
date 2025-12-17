# Category 7: Spatiotemporal Transformers

Integration of BaRISTA (Brain Scale Informed Spatiotemporal Representation) transformer architecture based on arXiv:2512.12135.

## Overview

This category implements transformer-based consciousness analysis using region-level encoding and attention mechanisms. The approach reveals which brain regions are important for different consciousness states.

## Core Concepts

1. **Region-Level Encoding**: Brain modules as tokens (not individual nodes)
2. **Multi-Head Attention**: Different aspects of region interactions
3. **Masked Reconstruction**: Self-supervised learning strategy
4. **Flexible Spatial Scales**: From nodes to communities to hemispheres

## Experiments

### exp1_region_level_encoding.py
- **Goal**: Compare region-level vs node-level encoding
- **Tests**: Modular networks with 2, 4, 6, 8 communities
- **Metrics**: Classification accuracy, computational efficiency

### exp2_masked_reconstruction.py
- **Goal**: Self-supervised learning via masking
- **Tests**: Mask 15% of regions, reconstruct activity
- **Target**: 10%+ improvement from pre-training

### exp3_transformer_consciousness.py
- **Goal**: Full transformer for C(t) prediction
- **Tests**: State classification, attention analysis
- **Target**: >85% accuracy, interpretable attention

### exp4_spatial_scale_analysis.py
- **Goal**: Optimal spatial granularity
- **Tests**: Nodes → communities → hemispheres
- **Target**: Find best scale for each metric

## Running Experiments

```bash
# Run individual experiment
cd experiments/category7_spatiotemporal_transformers
python exp1_region_level_encoding.py

# View results
ls results/exp1_region_level_encoding/
```

## Key Features

- ✅ Biologically meaningful region-level analysis
- ✅ Attention weights show important connections
- ✅ Self-supervised pre-training
- ✅ Hierarchical spatial analysis

## Success Criteria

- [ ] Region-level outperforms node-level encoding
- [ ] Transformer achieves >85% classification accuracy
- [ ] Attention highlights interpretable brain regions
- [ ] Optimal spatial scale identified per metric

## References

- Paper: arXiv:2512.12135 "Brain Scale Informed Spatiotemporal Representation"
- Documentation: `papers/neuroscience/2512.12135_barista.md`
- Core modules: `src/transformers/`
