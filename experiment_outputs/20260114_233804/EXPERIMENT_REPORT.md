# Consciousness Circuit Experiment Report

**Model**: Qwen/Qwen2.5-0.5B-Instruct
**Date**: 2026-01-14 23:38:16
**Output Directory**: experiment_outputs\20260114_233804

## Summary

| Experiment | Status | Duration |
|------------|--------|----------|
| Layerwise Analysis | ✓ | 1.6s |
| Consciousness Patching | ✓ | 0.6s |
| Thought Trajectory | ✓ | 1.7s |
| SAE Collection | ✓ | 1.4s |
| Steering Vectors | ✓ | 0.7s |

## Detailed Results

### Layerwise Analysis

```json
{
  "n_layers": 7,
  "peak_layer": 4
}
```

### Consciousness Patching

```json
{
  "n_layers_tested": 7,
  "most_impactful_layer": 24,
  "max_delta": 0.13555664062500017
}
```

### Thought Trajectory

```json
{
  "n_steps": 20,
  "final_score": 0.504748046875,
  "tokens": [
    " First",
    ",",
    " I",
    " need",
    " to",
    " understand",
    " what",
    " consciousness",
    " is",
    "."
  ],
  "generated_text": "Let me think about consciousness step by step. First, I need to understand what consciousness is. Co"
}
```

### SAE Collection

```json
{
  "n_samples": 60,
  "layers": [
    22,
    23,
    24
  ],
  "file": "experiment_outputs\\20260114_233804\\sae_activations.npz"
}
```

### Steering Vectors

```json
{
  "n_vectors": 3,
  "layers": [
    20,
    22,
    24
  ],
  "norms": {
    "20": 16.863527297973633,
    "22": 23.080663681030273,
    "24": 126.87596893310547
  },
  "file": "experiment_outputs\\20260114_233804\\steering_vectors.npz"
}
```


## Generated Files

- `layerwise_*.png` - Consciousness across layers
- `patching_impact.png` - Layer patching impact
- `trace_trajectory.png` - Token-by-token consciousness
- `sae_activations.npz` - Residual activations for SAE
- `steering_vectors.npz` - Consciousness steering vectors

## Next Steps

1. **Analyze layerwise plots** to find where consciousness emerges
2. **Review patching results** to identify causal decision layers
3. **Train SAE** on collected activations to discover new features
4. **Apply steering** with different alpha values to modulate consciousness
