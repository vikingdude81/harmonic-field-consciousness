# Consciousness Circuit Experiment Report

**Model**: unsloth/Qwen2.5-32B-Instruct-bnb-4bit
**Date**: 2026-01-15 00:06:32
**Output Directory**: experiment_outputs\20260115_000545

## Summary

| Experiment | Status | Duration |
|------------|--------|----------|
| Layerwise Analysis | ✓ | 1.3s |
| Consciousness Patching | ✓ | 0.6s |
| Thought Trajectory | ✓ | 5.4s |
| SAE Collection | ✓ | 5.1s |
| Steering Vectors | ✓ | 2.3s |

## Detailed Results

### Layerwise Analysis

```json
{
  "n_layers": 17,
  "peak_layer": 16
}
```

### Consciousness Patching

```json
{
  "n_layers_tested": 17,
  "most_impactful_layer": 0,
  "max_delta": 0.0
}
```

### Thought Trajectory

```json
{
  "n_steps": 20,
  "final_score": 0.0,
  "tokens": [
    " What",
    " is",
    " the",
    " first",
    " thing",
    " I",
    " should",
    " consider",
    " when",
    " thinking"
  ],
  "generated_text": "Let me think about consciousness step by step. What is the first thing I should consider when thinki"
}
```

### SAE Collection

```json
{
  "n_samples": 60,
  "layers": [
    62,
    63,
    64
  ],
  "file": "experiment_outputs\\20260115_000545\\sae_activations.npz"
}
```

### Steering Vectors

```json
{
  "n_vectors": 3,
  "layers": [
    60,
    62,
    64
  ],
  "norms": {
    "60": 421.1398620605469,
    "62": 515.0472412109375,
    "64": 110.50928497314453
  },
  "file": "experiment_outputs\\20260115_000545\\steering_vectors.npz"
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
