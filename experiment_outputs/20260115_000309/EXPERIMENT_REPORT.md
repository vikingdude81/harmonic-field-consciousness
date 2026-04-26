# Consciousness Circuit Experiment Report

**Model**: unsloth/Qwen2.5-32B-Instruct-bnb-4bit
**Date**: 2026-01-15 00:03:42
**Output Directory**: experiment_outputs\20260115_000309

## Summary

| Experiment | Status | Duration |
|------------|--------|----------|
| Thought Trajectory | ✓ | 4.9s |

## Detailed Results

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
