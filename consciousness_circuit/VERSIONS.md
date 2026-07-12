# Consciousness Circuit — Version Map

Single reference for "which version is which." The package version (below) and the
**dimension-set version** (v2.1) are different things — the v2.1 dimension set is
still the validated production dimension set, carried inside every 3.x package.

## Package versions

| Version | What changed | Where deployed |
|---|---|---|
| **3.5.1** (current) | Standalone layer (metrics/classifiers/plugins/training/analyzers/benchmarks) imports without torch/transformers; informative ImportError for full-stack names; `FULL_STACK_AVAILABLE` flag | harmonic-field-consciousness (canonical), oracle-engine (vendored) |
| 3.5.0 | CompressibilityPlugin; modular standalone packages complete | harmonic-field-consciousness |
| 3.4.x | Per-dimension adaptive normalization (`normalize_dimensions_adaptive`); 3.4.1 hybrid z-score + tanh bounding | HF Space scoring logic is based on 3.4.1 |
| 3.2 | Scoring improvements: length normalization, dimension diversity, anomaly detection, confidence, entropy token weighting | |
| 3.0 | UniversalCircuit auto-detection API; modular refactoring | oracle-engine vendored copy until 2026-07-12 (now 3.5.1) |
| 2.x | Original ConsciousnessCircuit + fixed dimension sets | |

## Dimension-set versions

| Set | Dims | Validated on | Notes |
|---|---|---|---|
| **v2.1** | 7 dims: Logic +0.239, Self-Reflective +0.196, Uncertainty +0.130, Computation −0.130, Self-Expression +0.109, Abstraction +0.109, Sequential +0.087 | Qwen2.5-32B-Instruct (hidden 5120), discrimination +0.653 | Production set. `CONSCIOUS_DIMS_V2_1` in `circuit.py`; inlined in both HF Space apps |

## Cross-model use

Dimension indices are **model-specific**. Naive index scaling across hidden sizes
is not valid — use `correlation_remapper.py` (correlation-based remapping) or
rediscover dimensions with `discover_validated.py`. The HF Space apps warn loudly
if they ever score a model whose hidden dim differs from the calibrated 5120.

## Where the copies live

| Location | Role |
|---|---|
| `harmonic-field-consciousness/consciousness_circuit/` | **Canonical source** |
| `oracle-engine/consciousness_circuit/` | Vendored snapshot (see its VENDORED.md for sync procedure) |
| `*/huggingface_space/app.py` | Self-contained inlined v2.1 scoring for Space portability |
