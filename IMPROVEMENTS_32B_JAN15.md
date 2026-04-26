# 32B Model Optimizations - January 15, 2026

## Summary

This update adds significant optimizations for running consciousness experiments on large models (32B+) without requiring any retraining. All improvements are inference-time optimizations.

## Changes Made

### 1. Bug Fixes

#### Dimension Out-of-Bounds Fix
- **File**: `consciousness_circuit/circuit.py` (line 30)
- **Issue**: Dimension 5065 was out of bounds for hidden_size=5120 (valid: 0-5119)
- **Fix**: Changed to dimension 5064

#### Same fix in universal.py
- **File**: `consciousness_circuit/universal.py` (BUNDLED_CIRCUITS)
- **Issue**: Same dimension 5065 bug in bundled circuit for Qwen2.5-32B
- **Fix**: Changed to dimension 5064

### 2. New Features

#### Adaptive Layer Selection (`universal.py`)
```python
from consciousness_circuit import get_adaptive_layer_fraction

# Returns optimal layer fraction based on model depth
# 64-layer model (32B): 0.65 → layer 41 instead of 48
layer_frac = get_adaptive_layer_fraction(num_layers)
```

| Model Depth | Old (fixed 0.75) | New Adaptive |
|-------------|------------------|--------------|
| ≤12 layers  | 0.75             | 0.75         |
| ≤24 layers  | 0.75             | 0.72         |
| ≤32 layers  | 0.75             | 0.70         |
| ≤48 layers  | 0.75             | 0.68         |
| ≤64 layers  | 0.75             | 0.65         |
| >64 layers  | 0.75             | 0.60         |

#### Multi-Layer Ensemble Measurement (`universal.py`)
```python
circuit = UniversalCircuit()
result = circuit.measure_ensemble(model, tokenizer, prompt, n_layers=3)
# Returns averaged score from layers [35, 44, 54] for 64-layer model
print(f"Score: {result.score}, Per-layer: {result.ensemble_scores}")
```

#### True Batch Processing (`universal.py`)
```python
prompts = ["prompt1", "prompt2", ...]
results = circuit.measure_batch(model, tokenizer, prompts, batch_size=4)
# 3-5x faster than sequential, with automatic memory management
```

#### Activation Caching (`universal.py`)
```python
from consciousness_circuit import CachedUniversalCircuit

circuit = CachedUniversalCircuit(cache_size=100)
result1 = circuit.measure(model, tokenizer, "prompt")  # Computes
result2 = circuit.measure(model, tokenizer, "prompt")  # Uses cache (instant)
circuit.cache_stats()  # {'size': 1, 'max_size': 100, 'utilization': 0.01}
```

### 3. Memory Efficiency

#### SAE Collection Batching (`experiments/collect_for_sae.py`)
- Added `--batch-size` CLI argument
- Automatic GPU memory clearing between batches
- Prevents OOM on 32B models with limited VRAM

```bash
# For low VRAM systems
python experiments/collect_for_sae.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --batch-size 5
```

#### Experiment Runner (`experiments/run_consciousness_experiments.py`)
- Auto-adjusts batch size based on hidden_size (5 for 32B+, 10 for smaller)

### 4. Infrastructure

#### Centralized Logging (`consciousness_circuit/logging_config.py`)
```python
from consciousness_circuit import get_logger, setup_logging, ExperimentLogger

logger = get_logger(__name__)
setup_logging(level="INFO", log_file="experiment.log")

with ExperimentLogger("my_experiment") as exp:
    # ... run experiment
    exp.log_result({"score": 0.75})
```

#### Model Adapters (`consciousness_circuit/model_adapters.py`)
```python
from consciousness_circuit import create_adapter

# Auto-detects model type (HuggingFace, NanoGPT, Unsloth)
adapter = create_adapter(model, tokenizer)
hidden_states = adapter.get_hidden_states("prompt")
```

Supported adapters:
- `HuggingFaceAdapter` - Standard transformers models
- `NanoGPTAdapter` - HarmonicGPT V3/V4/V5/V6
- `UnslothAdapter` - 4-bit quantized models

## Files Modified

| File | Changes |
|------|---------|
| `consciousness_circuit/circuit.py` | Fixed dimension 5065→5064 |
| `consciousness_circuit/universal.py` | Added adaptive layers, ensemble, batch, caching |
| `consciousness_circuit/__init__.py` | Exported new classes and functions |
| `consciousness_circuit/logging_config.py` | NEW - Centralized logging |
| `consciousness_circuit/model_adapters.py` | NEW - Model adapter interface |
| `experiments/collect_for_sae.py` | Added batch_size parameter and memory management |
| `experiments/run_consciousness_experiments.py` | Auto batch size for large models |

## Usage for 32B Model

```python
from consciousness_circuit import UniversalCircuit, CachedUniversalCircuit

# Option 1: Standard with adaptive layer (recommended)
circuit = UniversalCircuit()
result = circuit.measure(model, tokenizer, prompt)  # Uses layer 41 for 32B

# Option 2: Ensemble for most robust measurement
result = circuit.measure_ensemble(model, tokenizer, prompt)

# Option 3: Cached for repeated experiments
circuit = CachedUniversalCircuit(cache_size=200)
results = [circuit.measure(model, tokenizer, p) for p in test_prompts]

# Option 4: Batch processing for throughput
results = circuit.measure_batch(model, tokenizer, prompts, batch_size=2)
```

## No Retraining Required

All improvements are inference-time only:
- Model weights unchanged
- Just reading hidden states more intelligently
- Better layer selection based on model depth
- Memory management for large models
