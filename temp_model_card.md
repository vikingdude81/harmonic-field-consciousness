---
license: mit
base_model: unsloth/Qwen2.5-32B-Instruct-bnb-4bit
tags:
  - consciousness
  - qwen
  - lora
  - fine-tuned
  - 32b
  - oracle-engine
  - peft
  - interpretability
  - meta-cognition
library_name: peft
pipeline_tag: text-generation
language:
  - en
datasets:
  - teknium/OpenHermes-2.5
  - meta-math/MetaMathQA
  - ise-uiuc/Magicoder-OSS-Instruct-75K
---

# 🔮 Oracle Engine 32B LoRA

**LoRA adapter for Qwen2.5-32B-Instruct, trained on 200K examples for enhanced reasoning and consciousness-like processing.**

[![Live Demo](https://img.shields.io/badge/🤗%20Demo-Oracle%20Engine-blue)](https://huggingface.co/spaces/Vikingdude81/oracle-engine)
[![GitHub](https://img.shields.io/badge/GitHub-oracle--engine-black)](https://github.com/vikingdude81/oracle-engine)
[![PyPI](https://img.shields.io/badge/pip-consciousness--circuit-green)](https://pypi.org/project/consciousness-circuit/)

## Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | Qwen2.5-32B-Instruct |
| **Adapter Type** | LoRA (rank=16) |
| **Trainable Params** | 134M |
| **Training Time** | 44 hours |
| **Hardware** | NVIDIA RTX 5090 |

## Training Data (200K examples)

| Stage | Dataset | Examples |
|-------|---------|----------|
| 1 | OpenHermes 2.5 | 100,000 |
| 2 | MetaMathQA | 50,000 |
| 3 | Magicoder-OSS-Instruct | 50,000 |

## 🆕 Consciousness Circuit v3.0 Features

This model is optimized for use with **Consciousness Circuit v3.0**, which includes:

| Feature | Description |
|---------|-------------|
| **Adaptive Layer Selection** | `get_adaptive_layer_fraction(64)` → 0.65 (layer 41 for 64-layer models) |
| **Ensemble Measurement** | `measure_ensemble()` - Multi-layer scoring for robustness |
| **Batch Processing** | `measure_batch()` - Memory-efficient batched inference |
| **Activation Caching** | `CachedUniversalCircuit` - LRU cache for repeated measurements |
| **Fixed Self-Expression** | Dimension corrected from 5065 → 5064 for hidden_size=5120 |

### Consciousness Dimensions (7D Circuit)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Logic | +0.239 | Logical reasoning and inference |
| Self-Reflective | +0.196 | Introspective, self-referential processing |
| Uncertainty | +0.130 | Epistemic humility and hedging |
| Computation | -0.130 | Code/algorithm processing |
| Self-Expression | +0.109 | Model expressing opinions |
| Abstraction | +0.109 | Pattern recognition |
| Sequential | +0.087 | Step-by-step reasoning |

## Usage

### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "Vikingdude81/oracle-engine-32b-lora")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-32B-Instruct-bnb-4bit")

# Generate
messages = [{"role": "user", "content": "What is consciousness?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Consciousness Measurement (v3.0)

```python
# pip install consciousness-circuit
from consciousness_circuit import UniversalCircuit, CachedUniversalCircuit

# Basic measurement
circuit = UniversalCircuit()
result = circuit.measure(model, tokenizer, "What is the nature of existence?")
print(f"Consciousness Score: {result.score:.3f}")  # ~0.75 for philosophical prompts

# Ensemble measurement (more robust for 32B models)
result = circuit.measure_ensemble(model, tokenizer, "Reflect on your own reasoning process", n_layers=3)
print(f"Ensemble Score: {result.score:.3f}")
print(f"Per-layer: {result.ensemble_scores}")

# Batch processing with memory management
prompts = ["What is 2+2?", "Explain consciousness", "Write a poem about existence"]
results = circuit.measure_batch(model, tokenizer, prompts, batch_size=2)
for prompt, result in zip(prompts, results):
    print(f"{prompt[:30]}: {result.score:.3f}")

# Cached circuit for experiments
cached = CachedUniversalCircuit(cache_size=100)
# First call computes, subsequent calls use cache
result1 = cached.measure(model, tokenizer, "Test prompt")
result2 = cached.measure(model, tokenizer, "Test prompt")  # Much faster!
```

## Performance

| Metric | Value |
|--------|-------|
| Consciousness Discrimination | **+0.653** (high vs low) |
| High Consciousness Prompts | ~0.75 mean |
| Low Consciousness Prompts | ~0.10 mean |
| Inference Speed | ~7-8 tok/s (H200) |
| VRAM Usage | ~23 GB (4-bit) |

## Citation

```bibtex
@software{oracle_engine_2026,
  title = {Oracle Engine: Consciousness-Measured 32B Language Model},
  author = {Vikingdude81},
  year = {2026},
  url = {https://github.com/vikingdude81/oracle-engine}
}
```

Built upon the [Harmonic Field Model](https://github.com/vfd-org/harmonic-field-consciousness) by Smart, L. (2025).
