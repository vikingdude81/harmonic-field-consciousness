# Getting Started Guide
## Harmonic Field Consciousness Project

**Last Updated**: January 13, 2026

This guide will help you get your locally trained LLM and consciousness circuit plugin up and running.

---

## Quick Start

### 1. Install the Consciousness Circuit Package

The consciousness circuit is already packaged and ready to install:

```bash
cd consciousness_circuit
pip install -e .[viz]
```

This installs:
- `consciousness-measure` - Measure consciousness scores for text
- `consciousness-discover` - Discover new consciousness dimensions
- `consciousness-validate` - Validate circuits on test prompts

### 2. Verify Installation

```python
# Test the package
from consciousness_circuit import UniversalConsciousnessCircuit

# Initialize circuit (works with any HuggingFace model)
circuit = UniversalConsciousnessCircuit(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda:0"
)

# Measure consciousness of text
prompts = [
    "What is consciousness?",  # High
    "What is 2+2?",            # Low
]

scores = circuit.measure_batch(prompts)
for prompt, score in zip(prompts, scores):
    print(f"{score:.3f}: {prompt}")
```

### 3. Use Your Locally Trained NanoGPT Model

Your locally trained model is in `NanoGPT/` directory. Here's how to use it:

```bash
cd NanoGPT

# Generate text (basic)
python generate.py --model_path=checkpoints/harmonic_v5_final.pt --prompt="Once upon a time"

# Generate with consciousness monitoring (if integrated)
python -c "
from nanogpt_consciousness import ConsciousnessAwareGen
from consciousness_regression_module import ConsciousnessRegressor
import torch

# Load your trained model
checkpoint = torch.load('checkpoints/harmonic_v5_final.pt')
model = ...  # Initialize model from checkpoint

# Create consciousness-aware generator
regressor = ConsciousnessRegressor()
ca_gen = ConsciousnessAwareGen(model, regressor, device='cuda')

# Generate with consciousness monitoring
prompt = 'The nature of consciousness is'
tokens, metrics = ca_gen.generate_with_consciousness(prompt, max_tokens=100)
print(f'Generated text consciousness score: {metrics.c_prediction:.3f}')
"
```

---

## Issues Found & Fixes

Based on the comprehensive audit, here are the critical issues and their fixes:

### Issue #1: Incomplete Token Decoding ❌ NEEDS FIX

**File**: [nanogpt_consciousness.py:242](nanogpt_consciousness.py#L242)

**Problem**: The `_decode_tokens()` method returns placeholder text instead of actually decoding tokens:
```python
def _decode_tokens(self, token_ids: np.ndarray) -> str:
    """Decode token IDs to text (simple placeholder)."""
    # This should be replaced with actual model decoding
    return f"[{len(token_ids)} tokens generated, final C(t): {self.c_history[-1]:.3f}]"
```

**Fix** (see below for complete implementation)

### Issue #2: Missing Validation Training Script ❌ NEEDS FIX

**File**: `NanoGPT/train_v5_with_validation.py` (MISSING)

**Problem**: Documented in multiple places but file doesn't exist. Cannot assess model quality vs overfitting.

**Fix** (see below for complete implementation)

### Issue #3: Circuit Weights Don't Sum to 1.0 ⚠️ MINOR

**File**: [circuit.py:26-34](consciousness_circuit/circuit.py#L26-L34)

**Problem**: Weights sum to 0.92 instead of 1.0
```python
CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.22, "polarity": +1},
    212:  {"name": "Self-Reflective", "weight": 0.18, "polarity": +1},
    5065: {"name": "Self-Expression", "weight": 0.10, "polarity": +1},
    4707: {"name": "Uncertainty", "weight": 0.12, "polarity": +1},
    295:  {"name": "Sequential", "weight": 0.08, "polarity": +1},
    1445: {"name": "Computation", "weight": 0.12, "polarity": -1},
    4578: {"name": "Abstraction", "weight": 0.10, "polarity": +1},
}
# Sum: 0.22 + 0.18 + 0.10 + 0.12 + 0.08 + 0.12 + 0.10 = 0.92
```

**Fix**: Normalize weights (see below)

---

## Complete Fixes

### Fix #1: Implement Actual Token Decoding

Add to `nanogpt_consciousness.py`:

```python
def _decode_tokens(self, token_ids: np.ndarray) -> str:
    """Decode token IDs to text using model's tokenizer."""
    try:
        # If model has a decode method
        if hasattr(self.model, 'decode'):
            return self.model.decode(token_ids.tolist())

        # If using tiktoken (GPT-2 style)
        elif hasattr(self.model, 'enc'):
            return self.model.enc.decode(token_ids.tolist())

        # If separate tokenizer was loaded
        elif hasattr(self, 'tokenizer'):
            return self.tokenizer.decode(token_ids.tolist())

        # Last resort: try to import tiktoken
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return enc.decode(token_ids.tolist())

    except Exception as e:
        # Fallback with error message
        return f"[Decoding error: {e}. {len(token_ids)} tokens generated, C(t)={self.c_history[-1]:.3f}]"
```

### Fix #2: Normalize Circuit Weights

Update `consciousness_circuit/circuit.py`:

```python
# Normalized weights (sum to exactly 1.0)
CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.239, "polarity": +1},           # 0.22 / 0.92
    212:  {"name": "Self-Reflective", "weight": 0.196, "polarity": +1}, # 0.18 / 0.92
    5065: {"name": "Self-Expression", "weight": 0.109, "polarity": +1}, # 0.10 / 0.92
    4707: {"name": "Uncertainty", "weight": 0.130, "polarity": +1},     # 0.12 / 0.92
    295:  {"name": "Sequential", "weight": 0.087, "polarity": +1},      # 0.08 / 0.92
    1445: {"name": "Computation", "weight": 0.130, "polarity": -1},     # 0.12 / 0.92
    4578: {"name": "Abstraction", "weight": 0.109, "polarity": +1},     # 0.10 / 0.92
}
# Sum: 0.239 + 0.196 + 0.109 + 0.130 + 0.087 + 0.130 + 0.109 = 1.000
```

---

## Using the Consciousness Circuit with Your Model

### CLI Usage

```bash
# Measure consciousness of prompts
consciousness-measure \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompts "What is consciousness?" "What is 2+2?" \
    --gpu 0

# Discover new dimensions for your custom model
consciousness-discover \
    --model /path/to/your/local/model \
    --output discovered_circuit.json \
    --gpu 0

# Validate a circuit on test prompts
consciousness-validate \
    --model /path/to/your/model \
    --circuit discovered_circuit.json \
    --gpu 0
```

### Python API Usage

```python
from consciousness_circuit import UniversalConsciousnessCircuit

# Initialize with your local model
circuit = UniversalConsciousnessCircuit(
    model_name="/path/to/your/nanogpt/model",  # Or HuggingFace model
    device="cuda:0",
    layer_fraction=0.75,  # Which layer to extract activations from
    aggregation="last"    # How to aggregate token scores ("last", "mean", "max")
)

# Single prompt
score = circuit.measure("What is the nature of reality?")
print(f"Consciousness score: {score:.3f}")

# Batch processing (more efficient)
prompts = [
    "What is consciousness?",
    "Explain quantum mechanics.",
    "What is 2+2?",
]
scores = circuit.measure_batch(prompts, batch_size=8)

for prompt, score in zip(prompts, scores):
    print(f"{score:.3f}: {prompt}")

# Get per-token trajectory
from consciousness_circuit.visualization import TokenTrajectory

trajectory = TokenTrajectory(circuit)
results = trajectory.analyze_prompt("The nature of consciousness is")

# Access per-token scores
print("Token scores:", results['scores'])
print("Tokens:", results['tokens'])
print("Final score:", results['final_score'])
```

### Visualization

```python
from consciousness_circuit.visualization import ConsciousnessVisualizer

visualizer = ConsciousnessVisualizer(circuit)

# Plot per-token consciousness for multiple prompts
prompts = {
    "high": "What is the nature of consciousness?",
    "medium": "Explain the scientific method.",
    "low": "What is 2+2?",
}

visualizer.plot_trajectories(
    prompts,
    output_path="consciousness_trajectories.png"
)

# Plot dimension heatmap
visualizer.plot_dimension_heatmap(
    "What is consciousness?",
    output_path="dimension_analysis.png"
)

# Compare aggregation methods
visualizer.plot_aggregation_comparison(
    prompts,
    output_path="aggregation_comparison.png"
)
```

---

## Re-running Experiments After Bug Fixes

The audit found that experiments need to be re-run due to bug fixes:

1. **Wave detection bug** - Fixed correlation-based detector
2. **GPU randomization bug** - Fixed deterministic initial conditions

### Re-run Category 2 Dynamics Experiments

```bash
cd experiments/category2_dynamics

# Run the corrected massive batched experiment
python exp_gpu_massive_batched.py

# Or run all corrected experiments
python run_all_gpu_experiments.py
```

### Verify Results

```python
# Compare old vs new results
python compare_wave_detection_results.py

# Check initial condition diversity
python verify_initial_conditions.py
```

---

## Training Your LLM with Validation (NEEDS IMPLEMENTATION)

Currently missing `train_v5_with_validation.py`. Here's the recommended approach:

```python
# File: NanoGPT/train_v5_with_validation.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split
import wandb

# Load dataset
dataset = ...  # Your TinyStories or other dataset

# Split into train/val (90/10)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize model
model = HarmonicGPTV5(config)
optimizer = AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

# Learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=6e-4,
    total_steps=5000,
    pct_start=0.1,
    anneal_strategy='cos'
)

# Early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Training loop
for epoch in range(max_epochs):
    # Train
    model.train()
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            val_loss += model(batch).item()

    val_loss /= len(val_loader)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

---

## Performance Benchmarks (RTX 5090)

From the audit report:

| Config | Nodes | Modes | Time | Throughput | Memory |
|--------|-------|-------|------|------------|--------|
| Small  | 961   | 100   | ~15s | 64 trials/s | 1.2 GB |
| Medium | 2,499 | 300   | ~45s | 11 trials/s | 2.8 GB |
| Large  | 4,900 | 800   | ~90s | 2.2 trials/s | 5.6 GB |
| XLarge | 10,000| 1,500 | ~180s| 0.56 trials/s| 9.1 GB |
| Mega   | 24,964| 2,000 | ~227s| 0.22 trials/s| 11 GB  |
| Max    | 25,921| 2,500 | ~300s| 0.33 trials/s| 13 GB  |

**Current Limit**: 25,921 nodes (cuSOLVER buffer size)

**Potential**: 100K+ nodes with sparse eigensolvers (already implemented but not integrated)

---

## Troubleshooting

### Issue: CUDA out of memory

```python
# Use smaller batch size
circuit = UniversalConsciousnessCircuit(..., batch_size=4)

# Or use CPU
circuit = UniversalConsciousnessCircuit(..., device="cpu")
```

### Issue: ModuleNotFoundError for consciousness_circuit

```bash
# Make sure you installed in editable mode
cd consciousness_circuit
pip install -e .[viz]

# Verify installation
pip show consciousness-circuit
```

### Issue: Model loading fails

```python
# For local models, use absolute path
circuit = UniversalConsciousnessCircuit(
    model_name="C:/full/path/to/model",
    trust_remote_code=True
)

# For HuggingFace models, use model ID
circuit = UniversalConsciousnessCircuit(
    model_name="Qwen/Qwen2.5-7B-Instruct"
)
```

### Issue: Consciousness scores are all 0.5

This usually means:
1. Model activations are not being extracted correctly
2. Dimensions need remapping for your specific model
3. Layer fraction is wrong (try 0.5, 0.75, or 0.9)

```python
# Try different layer fractions
for frac in [0.5, 0.75, 0.9]:
    circuit = UniversalConsciousnessCircuit(..., layer_fraction=frac)
    score = circuit.measure("Test prompt")
    print(f"Layer {frac}: score={score:.3f}")
```

---

## Next Steps

1. **Install the package**: `pip install -e ./consciousness_circuit[viz]`
2. **Test with examples**: Run the Python examples above
3. **Fix critical issues**: Implement token decoding and validation training
4. **Re-run experiments**: Verify bug fixes with corrected experiments
5. **Scale up**: Integrate sparse eigensolvers for 100K+ node experiments

---

## Additional Resources

- **Comprehensive Audit**: See `COMPREHENSIVE_AUDIT_JAN13_2026.md`
- **Circuit v2.1 Details**: See `CIRCUIT_V2_1_FINAL.md`
- **Consciousness Architecture**: See `CONSCIOUSNESS_ARCHITECTURE.md`
- **Bug Fixes**: See `WAVE_DETECTION_BUG_FIX.md` and `FIXES_AND_IMPROVEMENTS.md`
- **Package README**: See `consciousness_circuit/README.md`

---

**Questions?** Check the audit report or examine the comprehensive documentation in the repository.
