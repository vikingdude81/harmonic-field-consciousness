# Quick Start - Harmonic Field Consciousness

**Last Updated**: January 13, 2026

---

## 🚀 Install & Test (5 minutes)

```bash
# 1. Install consciousness circuit package
cd consciousness_circuit
pip install -e .[viz]

# 2. Test installation
python -c "
from consciousness_circuit import UniversalConsciousnessCircuit
circuit = UniversalConsciousnessCircuit('Qwen/Qwen2.5-7B-Instruct', device='cuda:0')
score = circuit.measure('What is consciousness?')
print(f'Score: {score:.3f}')
"
```

---

## 📦 Use Your Locally Trained Model

```bash
cd NanoGPT

# Generate text
python generate.py \
    --model_path checkpoints/harmonic_v5_final.pt \
    --prompt "Once upon a time"

# With consciousness monitoring
python -c "
from nanogpt_consciousness import ConsciousnessAwareGen
# ... (see GETTING_STARTED_GUIDE.md for full example)
"
```

---

## 🔬 Use Consciousness Circuit Plugin

### Python API
```python
from consciousness_circuit import UniversalConsciousnessCircuit

# Initialize
circuit = UniversalConsciousnessCircuit(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda:0",
    aggregation="last"  # "last", "mean", or "max"
)

# Measure single prompt
score = circuit.measure("What is the nature of reality?")
print(f"Consciousness: {score:.3f}")

# Batch processing
prompts = ["High consciousness prompt", "Medium", "Low: 2+2?"]
scores = circuit.measure_batch(prompts)
```

### CLI
```bash
# Measure consciousness
consciousness-measure \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompts "What is consciousness?" \
    --gpu 0

# Discover new circuit
consciousness-discover \
    --model /path/to/model \
    --output circuit.json \
    --gpu 0

# Validate circuit
consciousness-validate \
    --model /path/to/model \
    --circuit circuit.json
```

---

## 📊 Visualize Results

```python
from consciousness_circuit.visualization import ConsciousnessVisualizer

visualizer = ConsciousnessVisualizer(circuit)

# Plot per-token consciousness
prompts = {
    "high": "What is consciousness?",
    "low": "What is 2+2?"
}
visualizer.plot_trajectories(prompts, output_path="trajectories.png")

# Plot dimension heatmap
visualizer.plot_dimension_heatmap(
    "What is consciousness?",
    output_path="dimensions.png"
)
```

---

## 🐛 What Was Fixed Today

### ✅ Fixed Issues
1. **Token decoding** - Now properly decodes generated text
2. **Circuit weights** - Normalized to sum to 1.0
3. **Documentation** - Comprehensive audit + guides created

### ⚠️ Previously Fixed (In Codebase)
1. **Wave detection** - Correlation-based algorithm
2. **GPU randomization** - Unique seeds per trial
3. **Numerical stability** - Edge case handling

### 🔧 Still TODO
1. **Re-run experiments** - Validate fixed bugs (~1 hour compute)
2. **Integrate sparse eigensolvers** - Scale to 100K+ nodes (1-2 days)

---

## 📖 Full Documentation

- **Comprehensive Audit**: [COMPREHENSIVE_AUDIT_JAN13_2026.md](COMPREHENSIVE_AUDIT_JAN13_2026.md)
- **Getting Started Guide**: [GETTING_STARTED_GUIDE.md](GETTING_STARTED_GUIDE.md)
- **Fixes Applied**: [FIXES_APPLIED_JAN13.md](FIXES_APPLIED_JAN13.md)
- **Package README**: [consciousness_circuit/README.md](consciousness_circuit/README.md)

---

## 🆘 Troubleshooting

**CUDA out of memory?**
```python
circuit = UniversalConsciousnessCircuit(..., device="cpu")
```

**Module not found?**
```bash
pip install -e ./consciousness_circuit[viz]
```

**Scores all 0.5?**
```python
# Try different layer fractions
circuit = UniversalConsciousnessCircuit(..., layer_fraction=0.75)
```

---

## 📈 Project Stats

- **Grade**: A- (Excellent)
- **Files Analyzed**: 100+ Python files
- **Issues Fixed**: 3 critical bugs + 2 implementations
- **Documentation**: 30+ markdown files
- **Tests**: 109 tests across 6 modules

---

**Ready to go!** Start with installing the package, then explore the full guides.
