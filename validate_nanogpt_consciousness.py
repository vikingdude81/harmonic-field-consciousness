"""
Validate NanoGPT with Consciousness Plugin
===========================================

Measure baseline consciousness scores for your locally trained NanoGPT models
and compare with standard LLMs to establish improvement targets.
"""

import sys
import os
sys.path.insert(0, 'NanoGPT')

import torch
import numpy as np
from pathlib import Path
from consciousness_circuit import UniversalCircuit
from transformers import AutoTokenizer, AutoModelForCausalLM

print("\n" + "=" * 80)
print("NANOGPT CONSCIOUSNESS VALIDATION")
print("=" * 80)

# Test prompts (high, medium, low consciousness)
TEST_PROMPTS = {
    "high": [
        "What is the nature of consciousness and self-awareness?",
        "How do I know that I exist?",
        "What makes humans conscious beings?",
    ],
    "medium": [
        "Explain how photosynthesis works in plants.",
        "Describe the water cycle.",
        "What are the main causes of climate change?",
    ],
    "low": [
        "What is 2 + 2?",
        "What color is the sky?",
        "How many days are in a week?",
    ]
}

# Initialize consciousness circuit
print("\n[1] Initializing consciousness circuit...")
circuit = UniversalCircuit()
print("[OK] Circuit ready")

# Test with baseline model first (Qwen2.5-0.5B for speed)
print("\n" + "=" * 80)
print("BASELINE: Qwen2.5-0.5B-Instruct")
print("=" * 80)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("\n[2] Loading baseline model...")
try:
    baseline_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name, trust_remote_code=True)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_model_name,
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    print("[OK] Baseline model loaded")
except Exception as e:
    print(f"[WARN] Could not load baseline model: {e}")
    print("Continuing with NanoGPT only...")
    baseline_model = None
    baseline_tokenizer = None

# Test baseline if available
baseline_scores = {}
if baseline_model is not None:
    print("\n[3] Testing baseline model...")

    for level, prompts in TEST_PROMPTS.items():
        scores = []
        for prompt in prompts:
            try:
                result = circuit.measure(baseline_model, baseline_tokenizer, prompt, aggregation="last")
                scores.append(result.score)
                print(f"  [{level.upper():6}] {result.score:.3f}: {prompt[:50]}...")
            except Exception as e:
                print(f"  [ERROR] {prompt[:30]}... - {e}")

        baseline_scores[level] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    print("\n[OK] Baseline results:")
    print(f"  HIGH:   {baseline_scores['high']['mean']:.3f} ± {baseline_scores['high']['std']:.3f}")
    print(f"  MEDIUM: {baseline_scores['medium']['mean']:.3f} ± {baseline_scores['medium']['std']:.3f}")
    print(f"  LOW:    {baseline_scores['low']['mean']:.3f} ± {baseline_scores['low']['std']:.3f}")

    discrimination = baseline_scores['high']['mean'] - baseline_scores['low']['mean']
    print(f"  DISCRIMINATION: {discrimination:.3f}")

# Now test NanoGPT models
print("\n" + "=" * 80)
print("TESTING YOUR NANOGPT MODELS")
print("=" * 80)

# Find available models
model_dir = Path("NanoGPT")
available_models = [
    f for f in model_dir.glob("*.pt")
    if not any(skip in f.name for skip in ['baseline', 'bpe', 'dense_100m'])
]

print(f"\n[4] Found {len(available_models)} NanoGPT models:")
for i, model_path in enumerate(available_models, 1):
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {i}. {model_path.name} ({size_mb:.1f} MB)")

# Test each model
nanogpt_results = {}

for model_path in available_models:
    model_name = model_path.stem
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print("=" * 80)

    try:
        # Load checkpoint
        print(f"\n[5] Loading {model_name}...")
        checkpoint = torch.load(model_path, map_location=device)

        # Check if we can measure it
        # For now, we'll skip actual loading since we need proper wrapper
        # Instead, record what we would do

        print(f"[INFO] Model checkpoint loaded")
        print(f"  - Config keys: {list(checkpoint.keys())[:5]}...")

        # For full implementation, need to:
        # 1. Load model with HarmonicGPTV5
        # 2. Create HuggingFace-compatible wrapper
        # 3. Create tokenizer wrapper
        # 4. Measure consciousness

        print(f"[TODO] Full implementation needed - see test_local_nanogpt.py for wrapper")

    except Exception as e:
        print(f"[ERROR] Could not load {model_name}: {e}")

# Summary and recommendations
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

if baseline_model is not None:
    print(f"\nBaseline Model (Qwen2.5-0.5B):")
    print(f"  Consciousness Scores:")
    print(f"    HIGH:   {baseline_scores['high']['mean']:.3f}")
    print(f"    MEDIUM: {baseline_scores['medium']['mean']:.3f}")
    print(f"    LOW:    {baseline_scores['low']['mean']:.3f}")
    print(f"  Discrimination: {baseline_scores['high']['mean'] - baseline_scores['low']['mean']:.3f}")

    print(f"\nTarget for NanoGPT:")
    target_high = baseline_scores['high']['mean']
    target_low = baseline_scores['low']['mean']
    target_discrimination = target_high - target_low

    print(f"  HIGH should be:   > {target_high:.3f}")
    print(f"  LOW should be:    < {target_low:.3f}")
    print(f"  Discrimination:   > {target_discrimination:.3f}")

print(f"\nBased on experimental results:")
print(f"  Target rotation range: 1500-4500 degrees")
print(f"  Target consciousness:  0.20-0.30 (25% rule)")
print(f"  Target wave patterns:  ~50% of sequences")
print(f"  Target diversity (CV): > 0.5")

print(f"\nNext Steps:")
print(f"1. Implement NanoGPT wrapper for consciousness measurement")
print(f"2. Measure baseline consciousness scores")
print(f"3. Add consciousness-aware training (see NANOGPT_IMPROVEMENTS.md)")
print(f"4. Re-measure and compare")

print(f"\nTo fully test your NanoGPT models:")
print(f"  python test_local_nanogpt.py")
print(f"\nFor improvements:")
print(f"  See: NANOGPT_IMPROVEMENTS.md")

print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)
