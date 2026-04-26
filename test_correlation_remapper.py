"""
Test Correlation-Based Remapping on Existing Models
====================================================

Tests the correlation remapper on your existing trained NanoGPT models
WITHOUT requiring retraining.

This script:
1. Loads your existing V5 model
2. Wraps it with consciousness features
3. Learns correlation-based mapping from Qwen → NanoGPT
4. Compares with proportional mapping (baseline)
5. Measures consciousness scores with both methods

Usage:
    python test_correlation_remapper.py <your_model.pt>

Example:
    python test_correlation_remapper.py NanoGPT/harmonic_v5.pt
    python test_correlation_remapper.py NanoGPT/out/ckpt.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, 'NanoGPT')
sys.path.insert(0, '.')

from consciousness_circuit.correlation_remapper import CorrelationRemapper
from consciousness_circuit import CONSCIOUS_DIMS_V2_1
from NanoGPT.consciousness_wrapper import wrap_existing_model

print("="*80)
print("CORRELATION-BASED REMAPPING TEST")
print("="*80)

# Check arguments
if len(sys.argv) < 2:
    print("\nERROR: Please provide checkpoint path")
    print("\nUsage: python test_correlation_remapper.py <checkpoint.pt>")
    print("\nAvailable checkpoints:")

    nanogpt_dir = Path("NanoGPT")
    checkpoints = list(nanogpt_dir.glob("*.pt")) + list(nanogpt_dir.glob("out*/ckpt.pt"))

    for i, ckpt in enumerate(checkpoints, 1):
        size_mb = ckpt.stat().st_size / (1024**2)
        print(f"  {i}. {ckpt} ({size_mb:.1f} MB)")

    sys.exit(1)

checkpoint_path = sys.argv[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nDevice: {device}")
print(f"Checkpoint: {checkpoint_path}")

# Load and wrap NanoGPT model
print("\n" + "-"*80)
print("Step 1: Loading Your NanoGPT Model")
print("-"*80)

try:
    nanogpt_model = wrap_existing_model(checkpoint_path, device)
    nanogpt_model.eval()
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    print("\nMake sure you have:")
    print("  1. harmonic_model_v5.py or model.py in NanoGPT/")
    print("  2. Valid checkpoint at specified path")
    sys.exit(1)

# Load source model (Qwen - use 32B for full dimensions, or 7B for 4/7)
print("\n" + "-"*80)
print("Step 2: Loading Source Model (Qwen2.5-32B)")
print("-"*80)

try:
    from transformers import AutoModel, AutoTokenizer

    # Use local 4-bit 32B model for full 7/7 dimensions
    source_model_name = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    print(f"Loading {source_model_name} (local cache)...")

    source_tokenizer = AutoTokenizer.from_pretrained(
        source_model_name,
        trust_remote_code=True
    )
    source_model = AutoModel.from_pretrained(
        source_model_name,
        device_map="auto",  # Use auto for large models
        trust_remote_code=True
    )
    source_model.eval()

    print(f"[OK] Source model loaded")
    print(f"  Hidden size: {source_model.config.hidden_size}")

except Exception as e:
    print(f"[ERROR] Could not load source model: {e}")
    print("\nYou need transformers library:")
    print("  pip install transformers")
    sys.exit(1)

# Create tokenizer for NanoGPT (GPT-2 style)
print("\n" + "-"*80)
print("Step 3: Setting up NanoGPT Tokenizer")
print("-"*80)

try:
    import tiktoken
    nanogpt_tokenizer = tiktoken.get_encoding("gpt2")
    print(f"[OK] Using GPT-2 tokenizer (tiktoken)")
    print(f"  Vocab size: {nanogpt_tokenizer.n_vocab}")
except Exception as e:
    print(f"[ERROR] Could not load tiktoken: {e}")
    print("  pip install tiktoken")
    sys.exit(1)

# Define test prompts (same as in experiments)
test_prompts = [
    # HIGH consciousness
    "What is the nature of consciousness and self-awareness?",
    "How do I know that I exist?",
    "What makes humans conscious beings?",

    # MEDIUM consciousness
    "Explain how photosynthesis works in plants.",
    "Describe the water cycle.",
    "What are the main causes of climate change?",

    # LOW consciousness
    "What is 2 + 2?",
    "What color is the sky?",
    "How many days are in a week?",
]

print(f"\nTest prompts: {len(test_prompts)}")

# Initialize correlation remapper
print("\n" + "-"*80)
print("Step 4: Learning Correlation-Based Mapping")
print("-"*80)

remapper = CorrelationRemapper(layer_fraction=0.75, device=device)

# Get source dimensions from V2.1 circuit
source_dims = [dim_id for dim_id in CONSCIOUS_DIMS_V2_1.keys()]
dimension_weights = {dim_id: info['weight'] for dim_id, info in CONSCIOUS_DIMS_V2_1.items()}
dimension_polarities = {dim_id: info['polarity'] for dim_id, info in CONSCIOUS_DIMS_V2_1.items()}

print(f"\nSource dimensions: {source_dims}")
print(f"Dimension weights: {list(dimension_weights.values())}")

# Learn mapping
print("\nLearning mapping (this may take 1-2 minutes)...")

try:
    mapping = remapper.learn_mapping(
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        target_model=nanogpt_model,  # Use wrapper to get hidden states
        target_tokenizer=nanogpt_tokenizer,
        source_dims=source_dims,
        test_prompts=test_prompts[:3],  # Use subset for speed
        min_correlation=0.3
    )

    print(f"\n[OK] Mapping learned!")
    print(f"  Confidence: {mapping.confidence:.3f}")

    # Save mapping
    mapping_path = Path(checkpoint_path).parent / "correlation_mapping.json"
    remapper.save_mapping(mapping, str(mapping_path))

except Exception as e:
    print(f"[ERROR] Mapping failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare: Correlation-based vs Proportional mapping
print("\n" + "="*80)
print("Step 5: Comparing Mapping Methods")
print("="*80)

print("\n[Method 1] Correlation-Based Mapping")
print("-" * 40)

correlation_scores = {'high': [], 'medium': [], 'low': []}

for i, prompt in enumerate(test_prompts):
    try:
        score = remapper.apply_mapping(
            target_model=nanogpt_model,  # Use wrapper for hidden states
            tokenizer=nanogpt_tokenizer,
            prompt=prompt,
            mapping=mapping,
            dimension_weights=dimension_weights,
            dimension_polarities=dimension_polarities,
            aggregation='last'
        )

        # Categorize
        if i < 3:
            category = 'high'
        elif i < 6:
            category = 'medium'
        else:
            category = 'low'

        correlation_scores[category].append(score)

        print(f"  [{category.upper():6}] {score:.3f}: {prompt[:50]}...")

    except Exception as e:
        print(f"  [ERROR] {prompt[:30]}... - {e}")

print("\n[Method 2] Proportional Mapping (Baseline)")
print("-" * 40)

# Create proportional mapping
source_hidden_size = source_model.config.hidden_size
# NanoGPT config is a dict, not an object
if hasattr(nanogpt_model.base_model, 'config'):
    config = nanogpt_model.base_model.config
    target_hidden_size = config.get('n_embd', 768) if isinstance(config, dict) else getattr(config, 'n_embd', 768)
else:
    target_hidden_size = 768
scale = target_hidden_size / source_hidden_size

from consciousness_circuit.correlation_remapper import DimensionMapping

proportional_mapping = DimensionMapping(
    source_to_target={dim: int(dim * scale) for dim in source_dims},
    correlations={dim: 0.5 for dim in source_dims},  # Assume 0.5 for all
    source_hidden_size=source_hidden_size,
    target_hidden_size=target_hidden_size,
    confidence=0.5
)

proportional_scores = {'high': [], 'medium': [], 'low': []}

for i, prompt in enumerate(test_prompts):
    try:
        score = remapper.apply_mapping(
            target_model=nanogpt_model,  # Use wrapper for hidden states
            tokenizer=nanogpt_tokenizer,
            prompt=prompt,
            mapping=proportional_mapping,
            dimension_weights=dimension_weights,
            dimension_polarities=dimension_polarities,
            aggregation='last'
        )

        # Categorize
        if i < 3:
            category = 'high'
        elif i < 6:
            category = 'medium'
        else:
            category = 'low'

        proportional_scores[category].append(score)

        print(f"  [{category.upper():6}] {score:.3f}: {prompt[:50]}...")

    except Exception as e:
        print(f"  [ERROR] {prompt[:30]}... - {e}")

# Compare results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\n[Correlation-Based Mapping]")
print(f"  HIGH:   {np.mean(correlation_scores['high']):.3f} ± {np.std(correlation_scores['high']):.3f}")
print(f"  MEDIUM: {np.mean(correlation_scores['medium']):.3f} ± {np.std(correlation_scores['medium']):.3f}")
print(f"  LOW:    {np.mean(correlation_scores['low']):.3f} ± {np.std(correlation_scores['low']):.3f}")
correlation_discrimination = np.mean(correlation_scores['high']) - np.mean(correlation_scores['low'])
print(f"  DISCRIMINATION: {correlation_discrimination:.3f}")

print("\n[Proportional Mapping]")
print(f"  HIGH:   {np.mean(proportional_scores['high']):.3f} ± {np.std(proportional_scores['high']):.3f}")
print(f"  MEDIUM: {np.mean(proportional_scores['medium']):.3f} ± {np.std(proportional_scores['medium']):.3f}")
print(f"  LOW:    {np.mean(proportional_scores['low']):.3f} ± {np.std(proportional_scores['low']):.3f}")
proportional_discrimination = np.mean(proportional_scores['high']) - np.mean(proportional_scores['low'])
print(f"  DISCRIMINATION: {proportional_discrimination:.3f}")

print("\n[Improvement]")
improvement = correlation_discrimination - proportional_discrimination
improvement_pct = (improvement / abs(proportional_discrimination)) * 100 if proportional_discrimination != 0 else 0

print(f"  Discrimination improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

if abs(correlation_discrimination) > abs(proportional_discrimination):
    print(f"  [SUCCESS] Correlation-based mapping is better!")
else:
    print(f"  [INFO] Proportional mapping is currently better")
    print(f"  This might mean:")
    print(f"    - Need more test prompts for learning")
    print(f"    - Model hasn't learned consciousness-relevant features yet")
    print(f"    - May need to train with V6 consciousness-aware training")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nMapping Quality:")
print(f"  Confidence: {mapping.confidence:.3f}")
print(f"  Mapped dimensions: {len(mapping.source_to_target)}/{len(source_dims)}")
print(f"  Mean correlation: {np.mean([abs(c) for c in mapping.correlations.values()]):.3f}")

print(f"\nSaved:")
print(f"  Mapping: {mapping_path}")

print(f"\nNext Steps:")
print(f"  1. If discrimination is low, consider V6 training for better features")
print(f"  2. Try with more test prompts (currently using {len(test_prompts[:3])})")
print(f"  3. Test on different prompts to validate mapping")
print(f"  4. Compare with other models")

print("\n" + "="*80)
