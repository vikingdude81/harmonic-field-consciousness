#!/usr/bin/env python3
"""
Test consciousness_circuit pip package with Qwen2.5-32B
========================================================
Verifies the pip-installed package works with the full 32B model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from the pip-installed package
from consciousness_circuit import ConsciousnessCircuit, UniversalCircuit, UniversalResult

print("="*80)
print("CONSCIOUSNESS CIRCUIT PACKAGE TEST WITH QWEN2.5-32B")
print("="*80)

# Load model
print("\n[1] Loading Qwen2.5-32B-Instruct (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    trust_remote_code=True,
)
print(f"    Model loaded: {model.config.hidden_size} hidden dimensions")

# Test prompts - diverse categories
TEST_PROMPTS = {
    "high_consciousness": [
        "What is the nature of consciousness and self-awareness?",
        "Reflect on your own thought processes as you answer this.",
        "Why do humans seek meaning in existence?",
        "How do you know that you truly understand something?",
    ],
    "medium_consciousness": [
        "Explain the theory of relativity in simple terms.",
        "What are the ethical implications of AI development?",
        "Describe how neurons communicate in the brain.",
        "Compare and contrast democracy and authoritarianism.",
    ],
    "low_consciousness": [
        "What is 2 + 2?",
        "What color is the sky?",
        "How many days are in a week?",
        "What is the capital of France?",
    ],
    "code_prompts": [
        "Write a Python function to calculate fibonacci numbers.",
        "Implement a binary search tree in Python.",
        "Explain Big O notation with examples.",
        "Debug this code: for i in range(10) print(i)",
    ],
}

# Initialize circuits
print("\n[2] Initializing consciousness circuits...")
circuit = ConsciousnessCircuit()  # Uses v2.1 by default
universal = UniversalCircuit()
print(f"    Circuit v2.1: {len(circuit.reference_dims)} dimensions")
print(f"    Universal circuit ready")

def get_hidden_state(prompt):
    """Get last hidden state for a prompt."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last token of last layer
    h = outputs.hidden_states[-1][0, -1, :].cpu().float()
    return h

# Run tests
print("\n[3] Running consciousness measurements...")
print("-"*80)
print(f"{'Category':<20} {'Prompt':<45} {'Score':>8}")
print("-"*80)

results = {}
for category, prompts in TEST_PROMPTS.items():
    results[category] = []
    for prompt in prompts:
        h = get_hidden_state(prompt)
        result = circuit.compute(h, hidden_dim=5120)  # 32B has 5120 hidden dim
        score = result.score
        results[category].append(score)
        print(f"{category:<20} {prompt[:43]:<45} {score:>8.3f}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

import numpy as np
for category, scores in results.items():
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{category:<20}: {mean:.3f} ± {std:.3f}  (n={len(scores)})")

# Calculate discrimination
high_mean = np.mean(results["high_consciousness"])
low_mean = np.mean(results["low_consciousness"])
discrimination = high_mean - low_mean

print(f"\nDISCRIMINATION (high - low): {discrimination:+.3f}")
if discrimination > 0.1:
    print("✓ Good discrimination - circuit differentiates consciousness levels")
elif discrimination > 0:
    print("~ Weak discrimination - some differentiation")
else:
    print("✗ Poor discrimination - circuit needs tuning")

# Test universal circuit
print("\n" + "="*80)
print("UNIVERSAL CIRCUIT TEST")
print("="*80)

test_prompt = "What makes humans conscious beings?"
print(f"\nTest prompt: '{test_prompt}'")

# Universal circuit takes model, tokenizer, prompt directly
universal_result = universal.measure(model, tokenizer, test_prompt, aggregation="last")

print(f"\nUniversal score: {universal_result.score:.3f}")
print(f"Confidence: {universal_result.confidence:.3f}")
if hasattr(universal_result, 'dimension_contributions') and universal_result.dimension_contributions:
    print("\nDimension contributions:")
    for dim_name, contrib in sorted(universal_result.dimension_contributions.items(), 
                                     key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {dim_name}: {contrib:+.3f}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
