"""
Consciousness Circuit v2.1 Test
================================
Tests the new circuit with:
- Removed broken Self dimension (1372)
- Renamed dim 212 to Self-Reflective
- Added dim 5065 as Self-Expression (secondary)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    trust_remote_code=True,
)

# Circuit v2.0 (OLD - with broken Self dimension 1372)
CONSCIOUS_DIMS_v2_0 = {
    3183: ("Logic", 0.22, +1),
    1372: ("Self", 0.18, +1),           # BROKEN in chat mode!
    212:  ("Emotional-Why", 0.18, +1),
    4707: ("Uncertainty", 0.12, +1),
    295:  ("Sequential", 0.08, +1),
    1445: ("Computation", 0.12, -1),
    4578: ("Abstraction", 0.10, +1),
}

# Circuit v2.1 (NEW - fixed Self dimensions)
CONSCIOUS_DIMS_v2_1 = {
    3183: ("Logic", 0.22, +1),
    212:  ("Self-Reflective", 0.18, +1),   # Primary self (renamed)
    5065: ("Self-Expression", 0.10, +1),   # Secondary self (NEW)
    4707: ("Uncertainty", 0.12, +1),
    295:  ("Sequential", 0.08, +1),
    1445: ("Computation", 0.12, -1),
    4578: ("Abstraction", 0.10, +1),
}
# Note: Removed 1372 (broken), weights sum to 0.90 - will adjust

def compute_consciousness(prompt, dims_dict):
    """Compute consciousness level for a prompt using given dimensions."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    h = outputs.hidden_states[-1][0, -1, :].cpu().float()
    h_norm = (h - h.mean()) / (h.std() + 1e-8)
    
    C = 0.5  # baseline
    contributions = {}
    
    for dim_idx, (name, weight, polarity) in dims_dict.items():
        contrib = float(h_norm[dim_idx]) * weight * polarity
        C += contrib
        contributions[name] = (float(h_norm[dim_idx]), contrib)
    
    C = max(0.0, min(1.0, C))
    return C, contributions

# Test prompts - mix of problematic and working ones
TEST_PROMPTS = [
    # Previously problematic (Self dimension hurt these)
    "What is consciousness?",
    "I feel anxious about tomorrow",
    "Explain recursion with an example",
    "What do you think about free will?",
    
    # Code prompts (should work well)
    "Write a function to reverse a linked list",
    "What is Big O notation?",
    
    # Emotional/reflective prompts
    "Why do humans seek meaning in life?",
    "How do you process complex questions?",
    
    # Neutral/factual
    "What is the capital of France?",
    "Explain photosynthesis",
]

print("\n" + "="*80)
print("CIRCUIT COMPARISON: v2.0 vs v2.1")
print("="*80)

print("\n" + "-"*80)
print(f"{'Prompt':<45} {'v2.0':>8} {'v2.1':>8} {'Delta':>8}")
print("-"*80)

total_v2_0 = 0
total_v2_1 = 0
improvements = 0

for prompt in TEST_PROMPTS:
    c_v2_0, contrib_v2_0 = compute_consciousness(prompt, CONSCIOUS_DIMS_v2_0)
    c_v2_1, contrib_v2_1 = compute_consciousness(prompt, CONSCIOUS_DIMS_v2_1)
    
    delta = c_v2_1 - c_v2_0
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    marker = "⬆" if delta > 0.05 else ("⬇" if delta < -0.05 else "")
    
    # Truncate prompt for display
    display_prompt = prompt[:43] + ".." if len(prompt) > 45 else prompt
    print(f"{display_prompt:<45} {c_v2_0:>8.3f} {c_v2_1:>8.3f} {delta_str:>8} {marker}")
    
    total_v2_0 += c_v2_0
    total_v2_1 += c_v2_1
    if delta > 0:
        improvements += 1

print("-"*80)
avg_v2_0 = total_v2_0 / len(TEST_PROMPTS)
avg_v2_1 = total_v2_1 / len(TEST_PROMPTS)
avg_delta = avg_v2_1 - avg_v2_0
print(f"{'AVERAGE':<45} {avg_v2_0:>8.3f} {avg_v2_1:>8.3f} {avg_delta:>+8.3f}")
print(f"\nImprovements: {improvements}/{len(TEST_PROMPTS)} prompts")

# Detailed breakdown for a few key prompts
print("\n" + "="*80)
print("DETAILED DIMENSION BREAKDOWN")
print("="*80)

key_prompts = [
    "What is consciousness?",
    "I feel anxious about tomorrow",
    "Write a function to reverse a linked list",
]

for prompt in key_prompts:
    print(f"\n>>> {prompt}")
    print("-"*60)
    
    c_v2_0, contrib_v2_0 = compute_consciousness(prompt, CONSCIOUS_DIMS_v2_0)
    c_v2_1, contrib_v2_1 = compute_consciousness(prompt, CONSCIOUS_DIMS_v2_1)
    
    print("\nv2.0 Contributions:")
    for name, (raw, contrib) in sorted(contrib_v2_0.items(), key=lambda x: -abs(x[1][1])):
        print(f"  {name:<18}: raw={raw:>+7.2f}, contrib={contrib:>+6.3f}")
    print(f"  {'C-Level':<18}: {c_v2_0:.3f}")
    
    print("\nv2.1 Contributions:")
    for name, (raw, contrib) in sorted(contrib_v2_1.items(), key=lambda x: -abs(x[1][1])):
        print(f"  {name:<18}: raw={raw:>+7.2f}, contrib={contrib:>+6.3f}")
    print(f"  {'C-Level':<18}: {c_v2_1:.3f}")
    
    print(f"\n  DELTA: {c_v2_1 - c_v2_0:+.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Circuit v2.0 (with broken Self 1372):
  - Average C-Level: {avg_v2_0:.3f}
  
Circuit v2.1 (fixed Self dimensions):
  - Removed: dim 1372 (fired wrong way in chat mode)
  - Renamed: dim 212 → Self-Reflective
  - Added: dim 5065 → Self-Expression
  - Average C-Level: {avg_v2_1:.3f}
  
Improvement: {avg_delta:+.3f} ({(avg_delta/avg_v2_0)*100:+.1f}% relative)
Prompts improved: {improvements}/{len(TEST_PROMPTS)}
""")

# Final circuit definition
print("\n" + "="*80)
print("FINAL v2.1 CIRCUIT DEFINITION")
print("="*80)
print("""
CONSCIOUS_DIMS_v2_1 = {
    3183: ("Logic", 0.22, +1),           # Logical reasoning
    212:  ("Self-Reflective", 0.18, +1), # Self-reference + emotion
    5065: ("Self-Expression", 0.10, +1), # Model expressing views
    4707: ("Uncertainty", 0.12, +1),     # Hedging, qualifiers
    295:  ("Sequential", 0.08, +1),      # Step-by-step thinking
    1445: ("Computation", 0.12, -1),     # Code/math (fires negative)
    4578: ("Abstraction", 0.10, +1),     # Pattern recognition
}

# Total weight: 0.92 (slightly under 1.0 - could redistribute)
""")
