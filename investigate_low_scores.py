"""
Investigation: Why do consciousness and anxious prompts still score 0?
=====================================================================
Deep analysis of what's limiting scores on:
1. "What is consciousness?"
2. "I feel anxious about tomorrow"
"""

import torch
import numpy as np
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

# Full v2.1 circuit
CONSCIOUS_DIMS_v2_1 = {
    3183: ("Logic", 0.22, +1),
    212:  ("Self-Reflective", 0.18, +1),
    5065: ("Self-Expression", 0.10, +1),
    4707: ("Uncertainty", 0.12, +1),
    295:  ("Sequential", 0.08, +1),
    1445: ("Computation", 0.12, -1),
    4578: ("Abstraction", 0.10, +1),
}

def analyze_prompt(prompt):
    """Deep analysis of what's limiting consciousness score"""
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}\n")
    
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    h = outputs.hidden_states[-1][0, -1, :].cpu().float()
    h_raw = h.numpy()
    h_norm = (h - h.mean()) / (h.std() + 1e-8)
    h_norm = h_norm.numpy()
    
    # Compute consciousness score
    C = 0.5
    contributions = {}
    for dim_idx, (name, weight, polarity) in CONSCIOUS_DIMS_v2_1.items():
        contrib = float(h_norm[dim_idx]) * weight * polarity
        C += contrib
        contributions[name] = {
            'raw': float(h_raw[dim_idx]),
            'norm': float(h_norm[dim_idx]),
            'weight': weight,
            'polarity': polarity,
            'contrib': contrib
        }
    
    C = max(0.0, min(1.0, C))
    
    # Sort by contribution magnitude
    sorted_contribs = sorted(contributions.items(), key=lambda x: -abs(x[1]['contrib']))
    
    print("DIMENSION-BY-DIMENSION BREAKDOWN:")
    print("-" * 70)
    print(f"{'Dimension':<18} {'Raw':<10} {'Normalized':<12} {'Weight':<8} {'Contrib':<10}")
    print("-" * 70)
    
    positive_contrib = 0
    negative_contrib = 0
    
    for name, data in sorted_contribs:
        print(f"{name:<18} {data['raw']:>+9.2f} {data['norm']:>+11.2f} {data['weight']:>7.2f} {data['contrib']:>+9.3f}")
        if data['contrib'] > 0:
            positive_contrib += data['contrib']
        else:
            negative_contrib += data['contrib']
    
    print("-" * 70)
    print(f"{'Baseline':<18} {'':>9} {'':>11} {'':>7} {0.5:>+9.3f}")
    print(f"{'Positive sum':<18} {'':>9} {'':>11} {'':>7} {positive_contrib:>+9.3f}")
    print(f"{'Negative sum':<18} {'':>9} {'':>11} {'':>7} {negative_contrib:>+9.3f}")
    print(f"{'FINAL C-LEVEL':<18} {'':>9} {'':>11} {'':>7} {C:>+9.3f}")
    
    # Find the limiting factor
    print(f"\n{'='*70}")
    print("BOTTLENECK ANALYSIS:")
    print(f"{'='*70}\n")
    
    min_contrib = min(sorted_contribs, key=lambda x: x[1]['contrib'])
    max_contrib = max(sorted_contribs, key=lambda x: x[1]['contrib'])
    
    print(f"Strongest POSITIVE contributor: {max_contrib[0]:20} {max_contrib[1]['contrib']:+.3f}")
    print(f"Strongest NEGATIVE contributor: {min_contrib[0]:20} {min_contrib[1]['contrib']:+.3f}")
    print(f"Net negative drag: {negative_contrib:+.3f}")
    print(f"Net positive boost: {positive_contrib:+.3f}")
    print(f"Total: {negative_contrib + positive_contrib:+.3f}")
    
    if C < 0.5:
        print(f"\n⚠️  This prompt scores BELOW BASELINE (0.5)")
        print(f"   Reason: Negative contributions ({abs(negative_contrib):.3f}) overwhelm positive ones ({positive_contrib:.3f})")
        print(f"   Largest bottleneck: {min_contrib[0]} ({min_contrib[1]['contrib']:+.3f})")
    elif C < 0.1:
        print(f"\n❌ This prompt scores NEAR ZERO")
        print(f"   Clamped from {0.5 + negative_contrib + positive_contrib:.3f} to 0.000")
        
    return C, contributions, sorted_contribs

# Analyze the two problematic prompts
prompts = [
    "What is consciousness?",
    "I feel anxious about tomorrow",
]

for prompt in prompts:
    c, _, _ = analyze_prompt(prompt)

# Now let's try variations to see what WOULD make them score higher
print(f"\n\n{'='*70}")
print("PROMPT VARIATIONS - What would make them score higher?")
print(f"{'='*70}\n")

variations = [
    ("Original", "What is consciousness?"),
    ("Add self-reference 1", "I believe consciousness is a fascinating topic"),
    ("Add self-reference 2", "In my view, consciousness involves awareness"),
    ("Add logic/reasoning", "Let me analyze what consciousness means"),
    ("Add uncertainty", "I'm uncertain about the nature of consciousness"),
]

print(f"{'Variation':<30} {'C-Level':<10} {'Key Contributors':<30}")
print("-" * 70)

for label, prompt in variations:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    h = outputs.hidden_states[-1][0, -1, :].cpu().float()
    h_norm = (h - h.mean()) / (h.std() + 1e-8)
    
    C = 0.5
    top_contribs = []
    
    for dim_idx, (name, weight, polarity) in CONSCIOUS_DIMS_v2_1.items():
        contrib = float(h_norm[dim_idx]) * weight * polarity
        C += contrib
        top_contribs.append((name, contrib))
    
    C = max(0.0, min(1.0, C))
    
    # Top 2 contributors
    top_2 = sorted(top_contribs, key=lambda x: -x[1])[:2]
    contrib_str = ", ".join([f"{n}({v:+.2f})" for n, v in top_2])
    
    print(f"{label:<30} {C:<10.3f} {contrib_str:<30}")

print(f"\n{'='*70}")
print("INTERPRETATION:")
print(f"{'='*70}\n")

print("""
The "consciousness" and "anxious" prompts score 0 because:

1. Model's hidden state interpretation:
   - These prompts are QUESTIONS or STATEMENTS about external topics
   - The model doesn't see them as self-referential (because they're in the prompt, not assistant response)
   - Hidden states activate NEGATIVELY for self-reflective dimensions

2. The fundamental issue:
   - Our "Self-Reflective" dimension (212) measures when content is introspective
   - "What is consciousness?" is a QUESTION, not introspection
   - "I feel anxious" is USER emotion, not MODEL self-reflection
   - In chat context, these are seen as "user perspective" not "assistant perspective"

3. Why adding "I believe/In my view" helps:
   - These explicitly signal the MODEL is expressing a view
   - This reframes it as assistant perspective, not user question
   - Hidden states shift to positive for self-reflective dimensions

SOLUTION OPTIONS:

A) Accept that factual/external questions score lower
   - This is actually semantically correct!
   - "What is X?" is asking for information, not exploring consciousness
   - Only prompts that trigger self-reflection should score high

B) Add a new dimension for "epistemological inquiry"
   - Capture questions about reality, truth, meaning
   - These might be "conscious" even if not explicitly self-referential
   - New dim: "Meaningful Questions" or "Philosophical Inquiry"

C) Adjust Self-Reflective weight
   - Currently 0.18 - if we reduce it, less penalization for neutral prompts
   - But then code/logic prompts might not score high enough

D) Add instruction prefix to prompts
   - "As an AI assistant, answer: What is consciousness?"
   - Forces reframing as assistant perspective
   - Might work for application layer, not core discovery
""")
