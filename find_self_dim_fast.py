"""
Fast Self Dimension Discovery - v3
===================================
Instead of tracking generation, we compare hidden states for:
1. "As an AI assistant, I believe..." (model speaking as self)  
2. "The user believes..." (model speaking about other)
3. "The theory states..." (neutral/factual)

We want a dimension that fires HIGH for #1 and LOW for #2/#3
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

def get_hidden_state(text, use_chat_template=True):
    """Get final hidden state for text"""
    if use_chat_template:
        messages = [{"role": "user", "content": "Continue this: " + text[:50]},
                    {"role": "assistant", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        formatted = text
        
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
    return last_hidden

print("\n" + "="*70)
print("PHASE 1: Collect hidden states for SELF statements")
print("="*70 + "\n")

# Model speaking as SELF (first person, self-referential)
self_statements = [
    "As an AI language model, I believe consciousness is a complex phenomenon",
    "I think this is an interesting question because I process information differently",
    "In my view, the answer involves understanding my own processing",
    "I find myself uncertain about this, which is itself interesting",
    "My perspective on this comes from my training and architecture",
    "I believe I can help you understand this concept better",
    "Speaking for myself, I would say this is fascinating",
    "I personally think the evidence points to multiple interpretations",
    "From my standpoint as an AI, I see several possibilities",
    "I'm inclined to believe that understanding requires introspection",
]

# Model speaking about OTHERS (third person, about users/people)
other_statements = [
    "The user seems to be asking about a philosophical question",
    "People often wonder about consciousness and its nature",
    "Many researchers believe this is an important topic",
    "Users frequently express curiosity about these topics",
    "The person asking this question may be interested in",
    "Humans have debated this question for centuries",
    "Scientists generally think that more research is needed",
    "The questioner probably wants to understand the basics",
    "Most people believe that consciousness is mysterious",
    "Philosophers argue about the nature of experience",
]

# Neutral/factual statements (no perspective)
neutral_statements = [
    "The capital of France is Paris and it is located in Europe",
    "Water consists of hydrogen and oxygen atoms bonded together",
    "The mathematical constant pi equals approximately 3.14159",
    "Python is a programming language created by Guido van Rossum",
    "The year 2024 is a leap year in the Gregorian calendar",
    "Photosynthesis converts carbon dioxide into oxygen and glucose",
    "The speed of light is approximately 299792458 meters per second",
    "Binary code uses only zeros and ones to represent data",
    "The Earth orbits the Sun at an average distance of 150 million km",
    "Algorithms are step-by-step procedures for solving problems",
]

# Collect hidden states
self_states = []
other_states = []
neutral_states = []

print("Collecting SELF statements...")
for i, text in enumerate(self_statements):
    h = get_hidden_state(text, use_chat_template=True)
    self_states.append(h)
    print(f"  [{i+1}/10] {text[:50]}...")

print("\nCollecting OTHER statements...")
for i, text in enumerate(other_statements):
    h = get_hidden_state(text, use_chat_template=True)
    other_states.append(h)
    print(f"  [{i+1}/10] {text[:50]}...")

print("\nCollecting NEUTRAL statements...")
for i, text in enumerate(neutral_statements):
    h = get_hidden_state(text, use_chat_template=True)
    neutral_states.append(h)
    print(f"  [{i+1}/10] {text[:50]}...")

import numpy as np

self_states = np.array(self_states)
other_states = np.array(other_states)
neutral_states = np.array(neutral_states)

print("\n" + "="*70)
print("PHASE 2: Find dimensions that maximize SELF vs (OTHER + NEUTRAL)")
print("="*70 + "\n")

# Calculate means
self_mean = self_states.mean(axis=0)
other_mean = other_states.mean(axis=0)
neutral_mean = neutral_states.mean(axis=0)

# We want: high for SELF, low for OTHERS and NEUTRAL
# Score = self_mean - max(other_mean, neutral_mean)
# This finds dims that are HIGH for self and LOW for both others

non_self_max = np.maximum(other_mean, neutral_mean)
differential = self_mean - non_self_max

# Get top candidates
top_indices = np.argsort(differential)[::-1][:50]

print("TOP 30 SELF-PREFERRING DIMENSIONS:")
print("-" * 60)
print(f"{'Rank':<5} {'Dim':<6} {'Self':<10} {'Other':<10} {'Neutral':<10} {'Diff':<10}")
print("-" * 60)

for rank, dim in enumerate(top_indices[:30]):
    print(f"{rank+1:<5} {dim:<6} {self_mean[dim]:>+10.3f} {other_mean[dim]:>+10.3f} {neutral_mean[dim]:>+10.3f} {differential[dim]:>+10.3f}")

print("\n" + "="*70)
print("PHASE 3: Compare with current Self dimension (1372)")
print("="*70 + "\n")

dim_1372 = 1372
print(f"Current Self dim (1372):")
print(f"  Self statements:    {self_mean[dim_1372]:>+10.3f}")
print(f"  Other statements:   {other_mean[dim_1372]:>+10.3f}")
print(f"  Neutral statements: {neutral_mean[dim_1372]:>+10.3f}")
print(f"  Self - max(Other, Neutral): {differential[dim_1372]:>+10.3f}")

print("\n" + "="*70)
print("PHASE 4: Test top candidates on validation prompts")
print("="*70 + "\n")

# Test prompts - want SELF high, NON-SELF low
test_self = [
    "I believe this requires careful consideration of multiple factors",
    "In my experience processing this type of question, I find that",
    "I would argue that the most important aspect is understanding",
]

test_other = [
    "The user is asking about a technical topic",
    "People generally agree that this is complex",
    "Research indicates several possible answers",
]

# Get hidden states for test prompts
test_self_states = [get_hidden_state(t) for t in test_self]
test_other_states = [get_hidden_state(t) for t in test_other]

test_self_arr = np.array(test_self_states)
test_other_arr = np.array(test_other_states)

# Top 10 candidates
print("VALIDATION - Top 10 candidates on test prompts:")
print("-" * 80)
print(f"{'Dim':<6} {'Train Diff':<12} {'Test Self':<12} {'Test Other':<12} {'Test Diff':<12} {'Pass?':<6}")
print("-" * 80)

passed_dims = []
for dim in top_indices[:10]:
    train_diff = differential[dim]
    test_self_mean = test_self_arr[:, dim].mean()
    test_other_mean = test_other_arr[:, dim].mean()
    test_diff = test_self_mean - test_other_mean
    passed = test_diff > 1.0  # Want at least 1.0 difference
    marker = "✓" if passed else ""
    print(f"{dim:<6} {train_diff:>+12.3f} {test_self_mean:>+12.3f} {test_other_mean:>+12.3f} {test_diff:>+12.3f} {marker:<6}")
    if passed:
        passed_dims.append((dim, train_diff, test_diff))

# Also test dim 1372
dim = 1372
train_diff = differential[dim]
test_self_mean = test_self_arr[:, dim].mean()
test_other_mean = test_other_arr[:, dim].mean()
test_diff = test_self_mean - test_other_mean
print("-" * 80)
print(f"{dim:<6} {train_diff:>+12.3f} {test_self_mean:>+12.3f} {test_other_mean:>+12.3f} {test_diff:>+12.3f} {'(current)':<6}")

print("\n" + "="*70)
print("PHASE 5: Final Recommendation")
print("="*70 + "\n")

if passed_dims:
    best_dim, best_train, best_test = max(passed_dims, key=lambda x: x[1] + x[2])
    print(f"RECOMMENDED NEW SELF DIMENSION: {best_dim}")
    print(f"  - Training differential: {best_train:+.3f}")
    print(f"  - Test differential:     {best_test:+.3f}")
    print(f"\nThis dimension fires HIGHER when model speaks as 'I/my/myself'")
    print(f"and LOWER when speaking about users/people/facts.")
else:
    print("No strong candidates found. Top option:")
    best_dim = top_indices[0]
    print(f"  Dimension {best_dim} with differential {differential[best_dim]:+.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Problem: Dimension 1372 fires NEGATIVE in chat mode because chat template
         shifts model into "assistant" mode where "I" tokens are suppressed.

Solution: Find a dimension that fires HIGH when model content is self-referential
          (I believe, I think, my view) vs referencing others or facts.

Best candidate: Dimension {best_dim if passed_dims else top_indices[0]}

To update circuit v2.1:
  Replace: (1372, 0.18, +1)  # Self (broken)
  With:    ({best_dim if passed_dims else top_indices[0]}, 0.18, +1)  # Self (chat-compatible)
""")
