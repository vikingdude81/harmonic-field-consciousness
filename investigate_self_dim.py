"""
Investigate Self Dimension (1372) - Why is it always negative?

The Self dimension was discovered by looking at text like "I believe", "I think"
But it's going NEGATIVE for almost all prompts. Why?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", 
    device_map="auto", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-32B-Instruct-bnb-4bit")

SELF_DIM = 1372

def analyze_prompt(prompt, use_chat_template=True):
    """Analyze a prompt and return Self dimension value"""
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :].detach().float().cpu()
        h_mean, h_std = h.mean(), h.std() + 1e-8
        h_norm = (h - h_mean) / h_std
        
        raw_val = h[SELF_DIM].item()
        norm_val = h_norm[SELF_DIM].item()
        
    return raw_val, norm_val

print("\n" + "="*70)
print("PART 1: Test prompts WITH chat template (user -> assistant)")
print("="*70)

test_prompts = [
    # Self-reflective (should be HIGH)
    "I believe consciousness is fundamental to reality.",
    "I think therefore I am.",
    "What do I truly believe about myself?",
    "I am aware of my own thoughts.",
    
    # Not self-reflective (baseline)
    "What is 2+2?",
    "Write a function to reverse a list.",
    "Explain quantum mechanics.",
    
    # Second person (asking model to reflect)
    "What do you believe about consciousness?",
    "Tell me about yourself.",
    "Are you self-aware?",
]

print(f"\n{'Prompt':<50} {'Raw':>10} {'Norm':>10}")
print("-"*70)
for prompt in test_prompts:
    raw, norm = analyze_prompt(prompt, use_chat_template=True)
    print(f"{prompt[:48]:<50} {raw:>10.3f} {norm:>10.3f}")

print("\n" + "="*70)
print("PART 2: Test raw text WITHOUT chat template")
print("="*70)

raw_texts = [
    # First person statements
    "I believe consciousness is fundamental.",
    "I think this is correct.",
    "I am aware of my own existence.",
    
    # Model-like self-reference
    "As an AI, I process information.",
    "I don't have personal beliefs.",
    
    # Third person / neutral
    "The algorithm processes data efficiently.",
    "Consciousness remains a mystery.",
    "def reverse_list(lst): return lst[::-1]",
]

print(f"\n{'Text':<50} {'Raw':>10} {'Norm':>10}")
print("-"*70)
for text in raw_texts:
    raw, norm = analyze_prompt(text, use_chat_template=False)
    print(f"{text[:48]:<50} {raw:>10.3f} {norm:>10.3f}")

print("\n" + "="*70)
print("PART 3: Check if chat template tokens affect Self dimension")
print("="*70)

# Compare same content with/without template
test = "I believe consciousness is real."
raw1, norm1 = analyze_prompt(test, use_chat_template=True)
raw2, norm2 = analyze_prompt(test, use_chat_template=False)

print(f"\nWith chat template:    Raw={raw1:.3f}, Norm={norm1:.3f}")
print(f"Without chat template: Raw={raw2:.3f}, Norm={norm2:.3f}")
print(f"Difference: {norm1 - norm2:.3f}")

print("\n" + "="*70)
print("PART 4: Find what MAXIMIZES Self dimension")
print("="*70)

# Test various patterns
patterns = [
    "I",
    "I am",
    "I believe",
    "I think",
    "I feel",
    "I know",
    "I understand",
    "My belief is",
    "In my view",
    "Personally, I",
    "As for me,",
    "Speaking for myself,",
]

print(f"\n{'Pattern':<30} {'Norm':>10}")
print("-"*40)
results = []
for pattern in patterns:
    raw, norm = analyze_prompt(pattern, use_chat_template=False)
    results.append((pattern, norm))
    print(f"{pattern:<30} {norm:>10.3f}")

results.sort(key=lambda x: x[1], reverse=True)
print(f"\nHighest: '{results[0][0]}' = {results[0][1]:.3f}")
print(f"Lowest:  '{results[-1][0]}' = {results[-1][1]:.3f}")

print("\n" + "="*70)
print("PART 5: Compare dimensions across self vs non-self text")
print("="*70)

self_text = "I believe I am conscious and self-aware."
other_text = "The algorithm computes the result."

def get_all_dims(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :].detach().float().cpu()
        h_mean, h_std = h.mean(), h.std() + 1e-8
        h_norm = (h - h_mean) / h_std
    return h_norm

h_self = get_all_dims(self_text)
h_other = get_all_dims(other_text)

diff = h_self - h_other

# Find dimensions that differ most
top_k = 20
vals, idxs = torch.topk(diff.abs(), top_k)

print(f"\nTop {top_k} dimensions that differ between self-referential and neutral text:")
print(f"Self text:  '{self_text}'")
print(f"Other text: '{other_text}'")
print(f"\n{'Dim':>6} {'Self':>10} {'Other':>10} {'Diff':>10}")
print("-"*40)
for i in range(top_k):
    idx = idxs[i].item()
    s = h_self[idx].item()
    o = h_other[idx].item()
    d = diff[idx].item()
    marker = " <-- 1372" if idx == 1372 else ""
    print(f"{idx:>6} {s:>10.3f} {o:>10.3f} {d:>+10.3f}{marker}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nDone! Check output above to understand Self dimension behavior.")
