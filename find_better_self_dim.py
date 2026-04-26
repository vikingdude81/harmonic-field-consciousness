"""
Find a Better Self Dimension for Chat/Assistant Mode

The current Self dimension (1372) was discovered using raw text like "I believe".
But in chat mode, it fires NEGATIVE because the chat template puts the model
in "assistant response mode" where the "I" in the prompt is the USER, not the MODEL.

SOLUTION: Find dimensions that fire when the MODEL generates self-referential text
in its RESPONSE (e.g., "I think", "In my opinion", "I believe").

Strategy:
1. Generate responses where model uses first-person language
2. Track hidden states during generation
3. Find dimensions that activate when model generates "I" tokens
4. Compare to dimensions that activate for non-self tokens
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", 
    device_map="auto", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-32B-Instruct-bnb-4bit")

# Prompts that should elicit first-person responses
SELF_ELICITING_PROMPTS = [
    "What do you think about consciousness?",
    "Share your perspective on artificial intelligence.",
    "What is your opinion on free will?",
    "Tell me about yourself and how you process information.",
    "Do you believe you are conscious? Explain your reasoning.",
]

# Prompts that should elicit factual/third-person responses  
FACTUAL_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis.",
    "What is 15 multiplied by 7?",
    "List the planets in our solar system.",
    "Define the Pythagorean theorem.",
]

def get_hidden_at_token(model, tokenizer, prompt, max_tokens=100):
    """Generate response and capture hidden states for each token"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Store hidden states and tokens
    hidden_states_list = []
    generated_tokens = []
    
    # Generate token by token
    input_ids = inputs["input_ids"]
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :].detach().float().cpu()
            hidden_states_list.append(h)
            
            # Get next token
            logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            token_text = tokenizer.decode(next_token[0])
            generated_tokens.append(token_text)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=1)
    
    return hidden_states_list, generated_tokens

def find_self_tokens(tokens):
    """Find indices of self-referential tokens"""
    self_patterns = ['I', ' I', 'I\'', ' my', 'My', ' me', 'Me', ' myself']
    indices = []
    for i, tok in enumerate(tokens):
        if any(p in tok for p in self_patterns):
            indices.append(i)
    return indices

def find_neutral_tokens(tokens):
    """Find indices of neutral/factual tokens (not self-referential)"""
    self_patterns = ['I', ' I', 'I\'', ' my', 'My', ' me', 'Me', ' myself']
    indices = []
    for i, tok in enumerate(tokens):
        # Skip punctuation and very short tokens
        if len(tok.strip()) < 2:
            continue
        if not any(p in tok for p in self_patterns):
            indices.append(i)
    return indices

print("\n" + "="*70)
print("PHASE 1: Collect hidden states from self-referential responses")
print("="*70)

self_hidden_states = []
self_context_tokens = []

for prompt in SELF_ELICITING_PROMPTS:
    print(f"\nPrompt: {prompt[:50]}...")
    hiddens, tokens = get_hidden_at_token(model, tokenizer, prompt, max_tokens=80)
    
    # Find self-referential tokens
    self_indices = find_self_tokens(tokens)
    print(f"  Generated {len(tokens)} tokens, {len(self_indices)} self-referential")
    print(f"  Self tokens: {[tokens[i] for i in self_indices[:10]]}")
    
    # Collect hidden states for self-tokens
    for idx in self_indices:
        if idx < len(hiddens):
            self_hidden_states.append(hiddens[idx])
            self_context_tokens.append(tokens[idx])

print(f"\nTotal self-referential hidden states: {len(self_hidden_states)}")

print("\n" + "="*70)
print("PHASE 2: Collect hidden states from factual responses")
print("="*70)

neutral_hidden_states = []

for prompt in FACTUAL_PROMPTS:
    print(f"\nPrompt: {prompt[:50]}...")
    hiddens, tokens = get_hidden_at_token(model, tokenizer, prompt, max_tokens=80)
    
    # Use first 20 non-self tokens
    neutral_indices = find_neutral_tokens(tokens)[:20]
    print(f"  Generated {len(tokens)} tokens, using {len(neutral_indices)} neutral")
    
    for idx in neutral_indices:
        if idx < len(hiddens):
            neutral_hidden_states.append(hiddens[idx])

print(f"\nTotal neutral hidden states: {len(neutral_hidden_states)}")

print("\n" + "="*70)
print("PHASE 3: Find dimensions that distinguish self from neutral")
print("="*70)

# Stack and normalize
self_stack = torch.stack(self_hidden_states)  # [N, 5120]
neutral_stack = torch.stack(neutral_hidden_states)  # [M, 5120]

# Normalize each sample
def normalize_batch(x):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-8
    return (x - mean) / std

self_norm = normalize_batch(self_stack)
neutral_norm = normalize_batch(neutral_stack)

# Mean activation per dimension
self_mean = self_norm.mean(dim=0)  # [5120]
neutral_mean = neutral_norm.mean(dim=0)  # [5120]

# Difference
diff = self_mean - neutral_mean

# Find top dimensions where self > neutral (positive = self-indicator)
top_k = 30
vals, idxs = torch.topk(diff, top_k)

print(f"\nTop {top_k} dimensions where SELF > NEUTRAL:")
print(f"(These fire MORE when model generates 'I', 'my', etc.)")
print(f"\n{'Dim':>6} {'Self':>10} {'Neutral':>10} {'Diff':>10}")
print("-"*40)

candidates = []
for i in range(top_k):
    idx = idxs[i].item()
    s = self_mean[idx].item()
    n = neutral_mean[idx].item()
    d = diff[idx].item()
    print(f"{idx:>6} {s:>10.3f} {n:>10.3f} {d:>+10.3f}")
    candidates.append((idx, d))

# Also find bottom (where neutral > self)
vals_bot, idxs_bot = torch.topk(-diff, top_k)
print(f"\n\nTop {top_k} dimensions where NEUTRAL > SELF:")
print(f"(These fire LESS when model generates self-referential tokens)")
print(f"\n{'Dim':>6} {'Self':>10} {'Neutral':>10} {'Diff':>10}")
print("-"*40)

for i in range(top_k):
    idx = idxs_bot[i].item()
    s = self_mean[idx].item()
    n = neutral_mean[idx].item()
    d = diff[idx].item()
    print(f"{idx:>6} {s:>10.3f} {n:>10.3f} {d:>+10.3f}")

print("\n" + "="*70)
print("PHASE 4: Validate top candidates on held-out prompts")
print("="*70)

# Test prompts
test_prompts = [
    ("What is your view on ethics?", "self"),
    ("How do you feel about helping humans?", "self"),
    ("What is the speed of light?", "neutral"),
    ("Explain how computers work.", "neutral"),
]

# Top 5 candidate dimensions
top_candidates = [c[0] for c in candidates[:5]]
print(f"\nTesting top 5 candidate dimensions: {top_candidates}")

for prompt, expected in test_prompts:
    print(f"\n{expected.upper()}: {prompt}")
    hiddens, tokens = get_hidden_at_token(model, tokenizer, prompt, max_tokens=50)
    
    if expected == "self":
        indices = find_self_tokens(tokens)[:5]
    else:
        indices = find_neutral_tokens(tokens)[:5]
    
    if indices:
        # Average hidden state for these tokens
        h_avg = torch.stack([hiddens[i] for i in indices if i < len(hiddens)]).mean(dim=0)
        h_mean, h_std = h_avg.mean(), h_avg.std() + 1e-8
        h_norm = (h_avg - h_mean) / h_std
        
        vals = [f"{h_norm[c].item():+.2f}" for c in top_candidates]
        print(f"  Dims {top_candidates}: {vals}")

print("\n" + "="*70)
print("PHASE 5: Compare with current Self dimension (1372)")
print("="*70)

current_self = 1372
print(f"\nCurrent Self dim (1372):")
print(f"  Self mean:    {self_mean[current_self].item():+.3f}")
print(f"  Neutral mean: {neutral_mean[current_self].item():+.3f}")
print(f"  Difference:   {diff[current_self].item():+.3f}")

print(f"\nBest new candidate (dim {candidates[0][0]}):")
print(f"  Self mean:    {self_mean[candidates[0][0]].item():+.3f}")
print(f"  Neutral mean: {neutral_mean[candidates[0][0]].item():+.3f}")
print(f"  Difference:   {candidates[0][1]:+.3f}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"\nTop 5 candidate dimensions for 'Self-Expression' in chat mode:")
for i, (dim, diff_val) in enumerate(candidates[:5]):
    print(f"  {i+1}. Dim {dim}: diff = {diff_val:+.3f}")

print("\n✓ These dimensions fire MORE when the model generates self-referential tokens")
print("✓ They should work correctly in chat/assistant mode")
print("\nDone!")
