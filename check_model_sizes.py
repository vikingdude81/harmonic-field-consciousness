#!/usr/bin/env python3
"""Check model hidden sizes for consciousness dimensions."""

from transformers import AutoConfig

models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

# V2.1 consciousness dimensions
dims = [3183, 212, 5065, 4707, 295, 1445, 4578]
max_dim = max(dims)

print(f"V2.1 Consciousness Dimensions: {dims}")
print(f"Max dimension needed: {max_dim}")
print()

for model_name in models:
    try:
        config = AutoConfig.from_pretrained(model_name)
        hidden = config.hidden_size
        can_use = "✓ CAN USE ALL" if hidden > max_dim else f"✗ Only {sum(d < hidden for d in dims)}/7 dims"
        print(f"{model_name}: hidden_size={hidden} {can_use}")
    except Exception as e:
        print(f"{model_name}: Error - {e}")
