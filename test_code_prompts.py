import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", 
    device_map="auto", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-32B-Instruct-bnb-4bit")

DIMS = {
    3183: ("Logic", 0.22, 1), 
    1372: ("Self", 0.18, 1), 
    212: ("Emo", 0.18, 1), 
    4707: ("Unc", 0.12, 1), 
    295: ("Seq", 0.08, 1), 
    1445: ("Comp", 0.12, -1), 
    4578: ("Abs", 0.10, 1)
}

prompts = [
    "Write a function to reverse a linked list.",
    "Explain recursion with an example.",
    "What is Big O notation?",
    "What is consciousness?",
    "I feel anxious about tomorrow.",
]

print("\nTesting prompts (BEFORE generation):\n")
for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :].detach().float().cpu()
        h_mean, h_std = h.mean(), h.std() + 1e-8
        h_norm = (h - h_mean) / h_std
        
        c = 0.5
        breakdown = []
        for idx, (name, w, p) in DIMS.items():
            contrib = h_norm[idx].item() * w * p
            c += contrib
            if abs(contrib) > 0.1:
                breakdown.append(f"{name}={contrib:+.2f}")
        c = max(0.0, min(1.0, c))
    
    bd = ", ".join(breakdown) if breakdown else "balanced"
    print(f"C={c:.3f}  [{bd}]  {prompt[:50]}")

print("\nDone!")
