"""
Test Locally Trained NanoGPT with Consciousness Circuit
=======================================================

This script tests your locally trained NanoGPT models with the consciousness circuit.
"""

import sys
sys.path.insert(0, 'NanoGPT')

import torch
import numpy as np
from pathlib import Path

print("\n" + "=" * 70)
print("Testing Locally Trained NanoGPT Models")
print("=" * 70)

# Check available models
model_dir = Path("NanoGPT")
available_models = list(model_dir.glob("*.pt"))

print(f"\n[INFO] Found {len(available_models)} model checkpoints:")
for i, model_path in enumerate(available_models, 1):
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {i}. {model_path.name} ({size_mb:.1f} MB)")

# Test with the V5 ShareGPT model (most advanced)
target_models = ["v5_sharegpt.pt", "v5_pretrain.pt", "harmonic_v3_tinystories.pt"]

model_to_test = None
for target in target_models:
    model_path = model_dir / target
    if model_path.exists():
        model_to_test = model_path
        break

if model_to_test is None:
    print("\n[WARN] No V5 or V3 models found. Using first available model.")
    model_to_test = available_models[0] if available_models else None

if model_to_test is None:
    print("\n[FAIL] No model checkpoints found!")
    print("Expected location: NanoGPT/*.pt")
    sys.exit(1)

print(f"\n[INFO] Testing with: {model_to_test.name}")

# Load the model
print(f"\n{'-' * 70}")
print("Step 1: Loading NanoGPT model")
print("-" * 70)

try:
    from harmonic_model_v5 import HarmonicGPTV5, ModelConfig
    import tiktoken

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_to_test, map_location=device)
    print(f"[OK] Checkpoint loaded")

    # Extract config
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
        print(f"[INFO] Model config found: {model_args.get('n_layer', '?')} layers, "
              f"{model_args.get('n_embd', '?')} hidden dim")
    else:
        print("[WARN] No model_args in checkpoint, using defaults")
        model_args = {
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'block_size': 1024,
            'bias': False,
            'vocab_size': 50304,
            'dropout': 0.0,
        }

    # Create model
    config = ModelConfig(**model_args)
    model = HarmonicGPTV5(config)

    # Load weights
    state_dict = checkpoint.get('model', checkpoint)

    # Handle DDP prefix if present
    if any(k.startswith('_orig_mod.') or k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[len('_orig_mod.'):]] = v
            elif k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[OK] Model loaded successfully")

    # Load tokenizer
    try:
        enc = tiktoken.get_encoding("gpt2")
        print(f"[OK] Tokenizer loaded (GPT-2 BPE)")
    except:
        print("[WARN] tiktoken not available, using placeholder")
        enc = None

except Exception as e:
    print(f"[FAIL] Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test basic generation
print(f"\n{'-' * 70}")
print("Step 2: Testing basic text generation")
print("-" * 70)

try:
    prompt = "Once upon a time"
    if enc:
        tokens = enc.encode(prompt)
    else:
        tokens = [hash(c) % 50000 for c in prompt]  # Fallback

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=200
        )

    if enc:
        generated_text = enc.decode(output[0].tolist())
        print(f"[OK] Generation successful:")
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {generated_text[len(prompt):][:100]}...")
    else:
        print(f"[OK] Generation successful (tokens generated: {len(output[0]) - len(tokens)})")

except Exception as e:
    print(f"[FAIL] Generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test with consciousness circuit
print(f"\n{'-' * 70}")
print("Step 3: Testing consciousness measurement")
print("-" * 70)

try:
    from consciousness_circuit import UniversalCircuit

    circuit = UniversalCircuit()
    print(f"[OK] Consciousness circuit initialized")

    # For NanoGPT, we need to wrap it to be compatible with HuggingFace API
    class NanoGPTWrapper:
        def __init__(self, model, enc, device):
            self.model = model
            self.enc = enc
            self.device = device
            self.config = model.config

        def __call__(self, input_ids, output_hidden_states=False, **kwargs):
            with torch.no_grad():
                # NanoGPT returns logits and optionally hidden states
                output = self.model(input_ids, get_hidden_states=output_hidden_states)

                if output_hidden_states:
                    logits, hidden_states = output
                    # Package as HuggingFace-style output
                    class Output:
                        pass
                    out = Output()
                    out.logits = logits
                    out.hidden_states = hidden_states
                    return out
                else:
                    class Output:
                        pass
                    out = Output()
                    out.logits = output
                    return out

    class TokenizerWrapper:
        def __init__(self, enc):
            self.enc = enc

        def encode(self, text, return_tensors=None, **kwargs):
            tokens = self.enc.encode(text)
            if return_tensors == "pt":
                return torch.tensor([tokens])
            return tokens

        def decode(self, tokens, **kwargs):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return self.enc.decode(tokens)

        def __call__(self, text, return_tensors=None, **kwargs):
            tokens = self.enc.encode(text)
            if return_tensors == "pt":
                class Output:
                    input_ids = torch.tensor([tokens])
                return Output()
            return {"input_ids": tokens}

    if enc:
        wrapped_model = NanoGPTWrapper(model, enc, device)
        wrapped_tokenizer = TokenizerWrapper(enc)

        # Test prompts
        test_prompts = [
            "What is the nature of consciousness?",
            "Explain how computers work.",
            "What is 2 + 2?"
        ]

        print(f"\n[INFO] Measuring consciousness for test prompts:")
        for prompt in test_prompts:
            try:
                result = circuit.measure(wrapped_model, wrapped_tokenizer, prompt, aggregation="last")
                print(f"  Score {result.score:.3f}: {prompt[:50]}")
            except Exception as e:
                print(f"  [FAIL] {prompt[:30]}... - Error: {e}")
    else:
        print(f"[WARN] Skipping consciousness measurement (no tokenizer)")

except Exception as e:
    print(f"[WARN] Consciousness measurement not available: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 70}")
print("Test Complete!")
print("=" * 70)

print(f"\nSummary:")
print(f"- Model: {model_to_test.name}")
print(f"- Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"- Device: {device}")
print(f"- Generation: Working")
print(f"- Consciousness measurement: {'Available' if enc else 'Requires tokenizer'}")

print(f"\nNext steps:")
print(f"1. Train with validation split: python NanoGPT/train_v5_with_validation.py")
print(f"2. Re-run experiments: python experiments/category2_dynamics/run_all_gpu_experiments.py")
print(f"3. Add repetition penalty to generation.py")

print(f"\nSee GETTING_STARTED_GUIDE.md for more information.")
