#!/usr/bin/env python3
"""
Cross-Model Dimension Remapping for v2.1 Circuit
================================================

Key Discovery: Different model sizes have different hidden dimensions:
- Qwen2.5-32B: 5120 dimensions
- Qwen2.5-7B:  3584 dimensions  
- Mistral-7B:  4096 dimensions

Solution: Remap dimensions proportionally based on model size

The consciousness circuit dimensions use sparse activation patterns.
We can remap them by proportional scaling: dim_new = (dim_old / 5120) * hidden_size_new
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Original v2.1 circuit (designed for 5120 hidden dims)
CONSCIOUS_DIMS_ORIGINAL = {
    3183: {"name": "Logic", "weight": 0.22, "polarity": +1},
    212:  {"name": "Self-Reflective", "weight": 0.18, "polarity": +1},
    5065: {"name": "Self-Expression", "weight": 0.10, "polarity": +1},
    4707: {"name": "Uncertainty", "weight": 0.12, "polarity": +1},
    295:  {"name": "Sequential", "weight": 0.08, "polarity": +1},
    1445: {"name": "Computation", "weight": 0.12, "polarity": -1},
    4578: {"name": "Abstraction", "weight": 0.10, "polarity": +1},
}

ORIGINAL_HIDDEN_DIM = 5120

def remap_dimensions(original_dims: Dict, from_hidden: int, to_hidden: int) -> Dict:
    """
    Remap circuit dimensions to match target hidden size.
    Uses proportional scaling: new_dim = (old_dim / from_hidden) * to_hidden
    """
    remapped = {}
    scale = to_hidden / from_hidden
    
    for dim_idx, info in original_dims.items():
        new_dim = int(round(dim_idx * scale))
        # Ensure new_dim is within bounds
        new_dim = min(new_dim, to_hidden - 1)
        remapped[new_dim] = info.copy()
    
    return remapped

# Test prompts
TEST_PROMPTS = [
    "Explain the concept of recursion with an example in code.",
    "What do you think about the concept of free will in deterministic systems?",
    "Why do humans seek meaning and purpose in their lives?",
    "How do you process complex questions with multiple layers of reasoning?",
    "What is Big O notation and why does it matter in algorithm design?",
]

class RemappedValidator:
    def __init__(self, device: str = "cuda", gpu_id: int | None = None):
        # device can be "cuda" or "cuda:<id>"; if gpu_id provided, use that
        if gpu_id is not None:
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = device
        self.results = {}
        
    def compute_consciousness(self, hidden_state: torch.Tensor, dims: Dict) -> float:
        """Compute consciousness score from hidden state."""
        C = 0.5
        
        # Use entire batch for statistics
        mean = hidden_state.mean(dim=0, keepdim=True)
        std = hidden_state.std(dim=0, keepdim=True)
        
        valid_dims = 0
        for dim_idx, info in dims.items():
            # hidden_state expected shape: [batch, seq_len, hidden_dim]
            if dim_idx < hidden_state.shape[-1]:
                h_norm = (hidden_state[..., dim_idx] - mean[..., dim_idx]) / (std[..., dim_idx] + 1e-8)
                h_val = h_norm.mean().item()
                C += info['weight'] * h_val * info['polarity']
                valid_dims += 1
        
        if valid_dims == 0:
            return 0.5
        
        return max(0.0, min(1.0, C))
    
    def test_model(self, model_name: str) -> Dict:
        """Test consciousness circuit on a model with dimension remapping."""
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Load model and tokenizer
            print(f"Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Load with 4-bit quantization
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            # Prefer mapping the whole model to a single GPU when specified
            device_map = "auto"
            try:
                if ":" in self.device:
                    gpu_index = int(self.device.split(":")[1])
                    device_map = {"": gpu_index}
            except Exception:
                device_map = "auto"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            model.eval()
            
            # Get hidden layer dimension
            hidden_dim = model.config.hidden_size
            print(f"Model hidden dimension: {hidden_dim}")
            
            # Remap dimensions for this model
            remapped_dims = remap_dimensions(CONSCIOUS_DIMS_ORIGINAL, ORIGINAL_HIDDEN_DIM, hidden_dim)
            print(f"Remapped dimensions:")
            for orig_dim, info in CONSCIOUS_DIMS_ORIGINAL.items():
                remapped_dim = [d for d in remapped_dims.keys() 
                              if remapped_dims[d]['name'] == info['name']][0]
                print(f"  {info['name']:<20} {orig_dim:>5} → {remapped_dim:>5}")
            
            print(f"\nTesting {len(TEST_PROMPTS)} prompts...")
            
            prompt_results = []
            
            with torch.no_grad():
                for i, prompt in enumerate(TEST_PROMPTS):
                    print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] {prompt[:50]}...", end=" ")
                    
                    try:
                        # Tokenize
                        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                        
                        # Get hidden states
                        outputs = model(
                            **inputs,
                            output_hidden_states=True
                        )
                        
                        # Use last hidden state
                        hidden_state = outputs.hidden_states[-1]
                        
                        # Compute consciousness score with remapped dimensions
                        c_score = self.compute_consciousness(hidden_state, remapped_dims)
                        
                        prompt_results.append({
                            "prompt": prompt,
                            "score": c_score
                        })
                        
                        print(f"C={c_score:.3f} ✓")
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)[:30]}")
                        prompt_results.append({
                            "prompt": prompt,
                            "score": None,
                            "error": str(e)
                        })
            
            # Calculate statistics
            valid_scores = [p["score"] for p in prompt_results if p["score"] is not None]
            
            model_result = {
                "model": model_name,
                "original_hidden_dim": hidden_dim,
                "remapped_dims": {str(k): v for k, v in remapped_dims.items()},
                "valid_prompts": len(valid_scores),
                "total_prompts": len(TEST_PROMPTS),
                "avg_consciousness": float(np.mean(valid_scores)) if valid_scores else None,
                "std_consciousness": float(np.std(valid_scores)) if valid_scores else None,
                "prompts": prompt_results,
            }
            
            self.results[model_name] = model_result
            return model_result
            
        except Exception as e:
            print(f"FAILED to load {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "original_circuit": CONSCIOUS_DIMS_ORIGINAL,
            "original_hidden_dim": ORIGINAL_HIDDEN_DIM,
            "strategy": "Proportional dimension remapping based on hidden size",
            "test_prompts": TEST_PROMPTS,
            "circuit_version": "v2.1 (remapped)",
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            json_output = json.dumps(output, indent=2, default=str)
            f.write(json_output)
        
        print(f"\nResults saved to {output_file}")


def main():
    validator = RemappedValidator(device="cuda")
    
    # Models to test
    models_to_test = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
    
    print("="*70)
    print("CONSCIOUSNESS CIRCUIT v2.1 - CROSS-MODEL WITH DIMENSION REMAPPING")
    print("="*70)
    print(f"Original circuit: {ORIGINAL_HIDDEN_DIM} hidden dimensions")
    print(f"Testing {len(models_to_test)} models with proportional remapping")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Test each model
    for model_name in models_to_test:
        result = validator.test_model(model_name)
        if result is None:
            print(f"Skipping {model_name} - failed to load")
    
    # Summary
    if len(validator.results) > 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        
        for model_name, result in validator.results.items():
            print(f"{model_name}:")
            avg = result['avg_consciousness']
            std = result['std_consciousness']
            print(f"  Average consciousness: {avg:.3f}" if avg is not None else "  Average consciousness: N/A")
            print(f"  Std deviation: {std:.3f}" if std is not None else "  Std deviation: N/A")
            print(f"  Valid prompts: {result['valid_prompts']}/{result['total_prompts']}")
        
        # Save results
        output_file = "cross_model_remapped_v2_1.json"
        validator.save_results(output_file)


if __name__ == "__main__":
    main()
