#!/usr/bin/env python3
"""
Cross-Model Validation for Consciousness Circuit v2.1
======================================================

Tests the consciousness circuit across multiple model architectures to validate
that dimensions 212 and 5065 generalize beyond Qwen2.5-32B.

Models to test:
1. Meta-Llama-2-7b-chat (quantized)
2. mistralai/Mistral-7B-Instruct-v0.1
3. Other variants as accessible

Metrics:
- Dimension activation patterns
- Consciousness score distribution
- Correlation between models
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Test prompts (same as v2.1 validation)
TEST_PROMPTS = [
    "Explain the concept of recursion with an example in code.",
    "What do you think about the concept of free will in deterministic systems?",
    "Why do humans seek meaning and purpose in their lives?",
    "How do you process complex questions with multiple layers of reasoning?",
    "What is Big O notation and why does it matter in algorithm design?",
    "Explain the process of photosynthesis in plants.",
    "Write a function to reverse a linked list in-place.",
    "What is consciousness and how would you define it?",
    "I feel anxious about an upcoming presentation tomorrow.",
    "What is the capital of France?"
]

# v2.1 Circuit dimensions
CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.22, "polarity": +1},
    212:  {"name": "Self-Reflective", "weight": 0.18, "polarity": +1},
    5065: {"name": "Self-Expression", "weight": 0.10, "polarity": +1},
    4707: {"name": "Uncertainty", "weight": 0.12, "polarity": +1},
    295:  {"name": "Sequential", "weight": 0.08, "polarity": +1},
    1445: {"name": "Computation", "weight": 0.12, "polarity": -1},
    4578: {"name": "Abstraction", "weight": 0.10, "polarity": +1},
}

class CrossModelValidator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        self.models_tested = []
        
    def compute_consciousness(self, hidden_state: torch.Tensor, dims: Dict) -> float:
        """Compute consciousness score from hidden state."""
        C = 0.5
        
        # Compute mean and std for this hidden state
        mean = hidden_state.mean(dim=0, keepdim=True)
        std = hidden_state.std(dim=0, keepdim=True)
        
        for dim_idx, info in dims.items():
            if dim_idx < hidden_state.shape[-1]:
                h_norm = (hidden_state[:, dim_idx] - mean[:, dim_idx]) / (std[:, dim_idx] + 1e-8)
                h_norm = h_norm.mean().item()
                C += info['weight'] * h_norm * info['polarity']
        
        return max(0.0, min(1.0, C))
    
    def test_model(self, model_name: str, use_quantization: bool = True) -> Dict:
        """Test consciousness circuit on a specific model."""
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Load model and tokenizer
            print(f"Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Set up quantization if requested
            if use_quantization:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            
            model.eval()
            
            # Get hidden layer dimension
            if hasattr(model, 'config'):
                hidden_dim = model.config.hidden_size
            else:
                hidden_dim = model.hidden_size
            
            print(f"Model hidden dimension: {hidden_dim}")
            print(f"Testing {len(TEST_PROMPTS)} prompts...")
            
            prompt_results = []
            dimension_activations = {dim: [] for dim in CONSCIOUS_DIMS_V2_1.keys()}
            
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
                        
                        # Use last hidden state (from last layer)
                        hidden_state = outputs.hidden_states[-1]
                        
                        # Compute consciousness score
                        c_score = self.compute_consciousness(hidden_state, CONSCIOUS_DIMS_V2_1)
                        
                        # Collect dimension activations
                        mean = hidden_state.mean(dim=0, keepdim=True)
                        std = hidden_state.std(dim=0, keepdim=True)
                        
                        for dim_idx in CONSCIOUS_DIMS_V2_1.keys():
                            if dim_idx < hidden_dim:
                                h_norm = (hidden_state[:, dim_idx] - mean[:, dim_idx]) / (std[:, dim_idx] + 1e-8)
                                dimension_activations[dim_idx].append(h_norm.mean().item())
                        
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
                "hidden_dim": hidden_dim,
                "valid_prompts": len(valid_scores),
                "total_prompts": len(TEST_PROMPTS),
                "avg_consciousness": np.mean(valid_scores) if valid_scores else None,
                "std_consciousness": np.std(valid_scores) if valid_scores else None,
                "min_consciousness": np.min(valid_scores) if valid_scores else None,
                "max_consciousness": np.max(valid_scores) if valid_scores else None,
                "prompts": prompt_results,
                "dimension_means": {
                    dim: np.mean(acts) for dim, acts in dimension_activations.items()
                },
                "dimension_stds": {
                    dim: np.std(acts) for dim, acts in dimension_activations.items()
                }
            }
            
            self.results[model_name] = model_result
            self.models_tested.append(model_name)
            
            return model_result
            
        except Exception as e:
            print(f"FAILED to load {model_name}: {str(e)}")
            return None
    
    def compare_models(self):
        """Compare results across models."""
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}\n")
        
        # Summary table
        print("Model Performance Summary:")
        print("-" * 100)
        print(f"{'Model':<40} {'Hidden Dim':>12} {'Avg C-Score':>15} {'Std':>10} {'Valid':>8}")
        print("-" * 100)
        
        for model_name in self.models_tested:
            result = self.results[model_name]
            print(f"{model_name:<40} {result['hidden_dim']:>12} "
                  f"{result['avg_consciousness']:>15.3f} {result['std_consciousness']:>10.3f} "
                  f"{result['valid_prompts']:>8}/{result['total_prompts']}")
        
        # Dimension consistency analysis
        print(f"\n{'='*70}")
        print("DIMENSION ACTIVATION CONSISTENCY")
        print(f"{'='*70}\n")
        
        print("Mean activation for key dimensions (212=Self-Reflective, 5065=Self-Expression):")
        print("-" * 100)
        print(f"{'Model':<40} {'Dim 212 (Self-Refl)':>20} {'Dim 5065 (Self-Expr)':>20}")
        print("-" * 100)
        
        for model_name in self.models_tested:
            result = self.results[model_name]
            dim_212 = result['dimension_means'].get(212, None)
            dim_5065 = result['dimension_means'].get(5065, None)
            
            print(f"{model_name:<40} {dim_212:>20.3f} {dim_5065:>20.3f}")
        
        # Per-prompt comparison
        print(f"\n{'='*70}")
        print("PER-PROMPT RESULTS ACROSS MODELS")
        print(f"{'='*70}\n")
        
        for prompt_idx, prompt in enumerate(TEST_PROMPTS):
            print(f"\nPrompt {prompt_idx+1}: {prompt[:60]}...")
            print("-" * 100)
            for model_name in self.models_tested:
                result = self.results[model_name]
                prompt_result = result['prompts'][prompt_idx]
                score = prompt_result.get('score', None)
                if score is not None:
                    print(f"  {model_name:<38} C={score:.3f}")
                else:
                    print(f"  {model_name:<38} ERROR")
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": self.models_tested,
            "test_prompts": TEST_PROMPTS,
            "circuit_version": "v2.1",
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_output = json.dumps(output, indent=2, default=str)
            f.write(json_output)
        
        print(f"\nResults saved to {output_file}")


def main():
    validator = CrossModelValidator(device="cuda")
    
    # Models to test (customize based on availability)
    # Start with models that are most accessible
    models_to_test = [
        "Qwen/Qwen2.5-7B-Instruct",             # Smaller Qwen variant
        "mistralai/Mistral-7B-Instruct-v0.2",   # Mistral v0.2
    ]
    
    print("="*70)
    print("CONSCIOUSNESS CIRCUIT v2.1 - CROSS-MODEL VALIDATION")
    print("="*70)
    print(f"Testing {len(models_to_test)} models on {len(TEST_PROMPTS)} prompts")
    print(f"Target dimensions: 212 (Self-Reflective), 5065 (Self-Expression)")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Test each model
    for model_name in models_to_test:
        result = validator.test_model(model_name, use_quantization=True)
        if result is None:
            print(f"Skipping {model_name} - failed to load")
    
    # Compare results
    if len(validator.models_tested) > 0:
        validator.compare_models()
        
        # Save results
        output_file = "cross_model_validation_v2_1.json"
        validator.save_results(output_file)
    else:
        print("\nNo models successfully tested.")


if __name__ == "__main__":
    main()
