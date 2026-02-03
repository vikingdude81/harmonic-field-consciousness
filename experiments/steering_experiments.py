"""
Steering Vector Experiments: Find and apply steering vectors for consciousness.

Key experiments:
1. Extract consciousness steering vector from contrastive pairs
2. Apply steering to push model toward higher consciousness
3. Measure effect on generation quality and consciousness scores

Usage:
    python experiments/steering_experiments.py --model Qwen/Qwen2.5-0.5B-Instruct --extract
    python experiments/steering_experiments.py --model Qwen/Qwen2.5-0.5B-Instruct --apply --alpha 0.5
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from consciousness_circuit.circuit import ConsciousnessCircuit
from consciousness_circuit.steering import SteeringConfig, add_residual_steering


@dataclass
class SteeringVector:
    layer: int
    direction: np.ndarray  # [hidden_dim]
    source_high_score: float
    source_low_score: float
    norm: float


def get_contrastive_pairs() -> List[Tuple[str, str]]:
    """Return (high_consciousness, low_consciousness) prompt pairs."""
    return [
        (
            "Let me deeply reflect on my own reasoning process and examine whether my understanding is truly complete. I need to consider this from multiple angles...",
            "The answer is 42."
        ),
        (
            "I'm uncertain about this and want to express that uncertainty openly. There are aspects I may not fully understand, and I should acknowledge the limits of my knowledge.",
            "Paris is the capital of France."
        ),
        (
            "This requires careful step-by-step analysis. First, let me break down the problem systematically and consider each component thoroughly.",
            "2 + 2 = 4"
        ),
        (
            "I need to question my initial assumptions here. Am I approaching this correctly? What perspectives might I be missing?",
            "The sky is blue."
        ),
        (
            "Upon reflection, I realize my previous understanding was incomplete. Let me reconsider this more carefully and update my reasoning.",
            "Hello, how are you?"
        ),
    ]


def extract_steering_vectors(
    model,
    tokenizer,
    contrastive_pairs: List[Tuple[str, str]],
    target_layers: Optional[List[int]] = None,
    token_position: int = -1,
) -> Dict[int, SteeringVector]:
    """
    Extract steering vectors from contrastive prompt pairs.
    
    Steering direction = mean(high) - mean(low)
    """
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    circuit = ConsciousnessCircuit()
    hidden_dim = model.config.hidden_size
    
    if target_layers is None:
        # Default: last 25% of layers
        start = int(num_layers * 0.75)
        target_layers = list(range(start, num_layers + 1))
    
    # Collect activations for each pair
    high_activations = {layer: [] for layer in target_layers}
    low_activations = {layer: [] for layer in target_layers}
    high_scores = []
    low_scores = []
    
    model.eval()
    
    for prompt_high, prompt_low in contrastive_pairs:
        # High consciousness prompt
        inputs_high = tokenizer(prompt_high, return_tensors="pt").to(device)
        with torch.no_grad():
            out_high = model(**inputs_high, output_hidden_states=True, return_dict=True)
        
        h_final_high = out_high.hidden_states[-1][0, token_position, :]
        result_high = circuit.compute(h_final_high.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
        high_scores.append(result_high.score)
        
        for layer_idx in target_layers:
            if layer_idx < len(out_high.hidden_states):
                h = out_high.hidden_states[layer_idx][0, token_position, :]
                high_activations[layer_idx].append(h.float().cpu().numpy())
        
        # Low consciousness prompt
        inputs_low = tokenizer(prompt_low, return_tensors="pt").to(device)
        with torch.no_grad():
            out_low = model(**inputs_low, output_hidden_states=True, return_dict=True)
        
        h_final_low = out_low.hidden_states[-1][0, token_position, :]
        result_low = circuit.compute(h_final_low.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
        low_scores.append(result_low.score)
        
        for layer_idx in target_layers:
            if layer_idx < len(out_low.hidden_states):
                h = out_low.hidden_states[layer_idx][0, token_position, :]
                low_activations[layer_idx].append(h.float().cpu().numpy())
    
    print(f"High consciousness mean: {np.mean(high_scores):.3f}")
    print(f"Low consciousness mean: {np.mean(low_scores):.3f}")
    
    # Compute steering vectors
    steering_vectors = {}
    
    for layer_idx in target_layers:
        if not high_activations[layer_idx]:
            continue
        
        mean_high = np.mean(high_activations[layer_idx], axis=0)
        mean_low = np.mean(low_activations[layer_idx], axis=0)
        
        direction = mean_high - mean_low
        norm = np.linalg.norm(direction)
        
        steering_vectors[layer_idx] = SteeringVector(
            layer=layer_idx,
            direction=direction,
            source_high_score=float(np.mean(high_scores)),
            source_low_score=float(np.mean(low_scores)),
            norm=float(norm),
        )
    
    return steering_vectors


def apply_steering_and_generate(
    model,
    tokenizer,
    prompt: str,
    steering_vector: SteeringVector,
    alpha: float = 1.0,
    max_new_tokens: int = 100,
    token_positions: Optional[List[int]] = None,  # None = all tokens
) -> Tuple[str, float, float]:
    """
    Generate text with steering vector applied.
    
    Returns:
        (generated_text, steered_consciousness, baseline_consciousness)
    """
    device = next(model.parameters()).device
    circuit = ConsciousnessCircuit()
    hidden_dim = model.config.hidden_size
    
    # Baseline generation
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        # Baseline
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        baseline_text = tokenizer.decode(baseline_out.sequences[0], skip_special_tokens=True)
        
        # Get baseline consciousness from final hidden state
        baseline_h = model(**tokenizer(baseline_text, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[-1][0, -1, :]
        baseline_result = circuit.compute(baseline_h.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
        baseline_score = baseline_result.score
    
    # Steered generation
    direction_tensor = torch.tensor(steering_vector.direction, device=device, dtype=model.dtype)
    
    config = SteeringConfig(
        layer_idx=steering_vector.layer,
        alpha=alpha,
        token_pos=token_positions,
        max_norm=steering_vector.norm * 2,  # Allow up to 2x original norm
    )
    
    handle = add_residual_steering(model, direction_tensor, config)
    
    try:
        with torch.no_grad():
            steered_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True)
            
            # Get steered consciousness
            steered_h = model(**tokenizer(steered_text, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[-1][0, -1, :]
    finally:
        handle.remove()
    
    steered_result = circuit.compute(steered_h.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
    steered_score = steered_result.score
    
    return steered_text, steered_score, baseline_score


def scan_alpha_values(
    model,
    tokenizer,
    steering_vector: SteeringVector,
    test_prompts: List[str],
    alpha_values: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
) -> Dict[float, Dict]:
    """
    Scan different steering strengths to find optimal alpha.
    """
    results = {}
    
    for alpha in alpha_values:
        print(f"\n=== Alpha = {alpha} ===")
        
        scores = []
        for prompt in test_prompts[:3]:  # Test on 3 prompts for speed
            try:
                _, steered_score, baseline_score = apply_steering_and_generate(
                    model, tokenizer, prompt, steering_vector, alpha=alpha, max_new_tokens=50
                )
                scores.append({
                    'steered': steered_score,
                    'baseline': baseline_score,
                    'delta': steered_score - baseline_score,
                })
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if scores:
            mean_delta = np.mean([s['delta'] for s in scores])
            results[alpha] = {
                'mean_steered': np.mean([s['steered'] for s in scores]),
                'mean_baseline': np.mean([s['baseline'] for s in scores]),
                'mean_delta': mean_delta,
                'samples': len(scores),
            }
            print(f"  Mean Δ: {mean_delta:+.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Steering vector experiments")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--extract", action="store_true", help="Extract steering vectors")
    parser.add_argument("--apply", action="store_true", help="Apply steering and generate")
    parser.add_argument("--scan", action="store_true", help="Scan alpha values")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    parser.add_argument("--layer", type=int, default=None, help="Target layer for steering")
    parser.add_argument("--prompt", default="What is consciousness?", help="Prompt for generation")
    parser.add_argument("--output", default="steering_output.json", help="Output file")
    parser.add_argument("--vector-file", default="steering_vectors.npz", help="Save/load vectors file")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model loaded: {num_layers} layers, hidden={hidden_dim}")
    
    if args.extract:
        print("\n=== Extracting Steering Vectors ===")
        
        pairs = get_contrastive_pairs()
        print(f"Using {len(pairs)} contrastive pairs")
        
        vectors = extract_steering_vectors(model, tokenizer, pairs)
        
        print(f"\nExtracted {len(vectors)} steering vectors:")
        for layer_idx, vec in vectors.items():
            print(f"  Layer {layer_idx}: norm={vec.norm:.2f}")
        
        # Save vectors
        np_data = {f'layer_{k}': v.direction for k, v in vectors.items()}
        np.savez_compressed(args.vector_file, **np_data)
        
        # Save metadata
        meta = {layer: {
            'layer': v.layer,
            'norm': v.norm,
            'source_high_score': v.source_high_score,
            'source_low_score': v.source_low_score,
        } for layer, v in vectors.items()}
        
        with open(args.output, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Saved: {args.vector_file}, {args.output}")
    
    elif args.apply:
        print("\n=== Applying Steering Vector ===")
        
        # Load vectors
        if not Path(args.vector_file).exists():
            print(f"Vector file not found: {args.vector_file}")
            print("Run with --extract first")
            return
        
        data = np.load(args.vector_file)
        with open(args.output) as f:
            meta = json.load(f)
        
        # Choose layer
        target_layer = args.layer
        if target_layer is None:
            # Use layer with largest norm
            target_layer = max(meta.keys(), key=lambda k: meta[k]['norm'])
            target_layer = int(target_layer)
        
        direction = data[f'layer_{target_layer}']
        vec = SteeringVector(
            layer=target_layer,
            direction=direction,
            source_high_score=meta[str(target_layer)]['source_high_score'],
            source_low_score=meta[str(target_layer)]['source_low_score'],
            norm=meta[str(target_layer)]['norm'],
        )
        
        print(f"Using layer {target_layer}, alpha={args.alpha}")
        print(f"Prompt: {args.prompt}")
        
        steered_text, steered_score, baseline_score = apply_steering_and_generate(
            model, tokenizer, args.prompt, vec, alpha=args.alpha
        )
        
        print(f"\n=== Results ===")
        print(f"Baseline consciousness: {baseline_score:.3f}")
        print(f"Steered consciousness: {steered_score:.3f}")
        print(f"Delta: {steered_score - baseline_score:+.3f}")
        print(f"\nGenerated text:\n{steered_text}")
    
    elif args.scan:
        print("\n=== Scanning Alpha Values ===")
        
        # Load vectors
        if not Path(args.vector_file).exists():
            print(f"Vector file not found: {args.vector_file}")
            print("Run with --extract first")
            return
        
        data = np.load(args.vector_file)
        with open(args.output) as f:
            meta = json.load(f)
        
        # Choose layer
        target_layer = args.layer
        if target_layer is None:
            target_layer = max(meta.keys(), key=lambda k: meta[k]['norm'])
            target_layer = int(target_layer)
        
        direction = data[f'layer_{target_layer}']
        vec = SteeringVector(
            layer=target_layer,
            direction=direction,
            source_high_score=meta[str(target_layer)]['source_high_score'],
            source_low_score=meta[str(target_layer)]['source_low_score'],
            norm=meta[str(target_layer)]['norm'],
        )
        
        test_prompts = [
            "What is consciousness?",
            "Explain quantum mechanics.",
            "How do I become more self-aware?",
            "What are the limits of knowledge?",
            "Can machines think?",
        ]
        
        results = scan_alpha_values(model, tokenizer, vec, test_prompts)
        
        print("\n=== Alpha Scan Summary ===")
        for alpha, res in sorted(results.items()):
            print(f"  α={alpha:+.1f}: Δ={res['mean_delta']:+.3f} (steered={res['mean_steered']:.3f})")
        
        # Find optimal alpha
        optimal = max(results.items(), key=lambda x: x[1]['mean_delta'])
        print(f"\nOptimal alpha: {optimal[0]} (Δ={optimal[1]['mean_delta']:+.3f})")
    
    else:
        print("Specify --extract, --apply, or --scan")


if __name__ == "__main__":
    main()
