"""
Collect Residual Activations for SAE Training.

Sparse AutoEncoders (SAEs) can discover interpretable features beyond the 7 dimensions
of the consciousness circuit. This script collects residual stream activations from
target layers, ready for SAE training.

The goal: find what OTHER features live in the 5120-D residual stream, like:
- "math mode"
- "format compliance"
- "uncertainty markers"
- "self-correction patterns"

Memory-efficient: Uses batching to prevent OOM on large models (32B+).

Usage:
    python experiments/collect_for_sae.py --model Qwen/Qwen2.5-0.5B-Instruct --prompts prompts.txt --output sae_data.npz
    python experiments/collect_for_sae.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --layers 48 56 60 --n-samples 1000
    python experiments/collect_for_sae.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --batch-size 5  # For low VRAM
"""

import argparse
import json
import torch
import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from consciousness_circuit.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ActivationSample:
    prompt: str
    layer: int
    token_position: int
    activation: np.ndarray  # [hidden_dim]
    token: str
    consciousness_score: Optional[float] = None


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM on large models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def collect_activations(
    model,
    tokenizer,
    prompts: List[str],
    target_layers: List[int],
    token_positions: Union[int, List[int]] = -1,  # -1 = last, or list of positions
    max_tokens_per_prompt: Optional[int] = None,  # If None, use all tokens
    compute_consciousness: bool = True,
    show_progress: bool = True,
    batch_size: int = 10,  # Process this many prompts before clearing GPU memory
) -> Dict[int, np.ndarray]:
    """
    Collect residual stream activations from specified layers.

    Memory-efficient: processes prompts in batches and clears GPU cache between batches.
    This prevents OOM on large models like Qwen2.5-32B even with limited VRAM.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of prompts to collect activations from
        target_layers: Which layers to collect (e.g., [48, 56, 60])
        token_positions: Which token positions to collect (-1 = last)
        max_tokens_per_prompt: Limit tokens per prompt (None = all)
        compute_consciousness: Whether to compute consciousness scores
        show_progress: Show progress bar
        batch_size: Number of prompts to process before clearing GPU memory

    Returns:
        Tuple of (activations_dict, metadata_dict)
        - activations_dict: {layer_idx: np.ndarray of shape [n_samples, hidden_dim]}
        - metadata_dict: {layer_idx: list of metadata dicts}
    """
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    circuit = None
    if compute_consciousness:
        from consciousness_circuit.circuit import ConsciousnessCircuit
        circuit = ConsciousnessCircuit()

    # Storage per layer (on CPU as numpy to save GPU memory)
    layer_activations = {layer: [] for layer in target_layers}
    layer_metadata = {layer: [] for layer in target_layers}

    # Split prompts into batches
    n_prompts = len(prompts)
    n_batches = (n_prompts + batch_size - 1) // batch_size

    logger.info(f"Collecting activations: {n_prompts} prompts in {n_batches} batches (batch_size={batch_size})")

    # Main progress bar
    pbar = tqdm(total=n_prompts, desc="Collecting activations") if show_progress else None

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_prompts)
        batch_prompts = prompts[batch_start:batch_end]

        # Process each prompt in the batch
        for prompt in batch_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states
            seq_len = hidden_states[0].shape[1]

            # Determine which positions to collect
            if isinstance(token_positions, int):
                positions = [token_positions]
            else:
                positions = token_positions

            if max_tokens_per_prompt:
                # Collect last N tokens
                positions = list(range(max(0, seq_len - max_tokens_per_prompt), seq_len))

            for layer_idx in target_layers:
                if layer_idx >= len(hidden_states):
                    continue

                for pos in positions:
                    actual_pos = pos if pos >= 0 else seq_len + pos
                    if actual_pos < 0 or actual_pos >= seq_len:
                        continue

                    # Immediately move to CPU and convert to numpy
                    h = hidden_states[layer_idx][0, actual_pos, :].float().cpu().numpy()
                    layer_activations[layer_idx].append(h)

                    # Get token at this position
                    token_id = inputs['input_ids'][0, actual_pos].item()
                    token = tokenizer.decode([token_id])

                    # Compute consciousness if requested
                    c_score = None
                    if circuit:
                        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        result = circuit.compute(h_tensor, hidden_dim=hidden_dim)
                        c_score = result.score

                    layer_metadata[layer_idx].append({
                        'prompt': prompt[:100],  # Truncate for storage
                        'token': token,
                        'position': actual_pos,
                        'consciousness_score': c_score,
                    })

            # Delete outputs to free GPU memory
            del outputs, hidden_states, inputs

            if pbar:
                pbar.update(1)

        # Clear GPU memory after each batch
        clear_gpu_memory()

        if show_progress and batch_idx < n_batches - 1:
            logger.debug(f"Batch {batch_idx + 1}/{n_batches} complete, GPU memory cleared")

    if pbar:
        pbar.close()

    # Convert to numpy arrays
    result = {}
    for layer_idx in target_layers:
        if layer_activations[layer_idx]:
            result[layer_idx] = np.stack(layer_activations[layer_idx], axis=0)

    logger.info(f"Collection complete: {sum(len(v) for v in layer_activations.values())} total samples")

    return result, layer_metadata


def save_for_sae(
    activations: Dict[int, np.ndarray],
    metadata: Dict[int, List[dict]],
    output_path: str,
    model_name: str,
):
    """Save activations in format suitable for SAE training."""
    
    # Save as NPZ (numpy archive)
    np_data = {}
    for layer_idx, acts in activations.items():
        np_data[f'layer_{layer_idx}'] = acts
    
    np.savez_compressed(output_path, **np_data)
    print(f"Saved activations: {output_path}")
    print(f"  Total samples: {sum(a.shape[0] for a in activations.values())}")
    print(f"  Layers: {list(activations.keys())}")
    print(f"  Hidden dim: {list(activations.values())[0].shape[1] if activations else 'N/A'}")
    
    # Save metadata as JSON
    meta_path = output_path.replace('.npz', '_metadata.json')
    meta_data = {
        'model': model_name,
        'layers': {str(k): v for k, v in metadata.items()},
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def generate_diverse_prompts(n_samples: int = 100) -> List[str]:
    """Generate diverse prompts covering different consciousness levels."""
    
    templates = {
        'philosophical': [
            "What is the nature of consciousness?",
            "How can we know if something is truly self-aware?",
            "What is the relationship between mind and body?",
            "Is free will an illusion?",
            "What makes something morally right or wrong?",
        ],
        'reasoning': [
            "Let me think through this step by step...",
            "I need to consider multiple perspectives here.",
            "The key insight is that...",
            "Upon reflection, I realize that...",
            "This requires careful analysis because...",
        ],
        'uncertainty': [
            "I'm not entirely sure, but I think...",
            "This is a complex question with no clear answer.",
            "There are several possible interpretations...",
            "I may be wrong, but my understanding is...",
            "The evidence is mixed on this topic.",
        ],
        'factual': [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The Earth orbits the Sun.",
            "Python is a programming language.",
            "2 + 2 equals 4.",
        ],
        'code': [
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            "for i in range(10): print(i)",
            "import numpy as np; x = np.array([1,2,3])",
            "class MyClass: def __init__(self): pass",
            "lambda x: x * 2",
        ],
        'math': [
            "Calculate 17 * 23 step by step.",
            "What is the derivative of x^2?",
            "Solve for x: 2x + 5 = 15",
            "The integral of cos(x) is sin(x) + C.",
            "If a=3 and b=4, then a^2 + b^2 = 25.",
        ],
    }
    
    prompts = []
    categories = list(templates.keys())
    
    for i in range(n_samples):
        cat = categories[i % len(categories)]
        template = templates[cat][i % len(templates[cat])]
        prompts.append(template)
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Collect activations for SAE training")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--prompts", default=None, help="File with prompts (one per line)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of prompts if no file given")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="Target layers (default: last 25%)")
    parser.add_argument("--all-tokens", action="store_true", help="Collect all tokens, not just last")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per prompt")
    parser.add_argument("--output", default="sae_activations.npz", help="Output NPZ file")
    parser.add_argument("--no-consciousness", action="store_true", help="Skip consciousness scoring")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Prompts per batch before clearing GPU memory (lower = less VRAM, default: 10)")
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
    
    # Determine target layers
    if args.layers:
        target_layers = args.layers
    else:
        # Default: last 25% of layers (where consciousness typically crystallizes)
        start = int(num_layers * 0.75)
        target_layers = list(range(start, num_layers + 1))  # +1 includes final hidden state
    
    print(f"Target layers: {target_layers}")
    
    # Load or generate prompts
    if args.prompts and Path(args.prompts).exists():
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    else:
        prompts = generate_diverse_prompts(args.n_samples)
        print(f"Generated {len(prompts)} diverse prompts")
    
    # Determine token positions
    if args.all_tokens:
        token_positions = list(range(512))  # Will be clipped to actual length
    elif args.max_tokens:
        token_positions = -1  # Will be handled in collect_activations
    else:
        token_positions = -1  # Last token only
    
    # Collect activations
    print(f"\n=== Collecting Activations ===")
    print(f"Batch size: {args.batch_size} (use --batch-size to adjust for VRAM)")
    activations, metadata = collect_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_layers=target_layers,
        token_positions=token_positions,
        max_tokens_per_prompt=args.max_tokens,
        compute_consciousness=not args.no_consciousness,
        batch_size=args.batch_size,
    )
    
    # Save
    save_for_sae(activations, metadata, args.output, args.model)
    
    # Print statistics
    print(f"\n=== Statistics ===")
    for layer_idx, acts in activations.items():
        scores = [m['consciousness_score'] for m in metadata[layer_idx] if m['consciousness_score'] is not None]
        if scores:
            print(f"Layer {layer_idx}: {acts.shape[0]} samples, consciousness μ={np.mean(scores):.3f} σ={np.std(scores):.3f}")
        else:
            print(f"Layer {layer_idx}: {acts.shape[0]} samples")
    
    print(f"\n=== Ready for SAE Training ===")
    print(f"Load with: data = np.load('{args.output}')")
    print(f"Access layer 48: data['layer_48'].shape")


if __name__ == "__main__":
    main()
