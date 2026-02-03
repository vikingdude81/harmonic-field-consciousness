"""
Consciousness Patching: Causal intervention using consciousness score as target metric.

Instead of patching for logit changes, we patch for CONSCIOUSNESS changes.
This reveals which layers causally control the consciousness signature.

Key questions:
- Which layers, when patched from high→low consciousness prompt, tank the score?
- Is there a "consciousness decision band" in late-middle layers?
- Does self-reflective come from different layers than computational?

Usage:
    python experiments/consciousness_patching.py --model Qwen/Qwen2.5-0.5B-Instruct
    python experiments/consciousness_patching.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from consciousness_circuit.circuit import ConsciousnessCircuit


@dataclass 
class PatchResult:
    layer: int
    original_score: float
    patched_score: float
    delta: float  # patched - original
    dimension_deltas: Dict[str, float]


def run_consciousness_patching(
    model,
    tokenizer,
    prompt_high: str,  # High consciousness prompt
    prompt_low: str,   # Low consciousness prompt
    layers_to_test: Optional[List[int]] = None,
    token_position: int = -1,
) -> List[PatchResult]:
    """
    Patch residual stream from low→high consciousness prompt.
    
    Process:
    1. Run high prompt, cache hidden states
    2. Run low prompt, cache hidden states
    3. For each layer: replace low's residual with high's, measure consciousness
    4. Compare to baseline (unpatched high)
    
    Returns:
        List of PatchResult showing impact of each layer patch
    """
    device = next(model.parameters()).device
    circuit = ConsciousnessCircuit()
    hidden_dim = model.config.hidden_size
    
    # Tokenize both prompts
    inputs_high = tokenizer(prompt_high, return_tensors="pt").to(device)
    inputs_low = tokenizer(prompt_low, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        # Get hidden states for both
        out_high = model(**inputs_high, output_hidden_states=True, return_dict=True)
        out_low = model(**inputs_low, output_hidden_states=True, return_dict=True)
    
    hidden_high = out_high.hidden_states  # tuple of [batch, seq, hidden]
    hidden_low = out_low.hidden_states
    num_layers = len(hidden_high)
    
    if layers_to_test is None:
        layers_to_test = list(range(num_layers))
    
    # Baseline: high prompt consciousness at last layer
    h_high_final = hidden_high[-1][0, token_position, :]
    baseline_result = circuit.compute(h_high_final.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
    baseline_score = baseline_result.score
    baseline_dims = baseline_result.dimension_contributions
    
    # Low prompt baseline
    h_low_final = hidden_low[-1][0, token_position, :]
    low_result = circuit.compute(h_low_final.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
    low_score = low_result.score
    
    print(f"High prompt baseline: {baseline_score:.3f}")
    print(f"Low prompt baseline: {low_score:.3f}")
    print(f"Gap: {baseline_score - low_score:.3f}")
    
    results = []
    
    for layer_idx in layers_to_test:
        # Create patched hidden state:
        # Use high prompt's residual at this layer, then continue with high's residuals
        # This is simplified - real patching would re-run forward from patched layer
        
        # For now: simulate by interpolating at this layer position
        # More accurate: use hooks to replace during forward pass
        
        h_patched = hidden_high[-1][0, token_position, :].clone()
        
        # Compute how much of this layer's contribution we're removing
        layer_contribution_high = hidden_high[layer_idx][0, token_position, :] 
        layer_contribution_low = hidden_low[layer_idx][0, token_position, :]
        
        # Approximate patching: replace high's layer contribution with low's
        # This is a linear approximation
        delta = layer_contribution_high - layer_contribution_low
        h_patched = h_patched - 0.5 * delta  # Attenuated patch
        
        patched_result = circuit.compute(h_patched.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
        
        dim_deltas = {}
        for k in baseline_dims:
            dim_deltas[k] = patched_result.dimension_contributions.get(k, 0) - baseline_dims.get(k, 0)
        
        results.append(PatchResult(
            layer=layer_idx,
            original_score=baseline_score,
            patched_score=patched_result.score,
            delta=patched_result.score - baseline_score,
            dimension_deltas=dim_deltas,
        ))
    
    return results


def run_full_activation_patching(
    model,
    tokenizer,
    prompt_high: str,
    prompt_low: str,
    layers_to_test: Optional[List[int]] = None,
    token_position: int = -1,
) -> List[PatchResult]:
    """
    Full activation patching using forward hooks.
    
    Runs the high prompt, but patches in residuals from low prompt at each layer.
    This is the proper causal intervention method.
    """
    device = next(model.parameters()).device
    circuit = ConsciousnessCircuit()
    hidden_dim = model.config.hidden_size
    
    inputs_high = tokenizer(prompt_high, return_tensors="pt").to(device)
    inputs_low = tokenizer(prompt_low, return_tensors="pt").to(device)
    
    model.eval()
    
    # First, cache all hidden states from low prompt
    with torch.no_grad():
        out_low = model(**inputs_low, output_hidden_states=True, return_dict=True)
    low_hidden = out_low.hidden_states
    
    # Baseline from high prompt
    with torch.no_grad():
        out_high = model(**inputs_high, output_hidden_states=True, return_dict=True)
    high_hidden = out_high.hidden_states
    num_layers = len(high_hidden)
    
    h_high_final = high_hidden[-1][0, token_position, :]
    baseline = circuit.compute(h_high_final.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
    baseline_score = baseline.score
    baseline_dims = baseline.dimension_contributions
    
    if layers_to_test is None:
        layers_to_test = list(range(num_layers))
    
    # Find transformer blocks
    layer_stack = None
    for name, module in model.named_modules():
        if hasattr(module, '__len__') and len(module) >= num_layers - 1:
            layer_stack = module
            break
    
    if layer_stack is None:
        # Fallback to simplified method
        return run_consciousness_patching(model, tokenizer, prompt_high, prompt_low, layers_to_test, token_position)
    
    results = []
    
    for layer_idx in layers_to_test:
        if layer_idx >= len(layer_stack) or layer_idx >= len(low_hidden):
            continue
            
        # Token position handling
        low_seq_len = low_hidden[layer_idx].shape[1]
        high_seq_len = high_hidden[layer_idx].shape[1]
        
        # Use minimum token position
        patch_pos = min(token_position % low_seq_len, token_position % high_seq_len)
        patch_value = low_hidden[layer_idx][0, patch_pos, :].clone()
        
        # Create patching hook
        def create_patch_hook(patch_vec, pos):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                out[0, pos, :] = patch_vec
                return output
            return hook
        
        handle = layer_stack[layer_idx].register_forward_hook(
            create_patch_hook(patch_value, token_position)
        )
        
        try:
            with torch.no_grad():
                patched_out = model(**inputs_high, output_hidden_states=True, return_dict=True)
            
            h_patched = patched_out.hidden_states[-1][0, token_position, :]
            patched_result = circuit.compute(h_patched.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
            
            dim_deltas = {}
            for k in baseline_dims:
                dim_deltas[k] = patched_result.dimension_contributions.get(k, 0) - baseline_dims.get(k, 0)
            
            results.append(PatchResult(
                layer=layer_idx,
                original_score=baseline_score,
                patched_score=patched_result.score,
                delta=patched_result.score - baseline_score,
                dimension_deltas=dim_deltas,
            ))
        finally:
            handle.remove()
    
    return results


def plot_patch_impact(results: List[PatchResult], save_path: Optional[str] = None, title: str = ""):
    """Plot impact of patching each layer on consciousness."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plot")
        return
    
    layers = [r.layer for r in results]
    deltas = [r.delta for r in results]
    
    dim_names = list(results[0].dimension_deltas.keys()) if results else []
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: overall consciousness delta by layer
    ax1 = axes[0]
    colors = ['red' if d < 0 else 'green' for d in deltas]
    ax1.bar(layers, deltas, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Consciousness Δ', fontsize=12)
    ax1.set_xlabel('Patched Layer', fontsize=12)
    ax1.set_title(f'Consciousness Impact of Layer Patching (high→low) {title}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Mark most impactful layers
    if deltas:
        min_idx = np.argmin(deltas)
        max_idx = np.argmax(deltas)
        ax1.annotate(f'Most disruptive: L{layers[min_idx]}', 
                    xy=(layers[min_idx], deltas[min_idx]),
                    xytext=(layers[min_idx], deltas[min_idx] - 0.05),
                    fontsize=9, ha='center')
    
    # Bottom: dimension-wise delta heatmap
    ax2 = axes[1]
    if dim_names:
        data = []
        for r in results:
            data.append([r.dimension_deltas.get(name, 0) for name in dim_names])
        data = np.array(data).T  # dims x layers
        
        sns.heatmap(data, ax=ax2, cmap='RdBu_r', center=0,
                   xticklabels=layers, yticklabels=dim_names,
                   cbar_kws={'label': 'Δ Contribution'})
        ax2.set_xlabel('Patched Layer', fontsize=12)
        ax2.set_ylabel('Dimension', fontsize=12)
        ax2.set_title('Per-Dimension Impact of Patching', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Consciousness-based activation patching")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--high", default=None, help="High consciousness prompt")
    parser.add_argument("--low", default=None, help="Low consciousness prompt")
    parser.add_argument("--full-patching", action="store_true", help="Use full activation patching (slower but accurate)")
    parser.add_argument("--layer-step", type=int, default=1, help="Test every N layers")
    parser.add_argument("--output", default="patching_output.json", help="Output JSON file")
    parser.add_argument("--plot", default="patching_plot.png", help="Output plot file")
    args = parser.parse_args()
    
    # Default prompts
    prompt_high = args.high or "I need to deeply reflect on my own reasoning process and examine whether my understanding is complete. Let me think about this carefully..."
    prompt_low = args.low or "The capital of France is Paris."
    
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
    print(f"Model loaded: {num_layers} layers, hidden={model.config.hidden_size}")
    
    layers_to_test = list(range(0, num_layers + 1, args.layer_step))  # +1 for embedding layer
    
    print(f"\n=== Consciousness Patching ===")
    print(f"High prompt: {prompt_high[:80]}...")
    print(f"Low prompt: {prompt_low[:80]}...")
    print(f"Testing {len(layers_to_test)} layers\n")
    
    if args.full_patching:
        print("Using full activation patching (hook-based)...")
        results = run_full_activation_patching(
            model, tokenizer, prompt_high, prompt_low, layers_to_test
        )
    else:
        print("Using approximate patching (faster)...")
        results = run_consciousness_patching(
            model, tokenizer, prompt_high, prompt_low, layers_to_test
        )
    
    print(f"\n=== Results ({len(results)} layers) ===")
    
    # Find most impactful layers
    sorted_by_impact = sorted(results, key=lambda r: abs(r.delta), reverse=True)
    print("\nMost impactful layers (patching causes largest consciousness change):")
    for r in sorted_by_impact[:5]:
        direction = "↓" if r.delta < 0 else "↑"
        print(f"  Layer {r.layer:2d}: {r.delta:+.4f} {direction}")
    
    # Find the "consciousness decision band"
    large_impact_layers = [r for r in results if abs(r.delta) > 0.01]
    if large_impact_layers:
        band_start = min(r.layer for r in large_impact_layers)
        band_end = max(r.layer for r in large_impact_layers)
        print(f"\nConsciousness decision band: layers {band_start} - {band_end}")
    
    # Save results
    output_data = {
        "prompt_high": prompt_high,
        "prompt_low": prompt_low,
        "model": args.model,
        "method": "full_patching" if args.full_patching else "approximate",
        "results": [asdict(r) for r in results],
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {args.output}")
    
    # Plot
    plot_patch_impact(results, save_path=args.plot, title=f"\n({args.model})")


if __name__ == "__main__":
    main()
