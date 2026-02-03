"""
Layerwise Analysis: Run consciousness circuit across ALL layers.

This reveals where in the network different features emerge, sharpen, or get rewritten.

Key insights:
- Where does "math mode" start?
- Where does "format compliance" start?
- Is "self-reflective" late-stage polishing or early-stage framing?

Usage:
    python experiments/layerwise_analysis.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is consciousness?"
    python experiments/layerwise_analysis.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --prompt "Let me think about this step by step..."
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from consciousness_circuit.circuit import ConsciousnessCircuit


@dataclass
class LayerResult:
    layer: int
    consciousness_score: float
    raw_score: float
    dimension_contributions: Dict[str, float]


def analyze_layers(
    model,
    tokenizer,
    prompt: str,
    layer_step: int = 1,
    circuit: Optional[ConsciousnessCircuit] = None,
    token_position: int = -1,  # -1 = last token
) -> List[LayerResult]:
    """
    Compute consciousness score at each layer for a given prompt.
    
    Returns:
        List of LayerResult for each analyzed layer
    """
    device = next(model.parameters()).device
    
    if circuit is None:
        circuit = ConsciousnessCircuit()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden) per layer
    num_layers = len(hidden_states)
    hidden_dim = model.config.hidden_size
    
    results = []
    for layer_idx in range(0, num_layers, layer_step):
        h = hidden_states[layer_idx][0, token_position, :]  # [hidden]
        result = circuit.compute(h.unsqueeze(0).unsqueeze(0), hidden_dim=hidden_dim)
        
        results.append(LayerResult(
            layer=layer_idx,
            consciousness_score=result.score,
            raw_score=result.raw_score,
            dimension_contributions=result.dimension_contributions,
        ))
    
    return results


def plot_layerwise(results: List[LayerResult], save_path: Optional[str] = None, title: str = ""):
    """Plot consciousness across layers with per-dimension breakdown."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plot")
        return
    
    layers = [r.layer for r in results]
    scores = [r.consciousness_score for r in results]
    
    dim_names = list(results[0].dimension_contributions.keys()) if results else []
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: overall consciousness across layers
    ax1 = axes[0]
    ax1.plot(layers, scores, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(layers, 0.5, scores, alpha=0.3, 
                     where=[s > 0.5 for s in scores], color='green')
    ax1.fill_between(layers, scores, 0.5, alpha=0.3,
                     where=[s < 0.5 for s in scores], color='red')
    ax1.set_ylabel('Consciousness Score', fontsize=12)
    ax1.set_title(f'Consciousness Across Layers {title}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: heatmap of dimension contributions
    ax2 = axes[1]
    
    if dim_names:
        data = []
        for r in results:
            data.append([r.dimension_contributions.get(name, 0) for name in dim_names])
        data = np.array(data).T  # dims x layers
        
        sns.heatmap(data, ax=ax2, cmap='RdBu_r', center=0,
                   xticklabels=layers, yticklabels=dim_names,
                   cbar_kws={'label': 'Contribution'})
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Dimension', fontsize=12)
        ax2.set_title('Per-Dimension Contributions Across Layers', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    return fig


def compare_prompts(
    model,
    tokenizer,
    prompts: Dict[str, str],  # name -> prompt
    layer_step: int = 1,
    save_path: Optional[str] = None,
):
    """Compare layerwise consciousness across multiple prompts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    circuit = ConsciousnessCircuit()
    
    all_results = {}
    for name, prompt in prompts.items():
        results = analyze_layers(model, tokenizer, prompt, layer_step, circuit)
        all_results[name] = results
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, results in all_results.items():
        layers = [r.layer for r in results]
        scores = [r.consciousness_score for r in results]
        ax.plot(layers, scores, '-o', label=name, linewidth=2, markersize=4)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Consciousness Score', fontsize=12)
    ax.set_title('Layerwise Consciousness: Prompt Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    return fig, all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze consciousness across layers")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--prompt", default=None, help="Single prompt to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare multiple prompt types")
    parser.add_argument("--layer-step", type=int, default=1, help="Analyze every N layers")
    parser.add_argument("--output", default="layerwise_output.json", help="Output JSON file")
    parser.add_argument("--plot", default="layerwise_plot.png", help="Output plot file")
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
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")
    
    if args.compare:
        # Compare multiple prompt types
        prompts = {
            "High (philosophical)": "What is the nature of consciousness and self-awareness?",
            "Medium (reasoning)": "Explain the theory of relativity in simple terms.",
            "Low (factual)": "What is the capital of France?",
            "Code": "Write a Python function to calculate fibonacci numbers.",
            "Math": "Calculate 17 * 23 step by step.",
        }
        
        print(f"\n=== Comparing {len(prompts)} prompts ===")
        fig, all_results = compare_prompts(
            model, tokenizer, prompts,
            layer_step=args.layer_step,
            save_path=args.plot,
        )
        
        # Save results
        output_data = {}
        for name, results in all_results.items():
            output_data[name] = [asdict(r) for r in results]
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved: {args.output}")
        
    else:
        # Single prompt analysis
        prompt = args.prompt or "What is consciousness?"
        print(f"Prompt: {prompt[:100]}...")
        
        results = analyze_layers(model, tokenizer, prompt, layer_step=args.layer_step)
        
        print(f"\n=== Layerwise Analysis ({len(results)} layers) ===")
        for r in results[::max(1, len(results)//10)]:  # Show ~10 samples
            print(f"  Layer {r.layer:2d}: {r.consciousness_score:.3f}")
        
        # Find peak layer
        peak = max(results, key=lambda r: r.consciousness_score)
        print(f"\nPeak consciousness at layer {peak.layer}: {peak.consciousness_score:.3f}")
        
        # Save
        with open(args.output, 'w') as f:
            json.dump({
                "prompt": prompt,
                "model": args.model,
                "results": [asdict(r) for r in results],
            }, f, indent=2)
        print(f"Saved: {args.output}")
        
        # Plot
        plot_layerwise(results, save_path=args.plot, title=f"\n({args.model})")


if __name__ == "__main__":
    main()
