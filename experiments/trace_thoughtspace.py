"""
Trace Thoughtspace: Record consciousness trajectory during generation.

This script captures the 5120-D "thought trajectory" token-by-token during
generation, showing how consciousness dimensions evolve as the model thinks.

Key insights:
- At what token did the model "commit" to an answer?
- Do uncertainty spikes come before corrections?
- Does computation dip when switching to style/wording?

Usage:
    python experiments/trace_thoughtspace.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --prompt "What is consciousness?"
    python experiments/trace_thoughtspace.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain quantum mechanics" --max-tokens 64
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

from consciousness_circuit.circuit import ConsciousnessCircuit, REFERENCE_HIDDEN_DIM


@dataclass
class TraceStep:
    step: int
    token_id: int
    token: str
    consciousness_score: float
    dimension_contributions: Dict[str, float]
    raw_score: float
    # Optional: store hidden state norm for debugging
    hidden_norm: float = 0.0


def trace_generation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    circuit: Optional[ConsciousnessCircuit] = None,
    store_hidden: bool = False,
    device: Optional[str] = None,
) -> tuple:
    """
    Generate tokens one-by-one and record consciousness at each step.
    
    Returns:
        (generated_text, trace_steps, hidden_states_list)
    """
    # Handle device detection for both regular and unsloth models
    if device is None:
        try:
            # Try model.device first (works for unsloth models)
            device = str(model.device)
        except AttributeError:
            try:
                device = str(next(model.parameters()).device)
            except (StopIteration, AttributeError):
                device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if circuit is None:
        hidden_dim = model.config.hidden_size
        circuit = ConsciousnessCircuit()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(generated))
    
    trace: List[TraceStep] = []
    hidden_states_list = [] if store_hidden else None
    
    model.eval()
    with torch.no_grad():
        for step in range(max_tokens):
            # Forward pass - no KV cache for unsloth compatibility
            out = model(
                input_ids=generated,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Last layer, last token: [batch, hidden]
            h_last = out.hidden_states[-1][:, -1, :]
            
            # Compute consciousness
            result = circuit.compute(h_last, hidden_dim=model.config.hidden_size)
            
            # Get next token
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            trace.append(TraceStep(
                step=step,
                token_id=int(next_id[0, 0]),
                token=tokenizer.decode(next_id[0, 0]),
                consciousness_score=result.score,
                dimension_contributions=result.dimension_contributions,
                raw_score=result.raw_score,
                hidden_norm=float(h_last.norm()),
            ))
            
            if store_hidden:
                hidden_states_list.append(h_last.float().cpu().numpy())
            
            generated = torch.cat([generated, next_id], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)
            
            if next_id.item() == tokenizer.eos_token_id:
                break
    
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, trace, hidden_states_list


def plot_trajectory(trace: List[TraceStep], save_path: Optional[str] = None):
    """Plot consciousness trajectory across tokens."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    steps = [t.step for t in trace]
    scores = [t.consciousness_score for t in trace]
    tokens = [t.token[:8] for t in trace]  # truncate long tokens
    
    # Get dimension names
    dim_names = list(trace[0].dimension_contributions.keys()) if trace else []
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    # Top: overall consciousness score
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.array(scores))
    ax1.bar(steps, scores, color=colors, edgecolor='black', linewidth=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='baseline')
    ax1.set_ylabel('Consciousness Score', fontsize=12)
    ax1.set_title('Consciousness Trajectory During Generation', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add token labels on x-axis
    ax1.set_xticks(steps)
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
    
    # Bottom: per-dimension contributions (stacked area)
    ax2 = axes[1]
    
    if dim_names:
        dim_data = {name: [] for name in dim_names}
        for t in trace:
            for name in dim_names:
                dim_data[name].append(t.dimension_contributions.get(name, 0))
        
        # Plot as lines
        for name, values in dim_data.items():
            ax2.plot(steps, values, label=name, linewidth=1.5, alpha=0.8)
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Generation Step (Token)', fontsize=12)
        ax2.set_ylabel('Dimension Contribution', fontsize=12)
        ax2.legend(loc='upper right', fontsize=8, ncol=4)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Trace consciousness during generation")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--output", default="trace_output.json", help="Output JSON file")
    parser.add_argument("--plot", default="trace_plot.png", help="Output plot file")
    parser.add_argument("--store-hidden", action="store_true", help="Store hidden states (uses more memory)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
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
    print(f"Prompt: {args.prompt[:100]}...")
    
    text, trace, hidden_states = trace_generation(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        store_hidden=args.store_hidden,
    )
    
    print(f"\n=== Generated ({len(trace)} tokens) ===")
    print(text)
    
    print(f"\n=== Consciousness Trajectory ===")
    for t in trace[:10]:
        print(f"  Step {t.step}: '{t.token}' → {t.consciousness_score:.3f}")
    if len(trace) > 10:
        print(f"  ... ({len(trace) - 10} more steps)")
    
    # Save trace
    trace_data = {
        "prompt": args.prompt,
        "model": args.model,
        "generated_text": text,
        "steps": [asdict(t) for t in trace],
    }
    with open(args.output, 'w') as f:
        json.dump(trace_data, f, indent=2)
    print(f"\nSaved trace: {args.output}")
    
    # Plot
    plot_trajectory(trace, save_path=args.plot)
    
    # Save hidden states if requested
    if hidden_states:
        np.savez(args.output.replace('.json', '_hidden.npz'), 
                 hidden_states=np.array(hidden_states))
        print(f"Saved hidden states: {args.output.replace('.json', '_hidden.npz')}")


if __name__ == "__main__":
    main()
