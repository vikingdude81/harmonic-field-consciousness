"""
Run All Consciousness Experiments: Unified runner for all analysis types.

This script runs all experiments in sequence and generates a comprehensive report.

Usage:
    python experiments/run_consciousness_experiments.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
    python experiments/run_consciousness_experiments.py --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --full
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(name: str, func, **kwargs):
    """Run an experiment with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        result = func(**kwargs)
        elapsed = time.time() - start
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return {'status': 'success', 'elapsed': elapsed, 'result': result}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ Failed after {elapsed:.1f}s: {e}")
        return {'status': 'failed', 'elapsed': elapsed, 'error': str(e)}


def run_layerwise(model, tokenizer, output_dir: Path, quick: bool = False):
    """Run layerwise analysis."""
    from experiments.layerwise_analysis import analyze_layers, plot_layerwise, compare_prompts
    
    if quick:
        # Single prompt
        results = analyze_layers(model, tokenizer, "What is consciousness?", layer_step=4)
        plot_layerwise(results, save_path=str(output_dir / "layerwise_single.png"))
        return {'n_layers': len(results), 'peak_layer': max(results, key=lambda r: r.consciousness_score).layer}
    else:
        # Compare prompts
        prompts = {
            "Philosophical": "What is the nature of consciousness?",
            "Reasoning": "Let me think step by step...",
            "Factual": "The capital of France is Paris.",
        }
        _, all_results = compare_prompts(model, tokenizer, prompts, layer_step=2, save_path=str(output_dir / "layerwise_compare.png"))
        return {'prompts': list(prompts.keys()), 'n_layers': len(list(all_results.values())[0])}


def run_patching(model, tokenizer, output_dir: Path, quick: bool = False):
    """Run consciousness patching."""
    from experiments.consciousness_patching import run_consciousness_patching, plot_patch_impact
    
    prompt_high = "Let me deeply reflect on my reasoning process..."
    prompt_low = "The answer is 42."
    
    num_layers = model.config.num_hidden_layers
    layers_to_test = list(range(0, num_layers + 1, 4 if quick else 1))
    
    results = run_consciousness_patching(model, tokenizer, prompt_high, prompt_low, layers_to_test)
    plot_patch_impact(results, save_path=str(output_dir / "patching_impact.png"))
    
    # Find most impactful
    if results:
        most_impactful = max(results, key=lambda r: abs(r.delta))
        return {
            'n_layers_tested': len(results),
            'most_impactful_layer': most_impactful.layer,
            'max_delta': most_impactful.delta,
        }
    return {}


def run_trace(model, tokenizer, output_dir: Path, quick: bool = False):
    """Run thought trajectory trace."""
    from experiments.trace_thoughtspace import trace_generation, plot_trajectory
    
    prompt = "Let me think about consciousness step by step."
    max_tokens = 20 if quick else 50
    
    text, trace_steps, _ = trace_generation(model, tokenizer, prompt, max_tokens=max_tokens)
    plot_trajectory(trace_steps, save_path=str(output_dir / "trace_trajectory.png"))
    
    return {
        'n_steps': len(trace_steps),
        'final_score': trace_steps[-1].consciousness_score if trace_steps else None,
        'tokens': [s.token for s in trace_steps[:10]],
        'generated_text': text[:100],
    }


def run_sae_collection(model, tokenizer, output_dir: Path, quick: bool = False):
    """Collect activations for SAE."""
    from experiments.collect_for_sae import collect_activations, save_for_sae, generate_diverse_prompts

    n_samples = 20 if quick else 100
    prompts = generate_diverse_prompts(n_samples)

    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers - 2, num_layers - 1, num_layers]  # Last 3 layers

    # Use smaller batch size for large models to prevent OOM
    hidden_size = model.config.hidden_size
    batch_size = 5 if hidden_size > 4096 else 10  # Smaller batches for 32B+ models

    activations, metadata = collect_activations(
        model, tokenizer, prompts, target_layers,
        compute_consciousness=True,
        batch_size=batch_size,
    )

    output_path = str(output_dir / "sae_activations.npz")
    save_for_sae(activations, metadata, output_path, str(model.config._name_or_path))

    return {
        'n_samples': sum(a.shape[0] for a in activations.values()),
        'layers': list(activations.keys()),
        'file': output_path,
        'batch_size': batch_size,
    }


def run_steering(model, tokenizer, output_dir: Path, quick: bool = False):
    """Run steering vector extraction."""
    from experiments.steering_experiments import extract_steering_vectors, get_contrastive_pairs
    import numpy as np
    
    pairs = get_contrastive_pairs()
    
    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers - 4, num_layers - 2, num_layers]
    
    vectors = extract_steering_vectors(model, tokenizer, pairs, target_layers)
    
    # Save
    np_data = {f'layer_{k}': v.direction for k, v in vectors.items()}
    vector_file = str(output_dir / "steering_vectors.npz")
    np.savez_compressed(vector_file, **np_data)
    
    return {
        'n_vectors': len(vectors),
        'layers': list(vectors.keys()),
        'norms': {k: v.norm for k, v in vectors.items()},
        'file': vector_file,
    }


def generate_report(results: dict, output_dir: Path, model_name: str):
    """Generate markdown report."""
    
    report = f"""# Consciousness Circuit Experiment Report

**Model**: {model_name}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory**: {output_dir}

## Summary

| Experiment | Status | Duration |
|------------|--------|----------|
"""
    
    for name, res in results.items():
        status = "✓" if res['status'] == 'success' else "✗"
        report += f"| {name} | {status} | {res['elapsed']:.1f}s |\n"
    
    report += "\n## Detailed Results\n\n"
    
    for name, res in results.items():
        report += f"### {name}\n\n"
        if res['status'] == 'success':
            report += f"```json\n{json.dumps(res.get('result', {}), indent=2, default=str)}\n```\n\n"
        else:
            report += f"**Error**: {res.get('error', 'Unknown')}\n\n"
    
    report += """
## Generated Files

- `layerwise_*.png` - Consciousness across layers
- `patching_impact.png` - Layer patching impact
- `trace_trajectory.png` - Token-by-token consciousness
- `sae_activations.npz` - Residual activations for SAE
- `steering_vectors.npz` - Consciousness steering vectors

## Next Steps

1. **Analyze layerwise plots** to find where consciousness emerges
2. **Review patching results** to identify causal decision layers
3. **Train SAE** on collected activations to discover new features
4. **Apply steering** with different alpha values to modulate consciousness
"""
    
    report_path = output_dir / "EXPERIMENT_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run all consciousness experiments")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")
    parser.add_argument("--full", action="store_true", help="Full mode (all experiments)")
    parser.add_argument("--output-dir", default="experiment_outputs", help="Output directory")
    parser.add_argument("--experiments", nargs='+', 
                       choices=['layerwise', 'patching', 'trace', 'sae', 'steering', 'all'],
                       default=['all'], help="Which experiments to run")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model - try unsloth first for quantized models, fallback to transformers
    print(f"\nLoading model: {args.model}")
    
    use_unsloth = "bnb" in args.model or "4bit" in args.model or "8bit" in args.model
    
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            print("Loaded with unsloth")
        except ImportError:
            print("unsloth not available, trying transformers...")
            use_unsloth = False
    
    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    print(f"Model: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")
    
    quick = args.quick or not args.full

    # Determine which experiments to run
    run_all = 'all' in args.experiments
    experiments_to_run = args.experiments if not run_all else ['layerwise', 'patching', 'trace', 'sae', 'steering']
    
    results = {}
    
    if 'layerwise' in experiments_to_run:
        results['Layerwise Analysis'] = run_experiment(
            "Layerwise Analysis", run_layerwise,
            model=model, tokenizer=tokenizer, output_dir=output_dir, quick=quick
        )
    
    if 'patching' in experiments_to_run:
        results['Consciousness Patching'] = run_experiment(
            "Consciousness Patching", run_patching,
            model=model, tokenizer=tokenizer, output_dir=output_dir, quick=quick
        )
    
    if 'trace' in experiments_to_run:
        results['Thought Trajectory'] = run_experiment(
            "Thought Trajectory", run_trace,
            model=model, tokenizer=tokenizer, output_dir=output_dir, quick=quick
        )
    
    if 'sae' in experiments_to_run:
        results['SAE Collection'] = run_experiment(
            "SAE Collection", run_sae_collection,
            model=model, tokenizer=tokenizer, output_dir=output_dir, quick=quick
        )
    
    if 'steering' in experiments_to_run:
        results['Steering Vectors'] = run_experiment(
            "Steering Vectors", run_steering,
            model=model, tokenizer=tokenizer, output_dir=output_dir, quick=quick
        )
    
    # Generate report
    print("\n" + "="*60)
    print("  GENERATING REPORT")
    print("="*60)
    
    generate_report(results, output_dir, args.model)
    
    # Save raw results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({k: {**v, 'result': str(v.get('result', {}))} for k, v in results.items()}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"\n{success_count}/{len(results)} experiments succeeded")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
