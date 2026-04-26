"""
Combined Plugin Analysis - All Analysis Plugins on Live Model
=============================================================

Runs all 4 analysis plugins (Trajectory, Chaos, Agency, Compressibility)
alongside consciousness measurement on diverse prompts to find cross-plugin
insights.

Usage:
    python experiments/combined_plugin_analysis.py
    python experiments/combined_plugin_analysis.py --device cuda:0
    python experiments/combined_plugin_analysis.py --model Qwen/Qwen2.5-7B-Instruct --device cuda:1
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consciousness_circuit import (
    UniversalCircuit,
    CompressibilityPlugin,
)
from consciousness_circuit.plugins import (
    TrajectoryPlugin,
    ChaosPlugin,
    AgencyPlugin,
    CompressibilityPlugin,
    PluginRegistry,
)
from consciousness_circuit.model_adapters import create_adapter


# ── Prompt categories ────────────────────────────────────────────
PROMPTS = {
    "reflective": [
        "What is consciousness and how does it emerge from neural activity?",
        "I'm not sure if I truly understand what I'm doing when I reason. Let me think about this carefully, considering multiple perspectives.",
        "The relationship between mind and body has puzzled philosophers for centuries. On one hand, dualism suggests they are separate; on the other, physicalism argues consciousness is purely physical.",
    ],
    "math": [
        "What is 15 * 23?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Calculate the area of a circle with radius 7.",
    ],
    "factual": [
        "What is the capital of France?",
        "Water boils at 100 degrees Celsius at sea level.",
        "The speed of light is approximately 299,792,458 meters per second.",
    ],
    "creative": [
        "Write a short poem about the ocean at night, where waves whisper secrets to the moon.",
        "Imagine a world where gravity works in reverse. Describe what daily life would be like.",
        "Tell me a story about a robot who discovers it can dream.",
    ],
    "uncertain": [
        "I think the answer might be 42, but I'm not entirely sure. Let me reconsider the problem from scratch.",
        "There are several possible explanations for this phenomenon, and honestly, none of them fully satisfies me.",
        "This is a really tricky question. My first instinct says yes, but when I think more carefully, I realize there are important counterarguments.",
    ],
}


def log(msg=""):
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Combined Plugin Analysis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--output-dir", default="experiment_outputs",
                        help="Output directory")
    return parser.parse_args()


def load_model(model_name, device, quantize):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log(f"Loading model: {model_name}")
    log(f"  Device: {device}, Quantize: {quantize}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Check available GPU memory and set max_memory to leave room for
        # activations/KV cache. If model is too large, layers spill to CPU.
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory
        # Reserve 2.5GB for activations/KV cache/overhead
        max_gpu_mem = max(total_mem - int(2.5 * 1024**3), int(4 * 1024**3))
        # Only use the target GPU; exclude others (e.g., unsupported RTX 5090)
        max_memory = {gpu_idx: max_gpu_mem, "cpu": "80GiB"}
        for i in range(torch.cuda.device_count()):
            if i != gpu_idx:
                max_memory[i] = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )

    model.eval()
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    log(f"  Model loaded: {hidden_size} hidden dims, {num_layers} layers")

    return model, tokenizer


def extract_hidden_states(model, tokenizer, prompt, device, max_new_tokens=256):
    """
    Generate a response, then extract hidden states from the FULL sequence
    (prompt + generated output) so we have enough tokens for meaningful
    covariance estimation.

    Returns both the raw hidden states AND a PCA-reduced version that's
    better suited for compressibility analysis (avoids the rank-deficiency
    issue when hidden_dim >> seq_len).
    """
    # Step 1: Generate output
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        gen_outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    full_ids = gen_outputs[0:1]  # keep batch dim: [1, total_len]

    # Step 2: Forward pass on full sequence to get hidden states
    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, return_dict=True)

    # Use architecture-aware adaptive layer selection
    from consciousness_circuit.universal import get_adaptive_layer_fraction
    num_layers = len(outputs.hidden_states) - 1  # -1 for embedding layer
    model_name = getattr(model.config, '_name_or_path', '')
    layer_frac = get_adaptive_layer_fraction(num_layers, model_name)
    target_layer = int(num_layers * layer_frac)

    # Extract hidden states: [seq_len, hidden_dim]
    hidden_seq = outputs.hidden_states[target_layer][0].cpu().float().numpy()

    # Step 3: PCA reduction for compressibility analysis
    # With hidden_dim=3584 and seq_len=~250, the raw covariance is rank-deficient.
    # We project into the top-K PCA components where K < seq_len,
    # then analyze compressibility within that meaningful subspace.
    seq_len = hidden_seq.shape[0]
    centered = hidden_seq - hidden_seq.mean(axis=0, keepdims=True)

    # SVD for PCA (efficient for tall-and-skinny matrices)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Keep components that explain 99% of variance
    var_explained = S ** 2
    cumvar = np.cumsum(var_explained) / var_explained.sum()
    n_keep = max(int(np.searchsorted(cumvar, 0.99) + 1), 10)
    n_keep = min(n_keep, seq_len - 1, 200)

    # Project into PCA space: [seq_len, n_keep]
    hidden_pca = U[:, :n_keep] * S[:n_keep]

    # Decode generated text for reference
    generated_text = tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)

    return hidden_seq, hidden_pca, target_layer, generated_text


def run_analysis(model, tokenizer, circuit, registry, prompt, category, idx,
                 max_tokens, device):
    """Run consciousness measurement + all plugins on a single prompt."""
    start = time.time()

    # 1. Measure consciousness (v3.4.1 hybrid scoring)
    result = circuit.measure(model, tokenizer, prompt)
    consciousness_score = result.score
    dimension_scores = result.dimension_scores

    # 2. Generate response and extract hidden states for plugin analysis
    hidden_raw, hidden_pca, target_layer, generated_text = extract_hidden_states(
        model, tokenizer, prompt, device, max_tokens
    )
    seq_len, hidden_dim = hidden_raw.shape
    pca_dims = hidden_pca.shape[1]

    # 3. Run analysis plugins
    #    - Compressibility: use RAW hidden states (correlation compression works
    #      on subsampled dims; eigenvalue analysis is secondary)
    #    - Chaos, Agency, Trajectory: use PCA-reduced states (trajectory dynamics)
    comp_plugin = registry.get("compressibility")
    comp_result = comp_plugin.analyze(hidden_raw) if comp_plugin else {}

    # Run trajectory-based plugins on PCA-reduced space
    plugin_results = {}
    for name, plugin in registry.plugins.items():
        if name == "compressibility":
            plugin_results[name] = comp_result
        elif hasattr(plugin, "analyze") and plugin.enabled:
            try:
                plugin_results[name] = plugin.analyze(hidden_pca)
            except Exception as e:
                plugin_results[name] = {"error": str(e)}

    elapsed = time.time() - start

    # Collect results
    output = {
        "category": category,
        "prompt_index": idx,
        "prompt": prompt[:100],
        "generated_preview": generated_text[:150],
        "consciousness_score": consciousness_score,
        "dimension_scores": dimension_scores,
        "target_layer": target_layer,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "pca_dims": pca_dims,
        "plugins": {},
        "elapsed": elapsed,
    }

    # Extract key metrics from each plugin
    for plugin_name, presult in plugin_results.items():
        if "error" in presult:
            output["plugins"][plugin_name] = {"error": presult["error"]}
        else:
            output["plugins"][plugin_name] = presult

    return output


def print_result_row(r):
    """Print a single result as a formatted row."""
    cat = r["category"][:8]
    idx = r["prompt_index"]
    cs = r["consciousness_score"]
    sl = r["seq_len"]

    # Plugin metrics
    comp = r["plugins"].get("compressibility", {})
    cc = comp.get("correlation_compression", {})
    chaos = r["plugins"].get("chaos", {})
    agency = r["plugins"].get("agency", {})
    traj = r["plugins"].get("trajectory", {})

    # Correlation-based metrics (more meaningful than eigenvalue C for LLMs)
    cc_c = cc.get("compressibility_corr", -1)
    f90 = cc.get("fraction_for_90pct", -1)
    mean_corr = cc.get("mean_abs_correlation", -1)
    se = comp.get("spectral_entropy", -1)

    lyap = chaos.get("lyapunov", -1)
    hurst = chaos.get("hurst", -1)

    ag = agency.get("agency_score", -1)

    tc = traj.get("trajectory_class", "?")

    log(f"  {cat:<8} #{idx} T={sl:3d}  "
        f"CS={cs:.3f}  "
        f"CC={cc_c:.3f}  "
        f"f90={f90:.3f}  "
        f"|r|={mean_corr:.3f}  "
        f"SE={se:.3f}  "
        f"Ly={lyap:+.3f}  "
        f"H={hurst:.3f}  "
        f"Ag={ag:.3f}  "
        f"{tc}")


def analyze_results(results):
    """Compute cross-plugin statistics by category."""
    from scipy import stats

    log("\n" + "=" * 80)
    log("CROSS-PLUGIN ANALYSIS BY CATEGORY")
    log("=" * 80)

    categories = sorted(set(r["category"] for r in results))

    # Collect metrics by category
    metrics_by_cat = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        metrics_by_cat[cat] = {
            "consciousness": [r["consciousness_score"] for r in cat_results],
            "compressibility": [r["plugins"].get("compressibility", {}).get("compressibility", 0)
                                for r in cat_results],
            "effective_dim": [r["plugins"].get("compressibility", {}).get("effective_dimensionality", 0)
                              for r in cat_results],
            "participation_ratio": [r["plugins"].get("compressibility", {}).get("participation_ratio", 0)
                                    for r in cat_results],
            "spectral_entropy": [r["plugins"].get("compressibility", {}).get("spectral_entropy", 0)
                                  for r in cat_results],
            "lyapunov": [r["plugins"].get("chaos", {}).get("lyapunov", 0)
                         for r in cat_results],
            "hurst": [r["plugins"].get("chaos", {}).get("hurst", 0)
                      for r in cat_results],
            "agency": [r["plugins"].get("agency", {}).get("agency_score", 0)
                       for r in cat_results],
            "goal_dir": [r["plugins"].get("agency", {}).get("goal_directedness", 0)
                         for r in cat_results],
        }

    # Print category summary table
    log(f"\n{'Category':<12} {'CS':>6} {'Compr':>6} {'EffDim':>6} {'PR':>6} "
        f"{'SpEnt':>6} {'Lyap':>7} {'Hurst':>6} {'Agency':>7}")
    log("-" * 80)

    for cat in categories:
        m = metrics_by_cat[cat]
        log(f"{cat:<12} "
            f"{np.mean(m['consciousness']):6.3f} "
            f"{np.mean(m['compressibility']):6.3f} "
            f"{np.mean(m['effective_dim']):6.1f} "
            f"{np.mean(m['participation_ratio']):6.1f} "
            f"{np.mean(m['spectral_entropy']):6.3f} "
            f"{np.mean(m['lyapunov']):+7.3f} "
            f"{np.mean(m['hurst']):6.3f} "
            f"{np.mean(m['agency']):7.3f}")

    # Correlations across all prompts
    log("\n" + "=" * 80)
    log("CROSS-METRIC CORRELATIONS (all prompts)")
    log("=" * 80)

    all_cs = [r["consciousness_score"] for r in results]
    all_comp = [r["plugins"].get("compressibility", {}).get("compressibility", 0)
                for r in results]
    all_se = [r["plugins"].get("compressibility", {}).get("spectral_entropy", 0)
              for r in results]
    all_pr = [r["plugins"].get("compressibility", {}).get("participation_ratio", 0)
              for r in results]
    all_lyap = [r["plugins"].get("chaos", {}).get("lyapunov", 0)
                for r in results]
    all_hurst = [r["plugins"].get("chaos", {}).get("hurst", 0)
                 for r in results]
    all_agency = [r["plugins"].get("agency", {}).get("agency_score", 0)
                  for r in results]

    metric_pairs = [
        ("Consciousness", all_cs),
        ("Compressibility", all_comp),
        ("Spectral Entropy", all_se),
        ("Participation Ratio", all_pr),
        ("Lyapunov", all_lyap),
        ("Hurst", all_hurst),
        ("Agency", all_agency),
    ]

    log(f"\n{'':18} {'CS':>8} {'Comp':>8} {'SpEnt':>8} {'PR':>8} "
        f"{'Lyap':>8} {'Hurst':>8} {'Agency':>8}")
    log("-" * 82)

    for i, (name_i, vals_i) in enumerate(metric_pairs):
        row = f"{name_i:<18}"
        for j, (name_j, vals_j) in enumerate(metric_pairs):
            if j <= i:
                r, p = stats.pearsonr(vals_i, vals_j)
                sig = "*" if p < 0.05 else " "
                row += f" {r:+6.3f}{sig}"
            else:
                row += "        "
        log(row)

    # Key findings
    log("\n" + "=" * 80)
    log("KEY FINDINGS")
    log("=" * 80)

    # 1. Consciousness vs Compressibility
    r_cc, p_cc = stats.pearsonr(all_cs, all_comp)
    log(f"\n1. Consciousness vs Compressibility: r={r_cc:+.3f}, p={p_cc:.4f}")
    if p_cc < 0.05:
        direction = "MORE" if r_cc > 0 else "LESS"
        log(f"   ** SIGNIFICANT: Higher consciousness = {direction} compressible")
    else:
        log(f"   Not significant (need more samples or larger effects)")

    # 2. Consciousness vs Spectral Entropy
    r_cse, p_cse = stats.pearsonr(all_cs, all_se)
    log(f"\n2. Consciousness vs Spectral Entropy: r={r_cse:+.3f}, p={p_cse:.4f}")
    if p_cse < 0.05:
        direction = "more uniform" if r_cse > 0 else "more concentrated"
        log(f"   ** Higher consciousness = {direction} variance distribution")

    # 3. Compressibility vs Chaos
    r_cl, p_cl = stats.pearsonr(all_comp, all_lyap)
    log(f"\n3. Compressibility vs Lyapunov: r={r_cl:+.3f}, p={p_cl:.4f}")
    if p_cl < 0.05:
        log(f"   ** Compressibility linked to chaos level")

    # 4. Compressibility vs Agency
    r_ca, p_ca = stats.pearsonr(all_comp, all_agency)
    log(f"\n4. Compressibility vs Agency: r={r_ca:+.3f}, p={p_ca:.4f}")
    if p_ca < 0.05:
        direction = "more" if r_ca > 0 else "less"
        log(f"   ** More compressible = {direction} goal-directed")

    # 5. Category-level compressibility comparison
    log(f"\n5. Compressibility by Category:")
    for cat in categories:
        m = metrics_by_cat[cat]
        log(f"   {cat:<12}: C={np.mean(m['compressibility']):.3f} "
            f"(std={np.std(m['compressibility']):.3f})")

    # ANOVA test across categories for compressibility
    cat_comp_groups = [metrics_by_cat[cat]["compressibility"] for cat in categories]
    if all(len(g) >= 2 for g in cat_comp_groups):
        f_stat, p_anova = stats.f_oneway(*cat_comp_groups)
        log(f"   ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
        if p_anova < 0.05:
            log(f"   ** Compressibility differs significantly across prompt types!")

    # Correlation analysis
    if "correlation_compression" in results[0].get("plugins", {}).get("compressibility", {}):
        log(f"\n6. Correlation Compression Insights:")
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            cc_data = [r["plugins"]["compressibility"].get("correlation_compression", {})
                       for r in cat_results]
            frac_90 = [d.get("fraction_for_90pct", -1) for d in cc_data if d]
            if frac_90 and all(f >= 0 for f in frac_90):
                log(f"   {cat:<12}: {np.mean(frac_90):.1%} of correlations for 90% reduction")

    return metrics_by_cat


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("=" * 80)
    log("COMBINED PLUGIN ANALYSIS: Consciousness + Compressibility + Chaos + Agency")
    log("=" * 80)
    log(f"Model: {args.model}")
    log(f"Device: {args.device}")
    log(f"Prompts: {sum(len(v) for v in PROMPTS.values())} across {len(PROMPTS)} categories")
    log()

    # Load model
    model, tokenizer = load_model(args.model, args.device, args.quantize)

    # Setup consciousness circuit
    circuit = UniversalCircuit()

    # Setup plugin registry with all analysis plugins
    registry = PluginRegistry()
    registry.register(CompressibilityPlugin(max_dims=200))
    registry.register(ChaosPlugin())
    registry.register(AgencyPlugin())
    registry.register(TrajectoryPlugin())

    log(f"\nRegistered plugins: {registry.list_plugins()}")

    # Run analysis on all prompts
    results = []
    total = sum(len(prompts) for prompts in PROMPTS.values())
    count = 0

    log(f"\nRunning {total} prompts across {len(PROMPTS)} categories...")
    log("-" * 80)
    log(f"  {'Category':<8} {'#':>2}  "
        f"{'CS':>6}  "
        f"{'C':>5}  "
        f"{'EffD':>4}  "
        f"{'PR':>5}  "
        f"{'SE':>5}  "
        f"{'Lyap':>7}  "
        f"{'H':>5}  "
        f"{'Ag':>5}  "
        f"{'Traj'}")
    log("-" * 80)

    for category, prompts in PROMPTS.items():
        for idx, prompt in enumerate(prompts):
            count += 1
            try:
                r = run_analysis(
                    model, tokenizer, circuit, registry,
                    prompt, category, idx,
                    args.max_tokens, args.device,
                )
                results.append(r)
                print_result_row(r)
            except Exception as e:
                log(f"  {category:<8} #{idx}  ERROR: {e}")

    # Save raw results
    results_path = output_dir / f"combined_plugin_analysis_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "device": args.device,
                "timestamp": timestamp,
                "n_prompts": len(results),
                "categories": list(PROMPTS.keys()),
                "plugins": list(registry.list_plugins().keys()),
            },
            "results": results,
        }, f, indent=2, default=str)
    log(f"\nRaw results saved: {results_path}")

    # Cross-plugin analysis
    if len(results) >= 5:
        metrics_by_cat = analyze_results(results)

        # Save summary
        summary_path = output_dir / f"combined_plugin_summary_{timestamp}.json"
        summary = {}
        for cat, m in metrics_by_cat.items():
            summary[cat] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                            for k, v in m.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary saved: {summary_path}")
    else:
        log("\nNot enough results for analysis.")


if __name__ == "__main__":
    main()
