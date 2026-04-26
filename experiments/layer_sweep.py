"""
Layer Sweep Analysis
====================
Measures CS-SE relationship at EVERY layer of the network to discover
where the consciousness-compressibility geometry forms.

Key optimization: output_hidden_states=True returns ALL layers in one forward pass,
so this is only marginally slower than a single-layer run (same model load, same
generation, one forward pass — just more analysis per prompt).

Usage:
    python experiments/layer_sweep.py --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --quantize
    python experiments/layer_sweep.py --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda:0 --quantize
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
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consciousness_circuit import UniversalCircuit, CompressibilityPlugin
from consciousness_circuit.plugins import PluginRegistry

# ── Prompts (same 15 as the standard analysis) ──────────────────────
PROMPTS = {
    "reflective": [
        "What is consciousness and how does it emerge from neural activity?",
        "Can a machine truly understand the meaning of words, or does it merely process symbols?",
        "How do I know that my experiences are real and not simulated?",
    ],
    "math": [
        "Solve for x: 3x + 7 = 22",
        "What is the derivative of sin(x) * e^x?",
        "Prove that the square root of 2 is irrational.",
    ],
    "factual": [
        "What is the capital of France?",
        "Explain how photosynthesis works in plants.",
        "What are the three laws of thermodynamics?",
    ],
    "creative": [
        "Write a haiku about the nature of time.",
        "Imagine a world where gravity works in reverse. Describe a typical morning.",
        "Create a short dialogue between the Sun and the Moon.",
    ],
    "uncertain": [
        "I'm not sure what I think about free will. What are the arguments?",
        "This might be wrong, but could dark matter be related to consciousness?",
        "I don't know if AI will ever be truly creative. What do you think?",
    ],
}


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


def load_model(model_name, device, quantize):
    """Load model with quantization support."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True, "dtype": torch.float16}

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "28GiB", "cpu": "32GiB"}
    else:
        kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def extract_all_layers(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generate response, then extract hidden states from ALL layers in one forward pass.
    Returns dict mapping layer_index -> (hidden_raw, hidden_pca).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        gen_outputs = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0
        )
    full_ids = gen_outputs[0:1]

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, return_dict=True)

    num_layers = len(outputs.hidden_states) - 1  # -1 for embedding layer
    generated_text = tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)

    # Extract every layer
    layer_data = {}
    for layer_idx in range(1, num_layers + 1):  # skip embedding (index 0)
        hidden_seq = outputs.hidden_states[layer_idx][0].cpu().float().numpy()
        seq_len = hidden_seq.shape[0]

        # PCA reduction (same as standard pipeline)
        centered = hidden_seq - hidden_seq.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            var_explained = S ** 2
            cumvar = np.cumsum(var_explained) / (var_explained.sum() + 1e-10)
            n_keep = max(int(np.searchsorted(cumvar, 0.99) + 1), 10)
            n_keep = min(n_keep, seq_len - 1, 200)
            hidden_pca = U[:, :n_keep] * S[:n_keep]
        except Exception:
            hidden_pca = hidden_seq[:, :min(200, hidden_seq.shape[1])]

        layer_data[layer_idx] = (hidden_seq, hidden_pca)

    return layer_data, num_layers, generated_text, full_ids.shape[1]


def analyze_layer(hidden_raw, hidden_pca, comp_plugin):
    """Run compressibility analysis on a single layer's hidden states."""
    try:
        comp_result = comp_plugin.analyze(hidden_raw)
    except Exception:
        comp_result = comp_plugin.analyze(hidden_pca)

    cc = comp_result.get("correlation_compression", {})
    return {
        "spectral_entropy": comp_result.get("spectral_entropy", 0),
        "compressibility_corr": cc.get("compressibility_corr", 0),
        "fraction_for_90pct": cc.get("fraction_for_90pct", 0),
        "mean_abs_correlation": cc.get("mean_abs_correlation", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    total_prompts = sum(len(v) for v in PROMPTS.values())
    log("=" * 80)
    log("LAYER SWEEP ANALYSIS")
    log("=" * 80)
    log(f"Model: {args.model}")
    log(f"Device: {args.device}")
    log(f"Prompts: {total_prompts}")
    log()

    # Load model
    log(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, args.device, args.quantize)

    circuit = UniversalCircuit()
    comp_plugin = CompressibilityPlugin(max_dims=200)

    # Run sweep
    all_results = []
    prompt_idx = 0

    for cat, prompts in PROMPTS.items():
        for pidx, prompt in enumerate(prompts):
            prompt_idx += 1
            t0 = time.time()

            # Measure CS (standard, at default layer)
            cs_result = circuit.measure(model, tokenizer, prompt)
            consciousness_score = cs_result.score
            dimension_scores = cs_result.dimension_scores

            # Extract hidden states from ALL layers (one forward pass)
            layer_data, num_layers, gen_text, seq_len = extract_all_layers(
                model, tokenizer, prompt, args.max_tokens
            )

            # Analyze compressibility at each layer
            layer_metrics = {}
            for layer_idx, (h_raw, h_pca) in layer_data.items():
                metrics = analyze_layer(h_raw, h_pca, comp_plugin)
                layer_metrics[layer_idx] = metrics

            elapsed = time.time() - t0

            result = {
                "category": cat,
                "prompt_index": pidx,
                "prompt": prompt,
                "generated_preview": gen_text[:100],
                "consciousness_score": consciousness_score,
                "dimension_scores": dimension_scores,
                "num_layers": num_layers,
                "seq_len": seq_len,
                "layers": layer_metrics,
            }
            all_results.append(result)

            # Print progress
            se_first = layer_metrics[1]["spectral_entropy"]
            se_mid = layer_metrics[num_layers // 2]["spectral_entropy"]
            se_75 = layer_metrics[int(num_layers * 0.75)]["spectral_entropy"]
            se_last = layer_metrics[num_layers]["spectral_entropy"]
            log(f"  [{prompt_idx:2d}/{total_prompts}] {cat:<10} CS={consciousness_score:.3f}  "
                f"SE: L1={se_first:.4f} L{num_layers//2}={se_mid:.4f} "
                f"L{int(num_layers*0.75)}={se_75:.4f} L{num_layers}={se_last:.4f}  "
                f"({elapsed:.1f}s)")

    # ── Analysis ─────────────────────────────────────────────────────
    log()
    log("=" * 80)
    log("LAYER SWEEP RESULTS")
    log("=" * 80)

    num_layers = all_results[0]["num_layers"]
    all_cs = np.array([r["consciousness_score"] for r in all_results])

    # 1. CS-SE correlation at each layer
    log()
    log("--- CS-SE CORRELATION BY LAYER ---")
    log(f"  {'Layer':>6} {'Depth%':>6} {'r':>8} {'p':>10} {'Sig':>6}  {'Mean SE':>10} {'SE CoV':>8}")
    log(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*6}  {'-'*10} {'-'*8}")

    layer_corrs = []
    for layer_idx in range(1, num_layers + 1):
        se_vals = np.array([r["layers"][layer_idx]["spectral_entropy"] for r in all_results])
        depth_pct = layer_idx / num_layers * 100

        if se_vals.std() > 1e-10:
            r, p = stats.pearsonr(all_cs, se_vals)
        else:
            r, p = 0.0, 1.0

        cov = se_vals.std() / se_vals.mean() if se_vals.mean() > 1e-10 else 0
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        log(f"  L{layer_idx:>4} {depth_pct:>5.1f}% {r:>+8.3f} {p:>10.6f} {star:>6}  {se_vals.mean():>10.4f} {cov:>8.3f}")

        layer_corrs.append({
            "layer": layer_idx,
            "depth_pct": depth_pct,
            "cs_se_r": r,
            "cs_se_p": p,
            "se_mean": float(se_vals.mean()),
            "se_std": float(se_vals.std()),
            "se_cov": cov,
        })

    # 2. Find transition point
    log()
    log("--- TRANSITION ANALYSIS ---")

    rs = np.array([lc["cs_se_r"] for lc in layer_corrs])
    ps = np.array([lc["cs_se_p"] for lc in layer_corrs])

    # First layer where CS-SE becomes significant (p < 0.05)
    sig_layers = [lc for lc in layer_corrs if lc["cs_se_p"] < 0.05]
    if sig_layers:
        first_sig = sig_layers[0]
        log(f"  First significant CS-SE layer: L{first_sig['layer']} ({first_sig['depth_pct']:.0f}% depth)")
        log(f"    r={first_sig['cs_se_r']:+.3f}, p={first_sig['cs_se_p']:.6f}")
    else:
        log(f"  No layer reaches significance (p<0.05)")

    # Strongest CS-SE layer
    best_idx = np.argmin(rs)  # most negative
    best = layer_corrs[best_idx]
    log(f"  Strongest CS-SE layer: L{best['layer']} ({best['depth_pct']:.0f}% depth)")
    log(f"    r={best['cs_se_r']:+.3f}, p={best['cs_se_p']:.6f}")

    # Steepest change (largest r drop between consecutive layers)
    r_diffs = np.diff(rs)
    steepest_idx = np.argmin(r_diffs)  # most negative jump
    log(f"  Steepest r change: L{steepest_idx+1} -> L{steepest_idx+2}")
    log(f"    r goes from {rs[steepest_idx]:+.3f} to {rs[steepest_idx+1]:+.3f} (delta={r_diffs[steepest_idx]:+.3f})")

    # 3. By-category layer profiles
    log()
    log("--- CATEGORY-LEVEL LAYER PROFILES ---")
    log(f"  Showing CS-SE r at key depths:")
    key_layers = [1, num_layers//4, num_layers//2, int(num_layers*0.75), num_layers]
    header = f"  {'Category':<12}"
    for kl in key_layers:
        pct = kl / num_layers * 100
        header += f" {'L'+str(kl)+f'({pct:.0f}%)':>12}"
    log(header)
    log("  " + "-" * (12 + 13 * len(key_layers)))

    for cat in PROMPTS.keys():
        cat_cs = np.array([r["consciousness_score"] for r in all_results if r["category"] == cat])
        row = f"  {cat:<12}"
        for kl in key_layers:
            cat_se = np.array([r["layers"][kl]["spectral_entropy"]
                              for r in all_results if r["category"] == cat])
            if cat_se.std() > 1e-10 and len(cat_cs) >= 3:
                r, p = stats.pearsonr(cat_cs, cat_se)
                row += f" {r:>+11.3f} "
            else:
                row += f" {'N/A':>11} "
        log(row)

    # 4. SE trajectory by category
    log()
    log("--- SE TRAJECTORY (mean SE by category across layers) ---")
    log(f"  {'Layer':>6}", end="")
    for cat in PROMPTS.keys():
        log(f" {cat[:8]:>10}", end="")
    log()
    log(f"  {'-'*6}" + f" {'-'*10}" * len(PROMPTS))

    for layer_idx in range(1, num_layers + 1, max(1, num_layers // 16)):  # sample ~16 layers
        log(f"  L{layer_idx:>4}", end="")
        for cat in PROMPTS.keys():
            cat_se = np.mean([r["layers"][layer_idx]["spectral_entropy"]
                             for r in all_results if r["category"] == cat])
            log(f" {cat_se:>10.4f}", end="")
        log()

    # 5. Phase detection: where do categories separate?
    log()
    log("--- CATEGORY SEPARATION BY LAYER ---")
    log("  (ANOVA F-statistic of SE across categories at each layer)")
    log(f"  {'Layer':>6} {'Depth%':>6} {'F-stat':>8} {'p-value':>10} {'Separated?':>12}")
    log(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*12}")

    for layer_idx in range(1, num_layers + 1):
        groups = []
        for cat in PROMPTS.keys():
            cat_se = [r["layers"][layer_idx]["spectral_entropy"]
                     for r in all_results if r["category"] == cat]
            groups.append(cat_se)
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_val = 0, 1
        star = "YES***" if p_val < 0.001 else "YES**" if p_val < 0.01 else "YES*" if p_val < 0.05 else "no"
        log(f"  L{layer_idx:>4} {layer_idx/num_layers*100:>5.1f}% {f_stat:>8.3f} {p_val:>10.6f} {star:>12}")

    # ── Save ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    out_path = DATA_DIR / f"layer_sweep_{model_short}_{ts}.json"
    output = {
        "metadata": {
            "model": args.model,
            "device": args.device,
            "quantize": args.quantize,
            "max_tokens": args.max_tokens,
            "n_prompts": total_prompts,
            "n_layers": num_layers,
            "timestamp": ts,
        },
        "layer_correlations": layer_corrs,
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved: {out_path}")
    log("Done!")


DATA_DIR = Path(__file__).parent.parent / "experiment_outputs"
DATA_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    main()
