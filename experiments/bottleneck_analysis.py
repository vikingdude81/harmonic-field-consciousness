"""
Representation Bottleneck Analysis
===================================
Deep dive into Qwen's extreme compression (SE 0.75 → 0.005 → 0.75)
vs Mistral's gradual refinement.

Measures at each layer:
  - Participation ratio (effective number of dimensions)
  - Top-k eigenvalue concentration
  - Rank of the representation (numerical rank at various thresholds)
  - Category-specific dimensionality differences

Usage:
    python experiments/bottleneck_analysis.py --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --quantize
    python experiments/bottleneck_analysis.py --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda:0 --quantize
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

from consciousness_circuit import UniversalCircuit

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

CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


def load_model(model_name, device, quantize):
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


def analyze_layer_geometry(hidden_seq):
    """
    Analyze the geometric properties of a hidden state matrix.

    Args:
        hidden_seq: numpy array [seq_len, hidden_dim]

    Returns:
        dict with geometric metrics
    """
    seq_len, hidden_dim = hidden_seq.shape

    # Center the data
    centered = hidden_seq - hidden_seq.mean(axis=0, keepdims=True)

    # SVD to get singular values (eigenvalues of covariance are S^2)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"error": "SVD failed"}

    # Eigenvalues (proportional to variance along each principal component)
    eigenvalues = S ** 2
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        return {"error": "Zero variance"}

    # Normalized eigenvalue distribution
    p = eigenvalues / total_var

    # 1. Participation Ratio (effective number of dimensions)
    # PR = (sum p_i)^2 / sum(p_i^2) = 1 / sum(p_i^2)  since sum(p_i) = 1
    pr = 1.0 / np.sum(p ** 2)

    # 2. Spectral entropy
    p_nonzero = p[p > 1e-12]
    se = -np.sum(p_nonzero * np.log(p_nonzero)) / np.log(len(p_nonzero)) if len(p_nonzero) > 1 else 0

    # 3. Top-k eigenvalue concentration
    cumvar = np.cumsum(p)
    dims_for_50 = int(np.searchsorted(cumvar, 0.50) + 1)
    dims_for_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dims_for_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    top1_frac = float(p[0])
    top5_frac = float(p[:5].sum()) if len(p) >= 5 else float(p.sum())
    top10_frac = float(p[:10].sum()) if len(p) >= 10 else float(p.sum())

    # 4. Numerical rank (number of eigenvalues above threshold)
    max_eig = eigenvalues[0]
    rank_1e2 = int(np.sum(eigenvalues > max_eig * 1e-2))  # within 1% of max
    rank_1e4 = int(np.sum(eigenvalues > max_eig * 1e-4))  # within 0.01% of max
    rank_1e6 = int(np.sum(eigenvalues > max_eig * 1e-6))  # within 0.0001% of max

    # 5. Eigenvalue decay rate (log-linear slope of top eigenvalues)
    n_fit = min(20, len(eigenvalues))
    log_eigs = np.log(eigenvalues[:n_fit] + 1e-12)
    x = np.arange(n_fit)
    slope, _ = np.polyfit(x, log_eigs, 1)

    return {
        "participation_ratio": float(pr),
        "spectral_entropy": float(se),
        "dims_for_50pct": dims_for_50,
        "dims_for_90pct": dims_for_90,
        "dims_for_99pct": dims_for_99,
        "top1_frac": top1_frac,
        "top5_frac": top5_frac,
        "top10_frac": top10_frac,
        "rank_1e2": rank_1e2,
        "rank_1e4": rank_1e4,
        "rank_1e6": rank_1e6,
        "eigenvalue_decay_slope": float(slope),
        "max_eigenvalue": float(max_eig),
        "total_variance": float(total_var),
        "top20_eigenvalues": eigenvalues[:20].tolist(),
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
    log("REPRESENTATION BOTTLENECK ANALYSIS")
    log("=" * 80)
    log(f"Model: {args.model}")
    log(f"Device: {args.device}")
    log(f"Prompts: {total_prompts}")
    log()

    log(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, args.device, args.quantize)
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    log(f"  Model loaded: {hidden_dim} hidden dims, {num_layers} layers")

    circuit = UniversalCircuit()

    all_results = []
    prompt_idx = 0

    for cat, prompts in PROMPTS.items():
        for pidx, prompt in enumerate(prompts):
            prompt_idx += 1
            t0 = time.time()

            # CS measurement
            cs_result = circuit.measure(model, tokenizer, prompt)

            # Generate + extract ALL layers
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                gen_outputs = model.generate(
                    input_ids, max_new_tokens=args.max_tokens,
                    do_sample=False, temperature=1.0
                )
            full_ids = gen_outputs[0:1]

            with torch.no_grad():
                outputs = model(full_ids, output_hidden_states=True, return_dict=True)

            gen_text = tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)

            # Analyze geometry at every layer
            layer_geometry = {}
            for layer_idx in range(1, num_layers + 1):
                hidden_seq = outputs.hidden_states[layer_idx][0].cpu().float().numpy()
                geom = analyze_layer_geometry(hidden_seq)
                layer_geometry[layer_idx] = geom

            elapsed = time.time() - t0

            result = {
                "category": cat,
                "prompt_index": pidx,
                "prompt": prompt,
                "generated_preview": gen_text[:100],
                "consciousness_score": cs_result.score,
                "dimension_scores": cs_result.dimension_scores,
                "num_layers": num_layers,
                "seq_len": int(full_ids.shape[1]),
                "hidden_dim": hidden_dim,
                "layers": layer_geometry,
            }
            all_results.append(result)

            # Progress: show key metrics at a few layers
            pr_first = layer_geometry[1]["participation_ratio"]
            pr_mid = layer_geometry[num_layers // 2]["participation_ratio"]
            pr_75 = layer_geometry[int(num_layers * 0.75)]["participation_ratio"]
            pr_last = layer_geometry[num_layers]["participation_ratio"]
            log(f"  [{prompt_idx:2d}/{total_prompts}] {cat:<10} CS={cs_result.score:.3f}  "
                f"PR: L1={pr_first:.1f} L{num_layers//2}={pr_mid:.1f} "
                f"L{int(num_layers*0.75)}={pr_75:.1f} L{num_layers}={pr_last:.1f}  "
                f"({elapsed:.1f}s)")

            # Free GPU memory
            del outputs
            torch.cuda.empty_cache()

    # ── Analysis ─────────────────────────────────────────────────────
    log()
    log("=" * 80)
    log("BOTTLENECK ANALYSIS RESULTS")
    log("=" * 80)

    # 1. Participation Ratio Profile
    log()
    log("--- PARTICIPATION RATIO BY LAYER ---")
    log(f"  (Effective number of dimensions used, out of {hidden_dim} total)")
    log(f"  {'Layer':>6} {'Depth%':>6} {'Mean PR':>10} {'PR/dim':>8} {'Top1%':>8} {'Top5%':>8} "
        f"{'d50':>6} {'d90':>6} {'d99':>6} {'Rank1e-4':>8}")
    log(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} "
        f"{'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    layer_profiles = []
    for layer_idx in range(1, num_layers + 1):
        metrics = [r["layers"][layer_idx] for r in all_results]
        depth_pct = layer_idx / num_layers * 100

        mean_pr = np.mean([m["participation_ratio"] for m in metrics])
        mean_top1 = np.mean([m["top1_frac"] for m in metrics])
        mean_top5 = np.mean([m["top5_frac"] for m in metrics])
        mean_d50 = np.mean([m["dims_for_50pct"] for m in metrics])
        mean_d90 = np.mean([m["dims_for_90pct"] for m in metrics])
        mean_d99 = np.mean([m["dims_for_99pct"] for m in metrics])
        mean_rank = np.mean([m["rank_1e4"] for m in metrics])
        mean_se = np.mean([m["spectral_entropy"] for m in metrics])
        mean_decay = np.mean([m["eigenvalue_decay_slope"] for m in metrics])

        pr_frac = mean_pr / hidden_dim * 100

        log(f"  L{layer_idx:>4} {depth_pct:>5.1f}% {mean_pr:>10.1f} {pr_frac:>7.2f}% "
            f"{mean_top1:>7.1%} {mean_top5:>7.1%} {mean_d50:>6.0f} {mean_d90:>6.0f} "
            f"{mean_d99:>6.0f} {mean_rank:>8.0f}")

        layer_profiles.append({
            "layer": layer_idx,
            "depth_pct": depth_pct,
            "mean_pr": float(mean_pr),
            "pr_fraction": float(pr_frac),
            "mean_top1": float(mean_top1),
            "mean_top5": float(mean_top5),
            "mean_d50": float(mean_d50),
            "mean_d90": float(mean_d90),
            "mean_d99": float(mean_d99),
            "mean_rank_1e4": float(mean_rank),
            "mean_se": float(mean_se),
            "mean_decay_slope": float(mean_decay),
        })

    # 2. Find bottleneck
    log()
    log("--- BOTTLENECK DETECTION ---")
    prs = np.array([lp["mean_pr"] for lp in layer_profiles])
    min_pr_idx = np.argmin(prs)
    max_pr_idx = np.argmax(prs)
    log(f"  Most compressed layer: L{min_pr_idx+1} ({layer_profiles[min_pr_idx]['depth_pct']:.0f}% depth)")
    log(f"    PR = {prs[min_pr_idx]:.1f} ({prs[min_pr_idx]/hidden_dim*100:.2f}% of {hidden_dim} dims)")
    log(f"    Top-1 eigenvalue captures: {layer_profiles[min_pr_idx]['mean_top1']:.1%}")
    log(f"    Dims for 90% variance: {layer_profiles[min_pr_idx]['mean_d90']:.0f}")
    log(f"  Least compressed layer: L{max_pr_idx+1} ({layer_profiles[max_pr_idx]['depth_pct']:.0f}% depth)")
    log(f"    PR = {prs[max_pr_idx]:.1f} ({prs[max_pr_idx]/hidden_dim*100:.2f}% of {hidden_dim} dims)")
    log(f"  Compression ratio (max/min PR): {prs[max_pr_idx]/prs[min_pr_idx]:.1f}x")

    # 3. By-category bottleneck differences
    log()
    log("--- CATEGORY-SPECIFIC BOTTLENECK ---")
    log(f"  (Participation ratio at the most compressed layer L{min_pr_idx+1})")
    log(f"  {'Category':<12} {'Mean PR':>10} {'d90':>6} {'Top1%':>8}")
    log(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*8}")

    bn_layer = min_pr_idx + 1
    for cat in CATEGORIES:
        cat_metrics = [r["layers"][bn_layer] for r in all_results if r["category"] == cat]
        if not cat_metrics:
            continue
        cat_pr = np.mean([m["participation_ratio"] for m in cat_metrics])
        cat_d90 = np.mean([m["dims_for_90pct"] for m in cat_metrics])
        cat_top1 = np.mean([m["top1_frac"] for m in cat_metrics])
        log(f"  {cat:<12} {cat_pr:>10.1f} {cat_d90:>6.0f} {cat_top1:>7.1%}")

    # 4. Category separation in PR space
    log()
    log("--- CATEGORY SEPARATION IN PR SPACE ---")
    log(f"  (ANOVA F-stat of PR across categories at each layer)")
    for layer_idx in range(1, num_layers + 1, max(1, num_layers // 8)):
        groups = []
        for cat in CATEGORIES:
            cat_pr = [r["layers"][layer_idx]["participation_ratio"]
                     for r in all_results if r["category"] == cat]
            groups.append(cat_pr)
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_val = 0, 1
        depth = layer_idx / num_layers * 100
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        log(f"  L{layer_idx:>3} ({depth:>4.0f}%): F={f_stat:.3f}, p={p_val:.4f} {star}")

    # ── Save ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    out_path = DATA_DIR / f"bottleneck_{model_short}_{ts}.json"
    output = {
        "metadata": {
            "model": args.model,
            "device": args.device,
            "quantize": args.quantize,
            "max_tokens": args.max_tokens,
            "n_prompts": total_prompts,
            "n_layers": num_layers,
            "hidden_dim": hidden_dim,
            "timestamp": ts,
        },
        "layer_profiles": layer_profiles,
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
