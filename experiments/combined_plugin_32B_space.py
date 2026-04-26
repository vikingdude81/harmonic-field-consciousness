"""
Combined Plugin Analysis — 32B Oracle Engine via HF Space API
==============================================================

Calls the experiment_measure API on the Vikingdude81/oracle-engine Space
(Qwen2.5-32B-Instruct on dedicated A10G GPU) to collect consciousness
scores AND compressibility metrics for the same prompt set used in the
local 7B experiment, enabling direct cross-scale comparison.

Usage:
    python experiments/combined_plugin_32B_space.py
    python experiments/combined_plugin_32B_space.py --reps 3
    python experiments/combined_plugin_32B_space.py --output-dir results/32B
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ── Prompt categories (same as 7B experiment) ──────────────────────
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
    parser = argparse.ArgumentParser(description="32B Combined Plugin Analysis via Space API")
    parser.add_argument("--space", default="Vikingdude81/oracle-engine",
                        help="HF Space ID")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max generation tokens per prompt")
    parser.add_argument("--reps", type=int, default=1,
                        help="Repetitions per prompt (for variance estimation)")
    parser.add_argument("--output-dir", default="experiment_outputs",
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout in seconds")
    return parser.parse_args()


def call_space_api(client, prompt, max_tokens=512, timeout=300, retries=3):
    """Call the experiment_measure API on the Space with retries."""
    for attempt in range(retries):
        try:
            result = client.predict(
                prompt,
                max_tokens,
                api_name="/experiment_measure",
            )
            return json.loads(result)
        except Exception as e:
            log(f"  [Attempt {attempt+1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                log(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log(f"  FAILED after {retries} attempts")
                return None


def compute_correlations(results):
    """Compute cross-metric Pearson correlations from results list."""
    # Extract scalar metrics
    metrics = {}
    for r in results:
        for key in ["consciousness_score"]:
            metrics.setdefault(key, []).append(r.get(key, 0))

        comp = r.get("compressibility", {})
        for key in [
            "spectral_entropy", "participation_ratio",
            "effective_dimensionality", "top1_variance_fraction",
        ]:
            metrics.setdefault(f"comp_{key}", []).append(comp.get(key, 0))

        # Correlation-based metrics (flat in Space API response)
        for key in ["compressibility_corr", "fraction_for_90pct", "mean_abs_correlation"]:
            metrics.setdefault(f"cc_{key}", []).append(comp.get(key, 0))

    # Convert to arrays
    metric_names = list(metrics.keys())
    n = len(metric_names)
    arrays = {k: np.array(v, dtype=float) for k, v in metrics.items()}

    # Pearson correlation matrix
    corr_matrix = np.zeros((n, n))
    p_values = np.zeros((n, n))

    from scipy import stats
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                r, p = stats.pearsonr(arrays[metric_names[i]], arrays[metric_names[j]])
                corr_matrix[i, j] = r
                p_values[i, j] = p

    return metric_names, corr_matrix, p_values, arrays


def category_summary(results):
    """Compute per-category means for key metrics."""
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    summary = {}
    for cat, items in categories.items():
        scores = [r["consciousness_score"] for r in items]
        comp_list = [r.get("compressibility", {}) for r in items]
        f90_list = [c.get("fraction_for_90pct", 0) for c in comp_list]
        cc_coef_list = [c.get("compressibility_corr", 0) for c in comp_list]
        mean_corr_list = [c.get("mean_abs_correlation", 0) for c in comp_list]
        spec_ent = [c.get("spectral_entropy", 0) for c in comp_list]

        summary[cat] = {
            "n": len(items),
            "consciousness_score": {"mean": float(np.mean(scores)), "std": float(np.std(scores))},
            "fraction_for_90pct": {"mean": float(np.mean(f90_list)), "std": float(np.std(f90_list))},
            "compression_coefficient": {"mean": float(np.mean(cc_coef_list)), "std": float(np.std(cc_coef_list))},
            "mean_abs_corr": {"mean": float(np.mean(mean_corr_list)), "std": float(np.std(mean_corr_list))},
            "spectral_entropy": {"mean": float(np.mean(spec_ent)), "std": float(np.std(spec_ent))},
        }
    return summary


def main():
    args = parse_args()

    log("=" * 70)
    log("32B Combined Plugin Analysis via HF Space API")
    log("=" * 70)
    log(f"Space: {args.space}")
    log(f"Max tokens: {args.max_tokens}")
    log(f"Reps per prompt: {args.reps}")
    log()

    # Connect to Space
    log("Connecting to HF Space...")
    try:
        from gradio_client import Client
        client = Client(args.space)
        log("Connected!")
    except Exception as e:
        log(f"ERROR connecting to Space: {e}")
        log("Make sure the Space is running with dedicated GPU.")
        sys.exit(1)

    log()

    # Collect results
    all_results = []
    total_prompts = sum(len(v) for v in PROMPTS.values()) * args.reps
    done = 0

    for category, prompts in PROMPTS.items():
        log(f"--- Category: {category} ({len(prompts)} prompts x {args.reps} reps) ---")
        for pi, prompt in enumerate(prompts):
            for rep in range(args.reps):
                done += 1
                label = f"[{done}/{total_prompts}]"
                log(f"  {label} {category}[{pi}] rep={rep}: {prompt[:60]}...")

                t0 = time.time()
                result = call_space_api(client, prompt, args.max_tokens, args.timeout)
                elapsed = time.time() - t0

                if result is None:
                    log(f"    SKIPPED (API failure)")
                    continue

                # Enrich with metadata
                result["category"] = category
                result["prompt_index"] = pi
                result["prompt"] = prompt
                result["rep"] = rep
                result["api_time"] = round(elapsed, 2)

                # Quick summary
                cs = result.get("consciousness_score", 0)
                comp = result.get("compressibility", {})
                f90 = comp.get("fraction_for_90pct", 0)
                cc_coef = comp.get("compressibility_corr", 0)
                se = comp.get("spectral_entropy", 0)

                log(f"    CS={cs:.4f}  CC={cc_coef:.4f}  f90={f90:.3f}  "
                    f"SE={se:.4f}  ({elapsed:.1f}s)")

                all_results.append(result)

        log()

    if not all_results:
        log("ERROR: No results collected!")
        sys.exit(1)

    log(f"Collected {len(all_results)} results total.")
    log()

    # ── Analysis ────────────────────────────────────────────────────
    log("=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Per-category summary
    log("\n--- Per-Category Summary ---")
    cat_summary = category_summary(all_results)
    for cat, stats in sorted(cat_summary.items()):
        log(f"\n  {cat.upper()} (n={stats['n']}):")
        log(f"    Consciousness:  {stats['consciousness_score']['mean']:.4f} +/- {stats['consciousness_score']['std']:.4f}")
        log(f"    Compress Coef:  {stats['compression_coefficient']['mean']:.4f} +/- {stats['compression_coefficient']['std']:.4f}")
        log(f"    f90 (correlations): {stats['fraction_for_90pct']['mean']:.3f} +/- {stats['fraction_for_90pct']['std']:.3f}")
        log(f"    Mean |corr|:    {stats['mean_abs_corr']['mean']:.4f} +/- {stats['mean_abs_corr']['std']:.4f}")
        log(f"    Spectral Entropy: {stats['spectral_entropy']['mean']:.4f} +/- {stats['spectral_entropy']['std']:.4f}")

    # Cross-metric correlations
    log("\n--- Cross-Metric Correlations ---")
    try:
        metric_names, corr_matrix, p_values, arrays = compute_correlations(all_results)

        # Print significant correlations
        sig_pairs = []
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                r = corr_matrix[i, j]
                p = p_values[i, j]
                if p < 0.05:
                    sig_pairs.append((metric_names[i], metric_names[j], r, p))

        if sig_pairs:
            log(f"\n  Significant correlations (p < 0.05):")
            for m1, m2, r, p in sorted(sig_pairs, key=lambda x: x[3]):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                log(f"    {m1} <-> {m2}: r={r:.3f}, p={p:.4f} {star}")
        else:
            log("  No significant correlations at p < 0.05")

        # Full matrix (abbreviated)
        log(f"\n  Full correlation matrix ({len(metric_names)} metrics):")
        header = "".ljust(30) + "  ".join(m[:8].ljust(8) for m in metric_names)
        log(f"    {header}")
        for i, m in enumerate(metric_names):
            row = m.ljust(30) + "  ".join(f"{corr_matrix[i,j]:+.3f}" if i != j else "  1.000"
                                           for j in range(len(metric_names)))
            log(f"    {row}")

    except ImportError:
        log("  scipy not available, skipping correlation analysis")
    except Exception as e:
        log(f"  Error in correlation analysis: {e}")

    # ── ANOVA across categories ──────────────────────────────────
    log("\n--- ANOVA (category effect on key metrics) ---")
    try:
        from scipy import stats as sp_stats

        for metric_key, extract_fn in [
            ("consciousness_score", lambda r: r.get("consciousness_score", 0)),
            ("compressibility_corr", lambda r: r.get("compressibility", {}).get("compressibility_corr", 0)),
            ("fraction_for_90pct", lambda r: r.get("compressibility", {}).get("fraction_for_90pct", 0)),
            ("spectral_entropy", lambda r: r.get("compressibility", {}).get("spectral_entropy", 0)),
            ("mean_abs_correlation", lambda r: r.get("compressibility", {}).get("mean_abs_correlation", 0)),
        ]:
            groups = {}
            for r in all_results:
                groups.setdefault(r["category"], []).append(extract_fn(r))

            group_values = list(groups.values())
            if len(group_values) >= 2 and all(len(g) >= 2 for g in group_values):
                F, p = sp_stats.f_oneway(*group_values)
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                log(f"  {metric_key}: F={F:.3f}, p={p:.4f} {star}")
            else:
                log(f"  {metric_key}: insufficient data for ANOVA")

    except ImportError:
        log("  scipy not available, skipping ANOVA")
    except Exception as e:
        log(f"  Error in ANOVA: {e}")

    # ── Save results ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"combined_plugin_32B_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "space": args.space,
            "timestamp": timestamp,
            "n_prompts": len(all_results),
            "reps_per_prompt": args.reps,
            "max_tokens": args.max_tokens,
            "categories": list(PROMPTS.keys()),
        },
        "results": all_results,
        "category_summary": cat_summary,
    }

    # Add correlations if computed
    try:
        output_data["correlations"] = {
            "metrics": metric_names,
            "matrix": corr_matrix.tolist(),
            "p_values": p_values.tolist(),
        }
        if sig_pairs:
            output_data["significant_correlations"] = [
                {"metric_1": m1, "metric_2": m2, "r": round(r, 4), "p": round(p, 6)}
                for m1, m2, r, p in sig_pairs
            ]
    except Exception:
        pass

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to: {output_file}")
    log()
    log("Done!")


if __name__ == "__main__":
    main()
