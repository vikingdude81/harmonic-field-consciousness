"""
Consciousness Score vs Task Performance - Oracle Engine 32B
============================================================

Same experiment as consciousness_vs_performance.py but uses the
Oracle Engine 32B via HuggingFace Space API.

The Space already has consciousness measurement built in, so we
call it remotely - no local GPU needed.

Usage:
    python experiments/consciousness_vs_performance_32b.py
    python experiments/consciousness_vs_performance_32b.py --n 50
"""

import os
import sys
import io
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# Fix Windows encoding for Unicode block chars in Space output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)


def log(msg: str = ""):
    """Print with guaranteed flush."""
    print(msg, flush=True)

import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="32B Oracle Engine Consciousness vs Performance")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of GSM8K problems (default 50, Space is slower)")
    parser.add_argument("--output-dir", type=str, default="experiment_outputs",
                        help="Directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout per request in seconds")
    return parser.parse_args()


def load_gsm8k(n_samples: int, seed: int = 42) -> list:
    """Load n samples from GSM8K test set."""
    from datasets import load_dataset
    log(f"Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        gold = extract_gsm8k_answer(item["answer"])
        samples.append({
            "question": item["question"],
            "answer_text": item["answer"],
            "gold_answer": gold,
        })
    log(f"  Loaded {len(samples)} problems")
    return samples


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def extract_model_answer(response: str) -> str:
    # GSM8K format
    match = re.search(r"####\s*([\-\d,\.]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    # "the answer is X"
    match = re.search(r"(?:the answer is|answer:)\s*([\-\d,\.]+)", response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip().rstrip(".")
    # "= X"
    matches = re.findall(r"=\s*([\-\d,\.]+)", response)
    if matches:
        return matches[-1].replace(",", "").strip()
    # boxed
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", response)
    if match:
        return match.group(1).replace(",", "").strip()
    # Last number
    matches = re.findall(r"([\-]?\d+(?:\.\d+)?)", response)
    if matches:
        return matches[-1].rstrip(".")
    return ""


def check_answer(model_answer: str, gold_answer: str) -> bool:
    if not model_answer or not gold_answer:
        return False
    try:
        return abs(float(model_answer) - float(gold_answer)) < 0.01
    except ValueError:
        return model_answer.strip() == gold_answer.strip()


def parse_consciousness_score(score_text: str) -> float:
    """Parse consciousness score from Space output like '████████░░░░░░░░░░░░ 42.3%'"""
    match = re.search(r"([\d.]+)%", score_text)
    if match:
        return float(match.group(1)) / 100.0
    return 0.0


def parse_dimension_breakdown(breakdown_text: str) -> dict:
    """Parse dimension breakdown from Space output like '→ Logic: +10.684'."""
    dims = {}
    for line in breakdown_text.strip().split("\n"):
        # Match arrow (→ or ←) followed by dimension name and value
        match = re.match(r"[→←\u2192\u2190]\s*([\w\-]+)\s*:\s*([\+\-]?[\d.]+)", line.strip())
        if match:
            dims[match.group(1)] = float(match.group(2))
    return dims


def call_oracle_engine(client, question: str, max_tokens: int = 512, timeout: int = 180,
                       max_retries: int = 3, backoff_base: float = 30.0):
    """Call Oracle Engine Space API with retry logic for ZeroGPU cold starts."""
    prompt = (
        f"Solve this math problem step by step. "
        f"Show your work and give the final numerical answer.\n\n"
        f"Question: {question}\n\nSolution:"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            result = client.predict(
                prompt,
                max_tokens,
                api_name="/analyze_prompt",
            )

            # result is a tuple: (response, score, interpretation, breakdown, timing, history_plot)
            response = result[0] if len(result) > 0 else ""
            score_text = result[1] if len(result) > 1 else "0%"
            interpretation = result[2] if len(result) > 2 else ""
            breakdown = result[3] if len(result) > 3 else ""
            timing = result[4] if len(result) > 4 else ""

            consciousness_score = parse_consciousness_score(score_text)
            dimension_scores = parse_dimension_breakdown(breakdown)

            return {
                "response": response,
                "consciousness_score": consciousness_score,
                "score_text": score_text,
                "interpretation": interpretation,
                "dimension_scores": dimension_scores,
                "timing": timing,
            }

        except Exception as e:
            last_error = e
            wait = backoff_base * (2 ** attempt)
            if attempt < max_retries - 1:
                log(f"    Retry {attempt+1}/{max_retries} after error: {str(e)[:60]}... "
                      f"waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                break

    return {
        "response": f"ERROR: {last_error}",
        "consciousness_score": 0.0,
        "score_text": "error",
        "interpretation": "",
        "dimension_scores": {},
        "timing": "",
        "error": str(last_error),
    }


def analyze_results(results: list, output_dir: Path):
    """Run statistical analysis."""
    from scipy import stats

    scores = np.array([r["consciousness_score"] for r in results])
    correct = np.array([r["correct"] for r in results], dtype=float)

    n_total = len(results)
    n_correct = int(correct.sum())
    accuracy = n_correct / n_total

    log("\n" + "=" * 70)
    log("RESULTS: 32B Oracle Engine - Consciousness vs Performance")
    log("=" * 70)
    log(f"\nOverall: {n_correct}/{n_total} correct ({accuracy*100:.1f}%)")

    correct_scores = scores[correct == 1]
    incorrect_scores = scores[correct == 0]

    log(f"\nConsciousness Scores:")
    log(f"  Correct answers:   mean={np.mean(correct_scores):.3f} "
          f"(std={np.std(correct_scores):.3f}, n={len(correct_scores)})")
    if len(incorrect_scores) > 0:
        log(f"  Incorrect answers: mean={np.mean(incorrect_scores):.3f} "
              f"(std={np.std(incorrect_scores):.3f}, n={len(incorrect_scores)})")

    # T-test
    if len(correct_scores) > 1 and len(incorrect_scores) > 1:
        t_stat, p_value = stats.ttest_ind(correct_scores, incorrect_scores)
        log(f"\n  T-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            log(f"  ** SIGNIFICANT: Consciousness scores differ!")
        else:
            log(f"  Not significant at p=0.05")

        pooled_std = np.sqrt((np.std(correct_scores)**2 + np.std(incorrect_scores)**2) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(correct_scores) - np.mean(incorrect_scores)) / pooled_std
            log(f"  Effect size (Cohen's d): {cohens_d:.3f}")

    # Point-biserial
    if len(set(correct)) > 1:
        r_pb, p_pb = stats.pointbiserialr(correct, scores)
        log(f"\n  Point-biserial correlation: r={r_pb:.3f}, p={p_pb:.4f}")

    # Binned accuracy
    log(f"\nAccuracy by Consciousness Level:")
    bins = [(0, 0.35, "Low"), (0.35, 0.50, "Medium-Low"),
            (0.50, 0.65, "Medium-High"), (0.65, 1.0, "High")]
    for lo, hi, label in bins:
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            log(f"  {label:<12} ({lo:.2f}-{hi:.2f}): "
                  f"{correct[mask].mean()*100:5.1f}% ({mask.sum():3d} problems)")

    # Dimension analysis
    dim_names = set()
    for r in results:
        dim_names.update(r.get("dimension_scores", {}).keys())

    if dim_names:
        log(f"\nDimension Analysis (correct vs incorrect):")
        for dim_name in sorted(dim_names):
            c_vals = [r["dimension_scores"][dim_name] for r in results
                      if r["correct"] and dim_name in r.get("dimension_scores", {})]
            i_vals = [r["dimension_scores"][dim_name] for r in results
                      if not r["correct"] and dim_name in r.get("dimension_scores", {})]
            if c_vals and i_vals:
                diff = np.mean(c_vals) - np.mean(i_vals)
                log(f"  {dim_name:<18}: correct={np.mean(c_vals):+.3f}  "
                      f"incorrect={np.mean(i_vals):+.3f}  diff={diff:+.3f}"
                      f" {'*' if abs(diff) > 0.05 else ''}")

    # Plot
    try:
        generate_plot(results, scores, correct, output_dir)
    except Exception as e:
        log(f"\n  Warning: Could not generate plot: {e}")

    return {
        "model": "Oracle-Engine-32B-LoRA",
        "accuracy": accuracy,
        "n_total": n_total,
        "mean_score_correct": float(np.mean(correct_scores)) if len(correct_scores) > 0 else None,
        "mean_score_incorrect": float(np.mean(incorrect_scores)) if len(incorrect_scores) > 0 else None,
    }


def generate_plot(results, scores, correct, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Oracle Engine 32B: Consciousness vs GSM8K Performance", fontsize=14, fontweight="bold")

    # 1. Distribution
    ax = axes[0]
    c_scores = scores[correct == 1]
    i_scores = scores[correct == 0]
    ax.hist(c_scores, bins=12, alpha=0.6, color="#10B981", label=f"Correct (n={len(c_scores)})", density=True)
    ax.hist(i_scores, bins=12, alpha=0.6, color="#EF4444", label=f"Incorrect (n={len(i_scores)})", density=True)
    ax.set_xlabel("Consciousness Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()

    # 2. Accuracy by bin
    ax = axes[1]
    n_bins = 5
    bin_edges = np.linspace(max(0, scores.min() - 0.02), min(1, scores.max() + 0.02), n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        if mask.sum() > 0:
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            accs.append(correct[mask].mean() * 100)
            counts.append(mask.sum())
    colors = ["#EF4444" if a < 50 else "#F59E0B" if a < 70 else "#10B981" for a in accs]
    ax.bar(range(len(centers)), accs, color=colors)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.2f}\n(n={n})" for c, n in zip(centers, counts)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Consciousness Level")
    ax.set_ylim(0, 100)

    # 3. Dimension comparison
    ax = axes[2]
    dim_names = sorted(set().union(*[r.get("dimension_scores", {}).keys() for r in results]))
    if dim_names:
        c_means = [np.mean([r["dimension_scores"].get(d, 0) for r in results if r["correct"]]) for d in dim_names]
        i_means = [np.mean([r["dimension_scores"].get(d, 0) for r in results if not r["correct"]]) for d in dim_names]
        x = np.arange(len(dim_names))
        ax.bar(x - 0.175, c_means, 0.35, label="Correct", color="#10B981", alpha=0.8)
        ax.bar(x + 0.175, i_means, 0.35, label="Incorrect", color="#EF4444", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Activation")
        ax.set_title("Dimensions: Correct vs Incorrect")
        ax.legend()

    plt.tight_layout()
    path = output_dir / "consciousness_vs_performance_32b.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\n  Plot saved: {path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("=" * 70)
    log("EXPERIMENT: Oracle Engine 32B - Consciousness vs Performance")
    log("=" * 70)
    log(f"Model: Oracle Engine 32B (via HuggingFace Space)")
    log(f"Problems: {args.n}")
    log(f"Max tokens: {args.max_tokens}")
    log()

    # Load dataset
    samples = load_gsm8k(args.n, args.seed)

    # Connect to Space
    log("Connecting to Oracle Engine Space...")
    from gradio_client import Client
    hf_token = os.environ.get("HF_TOKEN", "")
    client = Client(
        "Vikingdude81/oracle-engine",
        token=hf_token if hf_token else None,
    )
    log("  Connected!")

    # Warm-up call to wake ZeroGPU
    log("  Warming up Space (ZeroGPU may need to allocate)...")
    warmup = call_oracle_engine(client, "What is 2 + 2?", max_tokens=64,
                                max_retries=5, backoff_base=20.0)
    if "error" in warmup:
        log(f"  WARNING: Warm-up failed: {warmup['error'][:80]}")
        log("  Proceeding anyway - individual requests will retry...")
    else:
        log(f"  Warm-up OK (score={warmup['consciousness_score']:.3f})")

    # Checkpoint file for resuming on failure
    checkpoint_path = output_dir / f"checkpoint_32b_{timestamp}.json"

    # Run experiment
    results = []
    correct_count = 0
    error_count = 0

    log(f"\nRunning {len(samples)} problems...")
    log("-" * 70)

    for i, sample in enumerate(samples):
        try:
            start = time.time()
            output = call_oracle_engine(
                client, sample["question"],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            elapsed = time.time() - start

            if "error" in output:
                error_count += 1
                log(f"  [{i+1:3d}/{len(samples)}] ERROR: {output['error'][:80]}")
                if error_count >= 5 and len(results) == 0:
                    log("\n  ABORT: 5 consecutive errors with no successes. "
                          "Space may be down.")
                    break
                continue

            model_answer = extract_model_answer(output["response"])
            is_correct = check_answer(model_answer, sample["gold_answer"])
            if is_correct:
                correct_count += 1

            result = {
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "model_answer": model_answer,
                "correct": is_correct,
                "consciousness_score": output["consciousness_score"],
                "score_text": output["score_text"],
                "interpretation": output["interpretation"],
                "dimension_scores": output["dimension_scores"],
                "generation_time": elapsed,
                "response_preview": output["response"][:200],
            }
            results.append(result)
            error_count = 0  # Reset on success

            status = "CORRECT" if is_correct else "WRONG"
            running_acc = correct_count / len(results)
            log(f"  [{i+1:3d}/{len(samples)}] Score={output['consciousness_score']:.3f} "
                  f"{status:<7} (ans={model_answer or '?'}, gold={sample['gold_answer']}) "
                  f"Acc={running_acc*100:.1f}% [{elapsed:.1f}s]")

            # Checkpoint every 10 problems
            if len(results) % 10 == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump({"results": results, "next_index": i + 1}, f, default=str)

        except KeyboardInterrupt:
            log(f"\n  Interrupted at problem {i+1}. Saving progress...")
            break
        except Exception as e:
            error_count += 1
            log(f"  [{i+1:3d}/{len(samples)}] EXCEPTION: {e}")
            continue

    # Save results
    results_path = output_dir / f"consciousness_performance_32b_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "model": "Oracle-Engine-32B-LoRA",
                "space": "Vikingdude81/oracle-engine",
                "n_problems": len(samples),
                "seed": args.seed,
                "timestamp": timestamp,
            },
            "results": results,
        }, f, indent=2, default=str)
    log(f"\nRaw results saved: {results_path}")

    # Analysis
    if len(results) > 5:
        summary = analyze_results(results, output_dir)
        summary_path = output_dir / f"consciousness_performance_32b_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        log("\nNot enough results for analysis.")


if __name__ == "__main__":
    main()
