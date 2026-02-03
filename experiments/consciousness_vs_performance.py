"""
Consciousness Score vs Task Performance Experiment
====================================================

Tests the hypothesis: Do higher consciousness scores predict correct answers
on reasoning tasks?

Uses GSM8K (grade school math) as the benchmark because:
1. Clear right/wrong answers (numerical)
2. Requires multi-step reasoning (should correlate with consciousness)
3. Well-studied benchmark with known difficulty distribution

Usage:
    python experiments/consciousness_vs_performance.py

    # With custom model:
    python experiments/consciousness_vs_performance.py --model "Qwen/Qwen2.5-7B-Instruct"

    # Smaller sample for quick test:
    python experiments/consciousness_vs_performance.py --n 20

    # Use specific GPU:
    python experiments/consciousness_vs_performance.py --device cuda:0
"""

import os
import sys
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Consciousness vs Performance Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of GSM8K problems to test")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to load model on")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for generation")
    parser.add_argument("--output-dir", type=str, default="experiment_outputs",
                        help="Directory for results")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation (1 = sequential)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# ============================================================================
# GSM8K Dataset
# ============================================================================

def load_gsm8k(n_samples: int, seed: int = 42) -> list:
    """Load n samples from GSM8K test set."""
    from datasets import load_dataset

    print(f"Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Shuffle deterministically and take n samples
    ds = ds.shuffle(seed=seed)
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        samples.append({
            "question": item["question"],
            "answer_text": item["answer"],
            "gold_answer": extract_gsm8k_answer(item["answer"]),
        })

    print(f"  Loaded {len(samples)} problems")
    return samples


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K format (after ####)."""
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).replace(",", "").strip()
    return ""


def extract_model_answer(response: str) -> str:
    """
    Extract numerical answer from model's response.

    Tries multiple patterns:
    1. "#### <number>" (if model follows GSM8K format)
    2. "The answer is <number>"
    3. "= <number>" at end
    4. Last number in the response
    """
    # Pattern 1: GSM8K format
    match = re.search(r"####\s*([\-\d,\.]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Pattern 2: "the answer is X"
    match = re.search(r"(?:the answer is|answer:)\s*([\-\d,\.]+)", response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip().rstrip(".")

    # Pattern 3: "= X" near the end (last occurrence)
    matches = re.findall(r"=\s*([\-\d,\.]+)", response)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Pattern 4: boxed answer (common in math formatting)
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Pattern 5: Last number in the response
    matches = re.findall(r"([\-]?\d+(?:\.\d+)?)", response)
    if matches:
        return matches[-1].rstrip(".")

    return ""


def check_answer(model_answer: str, gold_answer: str) -> bool:
    """Check if model's answer matches gold answer."""
    if not model_answer or not gold_answer:
        return False

    try:
        model_val = float(model_answer)
        gold_val = float(gold_answer)
        # Allow small floating point differences
        return abs(model_val - gold_val) < 0.01
    except ValueError:
        return model_answer.strip() == gold_answer.strip()


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_id: str, device: str, quantize: bool):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    print(f"  Device: {device}, Quantize: {quantize}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if quantize:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": device}
        except ImportError:
            print("  Warning: bitsandbytes not available, loading without quantization")
            load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    print(f"  Model loaded: {model.config.hidden_size} hidden dims, "
          f"{model.config.num_hidden_layers} layers")

    return model, tokenizer


# ============================================================================
# Generation + Consciousness Measurement
# ============================================================================

def generate_and_measure(
    model,
    tokenizer,
    circuit,
    question: str,
    max_tokens: int = 512,
) -> dict:
    """
    Generate an answer and measure consciousness on the question.

    Returns dict with:
        response: model's text response
        consciousness: UniversalResult from v3.2 circuit
        generation_time: seconds for generation
    """
    # Format as chat
    prompt = f"Solve this math problem step by step. Show your work and give the final numerical answer.\n\nQuestion: {question}\n\nSolution:"

    # Measure consciousness on the QUESTION (before generation)
    # This tests: does the model's internal processing depth predict success?
    consciousness = circuit.measure(
        model, tokenizer, prompt,
        aggregation="mean",
        length_normalize=True,
    )

    # Generate response
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = prompt

    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - gen_start

    # Decode
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "response": response,
        "consciousness": consciousness,
        "generation_time": gen_time,
    }


# ============================================================================
# Statistical Analysis
# ============================================================================

def analyze_results(results: list, output_dir: Path):
    """Run statistical analysis and generate visualizations."""
    from scipy import stats

    # Extract arrays
    scores = np.array([r["consciousness_score"] for r in results])
    correct = np.array([r["correct"] for r in results], dtype=float)
    confidence = np.array([r["confidence"] for r in results])
    diversity = np.array([r["dimension_diversity"] for r in results])

    n_total = len(results)
    n_correct = int(correct.sum())
    accuracy = n_correct / n_total

    print("\n" + "=" * 70)
    print("RESULTS: Consciousness Score vs Task Performance")
    print("=" * 70)

    print(f"\nOverall: {n_correct}/{n_total} correct ({accuracy*100:.1f}%)")

    # Split into correct/incorrect groups
    correct_scores = scores[correct == 1]
    incorrect_scores = scores[correct == 0]

    print(f"\nConsciousness Scores:")
    print(f"  Correct answers:   mean={np.mean(correct_scores):.3f} "
          f"(std={np.std(correct_scores):.3f}, n={len(correct_scores)})")
    print(f"  Incorrect answers: mean={np.mean(incorrect_scores):.3f} "
          f"(std={np.std(incorrect_scores):.3f}, n={len(incorrect_scores)})")

    # T-test: are consciousness scores significantly different?
    if len(correct_scores) > 1 and len(incorrect_scores) > 1:
        t_stat, p_value = stats.ttest_ind(correct_scores, incorrect_scores)
        print(f"\n  T-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  ** SIGNIFICANT (p < 0.05): Consciousness scores differ between correct/incorrect")
        else:
            print(f"  Not significant at p=0.05")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(correct_scores)**2 + np.std(incorrect_scores)**2) / 2
        )
        if pooled_std > 0:
            cohens_d = (np.mean(correct_scores) - np.mean(incorrect_scores)) / pooled_std
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

    # Point-biserial correlation (continuous score vs binary outcome)
    r_pb, p_pb = stats.pointbiserialr(correct, scores)
    print(f"\n  Point-biserial correlation: r={r_pb:.3f}, p={p_pb:.4f}")

    # Bin analysis: accuracy at different consciousness levels
    print(f"\nAccuracy by Consciousness Level:")
    bins = [(0, 0.35, "Low"), (0.35, 0.50, "Medium-Low"),
            (0.50, 0.65, "Medium-High"), (0.65, 1.0, "High")]

    for lo, hi, label in bins:
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            print(f"  {label:<12} ({lo:.2f}-{hi:.2f}): "
                  f"{bin_acc*100:5.1f}% ({mask.sum():3d} problems)")

    # Confidence analysis
    print(f"\nConfidence Metrics:")
    high_conf = confidence >= 0.7
    low_conf = confidence < 0.7
    if high_conf.sum() > 0:
        print(f"  High confidence (>=0.7): accuracy={correct[high_conf].mean()*100:.1f}% "
              f"(n={high_conf.sum()})")
    if low_conf.sum() > 0:
        print(f"  Low confidence (<0.7):   accuracy={correct[low_conf].mean()*100:.1f}% "
              f"(n={low_conf.sum()})")

    # Dimension analysis: which dimensions predict correctness?
    print(f"\nDimension Analysis (mean activation for correct vs incorrect):")
    dim_names = set()
    for r in results:
        dim_names.update(r.get("dimension_scores", {}).keys())

    for dim_name in sorted(dim_names):
        c_vals = [r["dimension_scores"][dim_name] for r in results
                  if r["correct"] and dim_name in r.get("dimension_scores", {})]
        i_vals = [r["dimension_scores"][dim_name] for r in results
                  if not r["correct"] and dim_name in r.get("dimension_scores", {})]
        if c_vals and i_vals:
            c_mean = np.mean(c_vals)
            i_mean = np.mean(i_vals)
            diff = c_mean - i_mean
            print(f"  {dim_name:<18}: correct={c_mean:+.3f}  incorrect={i_mean:+.3f}  "
                  f"diff={diff:+.3f} {'*' if abs(diff) > 0.05 else ''}")

    # Generate visualization
    try:
        generate_plots(results, scores, correct, output_dir)
    except Exception as e:
        print(f"\n  Warning: Could not generate plots: {e}")

    return {
        "accuracy": accuracy,
        "n_total": n_total,
        "n_correct": n_correct,
        "mean_score_correct": float(np.mean(correct_scores)) if len(correct_scores) > 0 else None,
        "mean_score_incorrect": float(np.mean(incorrect_scores)) if len(incorrect_scores) > 0 else None,
        "point_biserial_r": float(r_pb),
        "point_biserial_p": float(p_pb),
    }


def generate_plots(results: list, scores: np.ndarray, correct: np.ndarray, output_dir: Path):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Consciousness Score vs Task Performance (GSM8K)", fontsize=14, fontweight="bold")

    # 1. Score distribution: correct vs incorrect
    ax = axes[0, 0]
    correct_scores = scores[correct == 1]
    incorrect_scores = scores[correct == 0]
    ax.hist(correct_scores, bins=15, alpha=0.6, color="#10B981", label=f"Correct (n={len(correct_scores)})", density=True)
    ax.hist(incorrect_scores, bins=15, alpha=0.6, color="#EF4444", label=f"Incorrect (n={len(incorrect_scores)})", density=True)
    ax.set_xlabel("Consciousness Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Correctness")
    ax.legend()

    # 2. Accuracy by consciousness bin
    ax = axes[0, 1]
    n_bins = 5
    bin_edges = np.linspace(scores.min(), scores.max(), n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accs.append(correct[mask].mean() * 100)
            bin_counts.append(mask.sum())

    bars = ax.bar(range(len(bin_centers)), bin_accs,
                  color=["#EF4444" if a < 50 else "#F59E0B" if a < 70 else "#10B981" for a in bin_accs])
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels([f"{c:.2f}\n(n={n})" for c, n in zip(bin_centers, bin_counts)])
    ax.set_xlabel("Consciousness Score (bin center)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Consciousness Level")
    ax.set_ylim(0, 100)

    # 3. Dimension heatmap: correct vs incorrect
    ax = axes[1, 0]
    dim_names = sorted(set().union(*[r.get("dimension_scores", {}).keys() for r in results]))
    if dim_names:
        correct_means = []
        incorrect_means = []
        for dim_name in dim_names:
            c_vals = [r["dimension_scores"][dim_name] for r in results
                      if r["correct"] and dim_name in r.get("dimension_scores", {})]
            i_vals = [r["dimension_scores"][dim_name] for r in results
                      if not r["correct"] and dim_name in r.get("dimension_scores", {})]
            correct_means.append(np.mean(c_vals) if c_vals else 0)
            incorrect_means.append(np.mean(i_vals) if i_vals else 0)

        x = np.arange(len(dim_names))
        width = 0.35
        ax.bar(x - width / 2, correct_means, width, label="Correct", color="#10B981", alpha=0.8)
        ax.bar(x + width / 2, incorrect_means, width, label="Incorrect", color="#EF4444", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Activation")
        ax.set_title("Dimension Activations: Correct vs Incorrect")
        ax.legend()

    # 4. Confidence vs accuracy
    ax = axes[1, 1]
    confidence = np.array([r["confidence"] for r in results])
    conf_bins = np.linspace(0, 1, 6)
    conf_centers = []
    conf_accs = []
    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
        if mask.sum() > 0:
            conf_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)
            conf_accs.append(correct[mask].mean() * 100)

    if conf_centers:
        ax.plot(conf_centers, conf_accs, "o-", color="#6366F1", linewidth=2, markersize=8)
        ax.set_xlabel("Measurement Confidence")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Does Higher Confidence Predict Better Accuracy?")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / "consciousness_vs_performance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("EXPERIMENT: Consciousness Score vs Task Performance")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Problems: {args.n}")
    print(f"Device: {args.device}")
    print(f"Quantize: {args.quantize}")
    print(f"Seed: {args.seed}")
    print()

    # Load dataset
    samples = load_gsm8k(args.n, args.seed)

    # Load model
    model, tokenizer = load_model(args.model, args.device, args.quantize)

    # Initialize consciousness circuit
    from consciousness_circuit.universal import UniversalCircuit
    circuit = UniversalCircuit()

    # Run experiment
    results = []
    correct_count = 0

    print(f"\nRunning {len(samples)} problems...")
    print("-" * 70)

    for i, sample in enumerate(samples):
        try:
            # Generate + measure
            output = generate_and_measure(
                model, tokenizer, circuit,
                sample["question"],
                max_tokens=args.max_tokens,
            )

            # Extract and check answer
            model_answer = extract_model_answer(output["response"])
            is_correct = check_answer(model_answer, sample["gold_answer"])
            if is_correct:
                correct_count += 1

            c = output["consciousness"]

            result = {
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "model_answer": model_answer,
                "correct": is_correct,
                "consciousness_score": c.score,
                "raw_score": c.raw_score,
                "confidence": c.confidence,
                "dimension_diversity": c.dimension_diversity,
                "dominant_dimension": c.dominant_dimension,
                "anomaly_flags": c.anomaly_flags,
                "dimension_scores": c.dimension_scores,
                "token_count": c.token_count,
                "generation_time": output["generation_time"],
                "response_preview": output["response"][:200],
            }
            results.append(result)

            # Progress
            status = "CORRECT" if is_correct else "WRONG"
            running_acc = correct_count / (i + 1)
            print(f"  [{i+1:3d}/{len(samples)}] Score={c.score:.3f} "
                  f"Conf={c.confidence:.2f} "
                  f"Div={c.dimension_diversity:.2f} "
                  f"{status:<7} "
                  f"(ans={model_answer or '?'}, gold={sample['gold_answer']}) "
                  f"Acc={running_acc*100:.1f}% "
                  f"[{output['generation_time']:.1f}s]")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(samples)}] ERROR: {e}")
            continue

    # Save raw results
    results_path = output_dir / f"consciousness_performance_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "n_problems": len(samples),
                "device": args.device,
                "quantize": args.quantize,
                "seed": args.seed,
                "timestamp": timestamp,
                "circuit_version": "3.2.0",
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nRaw results saved: {results_path}")

    # Statistical analysis
    if len(results) > 5:
        summary = analyze_results(results, output_dir)

        # Save summary
        summary_path = output_dir / f"consciousness_performance_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")
    else:
        print("\nNot enough results for statistical analysis.")


if __name__ == "__main__":
    main()
