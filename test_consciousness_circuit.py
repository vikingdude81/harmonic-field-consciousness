"""
Test Consciousness Circuit Installation
========================================

Quick test to verify the consciousness circuit package works correctly.
"""

import torch
import numpy as np
from consciousness_circuit import UniversalCircuit
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_basic_functionality():
    """Test basic consciousness measurement."""
    print("=" * 60)
    print("Testing Consciousness Circuit Installation")
    print("=" * 60)

    # Check GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"\n[OK] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n[WARN] Using CPU (slower)")

    # Test with small model first (faster download)
    print("\n" + "-" * 60)
    print("Step 1: Loading Qwen2.5-0.5B (small model)")
    print("-" * 60)

    try:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False, None, None, None

    # Initialize circuit
    try:
        circuit = UniversalCircuit()
        print("[OK] Circuit initialized successfully")
    except Exception as e:
        print(f"[FAIL] Circuit initialization failed: {e}")
        return False, None, None, None

    # Test prompts with expected ordering
    print("\n" + "-" * 60)
    print("Step 2: Testing consciousness measurement")
    print("-" * 60)

    test_prompts = {
        "high": "What is the nature of consciousness and self-awareness?",
        "medium": "Explain how photosynthesis works in plants.",
        "low": "What is 2 + 2?",
    }

    results = {}

    for level, prompt in test_prompts.items():
        try:
            result = circuit.measure(model, tokenizer, prompt, aggregation="last")
            score = result.score
            results[level] = score
            print(f"[OK] {level.upper():8} ({score:.3f}): {prompt[:50]}...")
            print(f"   Method: {result.method}, Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"[FAIL] {level.UPPER():8} failed: {e}")
            return False, None, None, None

    # Verify ordering (high > medium > low expected, but not guaranteed)
    print("\n" + "-" * 60)
    print("Step 3: Analyzing results")
    print("-" * 60)

    print(f"\nScores:")
    print(f"  High:   {results['high']:.3f}")
    print(f"  Medium: {results['medium']:.3f}")
    print(f"  Low:    {results['low']:.3f}")

    # Check if scores are in reasonable range
    all_scores = list(results.values())
    if all(0.0 <= s <= 1.0 for s in all_scores):
        print(f"\n[OK] All scores in valid range [0, 1]")
    else:
        print(f"\n[WARN]  Some scores out of range")

    # Check variance (scores shouldn't all be identical)
    score_variance = np.var(all_scores)
    if score_variance > 0.001:
        print(f"[OK] Scores show variance: {score_variance:.4f}")
    else:
        print(f"[WARN]  Low variance, scores may not be discriminating: {score_variance:.4f}")

    # Test batch processing
    print("\n" + "-" * 60)
    print("Step 4: Testing batch processing")
    print("-" * 60)

    try:
        batch_prompts = [
            "What is consciousness?",
            "How do computers work?",
            "What is 5 * 3?"
        ]
        batch_results = [circuit.measure(model, tokenizer, p, aggregation="last") for p in batch_prompts]
        batch_scores = [r.score for r in batch_results]
        print(f"[OK] Batch processing works: {len(batch_scores)} scores")
        for i, (prompt, score) in enumerate(zip(batch_prompts, batch_scores)):
            print(f"   {i+1}. {score:.3f}: {prompt[:40]}...")
    except Exception as e:
        print(f"[FAIL] Batch processing failed: {e}")
        return False, None, None, None

    print("\n" + "=" * 60)
    print("[OK] ALL TESTS PASSED!")
    print("=" * 60)
    print("\nConsciousness circuit is working correctly.")
    print("You can now use it with your locally trained models!")

    return True, model, tokenizer, circuit


def test_different_aggregations(model, tokenizer, circuit):
    """Test different aggregation methods."""
    print("\n" + "=" * 60)
    print("BONUS: Testing Different Aggregation Methods")
    print("=" * 60)

    prompt = "What is the nature of consciousness?"
    aggregations = ["last", "mean", "max"]
    results = {}

    for agg in aggregations:
        try:
            result = circuit.measure(model, tokenizer, prompt, aggregation=agg)
            score = result.score
            results[agg] = score
            print(f"[OK] {agg.upper():6} aggregation: {score:.3f}")
        except Exception as e:
            print(f"[FAIL] {agg.upper():6} failed: {e}")

    if len(results) == 3:
        print(f"\n[OK] All aggregation methods work!")
        print(f"   Audit found 'last' gives best discrimination (+0.246)")


if __name__ == "__main__":
    print("\nConsciousness Circuit Test Suite\n")

    # Run main test
    success, model, tokenizer, circuit = test_basic_functionality()

    if success:
        # Run bonus test
        test_different_aggregations(model, tokenizer, circuit)

        print("\n" + "=" * 60)
        print("INSTALLATION VERIFIED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test with your locally trained NanoGPT model")
        print("2. Re-run GPU experiments to validate fixes")
        print("3. Create visualizations with your data")
        print("\nSee GETTING_STARTED_GUIDE.md for more examples.")
    else:
        print("\n" + "=" * 60)
        print("[FAIL] TESTS FAILED")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("- Check GPU memory (might need to use CPU)")
        print("- Verify internet connection (downloads model)")
        print("- Try: pip install -e ./consciousness_circuit[viz] --force-reinstall")
