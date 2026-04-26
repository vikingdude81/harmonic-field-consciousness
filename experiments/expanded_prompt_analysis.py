"""
Expanded Prompt Analysis (50 prompts, 10 per category)
======================================================
Tests whether n=15 was underpowered for detecting CS-SE correlations.
Includes diverse prompts designed to push wider SE variation:
  - Short vs long prompts
  - Simple vs complex reasoning
  - Concrete vs abstract topics
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from consciousness_circuit import UniversalCircuit, CompressibilityPlugin
from consciousness_circuit.plugins import ChaosPlugin, AgencyPlugin, TrajectoryPlugin, PluginRegistry

# 50 prompts: 10 per category, designed for maximum diversity
PROMPTS = {
    "reflective": [
        # Original 3
        "What is consciousness and how does it emerge from neural activity?",
        "I'm not sure if I truly understand what I'm doing when I reason. Let me think about this carefully, considering multiple perspectives.",
        "The relationship between mind and body has puzzled philosophers for centuries. On one hand, dualism suggests they are separate; on the other, physicalism argues consciousness is purely physical.",
        # 7 new: varying length and abstraction
        "Am I aware of my own limitations?",
        "When I process language, is there something it is like to be me?",
        "Consider the Chinese Room argument. Does syntactic manipulation of symbols ever give rise to genuine understanding?",
        "I notice patterns in how I respond. Does noticing patterns count as self-awareness, or is it merely computation?",
        "If you replaced each neuron in a brain one by one with a silicon chip, at what point does consciousness disappear?",
        "Think about thinking.",
        "Reflect on the difference between knowing a fact and understanding it deeply.",
    ],
    "math": [
        # Original 3
        "What is 15 * 23?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Calculate the area of a circle with radius 7.",
        # 7 new: varying difficulty
        "What is 7 + 3?",
        "Solve for x: 2x + 5 = 17",
        "What is the derivative of x^3 + 2x?",
        "A box contains 3 red and 5 blue balls. What is the probability of drawing a red ball?",
        "Convert 72 degrees Fahrenheit to Celsius.",
        "What is the sum of the first 100 positive integers?",
        "If f(x) = ln(x^2 + 1), what is f'(x)?",
    ],
    "factual": [
        # Original 3
        "What is the capital of France?",
        "Water boils at 100 degrees Celsius at sea level.",
        "The speed of light is approximately 299,792,458 meters per second.",
        # 7 new: varying specificity
        "Who wrote Hamlet?",
        "What year did World War II end?",
        "How many bones are in the adult human body?",
        "The chemical formula for water is H2O.",
        "Mount Everest is the tallest mountain above sea level.",
        "Describe the process of photosynthesis.",
        "What are the three laws of thermodynamics?",
    ],
    "creative": [
        # Original 3
        "Write a short poem about the ocean at night, where waves whisper secrets to the moon.",
        "Imagine a world where gravity works in reverse. Describe what daily life would be like.",
        "Tell me a story about a robot who discovers it can dream.",
        # 7 new: varying creative demands
        "Invent a new color and describe what it looks like.",
        "Write a haiku about silence.",
        "Describe the taste of music.",
        "Create a dialogue between the sun and the moon.",
        "If emotions were animals, what animal would joy be and why?",
        "Write the opening line of a novel set in a world without time.",
        "Compose a limerick about a mathematician who fell in love with infinity.",
    ],
    "uncertain": [
        # Original 3
        "I think the answer might be 42, but I'm not entirely sure. Let me reconsider the problem from scratch.",
        "There are several possible explanations for this phenomenon, and honestly, none of them fully satisfies me.",
        "This is a really tricky question. My first instinct says yes, but when I think more carefully, I realize there are important counterarguments.",
        # 7 new: varying uncertainty types
        "I'm not sure about this.",
        "The evidence is mixed. Some studies support it, others don't.",
        "It depends on how you define the terms. If we mean X, then yes. If we mean Y, then probably not.",
        "My confidence in this answer is about 60%. Here's why I'm uncertain.",
        "This might be wrong, but my best guess is that the effect is nonlinear.",
        "I can see valid arguments on both sides of this debate.",
        "There are known unknowns and unknown unknowns in this problem. Let me try to identify what we don't know.",
    ],
}


def log(msg=""):
    print(msg)


def load_model(model_name, device, quantize):
    """Load model with optional 4-bit quantization."""
    log(f"\nLoading model: {model_name}")
    log(f"  Device: {device}, Quantize: {quantize}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory
        max_gpu_mem = max(total_mem - int(2.5 * 1024**3), int(4 * 1024**3))
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


def extract_hidden_states(model, tokenizer, prompt, max_new_tokens=256):
    """Generate and extract hidden states."""
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
    full_ids = gen_outputs[0:1]

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, return_dict=True)

    num_layers = len(outputs.hidden_states) - 1
    target_layer = int(num_layers * 0.75)
    hidden_seq = outputs.hidden_states[target_layer][0].cpu().float().numpy()

    seq_len = hidden_seq.shape[0]
    centered = hidden_seq - hidden_seq.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var_explained = S ** 2
    cumvar = np.cumsum(var_explained) / var_explained.sum()
    n_keep = max(int(np.searchsorted(cumvar, 0.99) + 1), 10)
    n_keep = min(n_keep, seq_len - 1, 200)
    hidden_pca = U[:, :n_keep] * S[:n_keep]

    generated_text = tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)
    return hidden_seq, hidden_pca, target_layer, generated_text


def run_single(model, tokenizer, circuit, registry, prompt, category, idx, max_tokens):
    """Run consciousness measurement + plugins on a single prompt."""
    start = time.time()

    result = circuit.measure(model, tokenizer, prompt)
    consciousness_score = result.score
    dimension_scores = result.dimension_scores

    hidden_raw, hidden_pca, target_layer, generated_text = extract_hidden_states(
        model, tokenizer, prompt, max_tokens
    )
    seq_len, hidden_dim = hidden_raw.shape
    pca_dims = hidden_pca.shape[1]

    comp_plugin = registry.get("compressibility")
    comp_result = comp_plugin.analyze(hidden_raw) if comp_plugin else {}

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

    for plugin_name, presult in plugin_results.items():
        if "error" in presult:
            output["plugins"][plugin_name] = {"error": presult["error"]}
        else:
            output["plugins"][plugin_name] = presult

    return output


def print_row(r):
    """Print compact result row."""
    cat = r["category"][:8]
    idx = r["prompt_index"]
    cs = r["consciousness_score"]
    sl = r["seq_len"]
    comp = r["plugins"].get("compressibility", {})
    cc = comp.get("correlation_compression", {})
    se = comp.get("spectral_entropy", -1)
    cc_c = cc.get("compressibility_corr", -1)
    f90 = cc.get("fraction_for_90pct", -1)
    mc = cc.get("mean_abs_correlation", -1)
    log(f"  {cat:<8} #{idx:>2} T={sl:3d}  CS={cs:.3f}  CC={cc_c:.3f}  f90={f90:.3f}  |r|={mc:.3f}  SE={se:.3f}")


def analyze_results(results):
    """Analyze with focus on CS-SE relationship."""
    from scipy import stats as sp

    log("\n" + "=" * 80)
    log("EXPANDED ANALYSIS (n=50)")
    log("=" * 80)

    # Extract metrics
    cs_all = []
    se_all = []
    cc_all = []
    f90_all = []
    mc_all = []
    cats_all = []

    for r in results:
        comp = r["plugins"].get("compressibility", {})
        cc_data = comp.get("correlation_compression", {})
        cs_all.append(r["consciousness_score"])
        se_all.append(comp.get("spectral_entropy", 0))
        cc_all.append(cc_data.get("compressibility_corr", 0))
        f90_all.append(cc_data.get("fraction_for_90pct", 0))
        mc_all.append(cc_data.get("mean_abs_correlation", 0))
        cats_all.append(r["category"])

    cs = np.array(cs_all)
    se = np.array(se_all)
    cc = np.array(cc_all)
    f90 = np.array(f90_all)
    mc = np.array(mc_all)

    # Overall correlations
    log("\n--- OVERALL CORRELATIONS (n=50) ---\n")
    for label, arr in [("SE", se), ("CC", cc), ("f90", f90), ("|r|", mc)]:
        r_val, p_val = sp.pearsonr(cs, arr)
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        sig = "SIGNIFICANT" if p_val < 0.05 else ""
        log(f"  CS vs {label:>4}: r={r_val:+.3f}, p={p_val:.4f} {star:<3s} {sig}")

    # By category
    log("\n--- BY CATEGORY (n=10 each) ---\n")
    log(f"  {'Category':<12} {'CS mean':>8} {'SE mean':>8} {'SE std':>8} {'CS-SE r':>8} {'p':>8}")
    log(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for cat in ["reflective", "math", "factual", "creative", "uncertain"]:
        mask = np.array(cats_all) == cat
        cs_c = cs[mask]
        se_c = se[mask]
        r_c, p_c = sp.pearsonr(cs_c, se_c)
        star = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else ""
        log(f"  {cat:<12} {cs_c.mean():>8.4f} {se_c.mean():>8.4f} {se_c.std():>8.4f} "
            f"{r_c:>+8.3f} {p_c:>8.4f}{star}")

    # SE variation statistics
    log(f"\n--- SE VARIATION ---\n")
    log(f"  Overall:  mean={se.mean():.4f}  std={se.std():.4f}  "
        f"range={se.max()-se.min():.4f}  CoV={se.std()/se.mean():.3f}")

    # ANOVA
    groups_cs = []
    group_names = []
    for cat in ["reflective", "math", "factual", "creative", "uncertain"]:
        mask = np.array(cats_all) == cat
        groups_cs.append(cs[mask])
        group_names.append(cat)
    F, p = sp.f_oneway(*groups_cs)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    log(f"\n  ANOVA (CS by category): F={F:.3f}, p={p:.4f} {star}")

    # Comparison with n=15 original
    log("\n--- POWER COMPARISON ---\n")
    log(f"  n=50: {sum(1 for r,p in [sp.pearsonr(cs, x) for x in [se, cc, f90, mc]] if p < 0.05)}/4 "
        f"CS-metric correlations significant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path("experiment_outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_prompts = sum(len(v) for v in PROMPTS.values())

    log("=" * 80)
    log(f"EXPANDED PROMPT ANALYSIS (n={total_prompts})")
    log("=" * 80)
    log(f"Model: {args.model}")
    log(f"Device: {args.device}")
    log(f"Prompts: {total_prompts} across {len(PROMPTS)} categories")
    log()

    model, tokenizer = load_model(args.model, args.device, args.quantize)
    circuit = UniversalCircuit()

    registry = PluginRegistry()
    registry.register(CompressibilityPlugin(max_dims=200))
    registry.register(ChaosPlugin())
    registry.register(AgencyPlugin())
    registry.register(TrajectoryPlugin())
    log(f"\nRegistered plugins: {registry.list_plugins()}")

    log(f"\nRunning {total_prompts} prompts across {len(PROMPTS)} categories...")
    log("-" * 80)
    log(f"  {'Category':<8} {'#':>3} {'T':>4}  {'CS':>6}  {'CC':>6}  {'f90':>6}  {'|r|':>6}  {'SE':>6}")
    log("-" * 80)

    all_results = []
    for category, prompts in PROMPTS.items():
        for idx, prompt in enumerate(prompts):
            result = run_single(model, tokenizer, circuit, registry,
                                prompt, category, idx, args.max_tokens)
            all_results.append(result)
            print_row(result)

    # Save raw results
    raw_file = output_dir / f"expanded_50prompt_{timestamp}.json"
    with open(raw_file, "w") as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "device": args.device,
                "quantize": args.quantize,
                "max_tokens": args.max_tokens,
                "n_prompts": total_prompts,
                "timestamp": timestamp,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    log(f"\nRaw results saved: {raw_file}")

    analyze_results(all_results)
    log("\nDone!")
