"""
Expanded Prompt Comparison: n=15 vs n=50
========================================
Compares the original 15-prompt results with expanded 50-prompt results
to determine whether n=15 was underpowered for detecting CS-SE correlations.
"""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# Data files
DATA = {
    "Mistral-7B (n=15)": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_002457.json",
        "is_local": True,
    },
    "Mistral-7B (n=50)": {
        "path": "experiment_outputs/expanded_50prompt_20260204_150427.json",
        "is_local": True,
    },
    "Qwen-7B (n=15)": {
        "path": "experiment_outputs/combined_plugin_analysis_20260203_231542.json",
        "is_local": True,
    },
    "Qwen-7B (n=50)": {
        "path": "experiment_outputs/expanded_50prompt_20260204_154322.json",
        "is_local": True,
    },
}


def extract_metrics(result, is_local=True):
    """Extract standardized metrics."""
    if is_local:
        comp = result.get("plugins", {}).get("compressibility", {})
        cc = comp.get("correlation_compression", {})
        return {
            "cs": result["consciousness_score"],
            "cc": cc.get("compressibility_corr", 0),
            "f90": cc.get("fraction_for_90pct", 0),
            "se": comp.get("spectral_entropy", 0),
            "mc": cc.get("mean_abs_correlation", 0),
            "category": result.get("category", ""),
        }
    else:
        comp = result.get("compressibility", {})
        return {
            "cs": result["consciousness_score"],
            "cc": comp.get("compressibility_corr", 0),
            "f90": comp.get("fraction_for_90pct", 0),
            "se": comp.get("spectral_entropy", 0),
            "mc": comp.get("mean_abs_correlation", 0),
            "category": result.get("category", ""),
        }


# Load all data
datasets = {}
for name, info in DATA.items():
    with open(info["path"]) as f:
        data = json.load(f)
    results = data["results"]
    metrics = [extract_metrics(r, info["is_local"]) for r in results]
    datasets[name] = metrics


def corr_str(a, b):
    """Pearson correlation with stars."""
    r, p = stats.pearsonr(a, b)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    sig = "SIGNIFICANT" if p < 0.05 else ""
    return r, p, star, sig


print("=" * 90)
print("EXPANDED PROMPT COMPARISON: n=15 vs n=50")
print("=" * 90)
print()

# ---- Section 1: Overall Correlations ----
print("--- OVERALL CS-METRIC CORRELATIONS ---")
print()
print(f"  {'Dataset':<25} {'CS vs SE':>22} {'CS vs CC':>22} {'CS vs |r|':>22}")
print(f"  {'-'*25} {'-'*22} {'-'*22} {'-'*22}")

for name, metrics in datasets.items():
    cs = np.array([m["cs"] for m in metrics])
    se = np.array([m["se"] for m in metrics])
    cc = np.array([m["cc"] for m in metrics])
    mc = np.array([m["mc"] for m in metrics])

    r_se, p_se, s_se, _ = corr_str(cs, se)
    r_cc, p_cc, s_cc, _ = corr_str(cs, cc)
    r_mc, p_mc, s_mc, _ = corr_str(cs, mc)

    print(f"  {name:<25} r={r_se:+.3f} p={p_se:.4f}{s_se:3s} "
          f"r={r_cc:+.3f} p={p_cc:.4f}{s_cc:3s} "
          f"r={r_mc:+.3f} p={p_mc:.4f}{s_mc:3s}")

print()

# ---- Section 2: SE Statistics Comparison ----
print("--- SE STATISTICS ---")
print()
print(f"  {'Dataset':<25} {'SE mean':>10} {'SE std':>10} {'SE range':>10} {'SE CoV':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name, metrics in datasets.items():
    se = np.array([m["se"] for m in metrics])
    cov = se.std() / se.mean() if se.mean() > 0 else 0
    print(f"  {name:<25} {se.mean():>10.4f} {se.std():>10.4f} "
          f"{se.max()-se.min():>10.4f} {cov:>10.3f}")

print()

# ---- Section 3: CS Statistics Comparison ----
print("--- CS STATISTICS ---")
print()
print(f"  {'Dataset':<25} {'CS mean':>10} {'CS std':>10} {'CS range':>10} {'CS CoV':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name, metrics in datasets.items():
    cs = np.array([m["cs"] for m in metrics])
    cov = cs.std() / cs.mean() if cs.mean() > 0 else 0
    print(f"  {name:<25} {cs.mean():>10.4f} {cs.std():>10.4f} "
          f"{cs.max()-cs.min():>10.4f} {cov:>10.3f}")

print()

# ---- Section 4: By-Category Analysis ----
print("--- CS-SE CORRELATION BY CATEGORY ---")
print()
CATS = ["reflective", "math", "factual", "creative", "uncertain"]

for model_prefix in ["Mistral-7B", "Qwen-7B"]:
    print(f"  {model_prefix}:")
    print(f"    {'Category':<12} {'n=15':>20} {'n=50':>20} {'Direction':>12}")
    print(f"    {'-'*12} {'-'*20} {'-'*20} {'-'*12}")

    for n_label in [f"{model_prefix} (n=15)", f"{model_prefix} (n=50)"]:
        metrics = datasets[n_label]

    m15 = datasets[f"{model_prefix} (n=15)"]
    m50 = datasets[f"{model_prefix} (n=50)"]

    for cat in CATS:
        cs15 = [m["cs"] for m in m15 if m["category"] == cat]
        se15 = [m["se"] for m in m15 if m["category"] == cat]
        cs50 = [m["cs"] for m in m50 if m["category"] == cat]
        se50 = [m["se"] for m in m50 if m["category"] == cat]

        if len(cs15) >= 3:
            r15, p15 = stats.pearsonr(cs15, se15)
            s15 = f"r={r15:+.3f} p={p15:.2f}"
        else:
            s15 = "N/A"

        if len(cs50) >= 3:
            r50, p50 = stats.pearsonr(cs50, se50)
            s50 = f"r={r50:+.3f} p={p50:.4f}"
            star50 = "**" if p50 < 0.01 else "*" if p50 < 0.05 else ""
            s50 += star50
        else:
            s50 = "N/A"

        consistent = "same" if (len(cs15) >= 3 and len(cs50) >= 3 and
                                (r15 > 0) == (r50 > 0)) else "flipped"
        print(f"    {cat:<12} {s15:>20} {s50:>20} {consistent:>12}")

    print()

# ---- Section 5: Power Analysis ----
print("--- POWER ANALYSIS: n=15 vs n=50 ---")
print()

for model_prefix in ["Mistral-7B", "Qwen-7B"]:
    m15 = datasets[f"{model_prefix} (n=15)"]
    m50 = datasets[f"{model_prefix} (n=50)"]

    cs15 = np.array([m["cs"] for m in m15])
    se15 = np.array([m["se"] for m in m15])
    cs50 = np.array([m["cs"] for m in m50])
    se50 = np.array([m["se"] for m in m50])

    r15, p15 = stats.pearsonr(cs15, se15)
    r50, p50 = stats.pearsonr(cs50, se50)

    # Effect size comparison (Fisher z-transform)
    z15 = np.arctanh(r15) if abs(r15) < 0.999 else np.sign(r15) * 3
    z50 = np.arctanh(r50) if abs(r50) < 0.999 else np.sign(r50) * 3
    se_diff = np.sqrt(1/(len(cs15)-3) + 1/(len(cs50)-3))
    z_test = (z15 - z50) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_test)))

    print(f"  {model_prefix}:")
    print(f"    n=15: CS-SE r={r15:+.3f}, p={p15:.4f} {'SIGNIFICANT' if p15 < 0.05 else 'not significant'}")
    print(f"    n=50: CS-SE r={r50:+.3f}, p={p50:.4f} {'SIGNIFICANT' if p50 < 0.05 else 'not significant'}")
    print(f"    Effect size change: |r| went from {abs(r15):.3f} to {abs(r50):.3f}")
    print(f"    Fisher z-test for difference: z={z_test:.3f}, p={p_diff:.4f}")
    print(f"    Conclusion: {'LARGER sample reveals signal' if p15 >= 0.05 and p50 < 0.05 else 'Signal direction consistent' if (r15 > 0) == (r50 > 0) else 'Signal changed direction'}")
    print()

# ---- Section 6: Shared Prompts Consistency ----
print("--- SHARED PROMPT CONSISTENCY (first 3 prompts per category) ---")
print("  (Comparing results for identical prompts between n=15 and n=50 runs)")
print()

for model_prefix in ["Mistral-7B", "Qwen-7B"]:
    m15 = datasets[f"{model_prefix} (n=15)"]
    m50 = datasets[f"{model_prefix} (n=50)"]

    # First 3 per category should be the same prompts
    cs_pairs_15 = []
    cs_pairs_50 = []

    for cat in CATS:
        cat15 = [m for m in m15 if m["category"] == cat]
        cat50 = [m for m in m50 if m["category"] == cat]
        for i in range(min(3, len(cat15), len(cat50))):
            cs_pairs_15.append(cat15[i]["cs"])
            cs_pairs_50.append(cat50[i]["cs"])

    cs_pairs_15 = np.array(cs_pairs_15)
    cs_pairs_50 = np.array(cs_pairs_50)

    r, p = stats.pearsonr(cs_pairs_15, cs_pairs_50)
    mae = np.mean(np.abs(cs_pairs_15 - cs_pairs_50))

    print(f"  {model_prefix}:")
    print(f"    Shared prompts (n={len(cs_pairs_15)}): CS correlation r={r:.3f}, p={p:.4f}")
    print(f"    Mean absolute CS difference: {mae:.4f}")
    print(f"    CS reproducibility: {'HIGH' if r > 0.7 else 'MODERATE' if r > 0.4 else 'LOW'}")
    print()

# ---- Section 7: Summary ----
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print()

# Key findings
m_r15, m_p15 = stats.pearsonr(
    [m["cs"] for m in datasets["Mistral-7B (n=15)"]],
    [m["se"] for m in datasets["Mistral-7B (n=15)"]]
)
m_r50, m_p50 = stats.pearsonr(
    [m["cs"] for m in datasets["Mistral-7B (n=50)"]],
    [m["se"] for m in datasets["Mistral-7B (n=50)"]]
)
q_r15, q_p15 = stats.pearsonr(
    [m["cs"] for m in datasets["Qwen-7B (n=15)"]],
    [m["se"] for m in datasets["Qwen-7B (n=15)"]]
)
q_r50, q_p50 = stats.pearsonr(
    [m["cs"] for m in datasets["Qwen-7B (n=50)"]],
    [m["se"] for m in datasets["Qwen-7B (n=50)"]]
)

print("  Key Findings:")
print()
print(f"  1. Mistral-7B CS-SE correlation:")
print(f"     n=15: r={m_r15:+.3f}, p={m_p15:.4f} → {'SIGNIFICANT' if m_p15 < 0.05 else 'NOT significant'}")
print(f"     n=50: r={m_r50:+.3f}, p={m_p50:.4f} → {'SIGNIFICANT' if m_p50 < 0.05 else 'NOT significant'}")
if m_p15 >= 0.05 and m_p50 < 0.05:
    print(f"     → n=15 WAS underpowered for Mistral. The CS-SE link exists but needed more data.")
elif m_p15 < 0.05 and m_p50 < 0.05:
    print(f"     → Both significant. n=15 was adequate.")
else:
    print(f"     → No CS-SE signal even with n=50.")

print()
print(f"  2. Qwen-7B CS-SE correlation:")
print(f"     n=15: r={q_r15:+.3f}, p={q_p15:.4f} → {'SIGNIFICANT' if q_p15 < 0.05 else 'NOT significant'}")
print(f"     n=50: r={q_r50:+.3f}, p={q_p50:.4f} → {'SIGNIFICANT' if q_p50 < 0.05 else 'NOT significant'}")
if q_p15 < 0.05 and q_p50 >= 0.05:
    print(f"     → n=15 result may have been a type I error (false positive)")
elif q_p15 >= 0.05 and q_p50 >= 0.05:
    print(f"     → Consistently not significant. Qwen-7B SE variation is too low at 7B scale.")
else:
    print(f"     → Signal consistent across sample sizes.")

print()

# SE variation comparison
m_se_cov15 = np.std([m["se"] for m in datasets["Mistral-7B (n=15)"]]) / np.mean([m["se"] for m in datasets["Mistral-7B (n=15)"]])
m_se_cov50 = np.std([m["se"] for m in datasets["Mistral-7B (n=50)"]]) / np.mean([m["se"] for m in datasets["Mistral-7B (n=50)"]])
q_se_cov15 = np.std([m["se"] for m in datasets["Qwen-7B (n=15)"]]) / np.mean([m["se"] for m in datasets["Qwen-7B (n=15)"]])
q_se_cov50 = np.std([m["se"] for m in datasets["Qwen-7B (n=50)"]]) / np.mean([m["se"] for m in datasets["Qwen-7B (n=50)"]])

print(f"  3. SE Variation (CoV):")
print(f"     Mistral: {m_se_cov15:.3f} (n=15) → {m_se_cov50:.3f} (n=50) — {'increased' if m_se_cov50 > m_se_cov15 else 'decreased'}")
print(f"     Qwen:    {q_se_cov15:.3f} (n=15) → {q_se_cov50:.3f} (n=50) — {'increased' if q_se_cov50 > q_se_cov15 else 'decreased'}")
print()

print(f"  4. Implications for the Consciousness-Compressibility Hypothesis:")
if m_p50 < 0.05:
    print(f"     - Mistral DOES show CS-SE correlation when given enough prompts (r={m_r50:+.3f})")
    print(f"     - The original 'Mistral anomaly' was a power issue, not an architecture difference")
else:
    print(f"     - Mistral still shows no CS-SE correlation with n=50")
    print(f"     - The architecture-dependent SE ceiling is real and not a power issue")

print()
print("Done!")
