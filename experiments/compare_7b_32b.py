"""Compare 7B vs 32B cross-plugin results."""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# Load results
with open("experiment_outputs/combined_plugin_analysis_20260203_231542.json") as f:
    data_7b = json.load(f)
with open("experiment_outputs/combined_plugin_32B_20260204_000325.json") as f:
    data_32b = json.load(f)

r7 = data_7b["results"]
r32 = data_32b["results"]
categories = ["reflective", "math", "factual", "creative", "uncertain"]


def get_7b_metrics(r):
    comp = r["plugins"]["compressibility"]
    cc = comp.get("correlation_compression", {})
    return {
        "cs": r["consciousness_score"],
        "cc": cc.get("compressibility_corr", 0),
        "f90": cc.get("fraction_for_90pct", 0),
        "se": comp.get("spectral_entropy", 0),
        "mc": cc.get("mean_abs_correlation", 0),
    }


def get_32b_metrics(r):
    comp = r["compressibility"]
    return {
        "cs": r["consciousness_score"],
        "cc": comp.get("compressibility_corr", 0),
        "f90": comp.get("fraction_for_90pct", 0),
        "se": comp.get("spectral_entropy", 0),
        "mc": comp.get("mean_abs_correlation", 0),
    }


print("=" * 70)
print("7B vs 32B Cross-Scale Comparison")
print("=" * 70)
print(f"7B model:  {data_7b['metadata']['model']}")
print(f"32B model: {data_32b['metadata']['model']}")
print()

# Overall means
m7 = [get_7b_metrics(r) for r in r7]
m32 = [get_32b_metrics(r) for r in r32]

print("--- OVERALL MEANS ---")
print()
metric_names = {"cs": "Consciousness", "cc": "Compress Coeff", "f90": "Fraction-90%",
                "se": "Spectral Entropy", "mc": "Mean |corr|"}
for key, label in metric_names.items():
    v7 = [m[key] for m in m7]
    v32 = [m[key] for m in m32]
    ratio = np.mean(v32) / np.mean(v7) if np.mean(v7) != 0 else float("inf")
    print(f"  {label:20s}: 7B={np.mean(v7):.4f} +/- {np.std(v7):.4f}  "
          f"32B={np.mean(v32):.4f} +/- {np.std(v32):.4f}  ratio={ratio:.3f}")
print()

# Per-category comparison
print("--- PER-CATEGORY COMPARISON ---")
print()
for cat in categories:
    items_7b = [get_7b_metrics(r) for r in r7 if r["category"] == cat]
    items_32b = [get_32b_metrics(r) for r in r32 if r["category"] == cat]
    print(f"  {cat.upper()}:")
    for key, label in metric_names.items():
        v7 = np.mean([m[key] for m in items_7b])
        v32 = np.mean([m[key] for m in items_32b])
        print(f"    {label:20s}: 7B={v7:.4f}  32B={v32:.4f}  delta={v32-v7:+.4f}")
    print()

# Statistical tests
print("--- STATISTICAL TESTS (paired t-test by prompt) ---")
print()
for key, label in metric_names.items():
    v7 = [m[key] for m in m7]
    v32 = [m[key] for m in m32]
    t, p = stats.ttest_rel(v7, v32)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:20s}: t={t:+.3f}, p={p:.4f} {star}")

print()

# 32B cross-metric correlations
print("--- 32B KEY CORRELATIONS ---")
print()
if "significant_correlations" in data_32b:
    for sc in data_32b["significant_correlations"]:
        star = "***" if sc["p"] < 0.001 else "**" if sc["p"] < 0.01 else "*"
        print(f"  {sc['metric_1']:30s} <-> {sc['metric_2']:30s}: r={sc['r']:+.3f}, p={sc['p']:.4f} {star}")
print()

# Category rankings
print("--- CATEGORY RANKINGS (by consciousness score) ---")
print()
cats_7b = {cat: np.mean([r["consciousness_score"] for r in r7 if r["category"] == cat]) for cat in categories}
cats_32b = {cat: np.mean([r["consciousness_score"] for r in r32 if r["category"] == cat]) for cat in categories}
rank_7b = sorted(cats_7b.items(), key=lambda x: x[1], reverse=True)
rank_32b = sorted(cats_32b.items(), key=lambda x: x[1], reverse=True)
print("  7B:  " + " > ".join(f"{c}({v:.3f})" for c, v in rank_7b))
print("  32B: " + " > ".join(f"{c}({v:.3f})" for c, v in rank_32b))
print()

# Variance
print("--- VARIANCE COMPARISON ---")
print()
cs_7b = [m["cs"] for m in m7]
cs_32b = [m["cs"] for m in m32]
print(f"  Consciousness: 7B std={np.std(cs_7b):.4f}  32B std={np.std(cs_32b):.4f}")
print(f"  7B range=[{min(cs_7b):.4f}, {max(cs_7b):.4f}]  32B range=[{min(cs_32b):.4f}, {max(cs_32b):.4f}]")
print()

cc_7b = [m["cc"] for m in m7]
cc_32b = [m["cc"] for m in m32]
print(f"  Compress Coeff: 7B std={np.std(cc_7b):.4f}  32B std={np.std(cc_32b):.4f}")
print(f"  7B range=[{min(cc_7b):.4f}, {max(cc_7b):.4f}]  32B range=[{min(cc_32b):.4f}, {max(cc_32b):.4f}]")
print()

# Architecture
print("--- ARCHITECTURE ---")
print(f"  7B:  hidden_dim=3584, target_layer=21 (75% of 28)")
print(f"  32B: hidden_dim=5120, target_layer=48 (75% of 64)")
seq7 = [r["seq_len"] for r in r7]
seq32 = [r["compressibility"].get("seq_len", 0) for r in r32]
print(f"  7B seq_len:  mean={np.mean(seq7):.0f}, range=[{min(seq7)}, {max(seq7)}]")
print(f"  32B seq_len: mean={np.mean(seq32):.0f}, range=[{min(seq32)}, {max(seq32)}]")
print()

# 32B-specific: consciousness vs compressibility
print("--- 32B: CONSCIOUSNESS-COMPRESSIBILITY RELATIONSHIP ---")
print()
r_cs_cc, p_cs_cc = stats.pearsonr(cs_32b, cc_32b)
r_cs_se, p_cs_se = stats.pearsonr(cs_32b, [m["se"] for m in m32])
r_cs_f90, p_cs_f90 = stats.pearsonr(cs_32b, [m["f90"] for m in m32])
r_cs_mc, p_cs_mc = stats.pearsonr(cs_32b, [m["mc"] for m in m32])
print(f"  CS vs CC:  r={r_cs_cc:+.3f}, p={p_cs_cc:.4f}")
print(f"  CS vs SE:  r={r_cs_se:+.3f}, p={p_cs_se:.4f}")
print(f"  CS vs f90: r={r_cs_f90:+.3f}, p={p_cs_f90:.4f}")
print(f"  CS vs |r|: r={r_cs_mc:+.3f}, p={p_cs_mc:.4f}")
print()

# Same for 7B
print("--- 7B: CONSCIOUSNESS-COMPRESSIBILITY RELATIONSHIP ---")
print()
r_cs_cc_7, p_cs_cc_7 = stats.pearsonr(cs_7b, cc_7b)
r_cs_se_7, p_cs_se_7 = stats.pearsonr(cs_7b, [m["se"] for m in m7])
r_cs_f90_7, p_cs_f90_7 = stats.pearsonr(cs_7b, [m["f90"] for m in m7])
r_cs_mc_7, p_cs_mc_7 = stats.pearsonr(cs_7b, [m["mc"] for m in m7])
print(f"  CS vs CC:  r={r_cs_cc_7:+.3f}, p={p_cs_cc_7:.4f}")
print(f"  CS vs SE:  r={r_cs_se_7:+.3f}, p={p_cs_se_7:.4f}")
print(f"  CS vs f90: r={r_cs_f90_7:+.3f}, p={p_cs_f90_7:.4f}")
print(f"  CS vs |r|: r={r_cs_mc_7:+.3f}, p={p_cs_mc_7:.4f}")
print()

print("Done!")
