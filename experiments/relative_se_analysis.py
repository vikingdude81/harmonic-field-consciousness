"""
Relative Spectral Entropy Analysis
====================================
Tests whether z-scoring SE within each model reveals a CS-SE
correlation in Mistral (where raw SE is at ceiling ~0.81).

Hypothesis: Mistral's high absolute SE masks prompt-level variation.
If we normalize to z-scores (how much does THIS prompt deviate from
this model's baseline?), the CS-SE link may emerge.
"""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

MODELS = {
    "Qwen-7B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260203_231542.json",
        "family": "Qwen", "params": 7, "is_local": True,
    },
    "Qwen-14B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_085223.json",
        "family": "Qwen", "params": 14, "is_local": True,
    },
    "Qwen-32B": {
        "path": "experiment_outputs/combined_plugin_32B_20260204_000325.json",
        "family": "Qwen", "params": 32, "is_local": False,
    },
    "Mistral-7B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_002457.json",
        "family": "Mistral", "params": 7, "is_local": True,
    },
    "Nemo-12B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_092157.json",
        "family": "Mistral", "params": 12, "is_local": True,
    },
    "Small-24B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_093203.json",
        "family": "Mistral", "params": 24, "is_local": True,
    },
    "Yi-9B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_003911.json",
        "family": "Yi", "params": 9, "is_local": True,
    },
    "Yi-34B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_095020.json",
        "family": "Yi", "params": 34, "is_local": True,
    },
    "Llama-8B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_003239.json",
        "family": "Llama", "params": 8, "is_local": True,
    },
}


def extract_metrics(result, is_local=True):
    if is_local:
        comp = result["plugins"]["compressibility"]
        cc = comp.get("correlation_compression", {})
        return {
            "cs": result["consciousness_score"],
            "se": comp.get("spectral_entropy", 0),
            "cc": cc.get("compressibility_corr", 0),
            "f90": cc.get("fraction_for_90pct", 0),
            "mc": cc.get("mean_abs_correlation", 0),
        }
    else:
        comp = result.get("compressibility", {})
        return {
            "cs": result["consciousness_score"],
            "se": comp.get("spectral_entropy", 0),
            "cc": comp.get("compressibility_corr", 0),
            "f90": comp.get("fraction_for_90pct", 0),
            "mc": comp.get("mean_abs_correlation", 0),
        }


def pearson(a, b):
    try:
        r, p = stats.pearsonr(a, b)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return r, p, star
    except Exception:
        return 0, 1, ""


# Load all data
model_data = {}
for name, info in MODELS.items():
    with open(info["path"]) as f:
        data = json.load(f)
    results = data["results"]
    metrics = [extract_metrics(r, info["is_local"]) for r in results]
    model_data[name] = {
        "metrics": metrics,
        "raw": results,
        "info": info,
    }


print("=" * 90)
print("RELATIVE SPECTRAL ENTROPY ANALYSIS")
print("=" * 90)
print()

# ====================================================================
# 1. RAW SE STATISTICS: Demonstrate the ceiling effect
# ====================================================================
print("--- 1. RAW SE STATISTICS (the ceiling effect) ---")
print()
print(f"  {'Model':<14} {'SE mean':>8} {'SE std':>8} {'SE min':>8} {'SE max':>8} {'SE range':>8} {'CoV':>8}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for name in MODELS:
    se = [x["se"] for x in model_data[name]["metrics"]]
    mean_se = np.mean(se)
    std_se = np.std(se)
    cov = std_se / mean_se if mean_se > 0 else 0
    print(f"  {name:<14} {mean_se:>8.4f} {std_se:>8.4f} {min(se):>8.4f} {max(se):>8.4f} "
          f"{max(se)-min(se):>8.4f} {cov:>8.3f}")

print()
print("  CoV = Coefficient of Variation (std/mean). Higher = more prompt-level variation.")
print("  Low CoV means SE is at ceiling/floor with little room for CS to correlate.")
print()

# ====================================================================
# 2. RAW vs Z-SCORED CS-SE CORRELATION
# ====================================================================
print("--- 2. RAW vs Z-SCORED CS-SE CORRELATION ---")
print()
print(f"  {'Model':<14} {'Raw CS-SE':>20}   {'Z-scored CS-SE':>20}   {'Change':>8}")
print(f"  {'-'*14} {'-'*20}   {'-'*20}   {'-'*8}")

for name in MODELS:
    metrics = model_data[name]["metrics"]
    cs = np.array([x["cs"] for x in metrics])
    se = np.array([x["se"] for x in metrics])

    # Raw correlation
    r_raw, p_raw, star_raw = pearson(cs, se)

    # Z-score within this model
    cs_z = (cs - cs.mean()) / max(cs.std(), 1e-12)
    se_z = (se - se.mean()) / max(se.std(), 1e-12)

    # Z-scored correlation (mathematically identical for Pearson, but
    # this sets up the pooled analysis below)
    r_z, p_z, star_z = pearson(cs_z, se_z)

    # Store z-scored values for pooling
    model_data[name]["cs_z"] = cs_z
    model_data[name]["se_z"] = se_z

    change = abs(r_z) - abs(r_raw)
    print(f"  {name:<14} r={r_raw:+.3f} p={p_raw:.4f}{star_raw:<3s}   "
          f"r={r_z:+.3f} p={p_z:.4f}{star_z:<3s}   {change:+.3f}")

print()
print("  Note: Within-model z-scoring doesn't change Pearson r (it's invariant")
print("  to linear transforms). The real value is in pooled analysis below.")
print()

# ====================================================================
# 3. POOLED ANALYSIS: Z-scored metrics across ALL models
# ====================================================================
print("--- 3. POOLED ANALYSIS: All 9 models, z-scored within each ---")
print("  Can we detect a universal CS-SE link when we remove per-model baselines?")
print()

# Pool z-scored values
all_cs_z = []
all_se_z = []
all_cc_z = []
all_families = []
all_names = []
all_categories = []

for name in MODELS:
    metrics = model_data[name]["metrics"]
    cs = np.array([x["cs"] for x in metrics])
    se = np.array([x["se"] for x in metrics])
    cc = np.array([x["cc"] for x in metrics])

    cs_z = (cs - cs.mean()) / max(cs.std(), 1e-12)
    se_z = (se - se.mean()) / max(se.std(), 1e-12)
    cc_z = (cc - cc.mean()) / max(cc.std(), 1e-12)

    all_cs_z.extend(cs_z)
    all_se_z.extend(se_z)
    all_cc_z.extend(cc_z)
    all_families.extend([MODELS[name]["family"]] * len(cs))
    all_names.extend([name] * len(cs))

    for r in model_data[name]["raw"]:
        all_categories.append(r["category"])

all_cs_z = np.array(all_cs_z)
all_se_z = np.array(all_se_z)
all_cc_z = np.array(all_cc_z)
all_families = np.array(all_families)
all_categories = np.array(all_categories)

# Overall pooled correlation
r_pool, p_pool, star_pool = pearson(all_cs_z, all_se_z)
r_cc_pool, p_cc_pool, star_cc_pool = pearson(all_cs_z, all_cc_z)

print(f"  All models pooled (n={len(all_cs_z)}):")
print(f"    CS_z vs SE_z: r={r_pool:+.3f}, p={p_pool:.4f} {star_pool}")
print(f"    CS_z vs CC_z: r={r_cc_pool:+.3f}, p={p_cc_pool:.4f} {star_cc_pool}")
print()

# By family (pooled)
print("  By family (pooled across scales):")
for family in ["Qwen", "Mistral", "Yi", "Llama"]:
    mask = all_families == family
    n = mask.sum()
    r_f, p_f, star_f = pearson(all_cs_z[mask], all_se_z[mask])
    r_fc, p_fc, star_fc = pearson(all_cs_z[mask], all_cc_z[mask])
    print(f"    {family:<10} (n={n:>3}): CS_z vs SE_z: r={r_f:+.3f} p={p_f:.4f}{star_f:<3s}   "
          f"CS_z vs CC_z: r={r_fc:+.3f} p={p_fc:.4f}{star_fc}")
print()

# ====================================================================
# 4. CATEGORY-LEVEL ANALYSIS (pooled across ALL models)
# ====================================================================
print("--- 4. CATEGORY PROFILES (z-scored, pooled across all models) ---")
print("  Are some categories universally 'more conscious' and 'less entropic'?")
print()

print(f"  {'Category':<12} {'n':>4} {'mean CS_z':>10} {'mean SE_z':>10} {'mean CC_z':>10}")
print(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")

cat_profiles = {}
for cat in ["reflective", "math", "factual", "creative", "uncertain"]:
    mask = all_categories == cat
    n = mask.sum()
    cs_m = np.mean(all_cs_z[mask])
    se_m = np.mean(all_se_z[mask])
    cc_m = np.mean(all_cc_z[mask])
    cat_profiles[cat] = {"cs_z": cs_m, "se_z": se_m, "cc_z": cc_m, "n": n}
    print(f"  {cat:<12} {n:>4} {cs_m:>+10.3f} {se_m:>+10.3f} {cc_m:>+10.3f}")

print()

# Category means correlation
cat_cs = [cat_profiles[c]["cs_z"] for c in ["reflective", "math", "factual", "creative", "uncertain"]]
cat_se = [cat_profiles[c]["se_z"] for c in ["reflective", "math", "factual", "creative", "uncertain"]]
r_cat, p_cat, star_cat = pearson(cat_cs, cat_se)
print(f"  Category-level CS_z vs SE_z (n=5 categories): r={r_cat:+.3f}, p={p_cat:.4f} {star_cat}")
if r_cat < 0:
    print("  -> Higher consciousness categories have LOWER relative spectral entropy")
    print("     (more concentrated representations)")
print()

# ====================================================================
# 5. MIXED-EFFECTS PERSPECTIVE: Does scale modulate the CS-SE link?
# ====================================================================
print("--- 5. SCALE-MODULATED CS-SE ANALYSIS ---")
print("  Do larger models show stronger within-model CS-SE coupling?")
print()

all_params = []
all_r_values = []
all_p_values = []

for name, info in MODELS.items():
    metrics = model_data[name]["metrics"]
    cs = [x["cs"] for x in metrics]
    se = [x["se"] for x in metrics]
    r, p, star = pearson(cs, se)
    all_params.append(info["params"])
    all_r_values.append(r)
    all_p_values.append(p)

all_params = np.array(all_params)
all_r_values = np.array(all_r_values)
log_params = np.log2(all_params)

# Does CS-SE |r| increase with log(params)?
r_meta, p_meta, star_meta = pearson(log_params, np.abs(all_r_values))
print(f"  Meta-correlation: |CS-SE r| vs log2(params) across 9 models:")
print(f"    r={r_meta:+.3f}, p={p_meta:.4f} {star_meta}")
if r_meta > 0:
    print("    -> Larger models show STRONGER CS-SE coupling (universal trend)")
else:
    print("    -> No universal scaling of CS-SE coupling with model size")
print()

# Same but only negative r values (expected direction)
neg_mask = all_r_values < 0
if neg_mask.sum() >= 3:
    r_neg, p_neg, star_neg = pearson(log_params[neg_mask], all_r_values[neg_mask])
    print(f"  Among models with negative CS-SE (n={neg_mask.sum()}):")
    print(f"    r(CS-SE) vs log2(params): r={r_neg:+.3f}, p={p_neg:.4f} {star_neg}")
    names_neg = [n for n, m in zip(MODELS.keys(), neg_mask) if m]
    print(f"    Models: {', '.join(names_neg)}")
print()

# ====================================================================
# 6. WITHIN-CATEGORY CONSISTENCY
# ====================================================================
print("--- 6. WITHIN-CATEGORY CS-SE CONSISTENCY ACROSS MODELS ---")
print("  For each category, is the CS-SE pattern consistent?")
print()

for cat in ["reflective", "math", "factual", "creative", "uncertain"]:
    cs_vals = []
    se_vals = []
    model_names = []
    for name in MODELS:
        metrics = model_data[name]["metrics"]
        raw = model_data[name]["raw"]
        for i, (m, r) in enumerate(zip(metrics, raw)):
            if r["category"] == cat:
                cs_vals.append(m["cs"])
                se_vals.append(model_data[name]["se_z"][i] if hasattr(model_data[name].get("se_z", None), '__len__') else 0)
                model_names.append(name)

    # Can't do much with 3 per model, but pooled across 9 models we get 27
    cs_arr = np.array(cs_vals)
    se_arr = np.array(se_vals)
    if len(cs_arr) >= 5:
        r_c, p_c, star_c = pearson(cs_arr, se_arr)
    else:
        r_c, p_c, star_c = 0, 1, ""

    print(f"  {cat:<12} (n={len(cs_vals)}): CS vs SE_z: r={r_c:+.3f}, p={p_c:.4f} {star_c}")

print()

# ====================================================================
# 7. EFFECTIVE DIMENSIONALITY AS ALTERNATIVE TO SE
# ====================================================================
print("--- 7. ALTERNATIVE METRICS: f90 and mean |corr| ---")
print("  f90 and |r| may be less architecture-dependent than SE")
print()

# f90 statistics
print(f"  {'Model':<14} {'f90 mean':>8} {'f90 std':>8} {'|r| mean':>8} {'|r| std':>8}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for name in MODELS:
    metrics = model_data[name]["metrics"]
    f90 = [x["f90"] for x in metrics]
    mc = [x["mc"] for x in metrics]
    print(f"  {name:<14} {np.mean(f90):>8.4f} {np.std(f90):>8.4f} "
          f"{np.mean(mc):>8.4f} {np.std(mc):>8.4f}")

print()

# CS vs f90 and CS vs |r| correlations
print(f"  {'Model':<14} {'CS vs f90':>20}   {'CS vs |r|':>20}")
print(f"  {'-'*14} {'-'*20}   {'-'*20}")

for name in MODELS:
    metrics = model_data[name]["metrics"]
    cs = [x["cs"] for x in metrics]
    f90 = [x["f90"] for x in metrics]
    mc = [x["mc"] for x in metrics]

    r_f, p_f, star_f = pearson(cs, f90)
    r_m, p_m, star_m = pearson(cs, mc)

    print(f"  {name:<14} r={r_f:+.3f} p={p_f:.4f}{star_f:<3s}   "
          f"r={r_m:+.3f} p={p_m:.4f}{star_m}")

print()

# Pooled f90 and |r|
all_f90_z = []
all_mc_z = []
for name in MODELS:
    metrics = model_data[name]["metrics"]
    f90 = np.array([x["f90"] for x in metrics])
    mc = np.array([x["mc"] for x in metrics])
    f90_z = (f90 - f90.mean()) / max(f90.std(), 1e-12)
    mc_z = (mc - mc.mean()) / max(mc.std(), 1e-12)
    all_f90_z.extend(f90_z)
    all_mc_z.extend(mc_z)

all_f90_z = np.array(all_f90_z)
all_mc_z = np.array(all_mc_z)

r_f90p, p_f90p, star_f90p = pearson(all_cs_z, all_f90_z)
r_mcp, p_mcp, star_mcp = pearson(all_cs_z, all_mc_z)
print(f"  Pooled z-scored (n={len(all_cs_z)}):")
print(f"    CS_z vs f90_z: r={r_f90p:+.3f}, p={p_f90p:.4f} {star_f90p}")
print(f"    CS_z vs |r|_z: r={r_mcp:+.3f}, p={p_mcp:.4f} {star_mcp}")

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("=" * 90)
print("FINDINGS")
print("=" * 90)
print()
print(f"  1. Pooled CS_z vs SE_z (all 9 models, n={len(all_cs_z)}): r={r_pool:+.3f}, p={p_pool:.4f} {star_pool}")
print(f"  2. Category-level CS_z vs SE_z: r={r_cat:+.3f}, p={p_cat:.4f} {star_cat}")
print(f"  3. |CS-SE r| vs log2(params) meta-correlation: r={r_meta:+.3f}, p={p_meta:.4f} {star_meta}")
print(f"  4. Pooled CS_z vs f90_z: r={r_f90p:+.3f}, p={p_f90p:.4f} {star_f90p}")
print(f"  5. Pooled CS_z vs |r|_z: r={r_mcp:+.3f}, p={p_mcp:.4f} {star_mcp}")
print()
print("Done!")
