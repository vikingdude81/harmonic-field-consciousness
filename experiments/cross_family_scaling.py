"""
Cross-Family Scaling Analysis
==============================
Compares how consciousness-compressibility metrics scale across
multiple architecture families (Qwen, Mistral, Yi, Llama).

Key question: Is the CS-SE correlation strengthening with scale
universal, or specific to Qwen?
"""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# All model results organized by family
FAMILIES = {
    "Qwen": [
        {
            "name": "Qwen-7B",
            "path": "experiment_outputs/combined_plugin_analysis_20260203_231542.json",
            "params": 7,
            "is_local": True,
        },
        {
            "name": "Qwen-14B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_085223.json",
            "params": 14,
            "is_local": True,
        },
        {
            "name": "Qwen-32B",
            "path": "experiment_outputs/combined_plugin_32B_20260204_000325.json",
            "params": 32,
            "is_local": False,
        },
    ],
    "Mistral": [
        {
            "name": "Mistral-7B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_002457.json",
            "params": 7,
            "is_local": True,
        },
        {
            "name": "Nemo-12B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_092157.json",
            "params": 12,
            "is_local": True,
        },
        {
            "name": "Small-24B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_093203.json",
            "params": 24,
            "is_local": True,
        },
    ],
    "Yi": [
        {
            "name": "Yi-9B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_003911.json",
            "params": 9,
            "is_local": True,
        },
        {
            "name": "Yi-34B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_095020.json",
            "params": 34,
            "is_local": True,
        },
    ],
    "Llama": [
        {
            "name": "Llama-8B",
            "path": "experiment_outputs/combined_plugin_analysis_20260204_003239.json",
            "params": 8,
            "is_local": True,
        },
    ],
}


def extract_metrics(result, is_local=True):
    """Extract standardized metrics from a result entry."""
    if is_local:
        comp = result["plugins"]["compressibility"]
        cc = comp.get("correlation_compression", {})
        chaos = result["plugins"].get("chaos", {})
        agency = result["plugins"].get("agency", {})
        return {
            "cs": result["consciousness_score"],
            "cc": cc.get("compressibility_corr", 0),
            "f90": cc.get("fraction_for_90pct", 0),
            "se": comp.get("spectral_entropy", 0),
            "mc": cc.get("mean_abs_correlation", 0),
            "lyap": chaos.get("lyapunov", 0),
            "hurst": chaos.get("hurst", 0),
            "agency": agency.get("agency_score", 0),
        }
    else:
        comp = result.get("compressibility", {})
        return {
            "cs": result["consciousness_score"],
            "cc": comp.get("compressibility_corr", 0),
            "f90": comp.get("fraction_for_90pct", 0),
            "se": comp.get("spectral_entropy", 0),
            "mc": comp.get("mean_abs_correlation", 0),
        }


def pearson(a, b):
    """Pearson correlation with significance."""
    try:
        r, p = stats.pearsonr(a, b)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return r, p, star
    except Exception:
        return 0, 1, ""


# Load all data
family_data = {}
for family, models in FAMILIES.items():
    family_data[family] = []
    for model_info in models:
        with open(model_info["path"]) as f:
            data = json.load(f)
        results = data["results"]
        metrics = [extract_metrics(r, model_info["is_local"]) for r in results]

        # Get hidden_dim and layers from results
        hidden_dim = results[0].get("hidden_dim", "?")
        target_layer = results[0].get("target_layer", "?")

        family_data[family].append({
            "name": model_info["name"],
            "params": model_info["params"],
            "metrics": metrics,
            "raw": results,
            "hidden_dim": hidden_dim,
            "target_layer": target_layer,
            "is_local": model_info["is_local"],
        })


print("=" * 90)
print("CROSS-FAMILY SCALING ANALYSIS")
print("Qwen (7B/14B/32B) | Mistral (7B/12B/24B) | Yi (9B/34B) | Llama (8B)")
print("=" * 90)
print()

# ====================================================================
# 1. OVERALL TABLE: All models sorted by parameter count
# ====================================================================
print("--- 1. ALL MODELS OVERVIEW ---")
print()
header = f"{'Family':<10} {'Model':<14} {'Params':>6} {'HidDim':>7} {'CS':>12} {'SE':>12} {'CC':>12} {'f90':>12}"
print(header)
print("-" * len(header))

all_models = []
for family, models in family_data.items():
    for m in models:
        metrics = m["metrics"]
        cs = [x["cs"] for x in metrics]
        se = [x["se"] for x in metrics]
        cc = [x["cc"] for x in metrics]
        f90 = [x["f90"] for x in metrics]
        print(f"{family:<10} {m['name']:<14} {m['params']:>5}B {m['hidden_dim']:>7} "
              f"{np.mean(cs):.3f}+{np.std(cs):.3f} "
              f"{np.mean(se):.3f}+{np.std(se):.3f} "
              f"{np.mean(cc):.3f}+{np.std(cc):.3f} "
              f"{np.mean(f90):.3f}+{np.std(f90):.3f}")
        all_models.append({
            "family": family,
            "name": m["name"],
            "params": m["params"],
            "cs_mean": np.mean(cs),
            "se_mean": np.mean(se),
            "cc_mean": np.mean(cc),
        })

print()

# ====================================================================
# 2. CS-SE CORRELATION AT EACH SCALE (the key test)
# ====================================================================
print("--- 2. CS-SE CORRELATION BY MODEL (key universality test) ---")
print("  Does the consciousness-compressibility link emerge at scale in ALL families?")
print()
print(f"  {'Family':<10} {'Model':<14} {'Params':>6}   {'CS vs SE':>25}   {'CS vs CC':>25}")
print(f"  {'-'*10} {'-'*14} {'-'*6}   {'-'*25}   {'-'*25}")

cs_se_data = {}
for family, models in family_data.items():
    cs_se_data[family] = []
    for m in models:
        cs = [x["cs"] for x in m["metrics"]]
        se = [x["se"] for x in m["metrics"]]
        cc = [x["cc"] for x in m["metrics"]]

        r_se, p_se, star_se = pearson(cs, se)
        r_cc, p_cc, star_cc = pearson(cs, cc)

        cs_se_data[family].append({
            "name": m["name"],
            "params": m["params"],
            "r_se": r_se,
            "p_se": p_se,
            "r_cc": r_cc,
            "p_cc": p_cc,
        })

        sig_se = "SIGNIFICANT" if p_se < 0.05 else ""
        sig_cc = "SIGNIFICANT" if p_cc < 0.05 else ""
        print(f"  {family:<10} {m['name']:<14} {m['params']:>5}B   "
              f"r={r_se:+.3f} p={p_se:.4f} {star_se:<3s} {sig_se:<11s}   "
              f"r={r_cc:+.3f} p={p_cc:.4f} {star_cc:<3s} {sig_cc}")

print()

# ====================================================================
# 3. SCALING CURVES WITHIN EACH FAMILY
# ====================================================================
print("--- 3. SCALING CURVES BY FAMILY ---")
print()

for family, models in family_data.items():
    if len(models) < 2:
        continue

    params = [m["params"] for m in models]
    log_params = np.log2(params)

    print(f"  {family} family ({' -> '.join(m['name'] for m in models)}):")

    for metric_key, label in [("cs", "Consciousness"), ("se", "Spectral Entropy"),
                               ("cc", "Compress Coeff"), ("f90", "Fraction-90%")]:
        means = [np.mean([x[metric_key] for x in m["metrics"]]) for m in models]
        if len(models) >= 3:
            r, p = stats.pearsonr(log_params, means)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            arrow = " -> ".join(f"{v:.4f}" for v in means)
            print(f"    {label:20s}: {arrow}  (r vs log2(P)={r:+.3f} {star})")
        else:
            arrow = " -> ".join(f"{v:.4f}" for v in means)
            diff = means[-1] - means[0]
            print(f"    {label:20s}: {arrow}  (delta={diff:+.4f})")

    # CS-SE correlation trend
    corrs = cs_se_data[family]
    r_vals = [c["r_se"] for c in corrs]
    p_vals = [c["p_se"] for c in corrs]
    print(f"    {'CS-SE corr trend':20s}: " +
          " -> ".join(f"r={r:+.3f}{'*' if p < 0.05 else ''}" for r, p in zip(r_vals, p_vals)))
    print()

# ====================================================================
# 4. UNIVERSALITY SCORE: Which models show significant CS-SE link?
# ====================================================================
print("--- 4. UNIVERSALITY ANALYSIS ---")
print()

# Group by parameter count buckets
small_models = []  # 7-9B
mid_models = []    # 12-14B
large_models = []  # 24-34B

for family, corrs in cs_se_data.items():
    for c in corrs:
        entry = {**c, "family": family}
        if c["params"] <= 10:
            small_models.append(entry)
        elif c["params"] <= 15:
            mid_models.append(entry)
        else:
            large_models.append(entry)

print("  Scale bucket       Models   Significant   Mean |r|   Verdict")
print("  " + "-" * 70)

for label, bucket in [("Small (7-9B)", small_models),
                       ("Mid (12-14B)", mid_models),
                       ("Large (24-34B)", large_models)]:
    n = len(bucket)
    if n == 0:
        continue
    n_sig = sum(1 for m in bucket if m["p_se"] < 0.05)
    mean_r = np.mean([abs(m["r_se"]) for m in bucket])
    names = ", ".join(f"{m['family']}-{m['params']}B" for m in bucket)
    verdict = "UNIVERSAL" if n_sig == n and n > 1 else f"{n_sig}/{n} significant"
    print(f"  {label:<20s} {n:>3}       {n_sig}/{n}           {mean_r:.3f}     {verdict}")
    print(f"    Models: {names}")

print()

# ====================================================================
# 5. CROSS-FAMILY COMPARISON AT SIMILAR SCALES
# ====================================================================
print("--- 5. CROSS-FAMILY AT ~7B SCALE ---")
print()

seven_b = []
for family, models in family_data.items():
    m = models[0]  # smallest model in each family
    metrics = m["metrics"]
    cs = np.mean([x["cs"] for x in metrics])
    se = np.mean([x["se"] for x in metrics])
    cc = np.mean([x["cc"] for x in metrics])
    r_se, p_se, _ = pearson([x["cs"] for x in metrics], [x["se"] for x in metrics])
    seven_b.append({
        "family": family,
        "name": m["name"],
        "params": m["params"],
        "cs": cs,
        "se": se,
        "cc": cc,
        "r_se": r_se,
        "p_se": p_se,
    })

print(f"  {'Family':<10} {'Model':<14} {'CS':>8} {'SE':>8} {'CC':>8} {'CS-SE r':>8} {'p':>8}")
print(f"  {'-'*10} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for m in seven_b:
    print(f"  {m['family']:<10} {m['name']:<14} {m['cs']:>8.4f} {m['se']:>8.4f} "
          f"{m['cc']:>8.4f} {m['r_se']:>+8.3f} {m['p_se']:>8.4f}")

# ANOVA on CS at ~7B
cs_groups = []
for family, models in family_data.items():
    m = models[0]
    cs_groups.append([x["cs"] for x in m["metrics"]])

if len(cs_groups) >= 3:
    F, p = stats.f_oneway(*cs_groups)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"\n  ANOVA (CS across families at ~7B): F={F:.3f}, p={p:.4f} {star}")
    if p >= 0.05:
        print("  -> CS is ARCHITECTURE-INDEPENDENT at small scale")

print()

# ====================================================================
# 6. LARGE SCALE COMPARISON
# ====================================================================
print("--- 6. CROSS-FAMILY AT LARGE SCALE (24-34B) ---")
print()

large_scale = []
for family, models in family_data.items():
    if len(models) < 2:
        continue
    m = models[-1]  # largest model in family
    if m["params"] >= 20:
        metrics = m["metrics"]
        r_se, p_se, star = pearson([x["cs"] for x in metrics], [x["se"] for x in metrics])
        r_cc, p_cc, star_cc = pearson([x["cs"] for x in metrics], [x["cc"] for x in metrics])
        large_scale.append({
            "family": family,
            "name": m["name"],
            "params": m["params"],
            "cs": np.mean([x["cs"] for x in metrics]),
            "se": np.mean([x["se"] for x in metrics]),
            "r_se": r_se,
            "p_se": p_se,
            "r_cc": r_cc,
            "p_cc": p_cc,
        })

if large_scale:
    print(f"  {'Family':<10} {'Model':<14} {'CS':>8} {'CS-SE r':>8} {'p':>8} {'CS-CC r':>8} {'p':>8}")
    print(f"  {'-'*10} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in large_scale:
        sig = "***" if m["p_se"] < 0.001 else "**" if m["p_se"] < 0.01 else "*" if m["p_se"] < 0.05 else ""
        print(f"  {m['family']:<10} {m['name']:<14} {m['cs']:>8.4f} "
              f"{m['r_se']:>+8.3f} {m['p_se']:>8.4f}{sig:<3s} "
              f"{m['r_cc']:>+8.3f} {m['p_cc']:>8.4f}")

    n_sig = sum(1 for m in large_scale if m["p_se"] < 0.05)
    print(f"\n  Significant CS-SE at large scale: {n_sig}/{len(large_scale)} families")
    if n_sig == len(large_scale):
        print("  -> CS-COMPRESSIBILITY LINK IS UNIVERSAL AT SCALE")

print()

# ====================================================================
# 7. CATEGORY RANKINGS STABILITY
# ====================================================================
print("--- 7. CATEGORY RANKINGS BY FAMILY AND SCALE ---")
print()

for family, models in family_data.items():
    for m in models:
        cats = {}
        for r in m["raw"]:
            cat = r["category"]
            cats.setdefault(cat, []).append(r["consciousness_score"])
        ranked = sorted(cats.items(), key=lambda x: np.mean(x[1]), reverse=True)
        ranking = " > ".join(f"{c}({np.mean(v):.3f})" for c, v in ranked)
        print(f"  {m['name']:<14}: {ranking}")
    print()

# ====================================================================
# 8. CHAOS/AGENCY SCALING (local models only)
# ====================================================================
print("--- 8. CHAOS/AGENCY AT SCALE (local models with full plugins) ---")
print()
header8 = f"  {'Family':<10} {'Model':<14} {'Lyap':>8} {'Hurst':>8} {'Agency':>8}   {'CC-Lyap':>16}   {'CC-Agency':>16}"
print(header8)
print(f"  {'-'*10} {'-'*14} {'-'*8} {'-'*8} {'-'*8}   {'-'*16}   {'-'*16}")

for family, models in family_data.items():
    for m in models:
        if not m["is_local"]:
            continue
        metrics = m["metrics"]
        if "lyap" not in metrics[0]:
            continue
        lyap = [x["lyap"] for x in metrics]
        hurst = [x["hurst"] for x in metrics]
        agency = [x["agency"] for x in metrics]
        cc = [x["cc"] for x in metrics]

        r_cl, p_cl, star_cl = pearson(cc, lyap)
        r_ca, p_ca, star_ca = pearson(cc, agency)

        print(f"  {family:<10} {m['name']:<14} {np.mean(lyap):>8.4f} {np.mean(hurst):>8.4f} "
              f"{np.mean(agency):>8.4f}   r={r_cl:+.3f} p={p_cl:.4f}{star_cl:<3s}   "
              f"r={r_ca:+.3f} p={p_ca:.4f}{star_ca}")

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("=" * 90)
print("SUMMARY: CROSS-FAMILY SCALING UNIVERSALITY")
print("=" * 90)
print()

# Collect CS-SE trend across all families with >= 2 models
print("  CS-SE Correlation Scaling by Family:")
print()
for family, corrs in cs_se_data.items():
    if len(corrs) < 2:
        continue
    print(f"  {family}:")
    for c in corrs:
        sig = "SIGNIFICANT" if c["p_se"] < 0.05 else "not sig."
        print(f"    {c['name']:>14}: r={c['r_se']:+.3f}, p={c['p_se']:.4f}  {sig}")

    # Check if correlation strengthens with scale
    r_vals = [abs(c["r_se"]) for c in corrs]
    if r_vals[-1] > r_vals[0]:
        print(f"    -> |r| strengthens: {r_vals[0]:.3f} -> {r_vals[-1]:.3f}")
    else:
        print(f"    -> |r| does NOT strengthen: {r_vals[0]:.3f} -> {r_vals[-1]:.3f}")
    print()

# Final verdict
print("  VERDICTS:")
print()

# Count families where the largest model shows significant CS-SE
families_with_scaling = []
for family, corrs in cs_se_data.items():
    if len(corrs) < 2:
        continue
    largest = corrs[-1]
    if largest["p_se"] < 0.05:
        families_with_scaling.append(family)

if families_with_scaling:
    print(f"  1. Families where CS-SE is significant at largest scale: "
          f"{', '.join(families_with_scaling)} ({len(families_with_scaling)} families)")
else:
    print("  1. No family shows significant CS-SE at large scale")

# Count families where CS-SE strengthens monotonically
strengthening = []
for family, corrs in cs_se_data.items():
    if len(corrs) < 2:
        continue
    r_abs = [abs(c["r_se"]) for c in corrs]
    if all(r_abs[i+1] >= r_abs[i] for i in range(len(r_abs)-1)):
        strengthening.append(family)

if strengthening:
    print(f"  2. Families where |CS-SE| strengthens monotonically: "
          f"{', '.join(strengthening)}")
else:
    print("  2. No family shows monotonic |CS-SE| strengthening")

# Architecture independence at small scale
small_cs = [np.mean([x["cs"] for x in models[0]["metrics"]]) for models in family_data.values()]
print(f"\n  3. CS range at small scale: {min(small_cs):.4f} - {max(small_cs):.4f} "
      f"(spread = {max(small_cs) - min(small_cs):.4f})")

# Strongest and weakest CS-SE at scale
if large_scale:
    best = max(large_scale, key=lambda x: abs(x["r_se"]))
    print(f"\n  4. Strongest CS-SE at scale: {best['family']} ({best['name']}) "
          f"r={best['r_se']:+.3f}, p={best['p_se']:.4f}")

print()
print("Done!")
