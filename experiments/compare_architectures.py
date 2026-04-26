"""
Cross-Architecture Comparison: 4 models x 7B scale + Qwen 32B
==============================================================
Compares consciousness + compressibility metrics across architectures.
"""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# Model result files
MODELS = {
    "Qwen-7B": "experiment_outputs/combined_plugin_analysis_20260203_231542.json",
    "Mistral-7B": "experiment_outputs/combined_plugin_analysis_20260204_002457.json",
    "Llama-8B": "experiment_outputs/combined_plugin_analysis_20260204_003239.json",
    "Yi-9B": "experiment_outputs/combined_plugin_analysis_20260204_003911.json",
    "Qwen-32B": "experiment_outputs/combined_plugin_32B_20260204_000325.json",
}

CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]


def extract_metrics(result, is_32b=False):
    """Extract standardized metrics from a result entry."""
    if is_32b:
        comp = result.get("compressibility", {})
        return {
            "cs": result["consciousness_score"],
            "cc": comp.get("compressibility_corr", 0),
            "f90": comp.get("fraction_for_90pct", 0),
            "se": comp.get("spectral_entropy", 0),
            "mc": comp.get("mean_abs_correlation", 0),
        }
    else:
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


# Load all data
model_data = {}
for name, path in MODELS.items():
    with open(path) as f:
        data = json.load(f)
    is_32b = "32B" in name
    results = data["results"]
    metrics = [extract_metrics(r, is_32b) for r in results]
    model_data[name] = {"raw": results, "metrics": metrics, "is_32b": is_32b}


print("=" * 80)
print("CROSS-ARCHITECTURE COMPARISON")
print("=" * 80)
print()

# ---- Overall comparison ----
print("--- OVERALL METRICS (mean +/- std) ---")
print()
header = f"{'Model':<15} {'CS':>12} {'CC':>12} {'f90':>12} {'SE':>12} {'|r|':>12}"
print(header)
print("-" * len(header))

for name in MODELS:
    m = model_data[name]["metrics"]
    cs = [x["cs"] for x in m]
    cc = [x["cc"] for x in m]
    f90 = [x["f90"] for x in m]
    se = [x["se"] for x in m]
    mc = [x["mc"] for x in m]
    print(f"{name:<15} "
          f"{np.mean(cs):.3f}+{np.std(cs):.3f} "
          f"{np.mean(cc):.3f}+{np.std(cc):.3f} "
          f"{np.mean(f90):.3f}+{np.std(f90):.3f} "
          f"{np.mean(se):.3f}+{np.std(se):.3f} "
          f"{np.mean(mc):.3f}+{np.std(mc):.3f}")

print()

# ---- 7B-only comparison with chaos/agency ----
print("--- 7B MODELS: FULL PLUGIN METRICS ---")
print()
header7 = f"{'Model':<15} {'CS':>8} {'CC':>8} {'f90':>8} {'SE':>8} {'Lyap':>8} {'Hurst':>8} {'Agency':>8}"
print(header7)
print("-" * len(header7))

for name in ["Qwen-7B", "Mistral-7B", "Llama-8B", "Yi-9B"]:
    m = model_data[name]["metrics"]
    print(f"{name:<15} "
          f"{np.mean([x['cs'] for x in m]):>8.4f} "
          f"{np.mean([x['cc'] for x in m]):>8.4f} "
          f"{np.mean([x['f90'] for x in m]):>8.4f} "
          f"{np.mean([x['se'] for x in m]):>8.4f} "
          f"{np.mean([x['lyap'] for x in m]):>8.4f} "
          f"{np.mean([x['hurst'] for x in m]):>8.4f} "
          f"{np.mean([x['agency'] for x in m]):>8.4f}")

print()

# ---- Category rankings per model ----
print("--- CATEGORY RANKINGS (by CS) ---")
print()
for name in MODELS:
    results = model_data[name]["raw"]
    is_32b = model_data[name]["is_32b"]
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, []).append(r["consciousness_score"])
    ranked = sorted(cats.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"  {name:<15}: " + " > ".join(f"{c}({np.mean(v):.3f})" for c, v in ranked))

print()

# ---- Cross-model correlation of metrics ----
print("--- CONSCIOUSNESS-COMPRESSIBILITY CORRELATION BY MODEL ---")
print()
print(f"  {'Model':<15} {'CS vs CC':>12} {'CS vs SE':>12} {'CS vs f90':>12} {'CS vs |r|':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for name in MODELS:
    m = model_data[name]["metrics"]
    cs = [x["cs"] for x in m]
    cc = [x["cc"] for x in m]
    se = [x["se"] for x in m]
    f90 = [x["f90"] for x in m]
    mc = [x["mc"] for x in m]

    def corr_str(a, b):
        try:
            r, p = stats.pearsonr(a, b)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            return f"r={r:+.3f}{star}"
        except Exception:
            return "N/A"

    print(f"  {name:<15} {corr_str(cs, cc):>12} {corr_str(cs, se):>12} "
          f"{corr_str(cs, f90):>12} {corr_str(cs, mc):>12}")

print()

# ---- Compressibility-Chaos correlation (7B only) ----
print("--- COMPRESSIBILITY-CHAOS CORRELATION (7B models) ---")
print()
for name in ["Qwen-7B", "Mistral-7B", "Llama-8B", "Yi-9B"]:
    m = model_data[name]["metrics"]
    cc = [x["cc"] for x in m]
    lyap = [x["lyap"] for x in m]
    agency = [x["agency"] for x in m]

    r_cl, p_cl = stats.pearsonr(cc, lyap)
    r_ca, p_ca = stats.pearsonr(cc, agency)
    star_cl = "***" if p_cl < 0.001 else "**" if p_cl < 0.01 else "*" if p_cl < 0.05 else ""
    star_ca = "***" if p_ca < 0.001 else "**" if p_ca < 0.01 else "*" if p_ca < 0.05 else ""
    print(f"  {name:<15} CC vs Lyap: r={r_cl:+.3f}, p={p_cl:.4f} {star_cl}"
          f"   CC vs Agency: r={r_ca:+.3f}, p={p_ca:.4f} {star_ca}")

print()

# ---- Universality test: do all models agree on which categories are most compressible? ----
print("--- UNIVERSALITY: CATEGORY COMPRESSIBILITY RANKINGS ---")
print()
for name in MODELS:
    results = model_data[name]["raw"]
    is_32b = model_data[name]["is_32b"]
    cats = {}
    for r in results:
        cat = r["category"]
        m = extract_metrics(r, is_32b)
        cats.setdefault(cat, []).append(m["f90"])
    ranked = sorted(cats.items(), key=lambda x: np.mean(x[1]))  # lower f90 = more compressible
    print(f"  {name:<15} (most->least compressible by f90): "
          + " > ".join(f"{c}({np.mean(v):.3f})" for c, v in ranked))

print()

# ---- ANOVA: does architecture matter? Pool all 7B models ----
print("--- ANOVA: DOES ARCHITECTURE AFFECT METRICS? (pooled 7B) ---")
print()
for metric_key in ["cs", "cc", "f90", "se"]:
    groups = []
    group_names = []
    for name in ["Qwen-7B", "Mistral-7B", "Llama-8B", "Yi-9B"]:
        m = model_data[name]["metrics"]
        groups.append([x[metric_key] for x in m])
        group_names.append(name)

    F, p = stats.f_oneway(*groups)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    labels = {"cs": "Consciousness", "cc": "Compress Coeff", "f90": "Fraction-90%", "se": "Spectral Entropy"}
    print(f"  {labels[metric_key]:20s}: F={F:.3f}, p={p:.4f} {star}")
    if p < 0.05:
        # Pairwise t-tests
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                t, tp = stats.ttest_ind(groups[i], groups[j])
                if tp < 0.05:
                    print(f"    {group_names[i]} vs {group_names[j]}: t={t:+.3f}, p={tp:.4f}")

print()

# ---- Summary statistics ----
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Count significant CS-compressibility correlations
sig_count = 0
for name in MODELS:
    m = model_data[name]["metrics"]
    cs = [x["cs"] for x in m]
    se = [x["se"] for x in m]
    r, p = stats.pearsonr(cs, se)
    if p < 0.05:
        sig_count += 1
        print(f"  {name}: CS vs SE significant (r={r:+.3f}, p={p:.4f})")

print(f"\n  Models with significant CS-SE correlation: {sig_count}/{len(MODELS)}")

# Architecture hidden dims
print("\n  Architecture details:")
for name in ["Qwen-7B", "Mistral-7B", "Llama-8B", "Yi-9B"]:
    results = model_data[name]["raw"]
    hd = results[0].get("hidden_dim", "?")
    tl = results[0].get("target_layer", "?")
    print(f"    {name}: hidden_dim={hd}, target_layer={tl}")

print(f"    Qwen-32B: hidden_dim=5120, target_layer=48")
print()
print("Done!")
