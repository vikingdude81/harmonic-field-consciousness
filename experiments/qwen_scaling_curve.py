"""
Qwen Scaling Curve: 7B → 14B → 32B
====================================
Analyzes how consciousness-compressibility metrics scale with model size
within the same architecture family (Qwen2.5), isolating parameter count
as the independent variable.
"""
import json
import numpy as np
import sys
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# Qwen results at three scales
MODELS = {
    "Qwen-7B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260203_231542.json",
        "params": 7,
        "layers": 28,
        "hidden_dim": 3584,
        "is_local": True,
    },
    "Qwen-14B": {
        "path": "experiment_outputs/combined_plugin_analysis_20260204_085223.json",
        "params": 14,
        "layers": 48,
        "hidden_dim": 5120,
        "is_local": True,
    },
    "Qwen-32B": {
        "path": "experiment_outputs/combined_plugin_32B_20260204_000325.json",
        "params": 32,
        "layers": 64,
        "hidden_dim": 5120,
        "is_local": False,  # from HF Space
    },
}

CATEGORIES = ["reflective", "math", "factual", "creative", "uncertain"]


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


def corr_str(a, b):
    """Format Pearson correlation with significance stars."""
    try:
        r, p = stats.pearsonr(a, b)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return r, p, f"r={r:+.3f}, p={p:.4f} {star}"
    except Exception:
        return 0, 1, "N/A"


# Load all data
model_data = {}
for name, info in MODELS.items():
    with open(info["path"]) as f:
        data = json.load(f)
    results = data["results"]
    metrics = [extract_metrics(r, info["is_local"]) for r in results]
    model_data[name] = {
        "raw": results,
        "metrics": metrics,
        "info": info,
    }


print("=" * 80)
print("QWEN SCALING CURVE: 7B → 14B → 32B")
print("=" * 80)
print()

# ---- Overall metrics table ----
print("--- OVERALL METRICS (mean +/- std) ---")
print()
header = f"{'Model':<12} {'Params':>6} {'Layers':>6} {'HidDim':>6} {'CS':>12} {'CC':>12} {'f90':>12} {'SE':>12} {'|r|':>12}"
print(header)
print("-" * len(header))

for name, info in MODELS.items():
    m = model_data[name]["metrics"]
    cs = [x["cs"] for x in m]
    cc = [x["cc"] for x in m]
    f90 = [x["f90"] for x in m]
    se = [x["se"] for x in m]
    mc = [x["mc"] for x in m]
    print(f"{name:<12} {info['params']:>5}B {info['layers']:>6} {info['hidden_dim']:>6} "
          f"{np.mean(cs):.3f}+{np.std(cs):.3f} "
          f"{np.mean(cc):.3f}+{np.std(cc):.3f} "
          f"{np.mean(f90):.3f}+{np.std(f90):.3f} "
          f"{np.mean(se):.3f}+{np.std(se):.3f} "
          f"{np.mean(mc):.3f}+{np.std(mc):.3f}")

print()

# ---- Scaling trends ----
print("--- SCALING TRENDS (how metrics change with size) ---")
print()

params = [MODELS[n]["params"] for n in MODELS]
log_params = np.log2(params)

for metric_key, label in [("cs", "Consciousness"), ("cc", "Compress Coeff"),
                           ("f90", "Fraction-90%"), ("se", "Spectral Entropy"),
                           ("mc", "Mean |corr|")]:
    means = [np.mean([x[metric_key] for x in model_data[n]["metrics"]]) for n in MODELS]
    r, p = stats.pearsonr(log_params, means)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    trend = "increases" if r > 0 else "decreases"
    print(f"  {label:20s}: {means[0]:.4f} → {means[1]:.4f} → {means[2]:.4f}  "
          f"(r vs log2(params)={r:+.3f} {star})")

print()

# ---- CS-Compressibility correlation at each scale ----
print("--- CS-COMPRESSIBILITY CORRELATION BY SCALE ---")
print("  (Does the consciousness-compressibility link strengthen with scale?)")
print()
print(f"  {'Model':<12} {'CS vs CC':>20} {'CS vs SE':>20} {'CS vs f90':>20} {'CS vs |r|':>20}")
print(f"  {'-'*12} {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

cs_se_corrs = []
for name in MODELS:
    m = model_data[name]["metrics"]
    cs = [x["cs"] for x in m]
    cc = [x["cc"] for x in m]
    se = [x["se"] for x in m]
    f90 = [x["f90"] for x in m]
    mc = [x["mc"] for x in m]

    _, _, s1 = corr_str(cs, cc)
    r_se, p_se, s2 = corr_str(cs, se)
    _, _, s3 = corr_str(cs, f90)
    _, _, s4 = corr_str(cs, mc)
    cs_se_corrs.append((r_se, p_se))

    print(f"  {name:<12} {s1:>20} {s2:>20} {s3:>20} {s4:>20}")

print()
print("  Scaling of CS-SE correlation:")
for i, name in enumerate(MODELS):
    r, p = cs_se_corrs[i]
    print(f"    {name}: r={r:+.3f}, p={p:.4f} {'← SIGNIFICANT' if p < 0.05 else ''}")

print()

# ---- Category rankings at each scale ----
print("--- CATEGORY RANKINGS (by CS) AT EACH SCALE ---")
print()
for name in MODELS:
    results = model_data[name]["raw"]
    is_local = model_data[name]["info"]["is_local"]
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, []).append(r["consciousness_score"])
    ranked = sorted(cats.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"  {name:<12}: " + " > ".join(f"{c}({np.mean(v):.3f})" for c, v in ranked))

print()

# ---- Chaos/Agency at 7B and 14B ----
print("--- CHAOS/AGENCY METRICS (local models only) ---")
print()
header = f"  {'Model':<12} {'Lyap':>8} {'Hurst':>8} {'Agency':>8}   CC vs Lyap       CC vs Agency"
print(header)
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}   {'-'*16}   {'-'*16}")

for name in ["Qwen-7B", "Qwen-14B"]:
    m = model_data[name]["metrics"]
    lyap = [x["lyap"] for x in m]
    hurst = [x["hurst"] for x in m]
    agency = [x["agency"] for x in m]
    cc = [x["cc"] for x in m]

    r_cl, p_cl = stats.pearsonr(cc, lyap)
    r_ca, p_ca = stats.pearsonr(cc, agency)
    star_cl = "***" if p_cl < 0.001 else "**" if p_cl < 0.01 else "*" if p_cl < 0.05 else ""
    star_ca = "***" if p_ca < 0.001 else "**" if p_ca < 0.01 else "*" if p_ca < 0.05 else ""

    print(f"  {name:<12} {np.mean(lyap):>8.4f} {np.mean(hurst):>8.4f} {np.mean(agency):>8.4f}   "
          f"r={r_cl:+.3f} p={p_cl:.4f}{star_cl:3s}   r={r_ca:+.3f} p={p_ca:.4f}{star_ca:3s}")

print()

# ---- Paired comparison (same prompts, different scale) ----
print("--- PAIRED TESTS: 7B vs 14B vs 32B (same prompts) ---")
print()

for key, label in [("cs", "Consciousness"), ("cc", "Compress Coeff"),
                    ("f90", "Fraction-90%"), ("se", "Spectral Entropy"),
                    ("mc", "Mean |corr|")]:
    v7 = [x[key] for x in model_data["Qwen-7B"]["metrics"]]
    v14 = [x[key] for x in model_data["Qwen-14B"]["metrics"]]
    v32 = [x[key] for x in model_data["Qwen-32B"]["metrics"]]

    t_7_14, p_7_14 = stats.ttest_rel(v7, v14)
    t_7_32, p_7_32 = stats.ttest_rel(v7, v32)
    t_14_32, p_14_32 = stats.ttest_rel(v14, v32)

    def star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print(f"  {label:20s}:")
    print(f"    7B vs 14B: t={t_7_14:+.3f}, p={p_7_14:.4f} {star(p_7_14)}")
    print(f"    7B vs 32B: t={t_7_32:+.3f}, p={p_7_32:.4f} {star(p_7_32)}")
    print(f"    14B vs 32B: t={t_14_32:+.3f}, p={p_14_32:.4f} {star(p_14_32)}")
    print()

# ---- Summary ----
print("=" * 80)
print("SCALING SUMMARY")
print("=" * 80)
print()

# Key observations
m7 = model_data["Qwen-7B"]["metrics"]
m14 = model_data["Qwen-14B"]["metrics"]
m32 = model_data["Qwen-32B"]["metrics"]

cs_means = [np.mean([x["cs"] for x in m]) for m in [m7, m14, m32]]
se_means = [np.mean([x["se"] for x in m]) for m in [m7, m14, m32]]
cc_means = [np.mean([x["cc"] for x in m]) for m in [m7, m14, m32]]

print(f"  Consciousness Score:   7B={cs_means[0]:.4f}  14B={cs_means[1]:.4f}  32B={cs_means[2]:.4f}")
print(f"  Spectral Entropy:      7B={se_means[0]:.4f}  14B={se_means[1]:.4f}  32B={se_means[2]:.4f}")
print(f"  Compress Coeff:        7B={cc_means[0]:.4f}  14B={cc_means[1]:.4f}  32B={cc_means[2]:.4f}")
print()

# CS-SE correlation scaling
print("  CS-SE correlation strengthening with scale:")
for i, name in enumerate(MODELS):
    r, p = cs_se_corrs[i]
    print(f"    {name:>12}: r={r:+.3f}  p={p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

# Architecture details
print()
print("  Architecture details:")
for name, info in MODELS.items():
    tl = int(info["layers"] * 0.75)
    print(f"    {name}: {info['params']}B params, {info['layers']} layers, "
          f"hidden_dim={info['hidden_dim']}, target_layer={tl}")

print()
print("Done!")
