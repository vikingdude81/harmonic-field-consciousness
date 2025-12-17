#!/usr/bin/env python3
"""
Advanced Cross-Experiment Analysis with Statistical Tests

1. Cross-experiment comparisons
2. Statistical tests (t-tests, ANOVA, effect sizes)
3. Deep dive into R (synchronization) as consciousness predictor
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED CROSS-EXPERIMENT ANALYSIS")
print("=" * 80)

# =============================================================================
# PART 1: CROSS-EXPERIMENT COMPARISONS
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: CROSS-EXPERIMENT COMPARISONS")
print("=" * 80)

# Load all datasets
df_hub = pd.read_csv('category1_network_topology/results/exp3_hub_disruption/hub_disruption_results.csv')
df_crit = pd.read_csv('category2_dynamics/results/exp4_criticality_tuning/state_criticality_results.csv')
df_coup = pd.read_csv('category2_dynamics/results/exp3_coupling_strength/coupling_sweep_results.csv')
df_drug = pd.read_csv('category2_dynamics/results/exp3_coupling_strength/drug_effects_results.csv')
df_metrics = pd.read_csv('category3_functional_modifications/results/exp2_new_metrics/sample_metrics_results.csv')
df_nn = pd.read_csv('category4_applications/results/exp1_neural_networks/architecture_comparison.csv')
df_social = pd.read_csv('category4_applications/results/exp2_social_networks/network_types_comparison.csv')

# 1.1 Consciousness across all experimental conditions
print("\n### 1.1 CONSCIOUSNESS (C) ACROSS ALL CONDITIONS ###")

all_conditions = []

# Hub disruption baseline (0% removal)
hub_baseline = df_hub[df_hub['removal_pct'] == 0].groupby('brain_state')['C'].mean()
for state, c in hub_baseline.items():
    all_conditions.append({'experiment': 'Hub_baseline', 'condition': state, 'C': c})

# Criticality states
for _, row in df_crit.iterrows():
    all_conditions.append({'experiment': 'Criticality', 'condition': row['state'], 'C': row['C']})

# Drug effects
for _, row in df_drug.iterrows():
    all_conditions.append({'experiment': 'Drug_effects', 'condition': row['drug'], 'C': row['C']})

# Coupling extremes
all_conditions.append({'experiment': 'Coupling', 'condition': 'Low_K', 'C': df_coup['C'].max()})
all_conditions.append({'experiment': 'Coupling', 'condition': 'High_K', 'C': df_coup['C'].min()})

# Neural networks
for _, row in df_nn.iterrows():
    all_conditions.append({'experiment': 'Neural_net', 'condition': row['architecture'], 'C': row['C_trained']})

df_all = pd.DataFrame(all_conditions)
print("\nGlobal C statistics:")
print(f"  Mean C across all conditions: {df_all['C'].mean():.4f}")
print(f"  Std C across all conditions:  {df_all['C'].std():.4f}")
print(f"  Min C: {df_all['C'].min():.4f} ({df_all.loc[df_all['C'].idxmin(), 'condition']})")
print(f"  Max C: {df_all['C'].max():.4f} ({df_all.loc[df_all['C'].idxmax(), 'condition']})")

# 1.2 Compare hub disruption across brain states
print("\n### 1.2 HUB DISRUPTION: STATE COMPARISON ###")
print("\nC(t) degradation rate (slope) by brain state:")
for state in df_hub['brain_state'].unique():
    state_data = df_hub[(df_hub['brain_state'] == state) & (df_hub['strategy'] == 'degree')]
    slope, intercept, r, p, se = stats.linregress(state_data['removal_pct'], state_data['C'])
    print(f"  {state:12s}: slope = {slope:.4f} per %removed (r² = {r**2:.3f})")

# 1.3 Network type vs brain state interaction
print("\n### 1.3 TOPOLOGY-STATE INTERACTION ###")
print("(Comparing consciousness functional components across network types)")
print("\nSocial network topology effects on consciousness components:")
topo_means = df_social.groupby('network_type')[['C', 'H_mode', 'PR', 'R', 'kappa']].mean()
print(topo_means.round(4).to_string())

# =============================================================================
# PART 2: STATISTICAL TESTS
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: STATISTICAL TESTS")
print("=" * 80)

# 2.1 T-test: Targeted vs Random lesions
print("\n### 2.1 T-TEST: TARGETED vs RANDOM LESIONS ###")
targeted = df_hub[df_hub['strategy'].isin(['degree', 'betweenness', 'eigenvector'])]['C']
random = df_hub[df_hub['strategy'] == 'random']['C']
t_stat, p_value = stats.ttest_ind(targeted, random)
cohens_d = (random.mean() - targeted.mean()) / np.sqrt((random.std()**2 + targeted.std()**2) / 2)
print(f"  Targeted mean C: {targeted.mean():.4f} ± {targeted.std():.4f}")
print(f"  Random mean C:   {random.mean():.4f} ± {random.std():.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")

# 2.2 ANOVA: Brain states
print("\n### 2.2 ONE-WAY ANOVA: BRAIN STATES ###")
groups = [df_hub[df_hub['brain_state'] == state]['C'].values for state in df_hub['brain_state'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_value:.2e}")

# Post-hoc: Tukey's HSD approximation via pairwise t-tests with Bonferroni
print("\n  Post-hoc pairwise comparisons (Bonferroni corrected):")
states = df_hub['brain_state'].unique()
n_comparisons = len(states) * (len(states) - 1) // 2
alpha_corrected = 0.05 / n_comparisons
significant_pairs = []
for i, s1 in enumerate(states):
    for s2 in states[i+1:]:
        g1 = df_hub[df_hub['brain_state'] == s1]['C']
        g2 = df_hub[df_hub['brain_state'] == s2]['C']
        t, p = stats.ttest_ind(g1, g2)
        sig = "***" if p < alpha_corrected else ""
        if p < alpha_corrected:
            significant_pairs.append((s1, s2, p))
        print(f"    {s1} vs {s2}: p = {p:.2e} {sig}")

# 2.3 Correlation: Criticality vs Consciousness
print("\n### 2.3 CORRELATION: CRITICALITY (κ) vs CONSCIOUSNESS (C) ###")
kappa_vals = df_crit['kappa'].values
c_vals = df_crit['C'].values
r, p = stats.pearsonr(kappa_vals, c_vals)
print(f"  Pearson r: {r:.4f}")
print(f"  p-value: {p:.4f}")
print(f"  R²: {r**2:.4f}")

# 2.4 Effect sizes for drug conditions
print("\n### 2.4 DRUG EFFECT SIZES (vs Baseline) ###")
baseline_c = df_drug[df_drug['drug'] == 'Baseline']['C'].values[0]
for _, row in df_drug.iterrows():
    if row['drug'] != 'Baseline':
        diff = row['C'] - baseline_c
        pct_change = (diff / baseline_c) * 100
        print(f"  {row['drug']:15s}: ΔC = {diff:+.4f} ({pct_change:+.1f}%)")

# 2.5 Coupling strength regression
print("\n### 2.5 COUPLING-CONSCIOUSNESS REGRESSION ###")
slope, intercept, r, p, se = stats.linregress(df_coup['K'], df_coup['C'])
print(f"  C = {intercept:.4f} + {slope:.4f} * K")
print(f"  R²: {r**2:.4f}")
print(f"  p-value: {p:.2e}")
print(f"  Interpretation: Each unit increase in K reduces C by {abs(slope):.4f}")

# =============================================================================
# PART 3: DEEP DIVE INTO R (SYNCHRONIZATION) AS PREDICTOR
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: R (SYNCHRONIZATION) AS CONSCIOUSNESS PREDICTOR")
print("=" * 80)

# 3.1 R vs C correlation across all data
print("\n### 3.1 R-C RELATIONSHIP ACROSS CONDITIONS ###")

# Combine R and C from multiple sources
r_c_data = []

# From hub disruption
for _, row in df_hub.iterrows():
    r_c_data.append({'source': 'hub_disruption', 'R': row['R'], 'C': row['C'], 
                     'state': row['brain_state'], 'condition': f"{row['strategy']}_{row['removal_pct']}%"})

# From coupling sweep
for _, row in df_coup.iterrows():
    r_c_data.append({'source': 'coupling', 'R': row['R'], 'C': row['C'],
                     'state': 'wake', 'condition': f"K={row['K']:.1f}"})

# From drugs
for _, row in df_drug.iterrows():
    r_c_data.append({'source': 'drug', 'R': row['R'], 'C': row['C'],
                     'state': 'modulated', 'condition': row['drug']})

df_rc = pd.DataFrame(r_c_data)

# Overall correlation
r_corr, p_corr = stats.pearsonr(df_rc['R'], df_rc['C'])
print(f"\nOverall R-C correlation:")
print(f"  Pearson r: {r_corr:.4f}")
print(f"  p-value: {p_corr:.2e}")

# 3.2 Non-linear relationship?
print("\n### 3.2 NON-LINEAR R-C RELATIONSHIP ###")
print("Testing quadratic fit: C = a + b*R + c*R²")
R = df_rc['R'].values
C = df_rc['C'].values
# Quadratic regression
coeffs = np.polyfit(R, C, 2)
fitted = np.polyval(coeffs, R)
ss_res = np.sum((C - fitted)**2)
ss_tot = np.sum((C - C.mean())**2)
r2_quad = 1 - ss_res / ss_tot
print(f"  Quadratic: C = {coeffs[0]:.4f}R² + {coeffs[1]:.4f}R + {coeffs[2]:.4f}")
print(f"  R² (quadratic): {r2_quad:.4f}")

# Linear for comparison
coeffs_lin = np.polyfit(R, C, 1)
fitted_lin = np.polyval(coeffs_lin, R)
ss_res_lin = np.sum((C - fitted_lin)**2)
r2_lin = 1 - ss_res_lin / ss_tot
print(f"  R² (linear):    {r2_lin:.4f}")
print(f"  Improvement:    {(r2_quad - r2_lin)*100:.1f}%")

# 3.3 R ranges and consciousness
print("\n### 3.3 OPTIMAL R RANGE FOR CONSCIOUSNESS ###")
df_rc['R_bin'] = pd.cut(df_rc['R'], bins=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], 
                        labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0'])
r_binned = df_rc.groupby('R_bin', observed=True)['C'].agg(['mean', 'std', 'count'])
print("\nMean C by R range:")
print(r_binned.round(4).to_string())

optimal_bin = r_binned['mean'].idxmax()
print(f"\n>>> Optimal R range: {optimal_bin} (C = {r_binned.loc[optimal_bin, 'mean']:.4f})")

# 3.4 R by experimental source
print("\n### 3.4 R-C RELATIONSHIP BY EXPERIMENTAL CONTEXT ###")
for source in df_rc['source'].unique():
    subset = df_rc[df_rc['source'] == source]
    if len(subset) > 2:
        r, p = stats.pearsonr(subset['R'], subset['C'])
        print(f"  {source:15s}: r = {r:+.4f}, p = {p:.2e}, n = {len(subset)}")

# 3.5 Interaction: R × kappa → C
print("\n### 3.5 INTERACTION: R × CRITICALITY → C ###")
# Merge with criticality data where possible
df_hub_crit = df_hub.copy()
print("Testing if R effect depends on criticality (κ):")
# Split by kappa quartiles
df_hub_crit['kappa_level'] = pd.qcut(df_hub_crit['kappa'], q=2, labels=['Low_κ', 'High_κ'])
for level in ['Low_κ', 'High_κ']:
    subset = df_hub_crit[df_hub_crit['kappa_level'] == level]
    r, p = stats.pearsonr(subset['R'], subset['C'])
    print(f"  {level}: R-C correlation = {r:+.4f} (p = {p:.2e})")

# 3.6 What modulates R?
print("\n### 3.6 WHAT CONTROLS SYNCHRONIZATION (R)? ###")
print("\nR by brain state:")
r_by_state = df_hub.groupby('brain_state')['R'].agg(['mean', 'std'])
print(r_by_state.round(4).to_string())

print("\nR by lesion strategy (at 20% removal):")
r_by_strat = df_hub[df_hub['removal_pct'] == 20].groupby('strategy')['R'].agg(['mean', 'std'])
print(r_by_strat.round(4).to_string())

print("\nR by coupling strength (K):")
print("  Low K (0-1):   R = {:.4f}".format(df_coup[df_coup['K'] <= 1]['R'].mean()))
print("  Med K (1-3):   R = {:.4f}".format(df_coup[(df_coup['K'] > 1) & (df_coup['K'] <= 3)]['R'].mean()))
print("  High K (3-5):  R = {:.4f}".format(df_coup[df_coup['K'] > 3]['R'].mean()))

# 3.7 Critical R threshold
print("\n### 3.7 CRITICAL R THRESHOLD FOR CONSCIOUSNESS ###")
# Find R value where C crosses 0.5 (consciousness threshold)
df_sorted = df_rc.sort_values('R')
# Interpolate
for i in range(len(df_sorted) - 1):
    c1, c2 = df_sorted.iloc[i]['C'], df_sorted.iloc[i+1]['C']
    r1, r2 = df_sorted.iloc[i]['R'], df_sorted.iloc[i+1]['R']
    if (c1 >= 0.5 and c2 < 0.5) or (c1 < 0.5 and c2 >= 0.5):
        # Linear interpolation
        r_critical = r1 + (0.5 - c1) * (r2 - r1) / (c2 - c1)
        print(f"  R threshold for C = 0.5: ~{r_critical:.3f}")
        break
else:
    print("  No clear threshold crossing found in data")

# 3.8 Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR FURTHER R ANALYSIS")
print("=" * 80)
print("""
1. TIME-SERIES ANALYSIS:
   - Compute R dynamics over time, not just steady-state
   - Test Granger causality: Does R change predict C change?
   - Analyze R fluctuations (variance) as consciousness marker

2. FREQUENCY DECOMPOSITION:
   - Compute R in different frequency bands (delta, theta, alpha, beta, gamma)
   - Test which band-specific R best predicts consciousness

3. SPATIAL PATTERNS:
   - Compute local R for network communities/modules
   - Test if R heterogeneity (variance across regions) matters

4. PHASE RELATIONSHIPS:
   - Beyond R (amplitude sync), analyze phase-locking value (PLV)
   - Test phase-amplitude coupling as predictor

5. DYNAMIC R:
   - Compute metastability (R variance over time)
   - Test if dynamic R patterns predict state transitions

6. MULTIVARIATE MODELS:
   - Build regression: C ~ R + R² + R*κ + R*H_mode
   - Test interaction terms for moderation effects

7. EXPERIMENTAL MANIPULATIONS:
   - Systematically vary network coupling to control R
   - Test causal relationship: Manipulate R, measure C change
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
