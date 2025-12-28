#!/usr/bin/env python3
"""
Comprehensive Results Analysis Script

Analyzes all experimental results and generates key insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("KEY INSIGHTS FROM ALL EXPERIMENTS")
print("=" * 70)

# 1. Hub Disruption Key Finding
df_hub = pd.read_csv('category1_network_topology/results/exp3_hub_disruption/hub_disruption_results.csv')
print("\n### 1. HUB VULNERABILITY ###")
print("Targeted (degree) vs Random lesions at 20% removal:")
for state in ['wake', 'psychedelic', 'anesthesia']:
    targeted = df_hub[(df_hub['brain_state']==state) & (df_hub['strategy']=='degree') & (df_hub['removal_pct']==20)]['C'].values[0]
    random_c = df_hub[(df_hub['brain_state']==state) & (df_hub['strategy']=='random') & (df_hub['removal_pct']==20)]['C'].values[0]
    diff = random_c - targeted
    print(f"  {state:12s}: Targeted C={targeted:.3f}, Random C={random_c:.3f}, Gap={diff:.3f}")
print(">>> Targeted lesions cause ~4-6% more consciousness loss than random lesions")

# 2. Criticality Sweet Spot
print("\n### 2. CRITICALITY & CONSCIOUSNESS ###")
df_crit = pd.read_csv('category2_dynamics/results/exp4_criticality_tuning/state_criticality_results.csv')
df_crit_sorted = df_crit.sort_values('C', ascending=False)
print("Consciousness ranking by criticality (kappa):")
for _, row in df_crit_sorted.iterrows():
    print(f"  {row['state']:12s}: C={row['C']:.3f}, kappa={row['kappa']:.3f}")
print(">>> States near criticality (kappa~0.5-0.6) show highest consciousness")

# 3. Drug Effects
print("\n### 3. PHARMACOLOGICAL MODULATION ###")
df_drug = pd.read_csv('category2_dynamics/results/exp3_coupling_strength/drug_effects_results.csv')
print("Drug effects on consciousness:")
for _, row in df_drug.iterrows():
    print(f"  {row['drug']:15s}: C={row['C']:.3f}, R={row['R']:.3f}")
print(">>> Psychedelics preserve consciousness; Propofol dramatically reduces it")

# 4. Network Type Effects  
print("\n### 4. NETWORK TOPOLOGY MATTERS ###")
df_social = pd.read_csv('category4_applications/results/exp2_social_networks/network_types_comparison.csv')
df_avg = df_social.groupby('network_type').agg({'C': 'mean', 'modularity': 'mean', 'clustering': 'mean'}).round(3)
print("Average C by network type:")
print(df_avg.to_string())
print(">>> Random networks have slightly higher C, but small-world has best modularity")

# 5. Feature Importance
print("\n### 5. WHAT PREDICTS CONSCIOUSNESS? ###")
df_feat = pd.read_csv('category3_functional_modifications/results/exp2_new_metrics/feature_importance.csv')
df_feat = df_feat.dropna().sort_values('abs_corr', ascending=False)
print("Top predictors of C(t):")
for _, row in df_feat.head(5).iterrows():
    metric = row['Unnamed: 0']
    corr = row['correlation']
    print(f"  {metric:10s}: r={corr:+.3f}")
print(">>> Synchronization (R), Participation Ratio (PR), and Mode Entropy (H_mode) are strongest predictors")

# 6. Neural Network Training
print("\n### 6. ARTIFICIAL CONSCIOUSNESS? ###")
df_nn = pd.read_csv('category4_applications/results/exp1_neural_networks/architecture_comparison.csv')
best = df_nn.loc[df_nn['C_trained'].idxmax()]
worst = df_nn.loc[df_nn['C_trained'].idxmin()]
print(f"Best architecture: {best['architecture']} (C={best['C_trained']:.3f})")
print(f"Worst architecture: {worst['architecture']} (C={worst['C_trained']:.3f})")
print(">>> Balanced architectures show highest consciousness-like metrics")

# 7. Coupling Dynamics
print("\n### 7. COUPLING STRENGTH PHASE TRANSITIONS ###")
df_coup = pd.read_csv('category2_dynamics/results/exp3_coupling_strength/coupling_sweep_results.csv')
max_c = df_coup.loc[df_coup['C'].idxmax()]
min_c = df_coup.loc[df_coup['C'].idxmin()]
print(f"Optimal coupling: K={max_c['K']:.2f} -> C={max_c['C']:.3f}")
print(f"Worst coupling:   K={min_c['K']:.2f} -> C={min_c['C']:.3f}")
print(">>> Low coupling preserves consciousness; high coupling causes synchronization collapse")

# 8. Modular Networks
print("\n### 8. MODULARITY EFFECTS ###")
try:
    df_mod = pd.read_csv('category1_network_topology/results/exp4_modular_networks/module_sweep_results.csv')
    print("Consciousness by number of modules:")
    print(df_mod.groupby('n_modules')['C'].mean().round(3).to_string())
except FileNotFoundError:
    print("Modular network detailed results not available")

print("\n" + "=" * 70)
print("SUMMARY: Key Findings")
print("=" * 70)
print("""
1. VULNERABILITY: Hub disruption more damaging than random lesions
2. CRITICALITY: Consciousness peaks near edge-of-chaos (kappa ≈ 0.5-0.6)
3. PHARMACOLOGY: Psychedelics maintain C; anesthetics reduce it dramatically
4. TOPOLOGY: Network structure affects consciousness metrics
5. PREDICTORS: R (synchronization), PR (participation), H_mode (entropy) predict C
6. AI SYSTEMS: Balanced neural architectures show highest "consciousness"
7. COUPLING: Low coupling optimal; over-synchronization reduces consciousness
""")
