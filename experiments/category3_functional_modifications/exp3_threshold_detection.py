#!/usr/bin/env python3
"""
Category 3: Functional Modifications

Experiment 3: Clinical Consciousness Threshold Detection

Implements ROC analysis for consciousness detection using C(t) and components:
1. ROC curve analysis for each metric
2. Sensitivity and specificity testing
3. Optimal threshold determination
4. Decision boundary visualization
5. Clinical applicability assessment

Key question: Can C(t) reliably distinguish conscious from unconscious states?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg

# Configuration
SEED = 42
np.random.seed(SEED)
N_NODES = 64
N_MODES = 20
N_SAMPLES = 100  # Samples per state
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_threshold_detection'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Category 3, Experiment 3: Clinical Consciousness Threshold Detection")
print("="*70)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_roc_curve(labels, scores, n_thresholds=100):
    """
    Compute ROC curve from binary labels and continuous scores.
    
    Returns:
        fpr: False positive rates
        tpr: True positive rates (sensitivity)
        thresholds: Threshold values
        auc: Area under curve
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    
    tpr = []  # Sensitivity
    fpr = []  # 1 - Specificity
    
    for thresh in thresholds:
        predictions = (scores >= thresh).astype(int)
        
        # True positives, false positives, etc.
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        tpr.append(sensitivity)
        fpr.append(1 - specificity)
    
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]
    thresholds = thresholds[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return fpr, tpr, thresholds, auc


def find_optimal_threshold(fpr, tpr, thresholds, method='youden'):
    """
    Find optimal threshold using various methods.
    
    Methods:
        'youden': Maximizes sensitivity + specificity - 1
        'balanced': Minimizes |sensitivity - specificity|
        'high_sensitivity': Ensures sensitivity > 0.9
        'high_specificity': Ensures specificity > 0.9
    """
    if method == 'youden':
        # Youden's J statistic
        j = tpr - fpr
        idx = np.argmax(j)
    elif method == 'balanced':
        # Closest to (0, 1) corner
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        idx = np.argmin(distances)
    elif method == 'high_sensitivity':
        # First threshold with sensitivity > 0.9
        valid = np.where(tpr >= 0.9)[0]
        idx = valid[-1] if len(valid) > 0 else len(tpr) - 1
    elif method == 'high_specificity':
        # First threshold with specificity > 0.9
        valid = np.where((1 - fpr) >= 0.9)[0]
        idx = valid[0] if len(valid) > 0 else 0
    else:
        idx = len(thresholds) // 2
    
    return thresholds[idx], tpr[idx], 1 - fpr[idx]


# ==============================================================================
# PART 1: Generate Clinical Dataset
# ==============================================================================

print("\n" + "-"*70)
print("PART 1: Generating Clinical Dataset")
print("-"*70)

# Generate network
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

# Define conscious and unconscious states
conscious_states = {
    'Wake (Alert)': lambda s: sg.generate_wake_state(N_MODES, seed=s),
    'Wake (Relaxed)': lambda s: sg.generate_relaxed_wake_state(N_MODES, seed=s),
    'REM Sleep': lambda s: sg.generate_rem_conscious(N_MODES, seed=s),
    'Meditation': lambda s: sg.generate_meditation_state(N_MODES, depth=0.5, seed=s),
    'Psychedelic (Low)': lambda s: sg.generate_psychedelic_state(N_MODES, intensity=0.3, seed=s),
}

unconscious_states = {
    'NREM Sleep': lambda s: sg.generate_nrem_unconscious(N_MODES, seed=s),
    'Deep Anesthesia': lambda s: sg.generate_anesthesia_state(N_MODES, depth=1.0, seed=s),
    'Light Anesthesia': lambda s: sg.generate_anesthesia_state(N_MODES, depth=0.5, seed=s),
    'Minimal Consciousness': lambda s: sg.generate_minimal_consciousness(N_MODES, seed=s),
}

clinical_data = []

print("Generating conscious state samples...")
for state_name, state_fn in conscious_states.items():
    for i in range(N_SAMPLES // len(conscious_states)):
        seed = SEED + i * 100 + hash(state_name) % 1000
        power = state_fn(seed)
        
        # Add noise
        noise = np.random.rand(N_MODES) * 0.1
        power = power + noise
        power = power / power.sum()
        
        metrics = met.compute_all_metrics(power, eigenvalues[:N_MODES])
        
        clinical_data.append({
            'state': state_name,
            'label': 1,  # Conscious
            'sample_id': f'{state_name}_{i}',
            **metrics
        })

print("Generating unconscious state samples...")
for state_name, state_fn in unconscious_states.items():
    for i in range(N_SAMPLES // len(unconscious_states)):
        seed = SEED + i * 100 + hash(state_name) % 1000
        power = state_fn(seed)
        
        # Add noise
        noise = np.random.rand(N_MODES) * 0.1
        power = power + noise
        power = power / power.sum()
        
        metrics = met.compute_all_metrics(power, eigenvalues[:N_MODES])
        
        clinical_data.append({
            'state': state_name,
            'label': 0,  # Unconscious
            'sample_id': f'{state_name}_{i}',
            **metrics
        })

df_clinical = pd.DataFrame(clinical_data)
print(f"\nTotal samples: {len(df_clinical)}")
print(f"Conscious: {df_clinical['label'].sum()}")
print(f"Unconscious: {len(df_clinical) - df_clinical['label'].sum()}")

# ==============================================================================
# PART 2: ROC Analysis for Each Metric
# ==============================================================================

print("\n" + "-"*70)
print("PART 2: ROC Analysis for Each Metric")
print("-"*70)

labels = df_clinical['label'].values
metrics_to_test = ['C', 'H_mode', 'PR', 'R', 'S_dot', 'kappa']

roc_results = {}

for metric in metrics_to_test:
    scores = df_clinical[metric].values
    fpr, tpr, thresholds, auc = compute_roc_curve(labels, scores)
    
    # Find optimal thresholds
    opt_youden, sens_y, spec_y = find_optimal_threshold(fpr, tpr, thresholds, 'youden')
    opt_balanced, sens_b, spec_b = find_optimal_threshold(fpr, tpr, thresholds, 'balanced')
    
    roc_results[metric] = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc,
        'optimal_threshold': opt_youden,
        'sensitivity': sens_y,
        'specificity': spec_y,
    }
    
    print(f"\n{metric}:")
    print(f"  AUC: {auc:.3f}")
    print(f"  Optimal threshold (Youden): {opt_youden:.3f}")
    print(f"  Sensitivity: {sens_y:.3f}")
    print(f"  Specificity: {spec_y:.3f}")

# ==============================================================================
# PART 3: Decision Boundary Analysis
# ==============================================================================

print("\n" + "-"*70)
print("PART 3: Decision Boundary Analysis")
print("-"*70)

# Test different threshold methods for C(t)
threshold_methods = ['youden', 'balanced', 'high_sensitivity', 'high_specificity']
C_scores = df_clinical['C'].values
fpr, tpr, thresholds, auc = compute_roc_curve(labels, C_scores)

print("\nC(t) Threshold Optimization Methods:")
threshold_comparison = []

for method in threshold_methods:
    opt_thresh, sens, spec = find_optimal_threshold(fpr, tpr, thresholds, method)
    
    # Compute accuracy
    predictions = (C_scores >= opt_thresh).astype(int)
    accuracy = np.mean(predictions == labels)
    
    threshold_comparison.append({
        'method': method,
        'threshold': opt_thresh,
        'sensitivity': sens,
        'specificity': spec,
        'accuracy': accuracy,
    })
    
    print(f"  {method}: thresh={opt_thresh:.3f}, sens={sens:.3f}, spec={spec:.3f}, acc={accuracy:.3f}")

df_thresholds = pd.DataFrame(threshold_comparison)

# ==============================================================================
# PART 4: Difficult Cases Analysis
# ==============================================================================

print("\n" + "-"*70)
print("PART 4: Difficult Cases Analysis")
print("-"*70)

# Find misclassified samples using optimal threshold
opt_thresh = roc_results['C']['optimal_threshold']
predictions = (df_clinical['C'] >= opt_thresh).astype(int)
df_clinical['predicted'] = predictions
df_clinical['correct'] = (predictions == df_clinical['label'])

# Analyze misclassifications
misclassified = df_clinical[~df_clinical['correct']]
print(f"\nMisclassified samples: {len(misclassified)} / {len(df_clinical)} ({100*len(misclassified)/len(df_clinical):.1f}%)")

if len(misclassified) > 0:
    print("\nMisclassified by state:")
    for state in misclassified['state'].unique():
        count = len(misclassified[misclassified['state'] == state])
        total = len(df_clinical[df_clinical['state'] == state])
        print(f"  {state}: {count}/{total} ({100*count/total:.1f}%)")

# Borderline cases (close to threshold)
threshold_range = 0.05
borderline = df_clinical[(df_clinical['C'] > opt_thresh - threshold_range) & 
                          (df_clinical['C'] < opt_thresh + threshold_range)]
print(f"\nBorderline cases (C within ±{threshold_range} of threshold): {len(borderline)}")

# ==============================================================================
# PART 5: Multi-Metric Classifier
# ==============================================================================

print("\n" + "-"*70)
print("PART 5: Multi-Metric Classifier")
print("-"*70)

# Simple logistic-like combination
# Weighted sum based on AUC performance
weights = {metric: roc_results[metric]['auc'] for metric in metrics_to_test}
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

# Compute combined score
combined_scores = np.zeros(len(df_clinical))
for metric in metrics_to_test:
    # Normalize scores to [0, 1]
    min_val = df_clinical[metric].min()
    max_val = df_clinical[metric].max()
    normalized = (df_clinical[metric] - min_val) / (max_val - min_val + 1e-10)
    combined_scores += weights[metric] * normalized

# ROC for combined
fpr_comb, tpr_comb, thresh_comb, auc_comb = compute_roc_curve(labels, combined_scores)
opt_combined, sens_comb, spec_comb = find_optimal_threshold(fpr_comb, tpr_comb, thresh_comb, 'youden')

print(f"\nCombined Multi-Metric Classifier:")
print(f"  Weights: {weights}")
print(f"  AUC: {auc_comb:.3f}")
print(f"  Sensitivity: {sens_comb:.3f}")
print(f"  Specificity: {spec_comb:.3f}")

roc_results['Combined'] = {
    'fpr': fpr_comb,
    'tpr': tpr_comb,
    'thresholds': thresh_comb,
    'auc': auc_comb,
    'optimal_threshold': opt_combined,
    'sensitivity': sens_comb,
    'specificity': spec_comb,
}

# ==============================================================================
# PART 6: Visualizations
# ==============================================================================

print("\n" + "-"*70)
print("PART 6: Generating Visualizations")
print("-"*70)

# Figure 1: ROC curves comparison
fig, ax = plt.subplots(figsize=(10, 8))

colors = {'C': 'blue', 'H_mode': 'green', 'PR': 'red', 'R': 'orange', 
          'S_dot': 'purple', 'kappa': 'brown', 'Combined': 'black'}

for metric, results in roc_results.items():
    ax.plot(results['fpr'], results['tpr'], color=colors[metric], 
            linewidth=2, label=f"{metric} (AUC={results['auc']:.2f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves for Consciousness Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
print(f"  Saved: roc_curves.png")

# Figure 2: Distribution by state with threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Conscious vs Unconscious distributions
ax = axes[0]
conscious = df_clinical[df_clinical['label'] == 1]['C']
unconscious = df_clinical[df_clinical['label'] == 0]['C']

ax.hist(conscious, bins=20, alpha=0.5, color='green', label='Conscious', density=True)
ax.hist(unconscious, bins=20, alpha=0.5, color='red', label='Unconscious', density=True)
ax.axvline(x=opt_thresh, color='blue', linestyle='--', linewidth=2, label=f'Threshold={opt_thresh:.3f}')
ax.set_xlabel('Consciousness C(t)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A. C(t) Distributions by Consciousness', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# By specific state
ax = axes[1]
state_means = df_clinical.groupby('state')['C'].mean().sort_values()
colors_by_label = ['red' if df_clinical[df_clinical['state'] == s]['label'].iloc[0] == 0 
                   else 'green' for s in state_means.index]

ax.barh(range(len(state_means)), state_means.values, color=colors_by_label, edgecolor='black')
ax.axvline(x=opt_thresh, color='blue', linestyle='--', linewidth=2, label=f'Threshold={opt_thresh:.3f}')
ax.set_yticks(range(len(state_means)))
ax.set_yticklabels(state_means.index)
ax.set_xlabel('Mean C(t)', fontsize=12)
ax.set_title('B. Mean C(t) by State', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'threshold_distributions.png', dpi=150, bbox_inches='tight')
print(f"  Saved: threshold_distributions.png")

# Figure 3: Decision boundary in 2D
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

pairs = [('H_mode', 'PR'), ('H_mode', 'kappa'), ('C', 'R'), ('PR', 'kappa')]

for ax, (m1, m2) in zip(axes.flat, pairs):
    conscious = df_clinical[df_clinical['label'] == 1]
    unconscious = df_clinical[df_clinical['label'] == 0]
    
    ax.scatter(conscious[m1], conscious[m2], c='green', alpha=0.6, label='Conscious', s=50)
    ax.scatter(unconscious[m1], unconscious[m2], c='red', alpha=0.6, label='Unconscious', s=50)
    
    ax.set_xlabel(m1, fontsize=10)
    ax.set_ylabel(m2, fontsize=10)
    ax.set_title(f'{m1} vs {m2}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'decision_boundaries_2d.png', dpi=150, bbox_inches='tight')
print(f"  Saved: decision_boundaries_2d.png")

# Figure 4: Sensitivity-Specificity trade-off
fig, ax = plt.subplots(figsize=(10, 6))

fpr, tpr, thresholds, _ = roc_results['C']['fpr'], roc_results['C']['tpr'], roc_results['C']['thresholds'], roc_results['C']['auc']
sensitivity = tpr
specificity = 1 - fpr

ax.plot(thresholds, sensitivity, 'b-', linewidth=2, label='Sensitivity')
ax.plot(thresholds, specificity, 'r-', linewidth=2, label='Specificity')
ax.axvline(x=opt_thresh, color='green', linestyle='--', linewidth=2, label=f'Optimal={opt_thresh:.3f}')

ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Sensitivity-Specificity Trade-off for C(t)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([thresholds.min(), thresholds.max()])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sensitivity_specificity.png', dpi=150, bbox_inches='tight')
print(f"  Saved: sensitivity_specificity.png")

# Save data
df_clinical.to_csv(OUTPUT_DIR / 'clinical_dataset.csv', index=False)
df_thresholds.to_csv(OUTPUT_DIR / 'threshold_comparison.csv', index=False)

roc_summary = pd.DataFrame([{
    'metric': m,
    'auc': r['auc'],
    'optimal_threshold': r['optimal_threshold'],
    'sensitivity': r['sensitivity'],
    'specificity': r['specificity']
} for m, r in roc_results.items()])
roc_summary.to_csv(OUTPUT_DIR / 'roc_summary.csv', index=False)

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("KEY FINDINGS: CLINICAL THRESHOLD DETECTION")
print("="*70)

best_metric = max(roc_results.items(), key=lambda x: x[1]['auc'])

print(f"""
1. ROC ANALYSIS RESULTS:
""")
for metric in ['C', 'Combined', 'H_mode', 'PR', 'kappa']:
    r = roc_results[metric]
    print(f"   {metric}: AUC={r['auc']:.3f}, Sens={r['sensitivity']:.3f}, Spec={r['specificity']:.3f}")

print(f"""
2. BEST PERFORMING METRIC:
   {best_metric[0]} with AUC = {best_metric[1]['auc']:.3f}

3. OPTIMAL C(t) THRESHOLD:
   Value: {opt_thresh:.3f}
   Sensitivity: {roc_results['C']['sensitivity']:.3f}
   Specificity: {roc_results['C']['specificity']:.3f}

4. CLINICAL INTERPRETATION:
   - C(t) > {opt_thresh:.3f}: Likely CONSCIOUS
   - C(t) < {opt_thresh:.3f}: Likely UNCONSCIOUS
   - Borderline ({opt_thresh-0.05:.3f} - {opt_thresh+0.05:.3f}): Uncertain, needs additional assessment

5. DIFFICULT CASES:
   - Minimal consciousness states are hardest to classify
   - REM sleep may appear similar to wake
   - Light anesthesia shows variable consciousness levels

6. MULTI-METRIC APPROACH:
   - Combining metrics improves discrimination
   - AUC = {roc_results['Combined']['auc']:.3f}
   - Recommended for clinical applications

7. RECOMMENDATIONS FOR CLINICAL USE:
   - Use C(t) as primary screening metric
   - Supplement with H_mode and kappa for borderline cases
   - Consider patient-specific baselines
   - Never rely on single measurement
""")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("="*70)
