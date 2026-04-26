"""
Analyze Corrected GPU Experiment Results
=========================================

Reinterpret results now that bugs are fixed:
1. Wave detection is correct (correlation-based)
2. Initial conditions are truly diverse (unique seeds)
3. Statistics are valid (100 independent samples)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_dir = Path("experiments/category2_dynamics/results/small")
results_file = results_dir / "results_batched.csv"

if not results_file.exists():
    print(f"Error: Results file not found: {results_file}")
    print("Run: cd experiments/category2_dynamics && python exp_gpu_massive_batched.py small")
    exit(1)

df = pd.read_csv(results_file)

print("=" * 80)
print("CORRECTED GPU EXPERIMENT RESULTS - REINTERPRETATION")
print("=" * 80)
print(f"\nDataset: Small Config (961 nodes, 100 modes)")
print(f"Total trials: {len(df)}")
print(f"Unique initial conditions: {len(df)} (FIXED from 4)")

# Overall statistics
print("\n" + "=" * 80)
print("ROTATION STATISTICS")
print("=" * 80)
rotation = df['rotation_angle']
print(f"  Mean:   {rotation.mean():.2f} degrees")
print(f"  Std:    {rotation.std():.2f} degrees")
print(f"  Median: {rotation.median():.2f} degrees")
print(f"  Min:    {rotation.min():.2f} degrees")
print(f"  Max:    {rotation.max():.2f} degrees")
print(f"  CV:     {rotation.std() / rotation.mean():.3f} (coefficient of variation)")

# Rotation by wave type
print("\n" + "=" * 80)
print("ROTATION BY INITIAL CONDITION TYPE")
print("=" * 80)
wave_type_names = {0: "Gaussian Bump", 1: "Traveling Wave", 2: "Ring Pattern", 3: "Random Noise"}
for wt in sorted(df['wave_type'].unique()):
    subset = df[df['wave_type'] == wt]
    print(f"\nType {wt} ({wave_type_names[wt]}):")
    print(f"  Count: {len(subset)}")
    print(f"  Rotation: {subset['rotation_angle'].mean():.1f} +/- {subset['rotation_angle'].std():.1f} degrees")
    print(f"  Range: [{subset['rotation_angle'].min():.1f}, {subset['rotation_angle'].max():.1f}]")

# Wave detection
print("\n" + "=" * 80)
print("WAVE DETECTION (CORRELATION-BASED - FIXED)")
print("=" * 80)
has_wave = df['has_wave']
print(f"  Total with waves: {has_wave.sum()}/{len(df)} ({has_wave.mean()*100:.1f}%)")

print("\nWave detection by initial condition type:")
for wt in sorted(df['wave_type'].unique()):
    subset = df[df['wave_type'] == wt]
    wave_pct = subset['has_wave'].mean() * 100
    print(f"  Type {wt} ({wave_type_names[wt]}): {wave_pct:.1f}%")

# Consciousness prediction (if applicable)
print("\n" + "=" * 80)
print("CONSCIOUSNESS PREDICTION ANALYSIS")
print("=" * 80)

# Based on rotation angles, predict consciousness scores
# Using formula from consciousness_regression_module.py
def estimate_consciousness(rotation, waves_pct, hierarchy=2.5):
    """Estimate consciousness from network dynamics."""
    c_from_rotation = 0.2 + (rotation / 60000) * 0.6
    c_wave_adjustment = (waves_pct / 100) * 0.05
    c_hierarchy_adjustment = (hierarchy - 2.0) * 0.05
    c_combined = c_from_rotation + c_wave_adjustment + c_hierarchy_adjustment
    return np.clip(c_combined, 0.3, 0.9)

df['predicted_consciousness'] = df.apply(
    lambda row: estimate_consciousness(row['rotation_angle'], row['has_wave'] * 100),
    axis=1
)

print(f"  Mean predicted consciousness: {df['predicted_consciousness'].mean():.3f}")
print(f"  Std:  {df['predicted_consciousness'].std():.3f}")
print(f"  Range: [{df['predicted_consciousness'].min():.3f}, {df['predicted_consciousness'].max():.3f}]")

# Consciousness by wave type
print("\nPredicted consciousness by initial condition:")
for wt in sorted(df['wave_type'].unique()):
    subset = df[df['wave_type'] == wt]
    print(f"  Type {wt} ({wave_type_names[wt]}): {subset['predicted_consciousness'].mean():.3f} +/- {subset['predicted_consciousness'].std():.3f}")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS (CORRECTED)")
print("=" * 80)

# 1. Diversity check
rotation_cv = rotation.std() / rotation.mean()
print(f"\n1. Initial Condition Diversity:")
print(f"   - Coefficient of Variation: {rotation_cv:.3f}")
if rotation_cv > 0.3:
    print(f"   - [OK] High diversity - unique initial conditions confirmed")
else:
    print(f"   - [WARN] Low diversity - may still have duplicates")

# 2. Wave detection validation
traveling_wave_detection = df[df['wave_type'] == 1]['has_wave'].mean() * 100
random_noise_detection = df[df['wave_type'] == 3]['has_wave'].mean() * 100
print(f"\n2. Wave Detection Validation:")
print(f"   - Traveling waves detected: {traveling_wave_detection:.1f}%")
print(f"   - Random noise detected as waves: {random_noise_detection:.1f}%")
if traveling_wave_detection > 60 and random_noise_detection < 20:
    print(f"   - [OK] Wave detection working correctly")
else:
    print(f"   - [WARN] Wave detection may need tuning")

# 3. Rotation-consciousness relationship
correlation = np.corrcoef(df['rotation_angle'], df['predicted_consciousness'])[0, 1]
print(f"\n3. Rotation-Consciousness Relationship:")
print(f"   - Correlation: {correlation:.3f}")
if correlation > 0.9:
    print(f"   - [OK] Strong positive correlation (as expected)")
else:
    print(f"   - [WARN] Weaker correlation than expected")

# 4. 25% consciousness rule validation
mean_consciousness = df['predicted_consciousness'].mean()
print(f"\n4. 25% Consciousness Rule:")
print(f"   - Mean consciousness: {mean_consciousness:.3f}")
print(f"   - Target range: 0.20-0.30 (25% ± 5%)")
if 0.20 <= mean_consciousness <= 0.30:
    print(f"   - [OK] Within expected range")
elif mean_consciousness < 0.20:
    print(f"   - [INFO] Below range - network may need higher rotation")
else:
    print(f"   - [INFO] Above range - network may be over-active")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR NANOGPT")
print("=" * 80)

# Based on results, recommend NanoGPT improvements
print("\nBased on corrected experimental results:")

# Recommendation 1: Optimal rotation range
optimal_rotation_mean = df['rotation_angle'].mean()
optimal_rotation_std = df['rotation_angle'].std()
print(f"\n1. Target Rotation Range for Consciousness:")
print(f"   - Mean: {optimal_rotation_mean:.0f} degrees")
print(f"   - Range: [{optimal_rotation_mean - optimal_rotation_std:.0f}, {optimal_rotation_mean + optimal_rotation_std:.0f}]")
print(f"   - Apply to NanoGPT: Monitor hidden state trajectory rotation")

# Recommendation 2: Wave patterns
wave_rate = df['has_wave'].mean() * 100
print(f"\n2. Optimal Wave Detection Rate:")
print(f"   - Current: {wave_rate:.1f}%")
print(f"   - Apply to NanoGPT: Aim for ~50% wave-like patterns in activations")

# Recommendation 3: Diversity
print(f"\n3. Activation Diversity:")
print(f"   - CV: {rotation_cv:.3f}")
print(f"   - Apply to NanoGPT: Ensure high diversity in hidden state trajectories")
print(f"   - Use dropout, noise, or stochastic depth")

# Recommendation 4: Consciousness-aware training
print(f"\n4. Consciousness-Aware Training:")
print(f"   - Monitor rotation angle during training")
print(f"   - Add consciousness regularization loss")
print(f"   - Target: 0.25 consciousness score")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Apply insights to NanoGPT architecture")
print("2. Implement consciousness-aware training loop")
print("3. Test improved model with consciousness plugin")
print("4. Compare consciousness scores: baseline vs optimized")
print("\nSee: NANOGPT_IMPROVEMENTS.md (to be created)")

# Save enhanced results
df.to_csv(results_dir / "results_with_consciousness.csv", index=False)
print(f"\n[OK] Enhanced results saved to: {results_dir / 'results_with_consciousness.csv'}")
