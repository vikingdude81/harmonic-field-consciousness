"""
Statistical analysis of full 180-trial Experiment 2 dataset
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load full dataset
df = pd.read_csv('results/exp_rotational_recovery_v2/exp2_rotation_recovery_correlation.csv')

print('='*80)
print('EXPERIMENT 2: FULL 180-TRIAL STATISTICAL ANALYSIS')
print('='*80)

print(f'\nDataset shape: {df.shape}')
print(f'Trials: {len(df)}')

# Group by perturbation strength
grouped = df.groupby('perturbation_strength')

print('\n--- BY PERTURBATION STRENGTH ---')
for strength, group in grouped:
    print(f'\nPerturbation: {strength}')
    print(f'  Trials: {len(group)}')
    print(f'  Rotation angle: {group["rotation_angle"].mean():.2f} ± {group["rotation_angle"].std():.2f}°')
    print(f'  Raw recovery: {group["recovery_pct"].mean():.2f} ± {group["recovery_pct"].std():.2f}%')
    print(f'  Normalized recovery: {group["normalized_recovery"].mean():.3f} ± {group["normalized_recovery"].std():.3f}')
    print(f'  Rotation quality: {group["rotation_quality"].mean():.3f} ± {group["rotation_quality"].std():.3f}')

# Overall correlations
print('\n--- CORRELATION MATRIX ---')
corr_cols = ['perturbation_strength', 'rotation_angle', 'recovery_pct', 'normalized_recovery', 'rotation_quality']
corr_matrix = df[corr_cols].corr()
print(corr_matrix.round(3))

# Test for significance
print('\n--- SIGNIFICANCE TESTS ---')
for col in ['rotation_angle', 'recovery_pct', 'normalized_recovery', 'rotation_quality']:
    r, p = stats.pearsonr(df['perturbation_strength'], df[col])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'Perturbation vs {col}: r={r:.3f}, p={p:.4f} {sig}')

# Count non-zero rotations
nonzero_angles = (df['rotation_angle'] > 0) & (df['rotation_angle'] < 180)
print(f'\nNon-zero, non-180° rotations: {nonzero_angles.sum()}/{len(df)} ({nonzero_angles.sum()/len(df)*100:.1f}%)')

# Binary vs continuous angles
binary = ((df['rotation_angle'] == 0) | (df['rotation_angle'] == 180)).sum()
continuous = len(df) - binary
print(f'Binary angles (0° or 180°): {binary}/{len(df)} ({binary/len(df)*100:.1f}%)')
print(f'Continuous angles: {continuous}/{len(df)} ({continuous/len(df)*100:.1f}%)')

# Distribution analysis
print('\n--- DISTRIBUTION STATISTICS ---')
print(f'\nRotation angles:')
print(f'  Mean: {df["rotation_angle"].mean():.2f}°')
print(f'  Median: {df["rotation_angle"].median():.2f}°')
print(f'  Std: {df["rotation_angle"].std():.2f}°')
print(f'  Range: [{df["rotation_angle"].min():.2f}, {df["rotation_angle"].max():.2f}]°')

print(f'\nNormalized recovery:')
print(f'  Mean: {df["normalized_recovery"].mean():.3f}')
print(f'  Median: {df["normalized_recovery"].median():.3f}')
print(f'  Std: {df["normalized_recovery"].std():.3f}')
print(f'  Range: [{df["normalized_recovery"].min():.3f}, {df["normalized_recovery"].max():.3f}]')

# Create enhanced visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Experiment 2: Full 180-Trial Analysis', fontsize=16, fontweight='bold')

# 1. Perturbation vs Rotation Angle
ax = axes[0, 0]
strengths = sorted(df['perturbation_strength'].unique())
means = [df[df['perturbation_strength']==s]['rotation_angle'].mean() for s in strengths]
stds = [df[df['perturbation_strength']==s]['rotation_angle'].std() for s in strengths]
ax.errorbar(strengths, means, yerr=stds, marker='o', capsize=5, linewidth=2)
ax.set_xlabel('Perturbation Strength', fontsize=12)
ax.set_ylabel('Rotation Angle (degrees)', fontsize=12)
ax.set_title('Perturbation vs Rotation Angle')
ax.grid(True, alpha=0.3)

# 2. Perturbation vs Recovery
ax = axes[0, 1]
means_rec = [df[df['perturbation_strength']==s]['normalized_recovery'].mean() for s in strengths]
stds_rec = [df[df['perturbation_strength']==s]['normalized_recovery'].std() for s in strengths]
ax.errorbar(strengths, means_rec, yerr=stds_rec, marker='s', capsize=5, linewidth=2, color='green')
ax.set_xlabel('Perturbation Strength', fontsize=12)
ax.set_ylabel('Normalized Recovery', fontsize=12)
ax.set_title('Perturbation vs Recovery')
ax.grid(True, alpha=0.3)

# 3. Rotation vs Recovery scatter
ax = axes[0, 2]
scatter = ax.scatter(df['rotation_angle'], df['normalized_recovery'], 
                    c=df['perturbation_strength'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Rotation Angle (degrees)', fontsize=12)
ax.set_ylabel('Normalized Recovery', fontsize=12)
ax.set_title('Rotation vs Recovery (colored by perturbation)')
plt.colorbar(scatter, ax=ax, label='Perturbation')

# 4. Rotation angle distribution
ax = axes[1, 0]
ax.hist(df['rotation_angle'], bins=30, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='0°')
ax.axvline(180, color='blue', linestyle='--', linewidth=2, label='180°')
ax.set_xlabel('Rotation Angle (degrees)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Rotation Angles')
ax.legend()

# 5. Normalized recovery distribution
ax = axes[1, 1]
ax.hist(df['normalized_recovery'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Normalized Recovery', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Normalized Recovery')

# 6. Quality vs Perturbation
ax = axes[1, 2]
means_qual = [df[df['perturbation_strength']==s]['rotation_quality'].mean() for s in strengths]
stds_qual = [df[df['perturbation_strength']==s]['rotation_quality'].std() for s in strengths]
ax.errorbar(strengths, means_qual, yerr=stds_qual, marker='^', capsize=5, linewidth=2, color='purple')
ax.set_xlabel('Perturbation Strength', fontsize=12)
ax.set_ylabel('Rotation Quality', fontsize=12)
ax.set_title('Perturbation vs Rotation Quality')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/exp_rotational_recovery_v2/exp2_full_analysis.png', dpi=150, bbox_inches='tight')
print(f'\n✓ Enhanced visualization saved to: results/exp_rotational_recovery_v2/exp2_full_analysis.png')
