#!/usr/bin/env python3
"""
Category 4, Experiment 3: Psychedelic States

Model altered consciousness states:
- Create psychedelic power distributions (enhanced high modes)
- Model different psychedelic intensities
- Compare to baseline wake state
- Visualize "ego dissolution" as reduced low-mode dominance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz

# Configuration
SEED = 42
N_NODES = 100
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp3_psychedelic_states'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 4, Experiment 3: Psychedelic States")
print("="*60)

# Generate network
print("\nGenerating network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

n_modes = 30
eigenvalues = eigenvalues[:n_modes]

# Generate states with varying psychedelic intensity
print("Generating psychedelic states...")
intensities = np.linspace(0, 1, 11)  # 0 = baseline, 1 = peak
states = {}

for intensity in intensities:
    power = sg.generate_psychedelic_state(n_modes=n_modes, intensity=intensity, seed=SEED)
    states[intensity] = power

# Also generate reference states
wake_power = sg.generate_wake_state(n_modes=n_modes, seed=SEED)
states['wake'] = wake_power

# Compute metrics for all states
print("Computing metrics...")
results = []

for key, power in tqdm(states.items(), desc="States"):
    if key == 'wake':
        intensity = -0.1  # For plotting
        state_name = 'Wake (baseline)'
    else:
        intensity = key
        state_name = f'Psychedelic ({intensity:.1f})'
    
    metrics = met.compute_all_metrics(power, eigenvalues)
    
    # Add low-mode dominance metric
    low_mode_power = power[:3].sum()  # First 3 modes
    high_mode_power = power[10:].sum()  # Modes 10+
    
    result = {
        'intensity': intensity,
        'state_name': state_name,
        'low_mode_power': low_mode_power,
        'high_mode_power': high_mode_power,
        'mode_balance': high_mode_power / (low_mode_power + 1e-12),
        **metrics
    }
    results.append(result)

df = pd.DataFrame(results)
df = df.sort_values('intensity')

# Save results
csv_path = OUTPUT_DIR / 'psychedelic_states_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Mode power distributions at different intensities
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

# Plot baseline wake + 11 psychedelic intensities
plot_intensities = [-0.1] + list(intensities)
for idx, intensity in enumerate(plot_intensities):
    ax = axes[idx]
    
    if intensity < 0:
        power = wake_power
        title = 'Baseline (Wake)'
        color = 'forestgreen'
    else:
        power = states[intensity]
        title = f'Intensity {intensity:.1f}'
        color = 'purple'
    
    k = np.arange(n_modes)
    ax.bar(k, power, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Power')
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 0.4)
    
    # Add metrics
    row = df[df['intensity'] == intensity].iloc[0]
    ax.text(0.95, 0.95, f"H={row['H_mode']:.2f}\nC={row['C']:.2f}",
           transform=ax.transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Mode Power Distributions Across Psychedelic Intensities', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'power_distributions.png')
print(f"  Saved: power_distributions.png")

# 2. Metrics vs intensity
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics_to_plot = ['H_mode', 'PR', 'R', 'C', 'low_mode_power', 'high_mode_power']
metric_labels = [
    'Mode Entropy $H_{mode}$',
    'Participation Ratio $PR$',
    'Phase Coherence $R$',
    'Consciousness Functional $C(t)$',
    'Low-Mode Power (0-2)',
    'High-Mode Power (10+)'
]

for ax, metric, label in zip(axes.flat, metrics_to_plot, metric_labels):
    # Plot psychedelic data
    psychedelic_data = df[df['intensity'] >= 0]
    ax.plot(psychedelic_data['intensity'], psychedelic_data[metric], 
           'o-', linewidth=2, markersize=8, color='purple', label='Psychedelic')
    
    # Add wake baseline as horizontal line
    wake_value = df[df['intensity'] < 0][metric].values[0]
    ax.axhline(wake_value, color='forestgreen', linestyle='--', 
              linewidth=2, label='Wake baseline')
    
    ax.set_xlabel('Psychedelic Intensity', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.05, 1.05)

plt.suptitle('Consciousness Metrics vs Psychedelic Intensity', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metrics_vs_intensity.png')
print(f"  Saved: metrics_vs_intensity.png")

# 3. "Ego dissolution" visualization (low-mode power reduction)
fig, ax = plt.subplots(figsize=(10, 6))

psychedelic_data = df[df['intensity'] >= 0]
wake_low_mode = df[df['intensity'] < 0]['low_mode_power'].values[0]

ax.plot(psychedelic_data['intensity'], psychedelic_data['low_mode_power'], 
       'o-', linewidth=3, markersize=10, color='purple', label='Psychedelic')
ax.axhline(wake_low_mode, color='forestgreen', linestyle='--', 
          linewidth=2, label='Wake baseline')

# Shade "ego dissolution" zone
ax.fill_between(psychedelic_data['intensity'], 
               psychedelic_data['low_mode_power'],
               wake_low_mode,
               where=psychedelic_data['low_mode_power'] < wake_low_mode,
               alpha=0.3, color='red', label='Ego dissolution zone')

ax.set_xlabel('Psychedelic Intensity', fontsize=12)
ax.set_ylabel('Low-Mode Power (modes 0-2)', fontsize=12)
ax.set_title('"Ego Dissolution": Reduced Low-Mode Dominance', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ego_dissolution.png')
print(f"  Saved: ego_dissolution.png")

# 4. Mode balance (high/low ratio)
fig, ax = plt.subplots(figsize=(10, 6))

psychedelic_data = df[df['intensity'] >= 0]
wake_balance = df[df['intensity'] < 0]['mode_balance'].values[0]

ax.plot(psychedelic_data['intensity'], psychedelic_data['mode_balance'], 
       'o-', linewidth=3, markersize=10, color='purple')
ax.axhline(wake_balance, color='forestgreen', linestyle='--', 
          linewidth=2, label='Wake baseline')

ax.set_xlabel('Psychedelic Intensity', fontsize=12)
ax.set_ylabel('Mode Balance (High/Low)', fontsize=12)
ax.set_title('Shift Toward High-Mode Activity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mode_balance.png')
print(f"  Saved: mode_balance.png")

# 5. Phase space trajectory (H_mode vs Low-mode power)
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.plasma(np.linspace(0, 1, len(psychedelic_data)))
for idx, (_, row) in enumerate(psychedelic_data.iterrows()):
    ax.scatter(row['H_mode'], row['low_mode_power'], 
              c=[colors[idx]], s=100, alpha=0.7,
              edgecolors='black', linewidths=1.5)
    if idx % 2 == 0:  # Label every other point
        ax.annotate(f"{row['intensity']:.1f}", 
                   (row['H_mode'], row['low_mode_power']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add wake baseline
wake_H = df[df['intensity'] < 0]['H_mode'].values[0]
wake_low = df[df['intensity'] < 0]['low_mode_power'].values[0]
ax.scatter([wake_H], [wake_low], c='green', s=200, marker='*',
          edgecolors='black', linewidths=2, label='Wake', zorder=10)

# Add arrow showing trajectory
for i in range(len(psychedelic_data) - 1):
    row1 = psychedelic_data.iloc[i]
    row2 = psychedelic_data.iloc[i + 1]
    ax.annotate('', xy=(row2['H_mode'], row2['low_mode_power']),
               xytext=(row1['H_mode'], row1['low_mode_power']),
               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

ax.set_xlabel('Mode Entropy $H_{mode}$', fontsize=12)
ax.set_ylabel('Low-Mode Power', fontsize=12)
ax.set_title('Psychedelic State Trajectory in Phase Space', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase_space_trajectory.png')
print(f"  Saved: phase_space_trajectory.png")

plt.close('all')

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
print(f"\nWake baseline:")
wake_row = df[df['intensity'] < 0].iloc[0]
print(f"  H_mode: {wake_row['H_mode']:.3f}")
print(f"  C(t):   {wake_row['C']:.3f}")
print(f"  Low-mode power: {wake_row['low_mode_power']:.3f}")

print(f"\nPeak psychedelic (intensity=1.0):")
peak_row = df[df['intensity'] == 1.0].iloc[0]
print(f"  H_mode: {peak_row['H_mode']:.3f}")
print(f"  C(t):   {peak_row['C']:.3f}")
print(f"  Low-mode power: {peak_row['low_mode_power']:.3f}")

print(f"\nChanges from wake to peak:")
print(f"  ΔH_mode: {peak_row['H_mode'] - wake_row['H_mode']:+.3f}")
print(f"  ΔC(t):   {peak_row['C'] - wake_row['C']:+.3f}")
print(f"  ΔLow-mode: {peak_row['low_mode_power'] - wake_row['low_mode_power']:+.3f}")

print("\n" + "="*60)
print("Experiment completed successfully!")
print(f"All results saved to: {OUTPUT_DIR}")
print("="*60)
