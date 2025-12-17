#!/usr/bin/env python3
"""
Category 2, Experiment 1: State Transitions

Animate transitions between conscious states:
- Implement smooth interpolation between states
- Create Wake → NREM → Dream → Wake cycle
- Track all metrics during transition
- Generate time series plots and phase space trajectories
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz

# Configuration
SEED = 42
N_NODES = 100
N_STEPS = 200
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp1_state_transitions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 2, Experiment 1: State Transitions")
print("="*60)

# Generate network
print("\nGenerating network...")
G = gg.generate_small_world(N_NODES, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

# Use first 20 modes (to match state generators)
n_modes = 20
eigenvalues = eigenvalues[:n_modes]

# Generate state transition sequence
print("Generating state transition sequence...")
state_sequence = ['wake', 'nrem', 'dream', 'wake']
power_sequence, labels = sg.generate_state_transition_sequence(
    state_sequence,
    n_steps=N_STEPS,
    seed=SEED
)

# Compute metrics for each time step
print("Computing metrics for each time step...")
metrics_history = []
for t in tqdm(range(N_STEPS), desc="Time steps"):
    power = power_sequence[t]
    power = power / power.sum()
    
    # Generate random phases matching power dimensions
    phases = np.random.uniform(0, 2 * np.pi, len(power))
    
    # Compute previous power with small change
    if t > 0:
        power_prev = power_sequence[t-1]
        power_prev = power_prev / power_prev.sum()
    else:
        power_prev = power + np.random.normal(0, 0.01, len(power))
        power_prev = np.clip(power_prev, 0, None)
        power_prev = power_prev / power_prev.sum()
    
    # Truncate eigenvalues to match power dimensions
    eig_trunc = eigenvalues[:len(power)]
    
    metrics = met.compute_all_metrics(
        power,
        eig_trunc,
        phases=phases,
        power_previous=power_prev,
        dt=1.0
    )
    metrics['time'] = t
    metrics['state'] = labels[t]
    metrics_history.append(metrics)

# Convert to DataFrame
df = pd.DataFrame(metrics_history)

# Save results
csv_path = OUTPUT_DIR / 'state_transitions_timeseries.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

# 1. Time series of all metrics
fig, axes = plt.subplots(6, 1, figsize=(12, 14))

metrics_to_plot = ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']
metric_labels = [
    'Mode Entropy $H_{mode}$',
    'Participation Ratio $PR$',
    'Phase Coherence $R$',
    'Entropy Production $\\dot{S}$',
    'Criticality Index $\\kappa$',
    'Consciousness Functional $C(t)$'
]

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    ax.plot(df['time'], df[metric], linewidth=2, color='steelblue')
    ax.set_ylabel(label, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_STEPS)
    ax.set_ylim(0, 1)
    
    # Add state labels
    for i, state in enumerate(state_sequence):
        start = i * (N_STEPS // len(state_sequence))
        end = (i + 1) * (N_STEPS // len(state_sequence))
        ax.axvspan(start, end, alpha=0.1, 
                  color=['green', 'blue', 'purple', 'green'][i],
                  label=state.upper() if ax == axes[0] else None)

axes[-1].set_xlabel('Time Step', fontsize=11)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('State Transitions: Wake → NREM → Dream → Wake', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'timeseries_all_metrics.png')
print(f"  Saved: timeseries_all_metrics.png")

# 2. Phase space trajectory (3D)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Use H_mode, PR, R as axes
x = df['H_mode'].values
y = df['PR'].values
z = df['R'].values

# Color by time
colors = np.arange(len(x))
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.6)
ax.plot(x, y, z, alpha=0.2, linewidth=0.5, color='gray')

# Mark start and end
ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=200, marker='o', 
          edgecolors='black', linewidths=2, label='Start (Wake)')
ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=200, marker='*', 
          edgecolors='black', linewidths=2, label='End (Wake)')

ax.set_xlabel('Mode Entropy $H_{mode}$', fontsize=11)
ax.set_ylabel('Participation Ratio $PR$', fontsize=11)
ax.set_zlabel('Phase Coherence $R$', fontsize=11)
ax.set_title('Phase Space Trajectory', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
plt.colorbar(scatter, ax=ax, label='Time', shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase_space_3d.png')
print(f"  Saved: phase_space_3d.png")

# 3. 2D phase space projections
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

projections = [
    ('H_mode', 'PR', 'Mode Entropy vs Participation Ratio'),
    ('H_mode', 'R', 'Mode Entropy vs Phase Coherence'),
    ('PR', 'R', 'Participation Ratio vs Phase Coherence'),
]

for ax, (x_metric, y_metric, title) in zip(axes, projections):
    x = df[x_metric].values
    y = df[y_metric].values
    colors = np.arange(len(x))
    
    scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=30, alpha=0.6)
    ax.plot(x, y, alpha=0.2, linewidth=0.5, color='gray')
    
    # Mark start
    ax.scatter([x[0]], [y[0]], c='green', s=150, marker='o', 
              edgecolors='black', linewidths=2, zorder=10)
    
    ax.set_xlabel(x_metric, fontsize=11)
    ax.set_ylabel(y_metric, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase_space_2d_projections.png')
print(f"  Saved: phase_space_2d_projections.png")

# 4. Mode power evolution over time
fig, ax = plt.subplots(figsize=(12, 6))

# Subsample time steps for clarity
time_indices = np.linspace(0, N_STEPS-1, 20, dtype=int)
mode_indices = np.arange(n_modes)

# Create heatmap
power_matrix = power_sequence[time_indices, :n_modes]
im = ax.imshow(power_matrix.T, aspect='auto', cmap='YlOrRd', 
              extent=[0, N_STEPS, n_modes, 0])

ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Mode Index', fontsize=11)
ax.set_title('Mode Power Evolution Over Time', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Power')

# Add state boundaries
for i in range(1, len(state_sequence)):
    boundary = i * (N_STEPS // len(state_sequence))
    ax.axvline(boundary, color='white', linestyle='--', linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mode_power_evolution.png')
print(f"  Saved: mode_power_evolution.png")

# 5. Create simple animation of C(t) evolution
print("\nCreating animation...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

def animate(frame):
    ax1.clear()
    ax2.clear()
    
    # Top: Mode power distribution
    power = power_sequence[frame, :n_modes]
    ax1.bar(mode_indices, power, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Power')
    ax1.set_title(f'Time: {frame}/{N_STEPS} | State: {labels[frame]}')
    ax1.set_ylim(0, power_sequence[:, :n_modes].max() * 1.1)
    
    # Bottom: C(t) time series
    ax2.plot(df['time'][:frame+1], df['C'][:frame+1], 'b-', linewidth=2)
    ax2.scatter([frame], [df['C'].iloc[frame]], c='red', s=100, zorder=10)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('C(t)')
    ax2.set_xlim(0, N_STEPS)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

# Create animation (subsample frames)
frames = range(0, N_STEPS, 5)
anim = FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)

# Save as GIF
anim_path = OUTPUT_DIR / 'state_transition_animation.gif'
writer = PillowWriter(fps=10)
anim.save(anim_path, writer=writer)
print(f"  Saved: state_transition_animation.gif")

plt.close('all')

print("\n" + "="*60)
print("Experiment completed successfully!")
print(f"All results saved to: {OUTPUT_DIR}")
print("="*60)
