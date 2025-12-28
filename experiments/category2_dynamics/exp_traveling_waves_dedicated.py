"""
Dedicated Traveling Waves Experiment
Properly integrates wave analysis with rotational dynamics

Changes from v2 Experiment 3:
- Fixed API compatibility with traveling_waves.py
- Correct lattice generation and dimension handling
- Proper function signatures for wave analysis
- Extended parameters for better data capture
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append('../../experiments')
from utils import graph_generators as gg
from utils import state_generators as sg
from utils.rotational_dynamics import analyze_rotational_dynamics
from utils.traveling_waves import (
    comprehensive_wave_analysis,
    detect_traveling_wave_correlation,
    compute_wave_speed
)

# Configuration
SEED = 42
np.random.seed(SEED)

# Extended parameter ranges for better data capture
NETWORK_SIZES = [100, 300, 500]  # Test multiple scales
LATTICE_CONNECTIVITIES = [4, 8]  # 4-connected vs 8-connected
N_TRIALS_PER_CONFIG = 50  # More trials for statistics
TRAJECTORY_LENGTH = 300  # Longer trajectories

OUTPUT_DIR = Path('results/exp_traveling_waves_dedicated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DEDICATED TRAVELING WAVES EXPERIMENT")
print("="*80)
print(f"Extended parameter space:")
print(f"  Network sizes: {NETWORK_SIZES}")
print(f"  Lattice types: {LATTICE_CONNECTIVITIES}-connected")
print(f"  Trials per config: {N_TRIALS_PER_CONFIG}")
print(f"  Trajectory length: {TRAJECTORY_LENGTH}")
print()

# Storage for all results
all_results = []

# Loop over configurations
for N_NODES in NETWORK_SIZES:
    for connectivity in LATTICE_CONNECTIVITIES:
        
        config_name = f"N{N_NODES}_conn{connectivity}"
        print(f"\n{'='*80}")
        print(f"Configuration: {config_name}")
        print(f"{'='*80}")
        
        # Create lattice
        lattice_side = int(np.sqrt(N_NODES))
        if lattice_side ** 2 != N_NODES:
            actual_n_nodes = lattice_side ** 2
            print(f"  Adjusted: {N_NODES} → {actual_n_nodes} nodes ({lattice_side}×{lattice_side})")
        else:
            actual_n_nodes = N_NODES
            
        # Generate lattice with specified connectivity
        if connectivity == 4:
            G_lattice = gg.generate_lattice(lattice_side, periodic=True)
        else:  # 8-connected
            G_lattice = gg.generate_lattice(lattice_side, periodic=True)
            # Add diagonal connections
            for i in range(lattice_side):
                for j in range(lattice_side):
                    node = i * lattice_side + j
                    # Add diagonal neighbors
                    diag_neighbors = [
                        ((i+1) % lattice_side) * lattice_side + ((j+1) % lattice_side),
                        ((i+1) % lattice_side) * lattice_side + ((j-1) % lattice_side),
                        ((i-1) % lattice_side) * lattice_side + ((j+1) % lattice_side),
                        ((i-1) % lattice_side) * lattice_side + ((j-1) % lattice_side)
                    ]
                    for neighbor in diag_neighbors:
                        if not G_lattice.has_edge(node, neighbor):
                            G_lattice.add_edge(node, neighbor)
        
        # Compute eigenmodes
        laplacian, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G_lattice)
        actual_nodes = eigenvectors.shape[0]
        N_MODES = min(80, actual_nodes // 2)
        
        print(f"  Actual nodes: {actual_nodes}")
        print(f"  Modes used: {N_MODES}")
        
        # Create 2D positions for wave analysis
        xx, yy = np.meshgrid(np.arange(lattice_side), np.arange(lattice_side))
        positions = np.column_stack([xx.ravel()[:actual_nodes], yy.ravel()[:actual_nodes]])
        
        # Run trials
        print(f"  Running {N_TRIALS_PER_CONFIG} trials...")
        for trial in tqdm(range(N_TRIALS_PER_CONFIG), desc=f"  {config_name}"):
            np.random.seed(SEED + trial * 1000 + N_NODES * 100 + connectivity * 10)
            
            # Generate wave-like initial condition
            # Multiple types for diversity
            wave_type = trial % 4
            
            if wave_type == 0:  # Gaussian bump (circular wave)
                center = positions.mean(axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                spread = np.std(distances) / 3
                initial_activity = np.exp(-distances**2 / spread**2)
                
            elif wave_type == 1:  # Plane wave (traveling in one direction)
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                projection = positions @ direction
                wavelength = (positions.max(axis=0) - positions.min(axis=0)).mean() / 4
                initial_activity = np.sin(2 * np.pi * projection / wavelength)
                
            elif wave_type == 2:  # Spiral wave
                center = positions.mean(axis=0)
                rel_pos = positions - center
                angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
                radii = np.linalg.norm(rel_pos, axis=1)
                initial_activity = np.sin(angles + radii * 0.5)
                
            else:  # Random excitation patch
                excite_center_idx = np.random.randint(actual_nodes)
                distances = np.linalg.norm(positions - positions[excite_center_idx], axis=1)
                spread = np.std(distances) / 5
                initial_activity = np.exp(-distances**2 / spread**2)
            
            # Normalize
            initial_activity = (initial_activity - initial_activity.mean()) / initial_activity.std()
            
            # Project to mode space
            initial_modes = eigenvectors[:, :N_MODES].T @ initial_activity
            
            # Simulate wave propagation
            trajectory_modes = np.zeros((TRAJECTORY_LENGTH, N_MODES))
            trajectory_modes[0] = initial_modes
            
            # Diffusion with slight nonlinearity
            for t in range(1, TRAJECTORY_LENGTH):
                # Decay by eigenvalues (wave equation approximation)
                decay = np.exp(-eigenvalues[:N_MODES] * 0.05)
                
                # Add weak nonlinearity (cubic term)
                nonlinear_coupling = 0.01 * trajectory_modes[t-1]**3
                
                # Small noise
                noise = np.random.randn(N_MODES) * 0.02
                
                trajectory_modes[t] = trajectory_modes[t-1] * decay - nonlinear_coupling + noise
            
            # Project back to spatial domain
            activity_spatial = trajectory_modes @ eigenvectors[:, :N_MODES].T
            
            # Analyze rotational dynamics in mode space
            rotation_result = analyze_rotational_dynamics(
                trajectory_modes,
                n_jpcs=min(10, N_MODES // 2)
            )
            
            mean_rotation_angle = rotation_result['total_rotation_degrees']
            mean_rotation_vel = rotation_result['mean_angular_velocity']
            
            # Analyze traveling waves - PROPER API CALL
            # Method 1: Comprehensive analysis (returns dict)
            wave_result = comprehensive_wave_analysis(
                activity_spatial,
                positions
            )
            
            # Method 2: Simple detection (if comprehensive fails)
            if not wave_result['has_traveling_wave']:
                # Try simpler detection
                has_wave_simple = False
                wave_speed_simple = 0.0
                
                for t in range(10, TRAJECTORY_LENGTH - 10):
                    wave_speed = compute_wave_speed(
                        activity_spatial[t-5:t+5],
                        positions
                    )
                    if wave_speed > 0:
                        has_wave_simple = True
                        wave_speed_simple = wave_speed
                        break
            else:
                has_wave_simple = wave_result['has_traveling_wave']
                wave_speed_simple = wave_result['wave_speed']
            
            # Compute wave-rotation correspondence
            # If waves detected, check correlation with rotation velocity
            if has_wave_simple and wave_speed_simple > 0:
                # Normalized correspondence score
                correspondence_score = np.abs(wave_speed_simple - mean_rotation_vel) / (wave_speed_simple + mean_rotation_vel + 1e-10)
            else:
                correspondence_score = 0.0
            
            # Store results
            all_results.append({
                'network_size': N_NODES,
                'actual_nodes': actual_nodes,
                'connectivity': connectivity,
                'trial': trial,
                'wave_type': wave_type,
                'has_wave_comprehensive': wave_result['has_traveling_wave'],
                'wave_speed_comprehensive': wave_result.get('wave_speed', 0.0),
                'has_wave_simple': has_wave_simple,
                'wave_speed_simple': wave_speed_simple,
                'mean_rotation_angle': mean_rotation_angle,
                'mean_rotation_velocity': mean_rotation_vel,
                'rotation_quality': rotation_result['rotation_quality'],
                'circularity': rotation_result['circularity'],
                'wave_rotation_correspondence': correspondence_score
            })

# Convert to DataFrame and save
df_results = pd.DataFrame(all_results)
df_results.to_csv(OUTPUT_DIR / 'wave_analysis_results.csv', index=False)

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

# Overall wave detection rates
print(f"\nOverall wave detection:")
print(f"  Comprehensive method: {df_results['has_wave_comprehensive'].sum()}/{len(df_results)} ({df_results['has_wave_comprehensive'].mean()*100:.1f}%)")
print(f"  Simple method: {df_results['has_wave_simple'].sum()}/{len(df_results)} ({df_results['has_wave_simple'].mean()*100:.1f}%)")

# By network size
print(f"\nBy network size:")
for size in NETWORK_SIZES:
    subset = df_results[df_results['network_size'] == size]
    print(f"  N={size}: {subset['has_wave_simple'].mean()*100:.1f}% waves detected ({subset['has_wave_simple'].sum()}/{len(subset)} trials)")

# By connectivity
print(f"\nBy connectivity:")
for conn in LATTICE_CONNECTIVITIES:
    subset = df_results[df_results['connectivity'] == conn]
    print(f"  {conn}-connected: {subset['has_wave_simple'].mean()*100:.1f}% waves detected ({subset['has_wave_simple'].sum()}/{len(subset)} trials)")

# By wave type
print(f"\nBy initial condition:")
wave_types = ['Gaussian', 'Plane wave', 'Spiral', 'Random patch']
for wt in range(4):
    subset = df_results[df_results['wave_type'] == wt]
    print(f"  {wave_types[wt]}: {subset['has_wave_simple'].mean()*100:.1f}% waves ({subset['has_wave_simple'].sum()}/{len(subset)} trials)")

# Wave-rotation correspondence
waves_detected = df_results[df_results['has_wave_simple']]
if len(waves_detected) > 0:
    print(f"\nWave-rotation correspondence (for trials with waves):")
    print(f"  Mean correspondence: {waves_detected['wave_rotation_correspondence'].mean():.3f}")
    print(f"  Mean wave speed: {waves_detected['wave_speed_simple'].mean():.3f}")
    print(f"  Mean rotation velocity: {waves_detected['mean_rotation_velocity'].mean():.3f}")

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'wave_analysis_results.csv'}")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Dedicated Traveling Waves Experiment - Extended Parameters', fontsize=16, fontweight='bold')

# 1. Wave detection by network size
ax = axes[0, 0]
detection_by_size = df_results.groupby('network_size')['has_wave_simple'].mean()
ax.bar(range(len(detection_by_size)), np.array(detection_by_size.values) * 100)
ax.set_xticks(range(len(detection_by_size)))
ax.set_xticklabels([f'N={s}' for s in detection_by_size.index])
ax.set_ylabel('Wave Detection Rate (%)', fontsize=12)
ax.set_title('Wave Detection by Network Size')
ax.grid(True, alpha=0.3, axis='y')

# 2. Wave detection by connectivity
ax = axes[0, 1]
detection_by_conn = df_results.groupby('connectivity')['has_wave_simple'].mean()
ax.bar(range(len(detection_by_conn)), np.array(detection_by_conn.values) * 100, color='orange')
ax.set_xticks(range(len(detection_by_conn)))
ax.set_xticklabels([f'{c}-conn' for c in detection_by_conn.index])
ax.set_ylabel('Wave Detection Rate (%)', fontsize=12)
ax.set_title('Wave Detection by Connectivity')
ax.grid(True, alpha=0.3, axis='y')

# 3. Wave detection by initial condition
ax = axes[0, 2]
detection_by_type = df_results.groupby('wave_type')['has_wave_simple'].mean()
ax.bar(range(len(detection_by_type)), np.array(detection_by_type.values) * 100, color='green')
ax.set_xticks(range(len(detection_by_type)))
ax.set_xticklabels(wave_types, rotation=45, ha='right')
ax.set_ylabel('Wave Detection Rate (%)', fontsize=12)
ax.set_title('Wave Detection by Initial Condition')
ax.grid(True, alpha=0.3, axis='y')

# 4. Wave speed distribution
ax = axes[1, 0]
waves_only = df_results[df_results['has_wave_simple']]
if len(waves_only) > 0:
    ax.hist(waves_only['wave_speed_simple'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Wave Speed', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Wave Speed Distribution (n={len(waves_only)})')
else:
    ax.text(0.5, 0.5, 'No waves detected', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Wave Speed Distribution')

# 5. Rotation velocity distribution  
ax = axes[1, 1]
ax.hist(df_results['mean_rotation_velocity'], bins=20, edgecolor='black', alpha=0.7, color='purple')
ax.set_xlabel('Mean Rotation Velocity', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Rotation Velocity Distribution')

# 6. Wave-rotation correspondence
ax = axes[1, 2]
if len(waves_only) > 0:
    ax.scatter(waves_only['wave_speed_simple'], waves_only['mean_rotation_velocity'], alpha=0.6)
    ax.set_xlabel('Wave Speed', fontsize=12)
    ax.set_ylabel('Rotation Velocity', fontsize=12)
    ax.set_title('Wave Speed vs Rotation Velocity')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(waves_only['wave_speed_simple'], waves_only['mean_rotation_velocity'])
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
else:
    ax.text(0.5, 0.5, 'No waves detected', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Wave Speed vs Rotation Velocity')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'wave_analysis_visualization.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {OUTPUT_DIR / 'wave_analysis_visualization.png'}")

print(f"\n{'='*80}")
print("EXPERIMENT COMPLETE")
print(f"{'='*80}")
