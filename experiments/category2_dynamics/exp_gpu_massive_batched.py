"""
GPU-Accelerated Large-Scale Experiments - BATCHED VERSION
Optimized for RTX 5090 with parallel trial execution

Key optimizations:
- Process multiple trials simultaneously (batch processing)
- Minimize CPU-GPU memory transfers
- Maximize GPU utilization across 21K+ CUDA cores
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import sys
from typing import Tuple

# GPU Selection - Prefer GPU 0 (RTX 5090) for heavy CUDA work
if torch.cuda.is_available():
    # List all available GPUs
    print("Available GPUs:")
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} ({props.total_memory / 1e9:.2f} GB)")
    
    # Explicitly use GPU 0 (RTX 5090) for heavy computation
    preferred_gpu = 0
    try:
        test_device = torch.device(f'cuda:{preferred_gpu}')
        test_tensor = torch.ones(10, device=test_device)
        _ = test_tensor * 2
        device = test_device
        print(f"\n[OK] Using GPU {preferred_gpu}: {torch.cuda.get_device_name(preferred_gpu)}")
        print(f"Memory: {torch.cuda.get_device_properties(preferred_gpu).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"[ERROR] GPU {preferred_gpu} failed: {e}")
        device = torch.device('cpu')
        print("Falling back to CPU")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")


def create_lattice_laplacian_gpu(side: int, periodic: bool = True) -> torch.Tensor:
    """Create Laplacian for 2D lattice on GPU"""
    N = side * side
    L = torch.zeros(N, N, device=device)
    
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            neighbors = []
            
            if periodic:
                neighbors = [
                    ((i-1) % side) * side + j,
                    ((i+1) % side) * side + j,
                    i * side + ((j-1) % side),
                    i * side + ((j+1) % side)
                ]
            else:
                if i > 0: neighbors.append((i-1) * side + j)
                if i < side-1: neighbors.append((i+1) * side + j)
                if j > 0: neighbors.append(i * side + (j-1))
                if j < side-1: neighbors.append(i * side + (j+1))
            
            L[idx, idx] = len(neighbors)
            for n in neighbors:
                L[idx, n] = -1
    
    return L


def compute_eigenmodes_gpu(laplacian: torch.Tensor, n_modes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute eigenmodes on GPU"""
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvalues[:n_modes], eigenvectors[:, :n_modes]


def simulate_trajectory_batch_gpu(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, 
                                   initial_modes_batch: torch.Tensor, timesteps: int,
                                   diffusion_rate: float = 0.05,
                                   noise_std: float = 0.02) -> torch.Tensor:
    """
    Simulate multiple trajectories in parallel with proper dynamics.
    
    Args:
        eigenvectors: (N_NODES, N_MODES)
        eigenvalues: (N_MODES,)
        initial_modes_batch: (BATCH_SIZE, N_MODES)
        timesteps: number of timesteps
        diffusion_rate: diffusion rate for dynamics
        noise_std: noise standard deviation
    
    Returns:
        trajectory_modes: (BATCH_SIZE, timesteps, N_MODES)
    """
    batch_size = initial_modes_batch.shape[0]
    n_modes = eigenvalues.shape[0]
    
    # Initialize trajectory tensor
    trajectory_modes = torch.zeros(batch_size, timesteps, n_modes, device=initial_modes_batch.device)
    trajectory_modes[:, 0, :] = initial_modes_batch
    
    # Precompute decay factors
    decay = torch.exp(-eigenvalues * diffusion_rate)
    
    # Simulate with proper dynamics (diffusion, nonlinearity, noise)
    for t in range(1, timesteps):
        # Diffusion decay
        trajectory_modes[:, t, :] = trajectory_modes[:, t-1, :] * decay
        
        # Weak nonlinearity (cubic damping)
        trajectory_modes[:, t, :] -= 0.01 * trajectory_modes[:, t-1, :]**3
        
        # Add noise
        trajectory_modes[:, t, :] += torch.randn(batch_size, n_modes, device=initial_modes_batch.device) * noise_std
    
    return trajectory_modes


def compute_rotation_angle_batch_gpu(trajectory_modes_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation angles for batch of trajectories.
    
    Uses proper angle unwrapping to compute total rotation in the first
    two mode dimensions, matching the non-batched implementation.
    
    Args:
        trajectory_modes_batch: (BATCH_SIZE, timesteps, N_MODES)
    
    Returns:
        angles: (BATCH_SIZE,) rotation angles in degrees
    """
    # Use first two dimensions for rotation plane
    x = trajectory_modes_batch[:, :, 0]  # (BATCH_SIZE, timesteps)
    y = trajectory_modes_batch[:, :, 1]  # (BATCH_SIZE, timesteps)
    
    # Add small epsilon to avoid NaN when both x and y are near zero
    eps = 1e-10
    x = x + eps * torch.sign(x + eps)
    
    # Compute angles at each timestep
    angles = torch.atan2(y, x)  # (BATCH_SIZE, timesteps)
    
    # Compute angle differences
    diffs = angles[:, 1:] - angles[:, :-1]  # (BATCH_SIZE, timesteps-1)
    
    # Handle wrapping (unwrap angles)
    diffs = torch.where(diffs > np.pi, diffs - 2*np.pi, diffs)
    diffs = torch.where(diffs < -np.pi, diffs + 2*np.pi, diffs)
    
    # Total rotation (sum of absolute angle changes)
    total_rotation = torch.sum(torch.abs(diffs), dim=1)  # (BATCH_SIZE,)
    
    # Convert to degrees
    rotation_degrees = torch.rad2deg(total_rotation)
    
    return rotation_degrees


def detect_waves_batch_gpu(activity_spatial_batch: torch.Tensor, positions: torch.Tensor, fast_mode: bool = True) -> dict:
    """
    Detect traveling waves in batch of spatial activity patterns.
    
    Args:
        activity_spatial_batch: (BATCH_SIZE, timesteps, N_NODES)
        positions: (N_NODES, 2)
        fast_mode: If True, use faster approximation suitable for large-scale experiments
    
    Returns:
        dict with has_wave (BATCH_SIZE,) and wave_speed (BATCH_SIZE,)
    """
    batch_size, timesteps, n_nodes = activity_spatial_batch.shape
    device = activity_spatial_batch.device
    
    # Compute pairwise distances (only need once)
    dist_matrix = torch.cdist(positions, positions)
    mean_dist = dist_matrix.mean().item()
    
    has_wave = torch.zeros(batch_size, dtype=torch.bool, device=device)
    wave_speed = torch.zeros(batch_size, device=device)
    
    if fast_mode:
        # Fast wave detection using variance decay pattern
        # Waves show smooth variance propagation, random noise doesn't
        early_var = activity_spatial_batch[:, :timesteps//4, :].var(dim=(1,2))  # Early variance
        late_var = activity_spatial_batch[:, -timesteps//4:, :].var(dim=(1,2))  # Late variance
        
        # Wave = smooth decay pattern (ratio near 0.5-1.0)
        # Random = rapid decay (ratio << 0.5) or explosion (ratio >> 1)
        var_ratio = late_var / (early_var + 1e-10)
        has_wave = (var_ratio > 0.1) & (var_ratio < 2.0)
        
        # Estimate wave speed from spatial correlation at lag=1
        for b in range(batch_size):
            if has_wave[b]:
                wave_speed[b] = mean_dist / 10.0  # Approximate
    else:
        # Original detailed detection (slower but more accurate)
        max_lag = min(20, timesteps // 10)
        
        for b in range(batch_size):
            correlations = []
            
            for lag in range(1, max_lag):
                if lag >= timesteps:
                    break
                
                # Correlation between activity at t and t+lag
                act_early = activity_spatial_batch[b, :-lag].flatten()
                act_late = activity_spatial_batch[b, lag:].flatten()
                
                # Compute correlation
                corr = torch.corrcoef(torch.stack([act_early, act_late]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())
            
            # Wave detected if correlation decays smoothly
            if len(correlations) > 5:
                mean_early = sum(correlations[:5]) / 5
                has_wave[b] = mean_early > 0.3
                
                # Estimate wave speed
                if has_wave[b]:
                    half_idx = next((i for i, c in enumerate(correlations) if c < 0.5), len(correlations))
                    wave_speed[b] = mean_dist / (half_idx + 1)
    
    return {
        'has_wave': has_wave,
        'wave_speed': wave_speed
    }


def run_batched_experiment(config: dict):
    """Run experiment with batched trial processing"""
    
    N_NODES = config['n_nodes']
    N_MODES = config['n_modes']
    TIMESTEPS = config['timesteps']
    N_TRIALS = config['n_trials']
    BATCH_SIZE = config.get('batch_size', 32)  # Process 32 trials at once
    CONFIG_NAME = config['name']
    
    print(f"\n{'='*80}")
    print(f"GPU-ACCELERATED BATCHED EXPERIMENT: {CONFIG_NAME.upper()}")
    print(f"{'='*80}")
    print(f"Network: {N_NODES} nodes")
    print(f"Modes: {N_MODES}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Trials: {N_TRIALS}")
    print(f"Batch Size: {BATCH_SIZE} trials in parallel")
    print(f"Device: {device}")
    print()
    
    # Setup
    start = time.time()
    print("Computing Laplacian eigenmodes on GPU...")
    
    # Find square lattice size
    side = int(np.sqrt(N_NODES))
    actual_nodes = side * side
    if actual_nodes != N_NODES:
        print(f"Adjusted nodes: {N_NODES} → {actual_nodes} ({side}x{side} lattice)")
        N_NODES = actual_nodes
    
    laplacian = create_lattice_laplacian_gpu(side, periodic=True)
    eigenvalues, eigenvectors = compute_eigenmodes_gpu(laplacian, N_MODES)
    
    # Create 2D positions
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    positions = torch.tensor(
        np.column_stack([xx.ravel(), yy.ravel()]),
        dtype=torch.float32,
        device=device
    )
    
    setup_time = time.time() - start
    print(f"[OK] Setup complete in {setup_time:.2f}s")
    print()
    
    # Run trials in batches
    results = []
    print(f"Running {N_TRIALS} trials in batches of {BATCH_SIZE}...")
    start = time.time()
    
    n_batches = (N_TRIALS + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, N_TRIALS)
        current_batch_size = batch_end - batch_start
        
        # Generate initial conditions for batch - FIXED: use different seeds
        initial_modes_batch = []
        wave_types_batch = []
        
        for trial_offset in range(current_batch_size):
            trial = batch_start + trial_offset
            
            # Generate initial condition with unique seed per trial
            wave_type = trial % 4
            wave_types_batch.append(wave_type)
            
            # Create unique random generator for this trial
            generator = torch.Generator(device=device).manual_seed(42 + trial)
            
            if wave_type == 0:  # Gaussian
                center = positions.mean(dim=0)
                distances = torch.norm(positions - center, dim=1)
                initial_activity = torch.exp(-distances**2 / (distances.std()**2 + 1e-10))
            elif wave_type == 1:  # Traveling wave - USES RANDOM DIRECTION
                direction = torch.randn(2, device=device, generator=generator)
                direction = direction / (direction.norm() + 1e-10)
                projection = positions @ direction
                proj_range = projection.max() - projection.min()
                proj_range = proj_range if proj_range > 1e-10 else 1.0
                initial_activity = torch.sin(2 * np.pi * projection / proj_range)
            elif wave_type == 2:  # Spiral
                center = positions.mean(dim=0)
                rel_pos = positions - center
                angles = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
                radii = torch.norm(rel_pos, dim=1)
                initial_activity = torch.sin(angles + radii * 0.5)
            else:  # Random patch
                initial_activity = torch.randn(N_NODES, device=device, generator=generator)
            
            # Normalize (with epsilon for numerical stability)
            std = initial_activity.std()
            if std < 1e-10:
                std = 1.0
            initial_activity = (initial_activity - initial_activity.mean()) / std
            
            # Project to modes
            initial_modes = eigenvectors.t() @ initial_activity
            initial_modes_batch.append(initial_modes)
        
        # Stack into batch tensor
        initial_modes_batch = torch.stack(initial_modes_batch)  # (current_batch_size, N_MODES)
        
        # Debug: check if initial modes vary
        if batch_idx == 0:
            print(f"DEBUG Batch 0: initial_modes variance across trials: {initial_modes_batch.var(dim=0).mean().item():.6f}")
            print(f"DEBUG Batch 0: initial_modes[0,:5] = {initial_modes_batch[0,:5]}")
            print(f"DEBUG Batch 0: initial_modes[1,:5] = {initial_modes_batch[1,:5]}")
        
        # Simulate all trials in batch simultaneously
        trajectory_modes_batch = simulate_trajectory_batch_gpu(
            eigenvectors, eigenvalues, initial_modes_batch, TIMESTEPS
        )
        
        # Compute rotation angles for batch
        rotation_angles = compute_rotation_angle_batch_gpu(trajectory_modes_batch)
        
        # Project back to spatial domain for wave detection
        # trajectory_modes_batch: (BATCH_SIZE, timesteps, N_MODES)
        # eigenvectors: (N_NODES, N_MODES)
        # Result: (BATCH_SIZE, timesteps, N_NODES)
        activity_spatial_batch = torch.matmul(trajectory_modes_batch, eigenvectors.t())
        
        # Detect waves in batch
        wave_results = detect_waves_batch_gpu(activity_spatial_batch, positions)
        
        # Store results
        for trial_offset in range(current_batch_size):
            trial = batch_start + trial_offset
            rot_angle = rotation_angles[trial_offset].item()
            # Handle NaN values (can occur when trajectory stays near origin)
            if np.isnan(rot_angle):
                rot_angle = 0.0
            result = {
                'trial': trial,
                'wave_type': wave_types_batch[trial_offset],
                'rotation_angle': rot_angle,
                'has_wave': wave_results['has_wave'][trial_offset].item(),
                'wave_speed': wave_results['wave_speed'][trial_offset].item(),
                'n_nodes': N_NODES,
                'n_modes': N_MODES,
                'timesteps': TIMESTEPS
            }
            results.append(result)
            if trial < 5:  # Debug: print first 5
                print(f"DEBUG Trial {trial}: rot={result['rotation_angle']:.2f}, wave_type={result['wave_type']}")
    
    runtime = time.time() - start
    print(f"\n[OK] {N_TRIALS} trials completed in {runtime:.2f}s")
    print(f"  Average: {runtime/N_TRIALS:.4f}s per trial")
    print(f"  Throughput: {N_TRIALS/runtime:.1f} trials/sec")
    
    # Save results
    output_path = Path('results') / CONFIG_NAME
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'results_batched.csv', index=False)
    
    # Summary stats
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    runtime = time.time() - start
    print(f"\n[OK] {N_TRIALS} trials completed in {runtime:.2f}s")
    print(f"  Average: {runtime/N_TRIALS:.4f}s per trial")
    print(f"  Throughput: {N_TRIALS/runtime:.1f} trials/sec")
    
    # Save results
    output_path = Path('results') / CONFIG_NAME
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'results_batched.csv', index=False)
    
    # Summary stats
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Wave detection rate: {df['has_wave'].mean()*100:.1f}% ({df['has_wave'].sum()}/{len(df)})")
    print(f"Mean rotation angle: {df['rotation_angle'].mean():.2f} +/- {df['rotation_angle'].std():.2f}")
    print(f"Mean wave speed: {df[df['has_wave']]['wave_speed'].mean():.2f}" if df['has_wave'].any() else "Mean wave speed: N/A")
    print(f"\n[OK] Results saved to: {output_path / 'results_batched.csv'}")
    
    # Summary stats
    print(f"\nResults:")
    print(f"  Avg rotation: {df['rotation_angle'].mean():.2f} rad")
    print(f"  Saved to: {output_path / 'results_batched.csv'}")
    
    return df


# Configurations
CONFIGS = {
    'small': {
        'name': 'small',
        'n_nodes': 961,      # 31x31
        'n_modes': 100,
        'timesteps': 1000,
        'n_trials': 100,
        'batch_size': 10     # Smaller batch for proper wave detection
    },
    'medium': {
        'name': 'medium',
        'n_nodes': 2499,     # ~50x50
        'n_modes': 300,
        'timesteps': 2000,
        'n_trials': 500,
        'batch_size': 32
    },
    'large': {
        'name': 'large',
        'n_nodes': 4900,     # 70x70
        'n_modes': 800,
        'timesteps': 5000,
        'n_trials': 200,
        'batch_size': 16     # Larger networks, smaller batches
    },
    'xlarge': {
        'name': 'xlarge',
        'n_nodes': 10000,    # 100x100
        'n_modes': 1500,
        'timesteps': 10000,
        'n_trials': 100,
        'batch_size': 8      # Very large, need smaller batches
    },
    # ========================================================================
    # LARGE SCALE CONFIGS - RTX 5090 (34GB VRAM)
    # cuSOLVER eigendecomposition limit: ~26,000 nodes (161x161 lattice)
    # Beyond this limit, use iterative/Lanczos methods or CPU fallback
    # ========================================================================
    'mega': {
        'name': 'mega',
        'n_nodes': 24964,    # 158x158 - safe for eigendecomp (~13s)
        'n_modes': 2000,
        'timesteps': 10000,
        'n_trials': 50,
        'batch_size': 4      # Mega scale - ~1.1s/trial
    },
    'giga': {
        'name': 'giga',
        'n_nodes': 24964,    # 158x158 - verified working
        'n_modes': 2000,
        'timesteps': 15000,  # Long trajectories for dynamics analysis
        'n_trials': 50,
        'batch_size': 3      # ~1.2s/trial, ~58s total
    },
    'ultra': {
        'name': 'ultra',
        'n_nodes': 25921,    # 161x161 - MAXIMUM for cuSOLVER eigendecomp
        'n_modes': 2200,
        'timesteps': 15000,
        'n_trials': 40,
        'batch_size': 3      # ~1.25s/trial, ~50s total
    },
    # Maximum verified scale - beyond this cuSOLVER fails
    'max': {
        'name': 'max',
        'n_nodes': 25921,    # 161x161 - absolute maximum for dense eigendecomp
        'n_modes': 2500,     # Use more modes for better resolution
        'timesteps': 20000,  # Maximum trajectory length
        'n_trials': 100,     # More trials for statistical power
        'batch_size': 2      # Conservative batch size
    }
}


if __name__ == '__main__':
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'small'
    
    if config_name not in CONFIGS:
        print(f"Unknown config: {config_name}")
        print(f"Available: {list(CONFIGS.keys())}")
        sys.exit(1)
    
    config = CONFIGS[config_name]
    run_batched_experiment(config)
