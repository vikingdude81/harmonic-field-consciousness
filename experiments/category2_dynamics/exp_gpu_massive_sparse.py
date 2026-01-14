"""
GPU-Accelerated Large-Scale Experiments - SPARSE EIGENSOLVER VERSION
Integration with SparseHarmonicBridge for >25K node networks

This version uses sparse eigensolvers instead of dense decomposition,
enabling experiments on networks that were previously too large:
- Previous limit: ~7,000 nodes (GPU memory constraint)
- New capability: 25K-100K+ nodes (sparse solver)

Key features:
- Sparse ARPACK eigensolver for CPU
- Optional CuPy GPU acceleration
- Memory-efficient sparse matrix storage
- Integration with existing experiment framework
"""

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import sys
from typing import Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neural_mass.sparse_harmonic_bridge import SparseHarmonicBridge, create_sparse_network

# GPU setup (for dynamics simulation, not eigendecomposition)
if torch.cuda.is_available():
    print("Available GPUs:")
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} ({props.total_memory / 1e9:.2f} GB)")

    device = torch.device('cuda:0')  # Use GPU 0 for dynamics
    print(f"\n[OK] Using GPU 0 for dynamics simulation")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")


def create_lattice_laplacian_sparse(side: int, periodic: bool = True) -> sp.csr_matrix:
    """Create sparse Laplacian for 2D lattice"""
    N = side * side

    row, col, data = [], [], []

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

            # Diagonal
            row.append(idx)
            col.append(idx)
            data.append(float(len(neighbors)))

            # Off-diagonal
            for n in neighbors:
                row.append(idx)
                col.append(n)
                data.append(-1.0)

    L = sp.csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float32)
    return L


def compute_eigenmodes_sparse(laplacian: sp.csr_matrix, n_modes: int,
                              use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenmodes using sparse solver.

    Args:
        laplacian: Sparse Laplacian matrix
        n_modes: Number of modes to compute
        use_gpu: Use GPU acceleration (requires CuPy)

    Returns:
        eigenvalues: (n_modes,) smallest eigenvalues
        eigenvectors: (N, n_modes) corresponding eigenvectors
    """
    N = laplacian.shape[0]

    # Use SparseHarmonicBridge for decomposition
    device_str = 'cuda' if use_gpu else 'cpu'
    bridge = SparseHarmonicBridge(
        adjacency_matrix=laplacian,
        n_modes=n_modes,
        device=device_str,
        solver='arpack',
        verbose=False
    )

    eigenvalues, eigenvectors = bridge.compute_harmonics()

    return eigenvalues, eigenvectors


def simulate_trajectory_batch_gpu(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor,
                                   initial_modes_batch: torch.Tensor, timesteps: int,
                                   diffusion_rate: float = 0.05,
                                   noise_std: float = 0.02) -> torch.Tensor:
    """
    Simulate multiple trajectories in parallel on GPU.

    This uses dense GPU operations for the actual dynamics simulation,
    but the eigenvectors come from sparse decomposition.
    """
    batch_size, n_modes = initial_modes_batch.shape

    # Preallocate trajectory storage
    trajectory_modes = torch.zeros(batch_size, timesteps, n_modes, device=device)
    trajectory_modes[:, 0, :] = initial_modes_batch

    # Precompute decay factors
    decay = torch.exp(-diffusion_rate * eigenvalues.to(device))

    # Simulate dynamics
    for t in range(1, timesteps):
        # Deterministic decay
        trajectory_modes[:, t, :] = trajectory_modes[:, t-1, :] * decay

        # Add noise
        if noise_std > 0:
            noise = torch.randn(batch_size, n_modes, device=device) * noise_std
            trajectory_modes[:, t, :] += noise

    return trajectory_modes


def detect_wave_patterns(modes_trajectory: torch.Tensor, threshold: float = 0.3) -> bool:
    """Detect if trajectory shows wave-like patterns"""
    # Simple heuristic: check if high-frequency modes dominate
    n_modes = modes_trajectory.shape[-1]
    high_freq_start = n_modes // 2

    high_freq_power = torch.mean(torch.abs(modes_trajectory[:, :, high_freq_start:]))
    low_freq_power = torch.mean(torch.abs(modes_trajectory[:, :, :high_freq_start]))

    return (high_freq_power / (low_freq_power + 1e-6)) > threshold


def detect_rotation(modes_trajectory: torch.Tensor, threshold: float = 0.1) -> bool:
    """Detect rotational patterns in mode space"""
    # Check for circular/spiral patterns in mode amplitudes
    diffs = torch.diff(modes_trajectory, dim=1)
    rotation_measure = torch.std(diffs) / (torch.mean(torch.abs(modes_trajectory)) + 1e-6)

    return rotation_measure > threshold


def run_sparse_experiment(side: int, n_modes: int, n_trials: int,
                          batch_size: int = 100, use_gpu_eigen: bool = False):
    """
    Run experiments with sparse eigensolver.

    Args:
        side: Lattice side length (N = side^2 nodes)
        n_modes: Number of eigenmodes
        n_trials: Number of trials
        batch_size: Batch size for parallel simulation
        use_gpu_eigen: Use GPU for eigendecomposition (requires CuPy)
    """
    N = side * side

    print("="*80)
    print(f"SPARSE EIGENSOLVER EXPERIMENT")
    print("="*80)
    print(f"Network size: {side}×{side} = {N:,} nodes")
    print(f"Modes: {n_modes}")
    print(f"Trials: {n_trials}")
    print(f"Batch size: {batch_size}")
    print(f"GPU eigendecomposition: {use_gpu_eigen}")
    print("="*80)

    # Create sparse Laplacian
    print("\n[1] Creating sparse Laplacian...")
    start = time.time()
    L = create_lattice_laplacian_sparse(side, periodic=True)
    elapsed = time.time() - start

    nnz = L.nnz
    sparsity = 100 * (1 - nnz / (N * N))
    print(f"    Created in {elapsed:.2f}s")
    print(f"    Non-zeros: {nnz:,} ({sparsity:.2f}% sparse)")
    print(f"    Memory: {L.data.nbytes / 1024 / 1024:.1f} MB")

    # Compute eigenmodes with sparse solver
    print(f"\n[2] Computing {n_modes} eigenmodes (sparse solver)...")
    start = time.time()
    eigenvalues, eigenvectors = compute_eigenmodes_sparse(L, n_modes, use_gpu=use_gpu_eigen)
    eigen_time = time.time() - start

    print(f"    Computed in {eigen_time:.2f}s ({N / eigen_time:.1f} nodes/sec)")
    print(f"    Eigenvalue range: [{eigenvalues[0]:.3f}, {eigenvalues[-1]:.3f}]")

    # Convert to GPU tensors for dynamics
    eigenvectors_gpu = torch.from_numpy(eigenvectors).float().to(device)
    eigenvalues_gpu = torch.from_numpy(eigenvalues).float().to(device)

    # Run batched simulations
    print(f"\n[3] Running {n_trials} trials (batch size {batch_size})...")
    results = []

    n_batches = (n_trials + batch_size - 1) // batch_size

    start = time.time()
    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        current_batch_size = min(batch_size, n_trials - batch_idx * batch_size)

        # Random initial conditions (different for each trial)
        initial_modes = torch.randn(current_batch_size, n_modes, device=device) * 0.1

        # Simulate
        trajectories = simulate_trajectory_batch_gpu(
            eigenvectors_gpu, eigenvalues_gpu,
            initial_modes, timesteps=100,
            diffusion_rate=0.05, noise_std=0.02
        )

        # Analyze each trial
        for trial_in_batch in range(current_batch_size):
            traj = trajectories[trial_in_batch:trial_in_batch+1]

            has_wave = detect_wave_patterns(traj)
            has_rotation = detect_rotation(traj)

            # Compute metrics
            final_energy = torch.sum(traj[:, -1, :] ** 2).item()
            max_amplitude = torch.max(torch.abs(traj)).item()

            results.append({
                'trial': batch_idx * batch_size + trial_in_batch,
                'has_wave': has_wave,
                'has_rotation': has_rotation,
                'final_energy': final_energy,
                'max_amplitude': max_amplitude
            })

    sim_time = time.time() - start

    # Summarize results
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total time: {eigen_time + sim_time:.2f}s")
    print(f"  - Eigendecomposition: {eigen_time:.2f}s")
    print(f"  - Simulations: {sim_time:.2f}s")
    print(f"  - Throughput: {n_trials / sim_time:.2f} trials/sec")

    print(f"\nPattern Detection:")
    print(f"  - Wave patterns: {df['has_wave'].sum()} / {n_trials} ({100*df['has_wave'].mean():.1f}%)")
    print(f"  - Rotations: {df['has_rotation'].sum()} / {n_trials} ({100*df['has_rotation'].mean():.1f}%)")

    print(f"\nMetrics:")
    print(f"  - Mean final energy: {df['final_energy'].mean():.4f} ± {df['final_energy'].std():.4f}")
    print(f"  - Mean max amplitude: {df['max_amplitude'].mean():.4f} ± {df['max_amplitude'].std():.4f}")

    print("="*80)

    return df


def main():
    """Run experiments on multiple scales"""

    # Test configurations (can now go much larger!)
    configs = [
        {'name': 'Large', 'side': 70, 'nodes': 4900, 'modes': 50, 'trials': 500},
        {'name': 'Very Large', 'side': 100, 'nodes': 10000, 'modes': 50, 'trials': 500},
        {'name': 'Huge', 'side': 141, 'nodes': 19881, 'modes': 50, 'trials': 500},
        {'name': 'Massive', 'side': 200, 'nodes': 40000, 'modes': 50, 'trials': 500},
    ]

    print("\n" + "="*80)
    print("SPARSE EIGENSOLVER - LARGE-SCALE EXPERIMENTS")
    print("="*80)
    print("\nConfigurations:")
    for cfg in configs:
        print(f"  {cfg['name']:15s}: {cfg['side']:3d}×{cfg['side']:<3d} = {cfg['nodes']:6,} nodes")

    # Run experiments
    all_results = {}

    for cfg in configs:
        print(f"\n\n{'='*80}")
        print(f"RUNNING: {cfg['name']} ({cfg['nodes']:,} nodes)")
        print(f"{'='*80}\n")

        try:
            df = run_sparse_experiment(
                side=cfg['side'],
                n_modes=cfg['modes'],
                n_trials=cfg['trials'],
                batch_size=100,
                use_gpu_eigen=False  # Set to True if CuPy installed
            )

            all_results[cfg['name']] = df

            # Save results
            output_dir = Path(__file__).parent / 'results'
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"sparse_experiment_{cfg['name'].lower().replace(' ', '_')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n[OK] Results saved to: {output_file}")

        except Exception as e:
            print(f"\n[ERROR] Failed on {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary across scales
    print("\n\n" + "="*80)
    print("CROSS-SCALE SUMMARY")
    print("="*80)

    for name, df in all_results.items():
        print(f"\n{name}:")
        print(f"  Wave patterns: {100*df['has_wave'].mean():.1f}%")
        print(f"  Rotations: {100*df['has_rotation'].mean():.1f}%")
        print(f"  Final energy: {df['final_energy'].mean():.4f}")

    print("\n" + "="*80)
    print("[OK] ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
