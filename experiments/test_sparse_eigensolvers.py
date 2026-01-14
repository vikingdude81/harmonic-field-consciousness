#!/usr/bin/env python3
"""
Sparse Eigensolver Development and Testing
For use on RTX A2000 (GPU 1) while LLM training runs on RTX 5090 (GPU 0)

Usage:
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 python experiments/test_sparse_eigensolvers.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import time
import argparse
from typing import Tuple, Optional

def generate_test_network(n_nodes: int, density: float = 0.01, seed: int = 42) -> sp.csr_matrix:
    """Generate a random sparse symmetric network."""
    np.random.seed(seed)
    A = sp.random(n_nodes, n_nodes, density=density, format='csr', dtype=np.float32)
    A = (A + A.T) / 2  # Make symmetric
    return A

def test_sparse_eigsh(
    A: sp.csr_matrix,
    k: int = 10,
    which: str = 'LM',
    maxiter: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Test scipy sparse eigensolver (ARPACK).

    Args:
        A: Sparse symmetric matrix
        k: Number of eigenvalues to compute
        which: Which eigenvalues to find ('LM' = largest magnitude)
        maxiter: Maximum iterations (None = auto)

    Returns:
        eigenvalues, eigenvectors, computation_time
    """
    start_time = time.time()

    try:
        eigenvalues, eigenvectors = eigsh(A, k=k, which=which, maxiter=maxiter)
        elapsed = time.time() - start_time
        return eigenvalues, eigenvectors, elapsed
    except ArpackNoConvergence as e:
        print(f"Warning: ARPACK did not converge. Returning partial results.")
        eigenvalues = e.eigenvalues
        eigenvectors = e.eigenvectors
        elapsed = time.time() - start_time
        return eigenvalues, eigenvectors, elapsed

def benchmark_sparse_solver(
    network_sizes: list = [1000, 5000, 10000, 15000, 20000, 25000],
    k: int = 50,
    density: float = 0.01
):
    """Benchmark sparse eigensolver on various network sizes."""
    print("="*80)
    print("SPARSE EIGENSOLVER BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Number of eigenvalues (k): {k}")
    print(f"  - Network density: {density}")
    print(f"  - Method: ARPACK (via scipy.sparse.linalg.eigsh)")
    print("\n" + "-"*80)

    results = []

    for n in network_sizes:
        print(f"\n[{n:,} nodes]")

        # Generate network
        print(f"  Generating sparse network... ", end="", flush=True)
        gen_start = time.time()
        A = generate_test_network(n, density=density)
        gen_time = time.time() - gen_start
        nnz = A.nnz
        sparsity = 100 * (1 - nnz / (n * n))
        print(f"done ({gen_time:.2f}s)")
        print(f"  - Non-zeros: {nnz:,} ({sparsity:.2f}% sparse)")
        print(f"  - Memory: ~{A.data.nbytes / 1024 / 1024:.1f} MB")

        # Compute eigenvalues
        print(f"  Computing top {k} eigenvalues... ", end="", flush=True)
        eigenvalues, eigenvectors, comp_time = test_sparse_eigsh(A, k=k, which='LM')
        print(f"done ({comp_time:.2f}s)")

        # Results
        print(f"  [OK] Top eigenvalue: {eigenvalues[-1]:.6f}")
        print(f"  [OK] Bottom eigenvalue: {eigenvalues[0]:.6f}")
        print(f"  [OK] Speed: {n / comp_time:.1f} nodes/sec")

        results.append({
            'n_nodes': n,
            'nnz': nnz,
            'gen_time': gen_time,
            'comp_time': comp_time,
            'top_eigenval': eigenvalues[-1],
            'speed': n / comp_time
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Nodes':>10} {'Non-zeros':>12} {'Gen Time':>10} {'Comp Time':>11} {'Speed':>15}")
    print(f"{'':>10} {'':>12} {'(sec)':>10} {'(sec)':>11} {'(nodes/sec)':>15}")
    print("-"*80)

    for r in results:
        print(f"{r['n_nodes']:>10,} {r['nnz']:>12,} {r['gen_time']:>10.2f} "
              f"{r['comp_time']:>11.2f} {r['speed']:>15.1f}")

    print("\n" + "="*80)
    print(f"[OK] All {len(results)} benchmarks completed successfully!")
    print("="*80)

def test_gpu_eigensolvers():
    """Test GPU-accelerated eigensolvers if available."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh

        print("\n" + "="*80)
        print("GPU EIGENSOLVER TEST (CuPy)")
        print("="*80)

        # Test on medium network
        n = 10000
        k = 50
        density = 0.01

        print(f"\nGenerating {n:,} node network on GPU...")
        A_cpu = generate_test_network(n, density=density)
        A_gpu = cpsp.csr_matrix(A_cpu)

        print(f"Computing top {k} eigenvalues on GPU...")
        start_time = time.time()
        eigenvalues, eigenvectors = cp_eigsh(A_gpu, k=k, which='LM')
        gpu_time = time.time() - start_time

        eigenvalues = cp.asnumpy(eigenvalues)

        print(f"\n[OK] GPU computation completed in {gpu_time:.2f}s")
        print(f"  - Top eigenvalue: {eigenvalues[-1]:.6f}")
        print(f"  - Speed: {n / gpu_time:.1f} nodes/sec")

        # Compare to CPU
        print(f"\nComparing to CPU...")
        start_time = time.time()
        eigenvalues_cpu, _, cpu_time = test_sparse_eigsh(A_cpu, k=k)

        print(f"[OK] CPU computation completed in {cpu_time:.2f}s")
        print(f"  - Speedup: {cpu_time / gpu_time:.2f}x")
        print(f"  - Max eigenvalue difference: {np.abs(eigenvalues[-1] - eigenvalues_cpu[-1]):.6e}")

    except ImportError:
        print("\n[INFO] CuPy not available. Skipping GPU tests.")
        print("       Install with: pip install cupy-cuda12x")

def main():
    parser = argparse.ArgumentParser(description='Test sparse eigensolvers')
    parser.add_argument('--sizes', nargs='+', type=int,
                       default=[1000, 5000, 10000, 15000, 20000, 25000],
                       help='Network sizes to test')
    parser.add_argument('--k', type=int, default=50,
                       help='Number of eigenvalues to compute')
    parser.add_argument('--density', type=float, default=0.01,
                       help='Network density (0-1)')
    parser.add_argument('--gpu', action='store_true',
                       help='Also test GPU eigensolvers')

    args = parser.parse_args()

    # CPU benchmarks
    benchmark_sparse_solver(
        network_sizes=args.sizes,
        k=args.k,
        density=args.density
    )

    # GPU benchmarks (if requested)
    if args.gpu:
        test_gpu_eigensolvers()

    print("\n[OK] Testing complete! Ready for integration with HarmonicBridge.")

if __name__ == "__main__":
    main()
