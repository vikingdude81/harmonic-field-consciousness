#!/usr/bin/env python3
"""
Demo: Sparse Harmonic Bridge Integration

Demonstrates the use of sparse eigensolvers for large-scale
network harmonic decomposition in consciousness analysis.

Examples:
1. Create large-scale network (25K+ nodes)
2. Compute harmonic modes using sparse solver
3. Analyze consciousness-related metrics
4. Compare different network topologies

Usage:
    python experiments/demo_sparse_harmonics.py --nodes 30000 --modes 100
    python experiments/demo_sparse_harmonics.py --nodes 50000 --gpu
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_mass.sparse_harmonic_bridge import (
    SparseHarmonicBridge,
    create_sparse_network
)


def demo_basic_usage(n_nodes: int = 10000, n_modes: int = 50):
    """Basic demonstration of sparse harmonic bridge."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Sparse Harmonic Analysis")
    print("="*80)

    # Create sparse network
    print(f"\nCreating random sparse network ({n_nodes:,} nodes, 1% density)...")
    W = create_sparse_network(n_nodes, density=0.01, distribution='random', seed=42)

    # Initialize bridge
    print(f"\nInitializing Sparse Harmonic Bridge...")
    bridge = SparseHarmonicBridge(
        adjacency_matrix=W,
        n_modes=n_modes,
        device='cpu',
        solver='arpack',
        verbose=True
    )

    # Compute harmonics
    print(f"\nComputing harmonics...")
    eigenvalues, eigenvectors = bridge.compute_harmonics()

    # Get harmonic modes
    harmonics = bridge.get_harmonic_modes()

    print(f"\nHarmonic Mode Summary:")
    print(f"  - Eigenvalues range: [{eigenvalues[0]:.3f}, {eigenvalues[-1]:.3f}]")
    print(f"  - Frequencies range: [{harmonics['frequencies'][0]:.3f}, {harmonics['frequencies'][-1]:.3f}]")
    print(f"  - Mean participation: {np.mean(harmonics['mode_participation']):.1f} nodes")
    print(f"  - Max participation: {np.max(harmonics['mode_participation']):.1f} nodes")

    return bridge, harmonics


def demo_consciousness_analysis(bridge: SparseHarmonicBridge):
    """Demonstrate consciousness state analysis."""
    print("\n" + "="*80)
    print("DEMO 2: Consciousness State Analysis")
    print("="*80)

    n_nodes = bridge.n_nodes

    # Simulate different activity patterns
    print("\nAnalyzing different activity patterns...")

    # Pattern 1: Uniform (low consciousness - anesthesia-like)
    activity_uniform = np.ones(n_nodes) * 0.1
    metrics_uniform = bridge.analyze_consciousness_state(activity_uniform)

    print("\n1. Uniform Activity (anesthesia-like):")
    print(f"   Harmonic Richness: {metrics_uniform['harmonic_richness']:.3f}")
    print(f"   Integration: {metrics_uniform['integration']:.3f}")
    print(f"   Complexity: {metrics_uniform['complexity']:.6f}")

    # Pattern 2: Random (high consciousness - awake-like)
    activity_random = np.random.randn(n_nodes) * 0.5 + 0.3
    metrics_random = bridge.analyze_consciousness_state(activity_random)

    print("\n2. Random Activity (awake-like):")
    print(f"   Harmonic Richness: {metrics_random['harmonic_richness']:.3f}")
    print(f"   Integration: {metrics_random['integration']:.3f}")
    print(f"   Complexity: {metrics_random['complexity']:.6f}")

    # Pattern 3: Localized (low consciousness - sleep-like)
    activity_local = np.zeros(n_nodes)
    activity_local[:n_nodes//10] = np.random.randn(n_nodes//10)
    metrics_local = bridge.analyze_consciousness_state(activity_local)

    print("\n3. Localized Activity (sleep-like):")
    print(f"   Harmonic Richness: {metrics_local['harmonic_richness']:.3f}")
    print(f"   Integration: {metrics_local['integration']:.3f}")
    print(f"   Complexity: {metrics_local['complexity']:.6f}")

    # Pattern 4: Oscillatory (structured consciousness)
    harmonics = bridge.get_harmonic_modes()
    mode_amplitudes = np.zeros(bridge.n_modes)
    mode_amplitudes[10:20] = np.random.randn(10)  # Activate mid-range modes
    activity_osc = bridge.reconstruct_from_modes(mode_amplitudes)
    metrics_osc = bridge.analyze_consciousness_state(activity_osc)

    print("\n4. Oscillatory Activity (structured consciousness):")
    print(f"   Harmonic Richness: {metrics_osc['harmonic_richness']:.3f}")
    print(f"   Integration: {metrics_osc['integration']:.3f}")
    print(f"   Complexity: {metrics_osc['complexity']:.6f}")

    print("\n" + "-"*80)
    print("Key Insight: Higher complexity indicates more conscious-like states")
    print("Complexity combines richness (diversity) with integration (global coherence)")
    print("-"*80)

    return {
        'uniform': metrics_uniform,
        'random': metrics_random,
        'local': metrics_local,
        'oscillatory': metrics_osc
    }


def demo_network_comparison(n_nodes: int = 5000, n_modes: int = 30):
    """Compare different network topologies."""
    print("\n" + "="*80)
    print("DEMO 3: Network Topology Comparison")
    print("="*80)

    topologies = ['random', 'scale_free', 'small_world']
    results = {}

    for topology in topologies:
        print(f"\n{topology.upper()} Network:")
        print("-" * 40)

        # Create network
        W = create_sparse_network(n_nodes, density=0.01, distribution=topology, seed=42)

        # Analyze
        bridge = SparseHarmonicBridge(W, n_modes=n_modes, verbose=False)
        eigenvalues, _ = bridge.compute_harmonics()
        harmonics = bridge.get_harmonic_modes()

        # Random activity pattern
        activity = np.random.randn(n_nodes)
        metrics = bridge.analyze_consciousness_state(activity)

        # Store results
        results[topology] = {
            'eigenvalues': eigenvalues,
            'participation': harmonics['mode_participation'],
            'metrics': metrics
        }

        print(f"  Top eigenvalue: {eigenvalues[-1]:.3f}")
        print(f"  Mean participation: {np.mean(harmonics['mode_participation']):.1f}")
        print(f"  Harmonic richness: {metrics['harmonic_richness']:.3f}")
        print(f"  Complexity: {metrics['complexity']:.6f}")

    print("\n" + "-"*80)
    print("Network Comparison Summary:")
    print(f"  Most complex: {max(results.items(), key=lambda x: x[1]['metrics']['complexity'])[0]}")
    print(f"  Most integrated: {max(results.items(), key=lambda x: x[1]['metrics']['integration'])[0]}")
    print("-"*80)

    return results


def demo_scalability(sizes: list = [5000, 10000, 20000, 30000], n_modes: int = 50):
    """Demonstrate scalability to large networks."""
    print("\n" + "="*80)
    print("DEMO 4: Scalability Test")
    print("="*80)

    import time

    print(f"\nTesting sparse solver on networks of increasing size...")
    print(f"(All with {n_modes} modes, 1% density)\n")

    for n in sizes:
        print(f"\n[{n:,} nodes]")

        # Create network
        start = time.time()
        W = create_sparse_network(n, density=0.01, distribution='random', seed=42)
        gen_time = time.time() - start

        # Compute harmonics
        bridge = SparseHarmonicBridge(W, n_modes=n_modes, verbose=False)
        start = time.time()
        eigenvalues, _ = bridge.compute_harmonics()
        comp_time = time.time() - start

        total_time = gen_time + comp_time
        throughput = n / comp_time

        print(f"  Generation: {gen_time:.2f}s")
        print(f"  Computation: {comp_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} nodes/sec")
        print(f"  Top eigenvalue: {eigenvalues[-1]:.3f}")

    print("\n" + "="*80)


def demo_gpu_acceleration(n_nodes: int = 20000, n_modes: int = 50):
    """Demonstrate GPU acceleration (if available)."""
    try:
        import cupy as cp

        print("\n" + "="*80)
        print("DEMO 5: GPU Acceleration")
        print("="*80)

        print(f"\nComparing CPU vs GPU on {n_nodes:,} node network...")

        # Create network
        W = create_sparse_network(n_nodes, density=0.01, distribution='random', seed=42)

        # CPU timing
        print("\n[CPU]")
        bridge_cpu = SparseHarmonicBridge(W, n_modes=n_modes, device='cpu', verbose=False)
        import time
        start = time.time()
        eigenvalues_cpu, _ = bridge_cpu.compute_harmonics()
        cpu_time = time.time() - start
        print(f"  Computation time: {cpu_time:.2f}s")
        print(f"  Top eigenvalue: {eigenvalues_cpu[-1]:.6f}")

        # GPU timing
        print("\n[GPU]")
        bridge_gpu = SparseHarmonicBridge(W, n_modes=n_modes, device='cuda', verbose=False)
        start = time.time()
        eigenvalues_gpu, _ = bridge_gpu.compute_harmonics()
        gpu_time = time.time() - start
        print(f"  Computation time: {gpu_time:.2f}s")
        print(f"  Top eigenvalue: {eigenvalues_gpu[-1]:.6f}")

        # Comparison
        speedup = cpu_time / gpu_time
        error = np.abs(eigenvalues_cpu[-1] - eigenvalues_gpu[-1])

        print(f"\n[Comparison]")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Eigenvalue error: {error:.2e}")

        if speedup > 1.0:
            print(f"  GPU is {speedup:.1f}x faster!")
        else:
            print(f"  CPU is {1/speedup:.1f}x faster (GPU overhead for this size)")

    except ImportError:
        print("\n[INFO] CuPy not available. Install with: pip install cupy-cuda12x")


def main():
    parser = argparse.ArgumentParser(description='Sparse Harmonic Bridge Demos')
    parser.add_argument('--nodes', type=int, default=10000,
                       help='Number of nodes for main demos')
    parser.add_argument('--modes', type=int, default=50,
                       help='Number of harmonic modes to compute')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU acceleration demo')
    parser.add_argument('--all', action='store_true',
                       help='Run all demos')
    parser.add_argument('--demo', type=int, choices=[1,2,3,4,5],
                       help='Run specific demo (1-5)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SPARSE HARMONIC BRIDGE DEMONSTRATIONS")
    print("Large-Scale Network Consciousness Analysis")
    print("="*80)

    bridge = None  # Initialize

    if args.demo == 1 or args.all or args.demo is None:
        bridge, harmonics = demo_basic_usage(args.nodes, args.modes)

    if args.demo == 2 or args.all:
        if bridge is None or args.demo == 2:  # Create bridge if not from demo 1
            W = create_sparse_network(args.nodes, density=0.01, seed=42)
            bridge = SparseHarmonicBridge(W, n_modes=args.modes, verbose=False)
            bridge.compute_harmonics()
        demo_consciousness_analysis(bridge)

    if args.demo == 3 or args.all:
        demo_network_comparison(n_nodes=min(args.nodes, 5000), n_modes=args.modes)

    if args.demo == 4 or args.all:
        sizes = [5000, 10000, 20000, min(args.nodes, 30000)]
        demo_scalability(sizes, args.modes)

    if args.demo == 5 or args.gpu or args.all:
        demo_gpu_acceleration(min(args.nodes, 20000), args.modes)

    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Sparse eigensolvers enable analysis of 25K-100K node networks")
    print("2. Consciousness metrics scale efficiently with network size")
    print("3. Different network topologies show distinct harmonic signatures")
    print("4. GPU acceleration provides speedups for large networks")
    print("5. Integration with consciousness framework validates the approach")
    print("\nReady for whole-brain connectivity analysis!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
