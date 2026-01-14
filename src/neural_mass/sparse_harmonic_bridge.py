"""
Sparse Harmonic Bridge - Scalable Network Eigendecomposition

Extends HarmonicBridge to support large-scale networks (>25K nodes) using
sparse eigensolvers. Enables analysis of whole-brain connectivity patterns
with hundreds of thousands of nodes.

Key features:
1. Sparse matrix eigendecomposition (ARPACK, LOBPCG)
2. GPU acceleration via CuPy (optional)
3. Memory-efficient storage and computation
4. Scales to 100K+ node networks

Integration with consciousness framework:
- Large networks → global harmonic modes
- Sparse structure → biologically realistic connectivity
- Eigenmodes → fundamental oscillation patterns
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg, ArpackNoConvergence
from typing import Dict, Tuple, Optional, Union, Literal
import warnings

# Optional GPU support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SparseHarmonicBridge:
    """
    Scalable harmonic decomposition for large neural networks.

    Uses sparse eigensolvers to compute harmonic modes from network
    connectivity matrices. Supports networks with 25K-100K+ nodes.
    """

    def __init__(
        self,
        adjacency_matrix: Union[np.ndarray, sp.spmatrix],
        n_modes: int = 50,
        device: Literal['cpu', 'cuda'] = 'cpu',
        solver: Literal['arpack', 'lobpcg'] = 'arpack',
        verbose: bool = True
    ):
        """
        Initialize sparse harmonic bridge.

        Args:
            adjacency_matrix: Network connectivity (dense or sparse)
            n_modes: Number of harmonic modes to compute
            device: 'cpu' or 'cuda' for GPU acceleration
            solver: 'arpack' (default) or 'lobpcg'
            verbose: Print progress messages
        """
        self.n_nodes = adjacency_matrix.shape[0]
        self.n_modes = min(n_modes, self.n_nodes - 2)  # ARPACK limitation
        self.device = device
        self.solver = solver
        self.verbose = verbose

        # Convert to sparse format if needed
        if not sp.issparse(adjacency_matrix):
            if self.verbose:
                print(f"Converting dense matrix ({self.n_nodes}×{self.n_nodes}) to sparse CSR format...")
            self.W = sp.csr_matrix(adjacency_matrix, dtype=np.float32)
        else:
            self.W = adjacency_matrix.astype(np.float32)

        # Ensure symmetric
        if not self._is_symmetric():
            if self.verbose:
                print("Matrix not symmetric, symmetrizing: W = (W + W.T) / 2")
            self.W = (self.W + self.W.T) / 2

        # GPU setup
        if device == 'cuda' and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available, falling back to CPU", RuntimeWarning)
            self.device = 'cpu'

        # Computed modes (lazy evaluation)
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._harmonics_computed = False

        if self.verbose:
            nnz = self.W.nnz
            sparsity = 100 * (1 - nnz / (self.n_nodes * self.n_nodes))
            mem_mb = self.W.data.nbytes / 1024 / 1024
            print(f"Sparse Harmonic Bridge initialized:")
            print(f"  - Nodes: {self.n_nodes:,}")
            print(f"  - Non-zeros: {nnz:,} ({sparsity:.2f}% sparse)")
            print(f"  - Memory: {mem_mb:.1f} MB")
            print(f"  - Modes to compute: {self.n_modes}")
            print(f"  - Device: {self.device}")
            print(f"  - Solver: {self.solver}")

    def _is_symmetric(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if matrix is symmetric."""
        diff = self.W - self.W.T
        return np.allclose(diff.data, 0, rtol=rtol, atol=atol)

    def compute_harmonics(
        self,
        maxiter: Optional[int] = None,
        tol: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute harmonic modes via sparse eigendecomposition.

        Args:
            maxiter: Maximum iterations (None = auto)
            tol: Convergence tolerance (0 = machine precision)

        Returns:
            eigenvalues: (n_modes,) array of eigenvalues
            eigenvectors: (n_nodes, n_modes) array of eigenvectors
        """
        if self._harmonics_computed:
            return self._eigenvalues, self._eigenvectors

        if self.verbose:
            print(f"\nComputing top {self.n_modes} harmonic modes...")

        if self.device == 'cuda' and CUPY_AVAILABLE:
            eigenvalues, eigenvectors = self._compute_gpu(maxiter, tol)
        else:
            eigenvalues, eigenvectors = self._compute_cpu(maxiter, tol)

        # Store results
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._harmonics_computed = True

        if self.verbose:
            print(f"[OK] Harmonic decomposition complete!")
            print(f"  - Top eigenvalue: {eigenvalues[-1]:.6f}")
            print(f"  - Bottom eigenvalue: {eigenvalues[0]:.6f}")
            print(f"  - Eigenvalue range: {eigenvalues[-1] - eigenvalues[0]:.6f}")

        return eigenvalues, eigenvectors

    def _compute_cpu(
        self,
        maxiter: Optional[int],
        tol: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU-based sparse eigendecomposition."""
        import time
        start_time = time.time()

        if self.solver == 'arpack':
            try:
                eigenvalues, eigenvectors = eigsh(
                    self.W,
                    k=self.n_modes,
                    which='LM',  # Largest magnitude
                    maxiter=maxiter,
                    tol=tol
                )
            except ArpackNoConvergence as e:
                warnings.warn(
                    f"ARPACK did not fully converge. "
                    f"Returning {len(e.eigenvalues)} partial results.",
                    RuntimeWarning
                )
                eigenvalues = e.eigenvalues
                eigenvectors = e.eigenvectors

        elif self.solver == 'lobpcg':
            # LOBPCG needs initial guess
            X = np.random.randn(self.n_nodes, self.n_modes).astype(np.float32)
            eigenvalues, eigenvectors = lobpcg(
                self.W,
                X,
                maxiter=maxiter if maxiter else 40,
                tol=tol if tol else 1e-8
            )
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"  - CPU computation time: {elapsed:.2f}s")

        return eigenvalues, eigenvectors

    def _compute_gpu(
        self,
        maxiter: Optional[int],
        tol: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated sparse eigendecomposition."""
        import time
        start_time = time.time()

        if self.verbose:
            print("  - Transferring matrix to GPU...")

        # Transfer to GPU
        W_gpu = cpsp.csr_matrix(self.W)

        if self.verbose:
            print("  - Computing on GPU...")

        # Compute on GPU (CuPy only supports eigsh, not lobpcg)
        eigenvalues_gpu, eigenvectors_gpu = cp_eigsh(
            W_gpu,
            k=self.n_modes,
            which='LM',
            maxiter=maxiter,
            tol=tol
        )

        # Transfer back to CPU
        eigenvalues = cp.asnumpy(eigenvalues_gpu)
        eigenvectors = cp.asnumpy(eigenvectors_gpu)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"  - GPU computation time: {elapsed:.2f}s")

        return eigenvalues, eigenvectors

    def get_harmonic_modes(self) -> Dict[str, np.ndarray]:
        """
        Get harmonic mode representation.

        Returns:
            Dictionary with:
                - eigenvalues: Harmonic frequencies (squared)
                - eigenvectors: Spatial mode patterns
                - frequencies: Effective frequencies (sqrt of eigenvalues)
                - mode_participation: Participation ratio for each mode
        """
        if not self._harmonics_computed:
            self.compute_harmonics()

        # Compute derived quantities
        frequencies = np.sqrt(np.abs(self._eigenvalues))

        # Participation ratio for each mode
        mode_participation = np.array([
            self._compute_mode_participation(self._eigenvectors[:, i])
            for i in range(self.n_modes)
        ])

        return {
            'eigenvalues': self._eigenvalues,
            'eigenvectors': self._eigenvectors,
            'frequencies': frequencies,
            'mode_participation': mode_participation,
            'n_nodes': self.n_nodes,
            'n_modes': self.n_modes
        }

    def _compute_mode_participation(self, mode_vector: np.ndarray) -> float:
        """
        Compute participation ratio for a single mode.

        PR = (Σ |ψ_i|^2)^2 / Σ |ψ_i|^4

        Measures how many nodes participate in the mode.
        PR ∈ [1, N] where N is number of nodes.
        """
        power = mode_vector ** 2
        sum_sq = np.sum(power) ** 2
        sum_fourth = np.sum(power ** 2)

        if sum_fourth > 1e-12:
            return sum_sq / sum_fourth
        else:
            return 1.0

    def project_onto_modes(
        self,
        activity_pattern: np.ndarray
    ) -> np.ndarray:
        """
        Project network activity onto harmonic modes.

        Args:
            activity_pattern: (n_nodes,) activity pattern

        Returns:
            mode_amplitudes: (n_modes,) projection onto each mode
        """
        if not self._harmonics_computed:
            self.compute_harmonics()

        # Project: a_k = <ψ_k | activity>
        mode_amplitudes = self._eigenvectors.T @ activity_pattern

        return mode_amplitudes

    def reconstruct_from_modes(
        self,
        mode_amplitudes: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct network activity from mode amplitudes.

        Args:
            mode_amplitudes: (n_modes,) mode coefficients

        Returns:
            activity_pattern: (n_nodes,) reconstructed activity
        """
        if not self._harmonics_computed:
            self.compute_harmonics()

        # Reconstruct: activity = Σ a_k * ψ_k
        activity_pattern = self._eigenvectors @ mode_amplitudes

        return activity_pattern

    def analyze_consciousness_state(
        self,
        activity_pattern: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze consciousness-related metrics from activity pattern.

        Args:
            activity_pattern: (n_nodes,) current network activity

        Returns:
            Dictionary with consciousness metrics:
                - harmonic_richness: Diversity of active modes
                - integration: Global connectivity strength
                - segregation: Local specialization
                - complexity: Balance between integration and segregation
        """
        # Project onto modes
        mode_amplitudes = self.project_onto_modes(activity_pattern)

        # Harmonic richness (entropy of mode distribution)
        power = mode_amplitudes ** 2
        power_norm = power / (np.sum(power) + 1e-12)
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-12))
        max_entropy = np.log(self.n_modes)
        harmonic_richness = entropy / max_entropy if max_entropy > 0 else 0.0

        # Integration (participation of high-frequency modes)
        high_freq_modes = self.n_modes // 2
        integration = np.sum(power[-high_freq_modes:]) / np.sum(power)

        # Segregation (participation of low-frequency modes)
        low_freq_modes = self.n_modes // 2
        segregation = np.sum(power[:low_freq_modes]) / np.sum(power)

        # Complexity (balance between integration and segregation)
        complexity = harmonic_richness * integration * segregation

        return {
            'harmonic_richness': float(harmonic_richness),
            'integration': float(integration),
            'segregation': float(segregation),
            'complexity': float(complexity),
            'total_power': float(np.sum(power))
        }

    def save_harmonics(self, filepath: str):
        """Save computed harmonic modes to file."""
        if not self._harmonics_computed:
            self.compute_harmonics()

        np.savez_compressed(
            filepath,
            eigenvalues=self._eigenvalues,
            eigenvectors=self._eigenvectors,
            n_nodes=self.n_nodes,
            n_modes=self.n_modes,
            adjacency_nnz=self.W.nnz
        )

        if self.verbose:
            print(f"Harmonics saved to: {filepath}")

    def load_harmonics(self, filepath: str):
        """Load precomputed harmonic modes from file."""
        data = np.load(filepath)

        self._eigenvalues = data['eigenvalues']
        self._eigenvectors = data['eigenvectors']
        self._harmonics_computed = True

        if self.verbose:
            print(f"Harmonics loaded from: {filepath}")
            print(f"  - Nodes: {data['n_nodes']}")
            print(f"  - Modes: {data['n_modes']}")


def create_sparse_network(
    n_nodes: int,
    density: float = 0.01,
    distribution: Literal['random', 'scale_free', 'small_world'] = 'random',
    seed: Optional[int] = None
) -> sp.csr_matrix:
    """
    Create a sparse network for testing.

    Args:
        n_nodes: Number of nodes
        density: Connection density (0-1)
        distribution: Network topology
        seed: Random seed

    Returns:
        Sparse symmetric adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)

    if distribution == 'random':
        # Random sparse network
        A = sp.random(n_nodes, n_nodes, density=density, format='csr', dtype=np.float32)
        A = (A + A.T) / 2  # Symmetrize

    elif distribution == 'scale_free':
        # Approximate scale-free using power law degree distribution
        from scipy.stats import powerlaw

        # Generate degree sequence
        degrees = powerlaw.rvs(2.5, size=n_nodes) * density * n_nodes
        degrees = np.clip(degrees.astype(int), 1, n_nodes // 2)

        # Build adjacency using configuration model (simplified)
        row, col, data = [], [], []
        for i in range(n_nodes):
            neighbors = np.random.choice(n_nodes, size=degrees[i], replace=False)
            for j in neighbors:
                if i != j:
                    row.append(i)
                    col.append(j)
                    data.append(np.random.rand())

        A = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
        A = (A + A.T) / 2

    elif distribution == 'small_world':
        # Watts-Strogatz-like small world
        k = int(density * n_nodes / 2)  # Average degree
        p_rewire = 0.1  # Rewiring probability

        row, col, data = [], [], []

        # Start with ring lattice
        for i in range(n_nodes):
            for j in range(1, k + 1):
                neighbor = (i + j) % n_nodes
                if np.random.rand() < p_rewire:
                    # Rewire
                    neighbor = np.random.randint(n_nodes)
                if i != neighbor:
                    row.append(i)
                    col.append(neighbor)
                    data.append(np.random.rand())

        A = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
        A = (A + A.T) / 2

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return A
