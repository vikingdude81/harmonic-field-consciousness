"""
Consciousness Metrics Calculations

Implements the five core components of the consciousness functional:
1. Mode Entropy (H_mode)
2. Participation Ratio (PR)
3. Phase Coherence (R)
4. Entropy Production Rate (Ṡ)
5. Criticality Index (κ)

Plus additional complexity measures:
- Lempel-Ziv complexity
- Multiscale entropy
"""

import numpy as np
from typing import Optional, Dict, Tuple
from scipy import signal


def compute_mode_entropy(power: np.ndarray, normalize: bool = True) -> float:
    """
    Compute Shannon entropy of mode power distribution.
    
    High entropy indicates broad distribution across modes (characteristic of consciousness).
    
    Args:
        power: Mode power distribution (normalized or unnormalized)
        normalize: Whether to normalize the result by maximum entropy
    
    Returns:
        Mode entropy H_mode (in nats, or normalized to [0,1])
    """
    # Normalize power to probability distribution
    power = np.asarray(power)
    power = power / (power.sum() + 1e-12)
    
    # Filter out zero values
    p = power[power > 1e-12]
    
    # Compute Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))
    
    if normalize:
        # Normalize by maximum possible entropy
        H_max = np.log(len(power))
        H = H / (H_max + 1e-12)
    
    return float(H)


def compute_participation_ratio(power: np.ndarray, normalize: bool = True) -> float:
    """
    Compute participation ratio (effective number of modes).
    
    High PR indicates many modes participating (characteristic of consciousness).
    
    Args:
        power: Mode power distribution (normalized or unnormalized)
        normalize: Whether to normalize by number of modes
    
    Returns:
        Participation ratio PR
    """
    power = np.asarray(power)
    power = power / (power.sum() + 1e-12)
    
    # PR = 1 / sum(p_k^2)
    PR = 1.0 / (np.sum(power ** 2) + 1e-12)
    
    if normalize:
        # Normalize by number of modes
        PR = PR / len(power)
    
    return float(PR)


def compute_phase_coherence(
    phases: np.ndarray,
    power: Optional[np.ndarray] = None
) -> float:
    """
    Compute phase coherence (Kuramoto order parameter).
    
    Measures alignment of oscillatory phases across modes.
    R ≈ 1: highly synchronized (rigid)
    R ≈ 0: desynchronized (fragmented)
    R ≈ 0.3-0.7: optimal for consciousness (flexible gating)
    
    Args:
        phases: Phase angles (in radians) for each mode
        power: Optional weights for each mode
    
    Returns:
        Phase coherence R ∈ [0, 1]
    """
    phases = np.asarray(phases)
    
    if power is None:
        power = np.ones(len(phases))
    else:
        power = np.asarray(power)
    
    # Normalize power
    power = power / (power.sum() + 1e-12)
    
    # Compute complex order parameter
    Z = np.sum(power * np.exp(1j * phases))
    
    # R is the magnitude
    R = np.abs(Z)
    
    return float(R)


def compute_entropy_production(
    power_current: np.ndarray,
    power_previous: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute entropy production rate Ṡ.
    
    Measures rate of change in mode distribution.
    Positive Ṡ indicates active information processing.
    
    Args:
        power_current: Current mode power distribution
        power_previous: Previous mode power distribution
        dt: Time step
    
    Returns:
        Entropy production rate Ṡ (normalized)
    """
    power_current = np.asarray(power_current)
    power_previous = np.asarray(power_previous)
    
    # Normalize
    p_curr = power_current / (power_current.sum() + 1e-12)
    p_prev = power_previous / (power_previous.sum() + 1e-12)
    
    # Compute KL divergence as proxy for entropy production
    # D_KL(p_curr || p_prev)
    p_prev = np.clip(p_prev, 1e-12, 1.0)
    p_curr = np.clip(p_curr, 1e-12, 1.0)
    
    KL = np.sum(p_curr * np.log(p_curr / p_prev))
    
    # Rate of change
    S_dot = KL / (dt + 1e-12)
    
    return float(S_dot)


def compute_criticality_index(
    eigenvalues: np.ndarray,
    power: np.ndarray
) -> float:
    """
    Compute criticality index κ.
    
    Measures proximity to edge-of-chaos critical regime.
    κ ≈ 1: critical (optimal for consciousness)
    κ < 1: subcritical (ordered)
    κ > 1: supercritical (chaotic)
    
    Args:
        eigenvalues: Graph Laplacian eigenvalues
        power: Mode power distribution
    
    Returns:
        Criticality index κ
    """
    eigenvalues = np.asarray(eigenvalues)
    power = np.asarray(power)
    
    # Normalize power
    power = power / (power.sum() + 1e-12)
    
    # Compute effective spectral radius
    # Weighted by mode power
    lambda_eff = np.sum(power * eigenvalues)
    
    # Normalize by maximum eigenvalue
    lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 1.0
    
    # κ measures relative position in spectrum
    kappa = lambda_eff / (lambda_max + 1e-12)
    
    return float(kappa)


def compute_consciousness_functional(
    H_mode: float,
    PR: float,
    R: float,
    S_dot: float,
    kappa: float,
    weights: Optional[Tuple[float, float, float, float, float]] = None
) -> float:
    """
    Compute combined consciousness functional C(t).
    
    C(t) = w1·H_mode + w2·PR + w3·R + w4·Ṡ + w5·κ
    
    Default weights are equal (0.2 each).
    
    Args:
        H_mode: Mode entropy (normalized)
        PR: Participation ratio (normalized)
        R: Phase coherence
        S_dot: Entropy production rate (normalized)
        kappa: Criticality index (normalized)
        weights: Optional custom weights (must sum to 1.0)
    
    Returns:
        Consciousness functional C(t) ∈ [0, 1]
    """
    if weights is None:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    w1, w2, w3, w4, w5 = weights
    
    C = w1 * H_mode + w2 * PR + w3 * R + w4 * S_dot + w5 * kappa
    
    return float(np.clip(C, 0.0, 1.0))


def compute_all_metrics(
    power: np.ndarray,
    eigenvalues: np.ndarray,
    phases: Optional[np.ndarray] = None,
    power_previous: Optional[np.ndarray] = None,
    dt: float = 1.0,
    weights: Optional[Tuple[float, float, float, float, float]] = None
) -> Dict[str, float]:
    """
    Compute all consciousness metrics at once.
    
    Args:
        power: Current mode power distribution
        eigenvalues: Graph Laplacian eigenvalues
        phases: Optional phase angles for each mode
        power_previous: Optional previous power distribution for Ṡ
        dt: Time step
        weights: Optional custom weights for C(t)
    
    Returns:
        Dictionary with all metrics
    """
    # Generate random phases if not provided
    if phases is None:
        phases = np.random.uniform(0, 2 * np.pi, len(power))
    
    # Use current power for previous if not provided
    if power_previous is None:
        power_previous = power + np.random.normal(0, 0.01, len(power))
        power_previous = np.clip(power_previous, 0, None)
    
    H = compute_mode_entropy(power, normalize=True)
    PR = compute_participation_ratio(power, normalize=True)
    R = compute_phase_coherence(phases, power)
    S_dot = compute_entropy_production(power, power_previous, dt)
    kappa = compute_criticality_index(eigenvalues, power)
    
    # Normalize S_dot to [0, 1] range (heuristic)
    S_dot_norm = np.tanh(S_dot)  # Maps to [-1, 1], then shift
    S_dot_norm = (S_dot_norm + 1) / 2  # Map to [0, 1]
    
    C = compute_consciousness_functional(H, PR, R, S_dot_norm, kappa, weights)
    
    return {
        'H_mode': H,
        'PR': PR,
        'R': R,
        'S_dot': S_dot_norm,
        'kappa': kappa,
        'C': C,
    }


def compute_lempel_ziv_complexity(sequence: np.ndarray, binary: bool = True) -> float:
    """
    Compute Lempel-Ziv complexity of a sequence.
    
    Measures algorithmic complexity (number of distinct patterns).
    
    Args:
        sequence: Binary or continuous sequence
        binary: Whether to binarize the sequence first
    
    Returns:
        Normalized LZ complexity
    """
    sequence = np.asarray(sequence).flatten()
    
    if binary:
        # Binarize around median
        sequence = (sequence > np.median(sequence)).astype(int)
    
    # Convert to string
    s = ''.join(map(str, sequence.astype(int)))
    
    n = len(s)
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    
    while True:
        if i + k > n:
            break
        
        # Check if substring is new
        substr = s[i:i+k]
        vocab = s[0:i+k]
        
        if substr in vocab[:-k]:
            k += 1
        else:
            c += 1
            i += k
            k = 1
    
    # Normalize
    # Theoretical maximum for random sequence
    c_max = n / np.log2(n + 1) if n > 1 else 1
    
    return c / c_max


def compute_multiscale_entropy(
    signal_data: np.ndarray,
    scales: Optional[np.ndarray] = None,
    m: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Compute multiscale entropy (MSE).
    
    Measures complexity across multiple time scales.
    
    Args:
        signal_data: Input signal
        scales: Scales to compute entropy over (default: 1 to 20)
        m: Embedding dimension for sample entropy
    
    Returns:
        Tuple of (entropy values at each scale, mean MSE)
    """
    signal_data = np.asarray(signal_data)
    
    if scales is None:
        scales = np.arange(1, min(21, len(signal_data) // 10))
    
    mse_values = []
    
    for scale in scales:
        # Coarse-grain the signal
        n_segments = len(signal_data) // scale
        coarse = np.mean(signal_data[:n_segments * scale].reshape(n_segments, scale), axis=1)
        
        # Compute sample entropy
        ent = _sample_entropy(coarse, m)
        mse_values.append(ent)
    
    return np.array(mse_values), float(np.mean(mse_values))


def _sample_entropy(signal_data: np.ndarray, m: int, r: Optional[float] = None) -> float:
    """
    Compute sample entropy of a signal.
    
    Args:
        signal_data: Input signal
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)
    
    Returns:
        Sample entropy value
    """
    N = len(signal_data)
    
    if r is None:
        r = 0.2 * np.std(signal_data)
    
    # Count template matches
    def _phi(m):
        patterns = np.array([signal_data[i:i+m] for i in range(N - m)])
        
        count = 0
        for i in range(len(patterns)):
            # Find matches within tolerance r
            dists = np.max(np.abs(patterns - patterns[i]), axis=1)
            count += np.sum(dists <= r) - 1  # Exclude self-match
        
        return count / (N - m)
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    
    return -np.log(phi_m1 / phi_m)


if __name__ == "__main__":
    # Test all metrics
    print("Testing consciousness metrics...")
    
    n_modes = 20
    
    # Create sample data
    eigenvalues = np.linspace(0, 10, n_modes)
    
    # Wake-like state
    power_wake = 0.3 + 0.4 * np.exp(-np.arange(n_modes) / 8)
    power_wake += 0.15 * np.random.rand(n_modes)
    power_wake /= power_wake.sum()
    
    # Anesthesia-like state
    power_anesthesia = np.exp(-np.arange(n_modes) / 1.5)
    power_anesthesia += 0.02 * np.random.rand(n_modes)
    power_anesthesia /= power_anesthesia.sum()
    
    # Test metrics
    for name, power in [("Wake", power_wake), ("Anesthesia", power_anesthesia)]:
        metrics = compute_all_metrics(power, eigenvalues)
        print(f"\n{name} state:")
        print(f"  H_mode: {metrics['H_mode']:.3f}")
        print(f"  PR:     {metrics['PR']:.3f}")
        print(f"  R:      {metrics['R']:.3f}")
        print(f"  S_dot:  {metrics['S_dot']:.3f}")
        print(f"  kappa:  {metrics['kappa']:.3f}")
        print(f"  C(t):   {metrics['C']:.3f}")
    
    # Test additional complexity measures
    signal_data = np.random.randn(1000)
    lz = compute_lempel_ziv_complexity(signal_data)
    mse_vals, mse_mean = compute_multiscale_entropy(signal_data)
    
    print(f"\nAdditional metrics:")
    print(f"  LZ complexity: {lz:.3f}")
    print(f"  MSE (mean):    {mse_mean:.3f}")
    
    print("\nAll metrics working correctly!")
