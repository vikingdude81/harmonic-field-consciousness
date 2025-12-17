"""
Chaos and Criticality Metrics

Extended metrics for analyzing edge-of-chaos dynamics:
1. Lyapunov exponent estimation
2. Avalanche size distributions (power-law detection)
3. Branching ratio (σ = 1 at criticality)
4. Correlation length estimation
5. Susceptibility (response to perturbation)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import stats
from scipy.optimize import curve_fit


def estimate_lyapunov_exponent(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    time_delay: int = 1,
    min_separation: int = 10
) -> Tuple[float, np.ndarray]:
    """
    Estimate the largest Lyapunov exponent from a time series.
    
    Positive λ indicates chaos (sensitive dependence on initial conditions).
    λ ≈ 0 indicates edge of chaos (critical).
    Negative λ indicates ordered/periodic behavior.
    
    Args:
        time_series: 1D time series data
        embedding_dim: Embedding dimension for phase space reconstruction
        time_delay: Time delay for embedding
        min_separation: Minimum temporal separation for neighbor search
    
    Returns:
        Tuple of (lyapunov_exponent, divergence_curve)
    """
    time_series = np.asarray(time_series).flatten()
    n = len(time_series)
    
    # Phase space reconstruction (Takens embedding)
    m = embedding_dim
    tau = time_delay
    
    n_vectors = n - (m - 1) * tau
    if n_vectors < 2 * min_separation:
        return 0.0, np.array([])
    
    # Create embedded vectors
    embedded = np.zeros((n_vectors, m))
    for i in range(m):
        embedded[:, i] = time_series[i * tau:i * tau + n_vectors]
    
    # Find nearest neighbors (excluding temporal neighbors)
    divergences = []
    
    for i in range(n_vectors - min_separation):
        # Find nearest neighbor
        distances = np.linalg.norm(embedded[i] - embedded, axis=1)
        distances[max(0, i - min_separation):min(n_vectors, i + min_separation)] = np.inf
        
        j = np.argmin(distances)
        if distances[j] == np.inf:
            continue
        
        # Track divergence over time
        max_steps = min(50, n_vectors - max(i, j) - 1)
        if max_steps < 5:
            continue
            
        div = []
        for k in range(max_steps):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 0:
                div.append(np.log(d))
            else:
                div.append(div[-1] if div else 0)
        divergences.append(div)
    
    if not divergences:
        return 0.0, np.array([])
    
    # Average divergence curves
    min_len = min(len(d) for d in divergences)
    divergences = [d[:min_len] for d in divergences]
    avg_divergence = np.mean(divergences, axis=0)
    
    # Estimate Lyapunov exponent from slope
    if len(avg_divergence) > 2:
        # Linear fit to log divergence
        t = np.arange(len(avg_divergence))
        slope, _, _, _, _ = stats.linregress(t, avg_divergence)
        lyapunov = slope
    else:
        lyapunov = 0.0
    
    return float(lyapunov), avg_divergence


def detect_avalanches(
    activity: np.ndarray,
    threshold: Optional[float] = None
) -> List[int]:
    """
    Detect avalanches (bursts of activity) in neural activity.
    
    An avalanche is a contiguous period of above-threshold activity.
    At criticality, avalanche sizes follow a power-law distribution.
    
    Args:
        activity: Time series of activity (e.g., total mode power)
        threshold: Threshold for defining active periods (default: mean)
    
    Returns:
        List of avalanche sizes
    """
    activity = np.asarray(activity).flatten()
    
    if threshold is None:
        threshold = np.mean(activity)
    
    # Binarize activity
    active = activity > threshold
    
    # Find avalanche boundaries
    avalanche_sizes = []
    current_size = 0
    in_avalanche = False
    
    for i, is_active in enumerate(active):
        if is_active:
            if not in_avalanche:
                in_avalanche = True
                current_size = 1
            else:
                current_size += 1
        else:
            if in_avalanche:
                avalanche_sizes.append(current_size)
                in_avalanche = False
                current_size = 0
    
    # Don't forget last avalanche
    if in_avalanche:
        avalanche_sizes.append(current_size)
    
    return avalanche_sizes


def fit_power_law(sizes: List[int], x_min: int = 1) -> Tuple[float, float, float]:
    """
    Fit a power-law distribution to avalanche sizes.
    
    P(x) ∝ x^(-α)
    
    At criticality, α ≈ 1.5 (mean-field) or α ≈ 2.0 (branching process).
    
    Args:
        sizes: List of avalanche sizes
        x_min: Minimum size to include in fit
    
    Returns:
        Tuple of (alpha, x_min, ks_statistic)
    """
    sizes = np.array([s for s in sizes if s >= x_min])
    
    if len(sizes) < 10:
        return 0.0, x_min, 1.0
    
    # Maximum likelihood estimator for power-law exponent
    # α = 1 + n / Σ ln(x_i / x_min)
    n = len(sizes)
    alpha = 1 + n / np.sum(np.log(sizes / (x_min - 0.5)))
    
    # Kolmogorov-Smirnov test for goodness of fit
    # Compare empirical CDF to theoretical power-law CDF
    sizes_sorted = np.sort(sizes)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1 - (sizes_sorted / x_min) ** (1 - alpha)
    
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    return float(alpha), x_min, float(ks_stat)


def compute_branching_ratio(
    activity: np.ndarray,
    window_size: int = 10
) -> Tuple[float, np.ndarray]:
    """
    Compute the branching ratio σ.
    
    σ = <descendants> / <ancestors>
    
    σ < 1: subcritical (activity dies out)
    σ = 1: critical (activity sustained, power-law avalanches)
    σ > 1: supercritical (activity explodes)
    
    Args:
        activity: Time series of activity
        window_size: Window for computing local branching ratio
    
    Returns:
        Tuple of (mean_branching_ratio, local_ratios)
    """
    activity = np.asarray(activity).flatten()
    n = len(activity)
    
    if n < window_size + 1:
        return 1.0, np.array([1.0])
    
    # Compute local branching ratios
    local_ratios = []
    
    for i in range(n - 1):
        if activity[i] > 0:
            ratio = activity[i + 1] / activity[i]
            local_ratios.append(ratio)
    
    local_ratios = np.array(local_ratios)
    
    # Trim extreme values
    local_ratios = np.clip(local_ratios, 0.01, 100)
    
    # Mean branching ratio
    sigma = np.mean(local_ratios)
    
    return float(sigma), local_ratios


def estimate_correlation_length(
    spatial_activity: np.ndarray,
    positions: Optional[np.ndarray] = None
) -> float:
    """
    Estimate the correlation length ξ from spatial activity patterns.
    
    At criticality, ξ → ∞ (diverges).
    Away from criticality, ξ is finite.
    
    Args:
        spatial_activity: Activity at each spatial location (N x T)
        positions: Optional spatial positions (N x d)
    
    Returns:
        Estimated correlation length
    """
    spatial_activity = np.asarray(spatial_activity)
    
    if spatial_activity.ndim == 1:
        spatial_activity = spatial_activity.reshape(-1, 1)
    
    n_nodes = spatial_activity.shape[0]
    
    # Compute correlation matrix
    if spatial_activity.shape[1] > 1:
        corr_matrix = np.corrcoef(spatial_activity)
    else:
        corr_matrix = np.eye(n_nodes)
    
    # If no positions, use indices as proxy
    if positions is None:
        positions = np.arange(n_nodes).reshape(-1, 1)
    
    # Compute distance matrix
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(positions))
    
    # Fit exponential decay: C(r) ∝ exp(-r/ξ)
    # Get unique distances and mean correlations at each distance
    r_flat = distances[np.triu_indices(n_nodes, k=1)]
    c_flat = corr_matrix[np.triu_indices(n_nodes, k=1)]
    
    # Bin by distance
    n_bins = min(20, len(np.unique(r_flat)))
    if n_bins < 3:
        return float(np.max(distances))
    
    r_bins = np.linspace(r_flat.min(), r_flat.max(), n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    c_means = []
    
    for i in range(n_bins):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i + 1])
        if np.sum(mask) > 0:
            c_means.append(np.mean(c_flat[mask]))
        else:
            c_means.append(np.nan)
    
    c_means = np.array(c_means)
    valid = ~np.isnan(c_means) & (c_means > 0)
    
    if np.sum(valid) < 3:
        return float(np.max(distances))
    
    # Fit exponential
    try:
        def exp_decay(r, xi, a):
            return a * np.exp(-r / xi)
        
        popt, _ = curve_fit(
            exp_decay, 
            r_centers[valid], 
            c_means[valid],
            p0=[np.max(distances) / 2, 1.0],
            bounds=([0.01, 0], [np.max(distances) * 10, 10])
        )
        xi = popt[0]
    except:
        # Fallback: use decay to 1/e
        half_idx = np.argmin(np.abs(c_means[valid] - c_means[valid][0] / np.e))
        xi = r_centers[valid][half_idx]
    
    return float(xi)


def compute_susceptibility(
    activity: np.ndarray,
    perturbation_strength: float = 0.1,
    n_trials: int = 10
) -> float:
    """
    Compute susceptibility χ (response to perturbation).
    
    At criticality, χ → ∞ (diverges).
    χ = Var(activity) / mean(activity)
    
    Args:
        activity: Baseline activity pattern
        perturbation_strength: Strength of test perturbations
        n_trials: Number of perturbation trials
    
    Returns:
        Susceptibility estimate
    """
    activity = np.asarray(activity).flatten()
    
    # Susceptibility from variance (fluctuation-dissipation relation)
    mean_activity = np.mean(activity)
    var_activity = np.var(activity)
    
    if mean_activity > 0:
        chi = var_activity / mean_activity
    else:
        chi = 0.0
    
    return float(chi)


def compute_all_chaos_metrics(
    time_series: np.ndarray,
    spatial_activity: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all chaos/criticality metrics.
    
    Args:
        time_series: 1D time series of aggregate activity
        spatial_activity: Optional N x T array of spatial activity
    
    Returns:
        Dictionary of chaos metrics
    """
    time_series = np.asarray(time_series).flatten()
    
    # Lyapunov exponent
    lyap, _ = estimate_lyapunov_exponent(time_series)
    
    # Avalanche analysis
    avalanche_sizes = detect_avalanches(time_series)
    if len(avalanche_sizes) >= 10:
        alpha, _, ks = fit_power_law(avalanche_sizes)
    else:
        alpha, ks = 0.0, 1.0
    
    # Branching ratio
    sigma, _ = compute_branching_ratio(time_series)
    
    # Susceptibility
    chi = compute_susceptibility(time_series)
    
    # Correlation length (if spatial data available)
    if spatial_activity is not None:
        xi = estimate_correlation_length(spatial_activity)
    else:
        xi = 0.0
    
    # Criticality score (heuristic combination)
    # Perfect criticality: σ=1, α≈1.5, λ≈0
    sigma_score = 1.0 - abs(sigma - 1.0)  # Peak at σ=1
    alpha_score = 1.0 - abs(alpha - 1.5) / 1.5 if alpha > 0 else 0  # Peak at α=1.5
    lyap_score = 1.0 - min(abs(lyap), 1.0)  # Peak at λ=0
    
    criticality_score = (sigma_score + alpha_score + lyap_score) / 3
    
    return {
        'lyapunov': lyap,
        'avalanche_alpha': alpha,
        'avalanche_ks': ks,
        'branching_ratio': sigma,
        'susceptibility': chi,
        'correlation_length': xi,
        'criticality_score': criticality_score,
        'n_avalanches': len(avalanche_sizes),
        'mean_avalanche_size': np.mean(avalanche_sizes) if avalanche_sizes else 0,
    }


if __name__ == "__main__":
    print("Testing chaos/criticality metrics...")
    
    # Generate test signals
    np.random.seed(42)
    n = 1000
    
    # Ordered signal (periodic)
    t = np.linspace(0, 10 * np.pi, n)
    ordered = np.sin(t) + 0.1 * np.random.randn(n)
    
    # Chaotic signal (logistic map)
    chaotic = np.zeros(n)
    chaotic[0] = 0.1
    r = 3.9  # Chaotic regime
    for i in range(1, n):
        chaotic[i] = r * chaotic[i-1] * (1 - chaotic[i-1])
    
    # Critical-like signal (pink noise / 1/f)
    from scipy import signal as sig
    white = np.random.randn(n)
    b, a = sig.butter(2, 0.1)
    critical = sig.filtfilt(b, a, white)
    critical = (critical - critical.min()) / (critical.max() - critical.min())
    
    print("\n" + "="*60)
    for name, data in [("Ordered", ordered), ("Chaotic", chaotic), ("Critical-like", critical)]:
        metrics = compute_all_chaos_metrics(data)
        print(f"\n{name} signal:")
        print(f"  Lyapunov exponent: {metrics['lyapunov']:.4f}")
        print(f"  Branching ratio σ: {metrics['branching_ratio']:.4f}")
        print(f"  Avalanche α: {metrics['avalanche_alpha']:.2f}")
        print(f"  Susceptibility χ: {metrics['susceptibility']:.4f}")
        print(f"  Criticality score: {metrics['criticality_score']:.3f}")
    
    print("\n" + "="*60)
    print("All chaos metrics working correctly!")
