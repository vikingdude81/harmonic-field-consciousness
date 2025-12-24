"""
Dynamic Stability Metrics

Implements metrics for measuring dynamic stability - the ability of neural
systems to maintain stable trajectories and recover from perturbations.

Based on:
- Batabyal et al. (2025) JOCN - Recovery dynamics in working memory
- Kozachkov et al. (2020) PLOS Comp Bio - Achieving stable dynamics
- Sussillo (2014) Current Opinion in Neurobiology - Neural dynamics

Key concepts:
- Dynamic stability = returning to proper trajectory after perturbation
- Different from static stability (fixed points)
- Critical for maintaining cognitive states despite noise
"""

import numpy as np
from scipy import linalg, signal
from scipy.spatial.distance import euclidean
from typing import Dict, Tuple, Optional, List


def compute_lyapunov_spectrum(trajectory: np.ndarray,
                              embedding_dim: int = 3,
                              delay: int = 1,
                              min_neighbors: int = 10) -> np.ndarray:
    """
    Compute Lyapunov exponent spectrum for trajectory stability analysis.
    
    Positive Lyapunov exponents indicate chaos/instability.
    Negative exponents indicate stability/convergence.
    
    Args:
        trajectory: State trajectory [time x dimensions]
        embedding_dim: Dimension for phase space reconstruction
        delay: Time delay for embedding
        min_neighbors: Minimum neighbors for local Jacobian estimation
        
    Returns:
        Lyapunov exponents (sorted descending)
    """
    T, N = trajectory.shape
    
    # Simple approximation using finite differences
    # Estimate local Jacobians
    jacobians = []
    
    for t in range(1, T-1):
        # Forward difference approximation
        dx = trajectory[t+1] - trajectory[t]
        x = trajectory[t]
        
        # Normalize by state magnitude to get relative change
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            jacobian_approx = dx / norm
            jacobians.append(jacobian_approx)
    
    if len(jacobians) == 0:
        return np.zeros(N)
    
    # Compute mean expansion rates
    jacobians = np.array(jacobians)
    lyapunov_exponents = np.mean(np.log(np.abs(jacobians) + 1e-10), axis=0)
    
    return np.sort(lyapunov_exponents)[::-1]


def compute_trajectory_stability(trajectory: np.ndarray,
                                reference_trajectory: Optional[np.ndarray] = None) -> Dict:
    """
    Measure stability of a trajectory relative to a reference.
    
    Args:
        trajectory: State trajectory [time x dimensions]
        reference_trajectory: Reference trajectory to compare against
                            (if None, uses unperturbed continuation)
        
    Returns:
        Dictionary with stability metrics
    """
    T, N = trajectory.shape
    
    if reference_trajectory is None:
        # Use linear extrapolation from first few points as reference
        t_ref = min(10, T // 4)
        velocity = np.mean(np.diff(trajectory[:t_ref], axis=0), axis=0)
        reference_trajectory = trajectory[0] + np.outer(np.arange(T), velocity)
    
    # Compute distance from reference over time
    distances = np.linalg.norm(trajectory - reference_trajectory, axis=1)
    
    # Stability metrics
    max_deviation = np.max(distances)
    mean_deviation = np.mean(distances)
    final_deviation = distances[-1]
    
    # Convergence rate (if trajectory is returning to reference)
    if len(distances) > 2:
        # Fit exponential decay: d(t) = A * exp(-λt)
        log_distances = np.log(distances + 1e-10)
        t = np.arange(len(distances))
        
        # Linear fit in log space
        try:
            slope, _ = np.polyfit(t, log_distances, 1)
            convergence_rate = -slope  # Positive = converging
        except:
            convergence_rate = 0.0
    else:
        convergence_rate = 0.0
    
    return {
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'final_deviation': final_deviation,
        'convergence_rate': convergence_rate,
        'is_stable': convergence_rate > 0 and final_deviation < mean_deviation,
        'distances_over_time': distances
    }


def compute_contraction_rate(dynamics_matrix: np.ndarray) -> float:
    """
    Compute contraction rate from dynamics matrix eigenvalues.
    
    For system dx/dt = M*x, negative eigenvalues indicate contraction.
    
    Args:
        dynamics_matrix: Linear dynamics matrix M
        
    Returns:
        Contraction rate (negative = contracting to fixed point)
    """
    eigenvalues = linalg.eigvals(dynamics_matrix)
    
    # Maximum real part determines growth/contraction
    max_real = np.max(np.real(eigenvalues))
    
    return max_real


def measure_perturbation_recovery(trajectory_unperturbed: np.ndarray,
                                  trajectory_perturbed: np.ndarray,
                                  perturbation_time: int) -> Dict:
    """
    Measure how well a system recovers from a perturbation.
    
    Implements metrics from Batabyal et al. (2025).
    
    Args:
        trajectory_unperturbed: Trajectory without perturbation [time x dims]
        trajectory_perturbed: Trajectory with perturbation [time x dims]
        perturbation_time: Time index when perturbation occurred
        
    Returns:
        Dictionary with recovery metrics
    """
    # Distance between trajectories over time
    distances = np.linalg.norm(
        trajectory_perturbed - trajectory_unperturbed, 
        axis=1
    )
    
    # Pre-perturbation baseline
    baseline_distance = np.mean(distances[:perturbation_time])
    
    # Peak perturbation (maximum distance after perturbation)
    post_pert_distances = distances[perturbation_time:]
    if len(post_pert_distances) == 0:
        return {'recovery_percentage': 0.0, 'recovery_time': None}
    
    peak_distance = np.max(post_pert_distances)
    peak_time = perturbation_time + np.argmax(post_pert_distances)
    
    # Final distance
    final_distance = distances[-1]
    
    # Recovery percentage
    if peak_distance > baseline_distance:
        recovery_pct = (1 - (final_distance - baseline_distance) / 
                       (peak_distance - baseline_distance)) * 100
        recovery_pct = np.clip(recovery_pct, 0, 100)
    else:
        recovery_pct = 100.0  # No significant perturbation
    
    # Recovery time (time to return within 2x baseline distance)
    threshold = baseline_distance * 2
    recovered_times = np.where(distances[peak_time:] <= threshold)[0]
    if len(recovered_times) > 0:
        recovery_time = recovered_times[0]
    else:
        recovery_time = None
    
    return {
        'recovery_percentage': recovery_pct,
        'recovery_time': recovery_time,
        'peak_distance': peak_distance,
        'peak_time': peak_time,
        'final_distance': final_distance,
        'baseline_distance': baseline_distance,
        'distances_over_time': distances
    }


def compute_basin_of_attraction_size(dynamics_matrix: np.ndarray,
                                    attractor_state: np.ndarray,
                                    n_samples: int = 1000,
                                    n_steps: int = 100,
                                    noise_scale: float = 0.1) -> float:
    """
    Estimate size of basin of attraction around an attractor.
    
    Larger basin = more stable/robust attractor.
    
    Args:
        dynamics_matrix: Linear dynamics M in dx/dt = M*x
        attractor_state: The attractor state
        n_samples: Number of random initial conditions to test
        n_steps: Number of integration steps
        noise_scale: Scale of initial perturbations
        
    Returns:
        Estimated basin size (fraction of samples that converge)
    """
    dim = len(attractor_state)
    converged = 0
    
    for _ in range(n_samples):
        # Random initial condition near attractor
        x0 = attractor_state + np.random.randn(dim) * noise_scale
        
        # Simulate dynamics
        x = x0.copy()
        for _ in range(n_steps):
            dx = dynamics_matrix @ (x - attractor_state)
            x = x + dx * 0.01  # Small time step
        
        # Check if converged to attractor
        distance = np.linalg.norm(x - attractor_state)
        if distance < noise_scale:
            converged += 1
    
    return converged / n_samples


def measure_trajectory_reproducibility(trajectories: List[np.ndarray],
                                      time_warping: bool = False) -> Dict:
    """
    Measure how reproducible trajectories are across trials/conditions.
    
    High reproducibility = stable dynamics.
    
    Args:
        trajectories: List of trajectories [each is time x dimensions]
        time_warping: Whether to use dynamic time warping for alignment
        
    Returns:
        Dictionary with reproducibility metrics
    """
    if len(trajectories) < 2:
        return {'reproducibility': 1.0, 'mean_pairwise_distance': 0.0}
    
    # Compute pairwise distances between trajectories
    pairwise_distances = []
    
    for i in range(len(trajectories)):
        for j in range(i+1, len(trajectories)):
            traj1, traj2 = trajectories[i], trajectories[j]
            
            # Align lengths
            min_len = min(len(traj1), len(traj2))
            traj1 = traj1[:min_len]
            traj2 = traj2[:min_len]
            
            if time_warping:
                # Use dynamic time warping (simplified version)
                distance = compute_dtw_distance(traj1, traj2)
            else:
                # Simple Euclidean distance
                distance = np.mean(np.linalg.norm(traj1 - traj2, axis=1))
            
            pairwise_distances.append(distance)
    
    mean_distance = np.mean(pairwise_distances)
    std_distance = np.std(pairwise_distances)
    
    # Reproducibility score (lower distance = higher reproducibility)
    # Normalize by average trajectory magnitude
    mean_magnitude = np.mean([np.mean(np.linalg.norm(t, axis=1)) 
                              for t in trajectories])
    
    if mean_magnitude > 0:
        reproducibility = 1 - np.clip(mean_distance / mean_magnitude, 0, 1)
    else:
        reproducibility = 0.0
    
    return {
        'reproducibility': reproducibility,
        'mean_pairwise_distance': mean_distance,
        'std_pairwise_distance': std_distance,
        'pairwise_distances': np.array(pairwise_distances)
    }


def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between trajectories.
    
    Simplified version for time series alignment.
    """
    n, m = len(traj1), len(traj2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(traj1[i-1], traj2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


def analyze_fixed_point_stability(dynamics_matrix: np.ndarray,
                                  fixed_point: np.ndarray) -> Dict:
    """
    Analyze stability of a fixed point.
    
    Args:
        dynamics_matrix: Jacobian at fixed point
        fixed_point: The fixed point coordinates
        
    Returns:
        Dictionary with stability classification
    """
    eigenvalues = linalg.eigvals(dynamics_matrix)
    
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Classify fixed point
    max_real = np.max(real_parts)
    has_complex = np.any(np.abs(imag_parts) > 1e-6)
    
    if max_real < -1e-6:
        if has_complex:
            stability_type = 'stable_spiral'
        else:
            stability_type = 'stable_node'
    elif max_real > 1e-6:
        if has_complex:
            stability_type = 'unstable_spiral'
        else:
            stability_type = 'unstable_node'
    else:
        stability_type = 'saddle_point'
    
    return {
        'stability_type': stability_type,
        'eigenvalues': eigenvalues,
        'max_real_eigenvalue': max_real,
        'is_stable': max_real < 0,
        'has_oscillations': has_complex
    }


def comprehensive_stability_analysis(trajectory: np.ndarray,
                                    reference: Optional[np.ndarray] = None,
                                    perturbation_time: Optional[int] = None) -> Dict:
    """
    Comprehensive stability analysis of neural dynamics.
    
    Args:
        trajectory: State trajectory [time x dimensions]
        reference: Reference trajectory (unperturbed)
        perturbation_time: When perturbation occurred
        
    Returns:
        Dictionary with all stability metrics
    """
    results = {}
    
    # 1. Lyapunov spectrum
    results['lyapunov_exponents'] = compute_lyapunov_spectrum(trajectory)
    results['max_lyapunov'] = results['lyapunov_exponents'][0]
    results['is_chaotic'] = results['max_lyapunov'] > 0
    
    # 2. Trajectory stability
    stability = compute_trajectory_stability(trajectory, reference)
    results.update({
        'trajectory_stability': stability,
        'is_stable': stability['is_stable']
    })
    
    # 3. Recovery from perturbation (if applicable)
    if reference is not None and perturbation_time is not None:
        recovery = measure_perturbation_recovery(
            reference, trajectory, perturbation_time
        )
        results['perturbation_recovery'] = recovery
        results['recovery_percentage'] = recovery['recovery_percentage']
    
    return results
