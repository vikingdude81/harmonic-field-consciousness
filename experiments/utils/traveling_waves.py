"""
Traveling Wave Detection and Analysis

Implements methods for detecting and analyzing traveling waves of activity
across spatial networks.

Based on:
- Batabyal et al. (2025) JOCN - Correspondence between rotations and traveling waves
- Muller et al. (2018) Nature Reviews Neuroscience - Traveling waves review
- Patel et al. (2012) Neuron - Traveling waves in hippocampus

Key concepts:
- Traveling waves reflect spatiotemporal organization of activity
- Correspond to rotational dynamics in state space
- Speed correlates with rotational angular velocity
"""

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple, Optional, List
import networkx as nx


def compute_optical_flow(activity: np.ndarray, 
                        positions: np.ndarray,
                        time_window: int = 5) -> Dict:
    """
    Compute optical flow of activity across spatial positions.
    
    Uses Lucas-Kanade-like method adapted for discrete spatial positions.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions of nodes [nodes x spatial_dims]
        time_window: Window size for computing flow
        
    Returns:
        Dictionary with flow vectors and principal direction
    """
    T, N = activity.shape
    n_dims = positions.shape[1]
    
    # Initialize flow vectors
    flows = np.zeros((T - time_window, N, n_dims))
    
    for t in range(T - time_window):
        # Get activity gradient over time
        dt_activity = activity[t + time_window] - activity[t]
        
        # For each node, estimate flow from its neighbors
        for i in range(N):
            # Find neighboring nodes
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbors = np.argsort(distances)[1:min(6, N)]  # Top 5 nearest neighbors
            
            if len(neighbors) == 0:
                continue
            
            # Compute spatial gradient from neighbors
            dx_activity = np.zeros(n_dims)
            for dim in range(n_dims):
                # Fit linear gradient in this dimension
                neighbor_positions = positions[neighbors, dim] - positions[i, dim]
                neighbor_activities = activity[t, neighbors] - activity[t, i]
                
                if np.std(neighbor_positions) > 1e-6:
                    slope, _ = np.polyfit(neighbor_positions, neighbor_activities, 1)
                    dx_activity[dim] = slope
            
            # Flow = -dt/dx (negative because wave moves opposite to gradient)
            if np.linalg.norm(dx_activity) > 1e-6:
                flows[t, i] = -dt_activity[i] * dx_activity / (np.linalg.norm(dx_activity)**2 + 1e-6)
    
    # Compute principal direction at each time point
    principal_directions = np.zeros((T - time_window, n_dims))
    angular_deviations = np.zeros(T - time_window - 1)
    
    for t in range(T - time_window):
        # Principal direction = mean flow direction
        mean_flow = np.mean(flows[t], axis=0)
        if np.linalg.norm(mean_flow) > 1e-6:
            principal_directions[t] = mean_flow / np.linalg.norm(mean_flow)
        
        # Compute angular deviation from previous time
        if t > 0:
            prev_dir = principal_directions[t-1]
            curr_dir = principal_directions[t]
            if np.linalg.norm(prev_dir) > 0 and np.linalg.norm(curr_dir) > 0:
                cos_angle = np.dot(prev_dir, curr_dir)
                cos_angle = np.clip(cos_angle, -1, 1)
                angular_deviations[t-1] = np.degrees(np.arccos(cos_angle))
    
    return {
        'flows': flows,
        'principal_directions': principal_directions,
        'angular_deviations': angular_deviations,
        'mean_angular_deviation': np.mean(angular_deviations)
    }


def detect_traveling_wave_correlation(activity: np.ndarray,
                                     positions: np.ndarray,
                                     time_lag: int = 5) -> Dict:
    """
    Detect traveling waves using correlation between activation latency and distance.
    
    Method from Patel et al. (2012) - simpler and more robust than optical flow.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions [nodes x spatial_dims]
        time_lag: Maximum time lag to consider
        
    Returns:
        Dictionary with correlation coefficient and wave properties
    """
    T, N = activity.shape
    
    # 1. Find peak activation time for each node
    peak_times = np.argmax(activity, axis=0)
    
    # 2. Compute pairwise distances between nodes
    distances = squareform(pdist(positions))
    
    # 3. Compute pairwise latency differences
    latency_diffs = np.abs(peak_times[:, None] - peak_times[None, :])
    
    # 4. Correlate distance with latency difference
    # Flatten upper triangle (avoid duplicate pairs)
    upper_tri = np.triu_indices(N, k=1)
    dist_flat = distances[upper_tri]
    latency_flat = latency_diffs[upper_tri]
    
    # Filter out zero distances and very large latencies
    valid = (dist_flat > 0) & (latency_flat <= time_lag)
    
    if np.sum(valid) < 10:
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'wave_speed': 0.0,
            'has_wave': False
        }
    
    # Compute correlation
    correlation, p_value = stats.pearsonr(dist_flat[valid], latency_flat[valid])
    
    # Estimate wave speed (distance per time unit)
    if correlation > 0:
        # Fit linear regression to get slope = speed
        slope, _ = np.polyfit(latency_flat[valid], dist_flat[valid], 1)
        wave_speed = slope
    else:
        wave_speed = 0.0
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'wave_speed': wave_speed,
        'has_wave': (correlation > 0.3) and (p_value < 0.05),
        'peak_times': peak_times,
        'distances': dist_flat[valid],
        'latencies': latency_flat[valid]
    }


def compute_wave_speed(activity: np.ndarray,
                      positions: np.ndarray,
                      dt: float = 1.0) -> float:
    """
    Compute traveling wave speed using phase gradient method.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions [nodes x spatial_dims]
        dt: Time step between samples
        
    Returns:
        Wave speed (distance units / time units)
    """
    # Get instantaneous phase using Hilbert transform
    analytic_signal = signal.hilbert(activity, axis=0)
    instantaneous_phase = np.angle(analytic_signal)
    
    # Compute spatial phase gradient at each time point
    T, N = activity.shape
    speeds = []
    
    for t in range(T):
        phases = instantaneous_phase[t]
        
        # Fit plane to phase vs position
        if positions.shape[1] == 2:
            # 2D case
            try:
                # phases = a*x + b*y + c
                X = np.column_stack([positions, np.ones(N)])
                coeffs, _, _, _ = np.linalg.lstsq(X, phases, rcond=None)
                
                # Phase gradient
                grad_phase = coeffs[:2]
                grad_magnitude = np.linalg.norm(grad_phase)
                
                if grad_magnitude > 1e-6:
                    # Speed = frequency / spatial gradient
                    # Use instantaneous frequency
                    freq = np.mean(np.abs(np.diff(phases)))
                    speed = freq / grad_magnitude
                    speeds.append(speed)
            except:
                continue
    
    if len(speeds) == 0:
        return 0.0
    
    return np.median(speeds)


def detect_spiral_waves(activity: np.ndarray,
                       positions: np.ndarray,
                       time_point: int = None) -> Dict:
    """
    Detect spiral/rotating wave patterns in activity.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions [nodes x 2] (requires 2D)
        time_point: Specific time to analyze (if None, averages over time)
        
    Returns:
        Dictionary with spiral detection metrics
    """
    if positions.shape[1] != 2:
        raise ValueError("Spiral detection requires 2D positions")
    
    if time_point is not None:
        act = activity[time_point]
    else:
        # Use time-averaged phase
        analytic = signal.hilbert(activity, axis=0)
        act = np.angle(np.mean(analytic, axis=0))
    
    # Compute center of mass of activity
    weights = act - np.min(act)
    if np.sum(weights) > 0:
        center = np.average(positions, axis=0, weights=weights)
    else:
        center = np.mean(positions, axis=0)
    
    # Compute angles from center
    rel_pos = positions - center
    angles_spatial = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
    
    # Get phase
    if time_point is not None:
        analytic = signal.hilbert(activity[:, :], axis=0)
        phases = np.angle(analytic[time_point])
    else:
        phases = act
    
    # Check if phase correlates with spatial angle (spiral pattern)
    correlation, p_value = stats.pearsonr(angles_spatial, phases)
    
    return {
        'spiral_correlation': correlation,
        'p_value': p_value,
        'is_spiral': np.abs(correlation) > 0.4 and p_value < 0.05,
        'center': center,
        'direction': 'clockwise' if correlation > 0 else 'counterclockwise'
    }


def analyze_wave_correspondence_to_rotation(activity: np.ndarray,
                                           positions: np.ndarray,
                                           rotational_velocity: np.ndarray,
                                           window_size: int = 10) -> Dict:
    """
    Analyze correspondence between traveling wave speed and rotational velocity.
    
    Based on Batabyal et al. (2025) finding that wave speed correlates with
    state-space rotational velocity.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions [nodes x spatial_dims]
        rotational_velocity: Angular velocity from jPCA [time]
        window_size: Window for computing wave speed
        
    Returns:
        Dictionary with correlation metrics
    """
    T = min(len(activity), len(rotational_velocity))
    
    # Compute wave speed in sliding windows
    wave_speeds = []
    rot_vels = []
    
    for t in range(0, T - window_size, window_size // 2):
        window_activity = activity[t:t+window_size]
        
        # Detect wave and get speed
        wave_result = detect_traveling_wave_correlation(
            window_activity, positions, time_lag=window_size//2
        )
        
        if wave_result['has_wave']:
            wave_speeds.append(wave_result['wave_speed'])
            rot_vels.append(np.mean(np.abs(rotational_velocity[t:t+window_size])))
    
    if len(wave_speeds) < 3:
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'correspondence': False
        }
    
    # Correlate wave speed with rotational velocity
    correlation, p_value = stats.pearsonr(wave_speeds, rot_vels)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'correspondence': (correlation > 0.5) and (p_value < 0.05),
        'wave_speeds': np.array(wave_speeds),
        'rotational_velocities': np.array(rot_vels)
    }


def generate_surrogate_data(activity: np.ndarray, 
                           method: str = 'shuffle') -> np.ndarray:
    """
    Generate surrogate data for statistical testing of traveling waves.
    
    Args:
        activity: Original activity matrix [time x nodes]
        method: 'shuffle' (shuffle nodes) or 'phase' (phase randomization)
        
    Returns:
        Surrogate activity with same statistics but no wave structure
    """
    if method == 'shuffle':
        # Shuffle node identities (breaks spatial structure)
        surrogate = activity[:, np.random.permutation(activity.shape[1])]
    
    elif method == 'phase':
        # Phase randomization (preserves power spectrum)
        surrogate = np.zeros_like(activity)
        for i in range(activity.shape[1]):
            fft = np.fft.fft(activity[:, i])
            # Randomize phases but keep amplitudes
            random_phases = np.random.uniform(0, 2*np.pi, len(fft))
            fft_random = np.abs(fft) * np.exp(1j * random_phases)
            surrogate[:, i] = np.real(np.fft.ifft(fft_random))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return surrogate


def comprehensive_wave_analysis(activity: np.ndarray,
                                positions: np.ndarray,
                                graph: Optional[nx.Graph] = None) -> Dict:
    """
    Comprehensive traveling wave analysis.
    
    Args:
        activity: Activity matrix [time x nodes]
        positions: Spatial positions [nodes x spatial_dims]
        graph: Optional network graph for topology-aware analysis
        
    Returns:
        Dictionary with all wave metrics
    """
    # 1. Correlation-based detection
    wave_corr = detect_traveling_wave_correlation(activity, positions)
    
    # 2. Optical flow
    optical_flow = compute_optical_flow(activity, positions)
    
    # 3. Wave speed
    speed = compute_wave_speed(activity, positions)
    
    # 4. Spiral detection (if 2D)
    spiral_result = None
    if positions.shape[1] == 2:
        try:
            spiral_result = detect_spiral_waves(activity, positions)
        except:
            pass
    
    # 5. Compare to surrogate
    surrogate = generate_surrogate_data(activity, method='shuffle')
    surrogate_wave = detect_traveling_wave_correlation(surrogate, positions)
    
    results = {
        'has_traveling_wave': wave_corr['has_wave'],
        'wave_correlation': wave_corr['correlation'],
        'wave_p_value': wave_corr['p_value'],
        'wave_speed': speed,
        'mean_angular_deviation': optical_flow['mean_angular_deviation'],
        'surrogate_correlation': surrogate_wave['correlation'],
        'wave_strength': wave_corr['correlation'] - surrogate_wave['correlation'],
        'optical_flow': optical_flow
    }
    
    if spiral_result is not None:
        results['spiral_detection'] = spiral_result
    
    return results
