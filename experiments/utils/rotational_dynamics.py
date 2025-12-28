"""
Rotational Dynamics Analysis

Implements jPCA (jittered Principal Component Analysis) and related methods
for analyzing rotational dynamics in neural population activity.

Based on:
- Churchland et al. (2012) Nature - Original jPCA paper
- Batabyal et al. (2025) JOCN - Rotational dynamics in working memory

Key concepts:
- Rotational dynamics reflect sequential activation patterns
- Fuller rotations correlate with better cognitive performance
- Correspond to traveling waves across cortical surface
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Dict, Optional
import warnings


def jpca(data: np.ndarray, n_components: int = 6) -> Dict:
    """
    Perform jittered Principal Component Analysis (jPCA).
    
    jPCA finds the plane of maximum rotational dynamics by looking for
    skew-symmetric structure in the dynamics matrix.
    
    Args:
        data: Neural activity data [time x neurons] or [trials x time x neurons]
        n_components: Number of jPCA components to extract (should be even)
        
    Returns:
        Dictionary containing:
            - jpcs: jPC projection matrix [neurons x n_components]
            - projections: Data projected onto jPCs [time x n_components]
            - rotation_quality: R² measure of rotational quality
            - eigenvalues: Eigenvalues of skew-symmetric dynamics
            - M_skew: Skew-symmetric dynamics matrix
    """
    # Ensure n_components is even for rotation planes
    if n_components % 2 != 0:
        n_components += 1
        warnings.warn(f"n_components must be even for rotation planes, using {n_components}")
    
    # Handle trial-averaged data
    if data.ndim == 3:
        data = np.mean(data, axis=0)  # Average across trials
    
    T, N = data.shape
    
    # 1. Get regular PCA to reduce dimensionality first
    data_centered = data - np.mean(data, axis=0)
    
    # Compute velocity (derivative)
    dt = 1  # Assume unit time steps
    velocity = np.diff(data_centered, axis=0) / dt
    data_for_dynamics = data_centered[:-1]  # Match velocity length
    
    # 2. Estimate dynamics matrix: dX/dt = M * X
    # Using least squares: M = (dX/dt) * X^T * (X * X^T)^-1
    try:
        M = velocity.T @ data_for_dynamics @ linalg.pinv(data_for_dynamics.T @ data_for_dynamics)
    except linalg.LinAlgError:
        # If singular, use more robust pseudo-inverse
        M = velocity.T @ linalg.pinv(data_for_dynamics.T)
    
    # 3. Extract skew-symmetric component
    M_skew = (M - M.T) / 2
    
    # 4. Find eigenvectors of M_skew (these are the jPCs)
    eigenvalues, eigenvectors = linalg.eig(M_skew)
    
    # Sort by magnitude of imaginary part (rotation frequency)
    idx = np.argsort(np.abs(np.imag(eigenvalues)))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top components (real parts of eigenvectors)
    jpcs = np.real(eigenvectors[:, :n_components])
    
    # 5. Project data onto jPCs
    projections = data_centered @ jpcs
    
    # 6. Compute rotation quality (R² variance explained by rotational dynamics)
    rotation_quality = compute_rotation_quality(projections, velocity @ jpcs)
    
    return {
        'jpcs': jpcs,
        'projections': projections,
        'rotation_quality': rotation_quality,
        'eigenvalues': eigenvalues[:n_components],
        'M_skew': M_skew,
        'data_centered': data_centered
    }


def compute_rotation_quality(projections: np.ndarray, velocity_proj: np.ndarray) -> float:
    """
    Compute R² measure of how well dynamics are captured by rotation.
    
    For perfect rotation, velocity should be orthogonal to position.
    
    Args:
        projections: Position in jPC space [time x components]
        velocity_proj: Velocity in jPC space [time x components]
        
    Returns:
        R²_d: Rotation quality metric (higher = more rotational)
    """
    # Match dimensions
    min_len = min(len(projections), len(velocity_proj))
    projections = projections[:min_len]
    velocity_proj = velocity_proj[:min_len]
    
    # For rotation, velocity should be perpendicular to position
    # Compute angle between velocity and position vectors
    dots = np.sum(projections * velocity_proj, axis=1)
    pos_norms = np.linalg.norm(projections, axis=1)
    vel_norms = np.linalg.norm(velocity_proj, axis=1)
    
    # Avoid division by zero
    valid = (pos_norms > 0) & (vel_norms > 0)
    if not np.any(valid):
        return 0.0
    
    cos_angles = dots[valid] / (pos_norms[valid] * vel_norms[valid])
    
    # R²_d = variance explained by orthogonal component
    r_squared = 1 - np.mean(cos_angles**2)
    
    return max(0, r_squared)  # Ensure non-negative


def compute_rotation_angle(trajectory: np.ndarray, plane_dims: Tuple[int, int] = (0, 1)) -> float:
    """
    Compute total angle traversed by a trajectory in a 2D plane.
    
    Args:
        trajectory: Neural trajectory [time x dimensions]
        plane_dims: Which two dimensions define the rotation plane
        
    Returns:
        Total angle traversed in degrees
    """
    # Extract 2D trajectory
    traj_2d = trajectory[:, list(plane_dims)]
    
    # Compute angles relative to origin
    angles = np.arctan2(traj_2d[:, 1], traj_2d[:, 0])
    
    # Unwrap to get cumulative angle
    angles_unwrapped = np.unwrap(angles)
    
    # Total rotation is difference between first and last
    total_rotation = np.abs(angles_unwrapped[-1] - angles_unwrapped[0])
    
    return np.degrees(total_rotation)


def compute_angular_velocity(trajectory: np.ndarray, dt: float = 1.0,
                            plane_dims: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Compute instantaneous angular velocity of a trajectory.
    
    Args:
        trajectory: Neural trajectory [time x dimensions]
        dt: Time step between samples
        plane_dims: Which two dimensions define the rotation plane
        
    Returns:
        Angular velocity at each time point [degrees/time]
    """
    # Extract 2D trajectory
    traj_2d = trajectory[:, list(plane_dims)]
    
    # Compute angles
    angles = np.arctan2(traj_2d[:, 1], traj_2d[:, 0])
    angles_unwrapped = np.unwrap(angles)
    
    # Compute derivative
    angular_vel = np.gradient(angles_unwrapped, dt)
    
    return np.degrees(angular_vel)


def compute_trajectory_circularity(trajectory: np.ndarray, 
                                  plane_dims: Tuple[int, int] = (0, 1)) -> float:
    """
    Measure how circular a trajectory is (vs elliptical or irregular).
    
    Fits an ellipse and computes ratio of minor to major axis.
    
    Args:
        trajectory: Neural trajectory [time x dimensions]
        plane_dims: Which two dimensions define the rotation plane
        
    Returns:
        Circularity score [0-1], where 1 is perfect circle
    """
    # Extract 2D trajectory and center it
    traj_2d = trajectory[:, list(plane_dims)]
    traj_centered = traj_2d - np.mean(traj_2d, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(traj_centered.T)
    
    # Eigenvalues of covariance give squared semi-axes lengths
    eigenvalues = linalg.eigvalsh(cov)
    
    # Avoid division by zero
    if eigenvalues[-1] == 0:
        return 0.0
    
    # Circularity = ratio of smaller to larger axis
    circularity = np.sqrt(eigenvalues[0] / eigenvalues[-1])
    
    return circularity


def compute_recovery_percentage(trajectory: np.ndarray, 
                               pre_perturbation_idx: int = 0,
                               post_perturbation_start: int = None) -> float:
    """
    Compute how much a trajectory recovers to its pre-perturbation state.
    
    Based on Batabyal et al. (2025) - measures partial recovery after distraction.
    
    Args:
        trajectory: Neural trajectory [time x dimensions]
        pre_perturbation_idx: Index of state before perturbation
        post_perturbation_start: Index where recovery begins (if None, uses middle)
        
    Returns:
        Recovery percentage [0-100], where 100 is complete recovery
    """
    if post_perturbation_start is None:
        post_perturbation_start = len(trajectory) // 2
    
    # Reference state (before perturbation)
    ref_state = trajectory[pre_perturbation_idx]
    
    # Initial distance (peak of perturbation)
    perturbation_states = trajectory[pre_perturbation_idx+1:post_perturbation_start]
    if len(perturbation_states) == 0:
        return 0.0
    
    distances = np.linalg.norm(perturbation_states - ref_state, axis=1)
    max_distance = np.max(distances)
    
    if max_distance == 0:
        return 100.0
    
    # Final distance (end of trajectory)
    final_distance = np.linalg.norm(trajectory[-1] - ref_state)
    
    # Recovery percentage
    recovery = (1 - final_distance / max_distance) * 100
    
    return np.clip(recovery, 0, 100)


def analyze_rotational_dynamics(data: np.ndarray, 
                               perturbation_time: Optional[int] = None,
                               n_jpcs: int = 6) -> Dict:
    """
    Comprehensive analysis of rotational dynamics in neural data.
    
    Performs jPCA and computes multiple rotation-related metrics.
    
    Args:
        data: Neural activity [time x neurons] or [trials x time x neurons]
        perturbation_time: Time point of perturbation (if applicable)
        n_jpcs: Number of jPCA components
        
    Returns:
        Dictionary with comprehensive rotation metrics
    """
    # Perform jPCA
    jpca_result = jpca(data, n_components=n_jpcs)
    projections = jpca_result['projections']
    
    # Compute rotation metrics on first 2 jPCs (primary rotation plane)
    total_angle = compute_rotation_angle(projections, plane_dims=(0, 1))
    angular_velocity = compute_angular_velocity(projections, plane_dims=(0, 1))
    circularity = compute_trajectory_circularity(projections, plane_dims=(0, 1))
    
    results = {
        **jpca_result,
        'total_rotation_degrees': total_angle,
        'angular_velocity': angular_velocity,
        'mean_angular_velocity': np.mean(np.abs(angular_velocity)),
        'peak_angular_velocity': np.max(np.abs(angular_velocity)),
        'circularity': circularity,
    }
    
    # If perturbation time is provided, compute recovery metrics
    if perturbation_time is not None and perturbation_time > 0:
        recovery = compute_recovery_percentage(
            projections, 
            pre_perturbation_idx=perturbation_time-1,
            post_perturbation_start=perturbation_time
        )
        results['recovery_percentage'] = recovery
    
    return results


def compare_rotational_dynamics(data1: np.ndarray, data2: np.ndarray,
                               labels: Tuple[str, str] = ('Condition 1', 'Condition 2'),
                               n_jpcs: int = 6) -> Dict:
    """
    Compare rotational dynamics between two conditions.
    
    Useful for comparing correct vs error trials, different brain states, etc.
    
    Args:
        data1: Neural activity for condition 1 [time x neurons]
        data2: Neural activity for condition 2 [time x neurons]
        labels: Names for the two conditions
        n_jpcs: Number of jPCA components
        
    Returns:
        Dictionary with comparative metrics
    """
    results1 = analyze_rotational_dynamics(data1, n_jpcs=n_jpcs)
    results2 = analyze_rotational_dynamics(data2, n_jpcs=n_jpcs)
    
    comparison = {
        labels[0]: {
            'rotation_quality': results1['rotation_quality'],
            'total_angle': results1['total_rotation_degrees'],
            'circularity': results1['circularity'],
            'mean_angular_velocity': results1['mean_angular_velocity']
        },
        labels[1]: {
            'rotation_quality': results2['rotation_quality'],
            'total_angle': results2['total_rotation_degrees'],
            'circularity': results2['circularity'],
            'mean_angular_velocity': results2['mean_angular_velocity']
        },
        'differences': {
            'rotation_quality_diff': results1['rotation_quality'] - results2['rotation_quality'],
            'total_angle_diff': results1['total_rotation_degrees'] - results2['total_rotation_degrees'],
            'circularity_diff': results1['circularity'] - results2['circularity'],
            'velocity_diff': results1['mean_angular_velocity'] - results2['mean_angular_velocity']
        }
    }
    
    return comparison
