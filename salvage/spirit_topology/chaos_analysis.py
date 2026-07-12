"""
Chaos Analysis - Harmonic Field Chaos Metrics.

This module provides chaos analysis patterns adapted from helios-trajectory-analysis,
applied to spirit topology and consciousness emergence.
"""

import numpy as np
from typing import List, Tuple

class ChaosAnalysis:
    """Chaos analysis for spiritual/consciousness systems."""
    
    def __init__(self):
        self.embedding_dim = 3
        self.delay_time = 1
    
    def compute_lyapunov_exponent(self, time_series: np.ndarray) -> float:
        """Compute Lyapunov exponent for a time series.
        
        Args:
            time_series: 1D array of observations
            
        Returns:
            Largest Lyapunov exponent (positive = chaotic)
        """
        if len(time_series) < 2 * self.embedding_dim:
            return 0.0
        
        # Reconstruct phase space
        reconstructed = self._reconstruct_phase_space(time_series)
        
        # Compute divergence rates
        exponents = []
        for i in range(len(reconstructed) - self.embedding_dim):
            x1 = reconstructed[i]
            x2 = reconstructed[i + self.embedding_dim]
            
            if np.linalg.norm(x2 - x1) > 0:
                divergence = np.log(np.linalg.norm(x2 - x1))
                exponents.append(divergence)
        
        if not exponents:
            return 0.0
        
        return np.mean(exponents)
    
    def _reconstruct_phase_space(self, time_series: np.ndarray) -> np.ndarray:
        """Reconstruct phase space using delay embedding."""
        n = len(time_series)
        reconstructed = np.zeros((n - self.delay_time * (self.embedding_dim - 1), self.embedding_dim))
        
        for i in range(n):
            idx = i % (n - self.delay_time * (self.embedding_dim - 1))
            reconstructed[idx, 0] = time_series[i]
        
        return reconstructed
    
    def compute_correlation_dimension(self, time_series: np.ndarray) -> float:
        """Compute correlation dimension (Grassberger-Procaccia algorithm).
        
        Args:
            time_series: 1D array of observations
            
        Returns:
            Correlation dimension estimate
        """
        if len(time_series) < 50:
            return 0.0
        
        # Compute pairwise distances
        n = len(time_series)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = abs(time_series[i] - time_series[j])
                distances[j, i] = distances[i, j]
        
        # Sort distances
        sorted_distances = np.sort(distances.flatten())
        
        # Compute correlation integral C(epsilon)
        epsilons = np.logspace(-2, 0, 10)
        correlations = []
        
        for epsilon in epsilons:
            count = np.sum(sorted_distances < epsilon)
            if count > 0 and count < n * (n - 1):
                correlations.append(np.log(count) / np.log(n * (n - 1)))
        
        # Slope of log-log plot
        if len(correlations) >= 2:
            x = np.log(epsilons[:len(correlations)])
            y = correlations
            slope, _, _, _ = np.polyfit(x, y, 1)
            return max(0.0, min(slope, self.embedding_dim))
        
        return 0.0
    
    def detect_attractor_dimension(self, time_series: np.ndarray) -> Tuple[float, str]:
        """Detect attractor dimension and type.
        
        Args:
            time_series: 1D array of observations
            
        Returns:
            Tuple of (dimension, attractor_type)
        """
        lyapunov = self.compute_lyapunov_exponent(time_series)
        correlation_dim = self.compute_correlation_dimension(time_series)
        
        if lyapunov > 0.1:
            return (correlation_dim, 'chaotic')
        elif correlation_dim > 2:
            return (correlation_dim, 'strange_attractor')
        else:
            return (correlation_dim, 'periodic_or_fixed_point')


def analyze_spirit_chaos(spirits: List[np.ndarray]) -> Dict:
    """Analyze chaos properties of a spirit ensemble.
    
    Args:
        spirits: List of spirit attribute vectors
        
    Returns:
        Chaos analysis results
    """
    if not spirits:
        return {}
    
    # Convert to time series (first dimension as temporal proxy)
    time_series = np.array([s[0] for s in spirits])
    
    chaos = ChaosAnalysis()
    lyapunov = chaos.compute_lyapunov_exponent(time_series)
    correlation_dim = chaos.compute_correlation_dimension(time_series)
    attractor_info = chaos.detect_attractor_dimension(time_series)
    
    return {
        'lyapunov_exponent': lyapunov,
        'correlation_dimension': correlation_dim,
        'attractor_type': attractor_info[1],
        'is_chaos': lyapunov > 0
    }


# Example usage
if __name__ == "__main__":
    # Sample spirit entropy values as time series
    entropy_series = np.array([0.72, 0.90, 0.72, 0.44, 1.00, 0.50, 0.54, 0.54, 0.60, 0.60, 0.46])
    
    chaos = ChaosAnalysis()
    lyapunov = chaos.compute_lyapunov_exponent(entropy_series)
    print(f"Lyapunov Exponent: {lyapunov}")
    
    correlation_dim = chaos.compute_correlation_dimension(entropy_series)
    print(f"Correlation Dimension: {correlation_dim}")
