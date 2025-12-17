"""
Real-time Decoder for Consciousness State Prediction

Implements recursive decoding for real-time consciousness state (C(t)) prediction
from streaming harmonic mode data.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import time


class RealtimeDecoder:
    """
    Real-time decoder for consciousness state prediction.
    
    Performs recursive decoding with minimal latency for streaming
    harmonic mode analysis.
    """
    
    def __init__(self, n_modes: int, buffer_size: int = 100):
        """
        Initialize real-time decoder.
        
        Args:
            n_modes: Number of harmonic modes
            buffer_size: Size of temporal buffer for streaming analysis
        """
        self.n_modes = n_modes
        self.buffer_size = buffer_size
        
        # Streaming buffer for mode amplitudes
        self.buffer = np.zeros((n_modes, buffer_size))
        self.buffer_index = 0
        self.buffer_full = False
        
        # Running statistics for incremental computation
        self.running_mean = np.zeros(n_modes)
        self.running_var = np.zeros(n_modes)
        self.n_samples = 0
        
        # Timing for latency benchmarking
        self.latencies = []
    
    def update(self, new_sample: np.ndarray) -> Dict[str, float]:
        """
        Update decoder with new sample and predict consciousness state.
        
        Args:
            new_sample: New harmonic mode amplitudes (n_modes,)
        
        Returns:
            Dictionary of consciousness metrics computed in real-time
        """
        start_time = time.time()
        
        # Update buffer
        self.buffer[:, self.buffer_index] = new_sample
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        
        if self.buffer_index == 0:
            self.buffer_full = True
        
        # Update running statistics incrementally
        self.n_samples += 1
        delta = new_sample - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = new_sample - self.running_mean
        self.running_var += delta * delta2
        
        # Compute consciousness metrics
        metrics = self._compute_metrics_incremental(new_sample)
        
        # Record latency
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.latencies.append(latency)
        
        metrics['latency_ms'] = latency
        metrics['mean_latency_ms'] = np.mean(self.latencies[-100:])  # Last 100 samples
        
        return metrics
    
    def _compute_metrics_incremental(self, current_sample: np.ndarray) -> Dict[str, float]:
        """
        Compute consciousness metrics incrementally.
        
        Args:
            current_sample: Current harmonic mode amplitudes
        
        Returns:
            Dictionary of metrics (H_mode, PR, etc.)
        """
        metrics = {}
        
        # Get current buffer data
        if self.buffer_full:
            buffer_data = self.buffer
        else:
            buffer_data = self.buffer[:, :self.buffer_index]
        
        if buffer_data.shape[1] == 0:
            return metrics
        
        # Compute power distribution
        power = np.mean(buffer_data ** 2, axis=1)
        power = power / (np.sum(power) + 1e-12)
        
        # Mode Entropy (H_mode)
        p = power[power > 1e-12]
        H_mode = -np.sum(p * np.log(p + 1e-12))
        H_max = np.log(len(power))
        metrics['H_mode'] = H_mode / (H_max + 1e-12)
        
        # Participation Ratio (PR)
        PR = 1.0 / (np.sum(power ** 2) + 1e-12)
        metrics['PR'] = PR / self.n_modes  # Normalize by number of modes
        
        # Phase Coherence (R) - using current sample
        phases = np.angle(current_sample + 1e-12)
        mean_phase = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
        R = np.abs(np.mean(np.exp(1j * (phases - mean_phase))))
        metrics['R'] = R
        
        # Entropy Production Rate (estimated from variance)
        if self.n_samples > 1:
            variance = self.running_var / (self.n_samples - 1)
            entropy_rate = 0.5 * np.log(2 * np.pi * np.e * np.mean(variance) + 1e-12)
            metrics['S_dot'] = max(0.0, entropy_rate)
        else:
            metrics['S_dot'] = 0.0
        
        # Criticality Index (κ) - simplified for real-time
        if buffer_data.shape[1] > 1:
            fluctuations = np.std(buffer_data, axis=1)
            mean_activity = np.mean(buffer_data, axis=1)
            kappa = np.mean(fluctuations / (mean_activity + 1e-12))
            metrics['kappa'] = np.clip(kappa, 0, 2)
        else:
            metrics['kappa'] = 1.0
        
        # Overall consciousness functional C(t)
        # Weighted combination of metrics
        C_t = (0.3 * metrics.get('H_mode', 0) + 
               0.25 * metrics.get('PR', 0) +
               0.25 * metrics.get('R', 0) +
               0.1 * min(metrics.get('S_dot', 0), 1.0) +
               0.1 * metrics.get('kappa', 0) / 2.0)
        metrics['C_t'] = C_t
        
        return metrics
    
    def predict_state(self, metrics: Dict[str, float]) -> str:
        """
        Predict consciousness state from metrics.
        
        Args:
            metrics: Dictionary of consciousness metrics
        
        Returns:
            Predicted state label ('wake', 'nrem', 'rem', 'anesthesia')
        """
        C_t = metrics.get('C_t', 0)
        H_mode = metrics.get('H_mode', 0)
        PR = metrics.get('PR', 0)
        
        # Simple threshold-based classification
        if C_t > 0.6 and H_mode > 0.6:
            return 'wake'
        elif C_t < 0.3 and H_mode < 0.3:
            return 'anesthesia'
        elif C_t > 0.4 and H_mode > 0.4:
            return 'rem'
        else:
            return 'nrem'
    
    def reset(self):
        """Reset decoder state."""
        self.buffer = np.zeros((self.n_modes, self.buffer_size))
        self.buffer_index = 0
        self.buffer_full = False
        self.running_mean = np.zeros(self.n_modes)
        self.running_var = np.zeros(self.n_modes)
        self.n_samples = 0
        self.latencies = []
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency statistics
        """
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        }
