"""
Multiscale Encoder for Harmonic Field Consciousness Model

Implements encoder that handles different harmonic mode frequencies and
temporal resolutions for consciousness state modeling.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import signal


class MultiscaleEncoder:
    """
    Encoder for multiscale harmonic mode analysis.
    
    Handles different temporal resolutions and sampling rates across
    harmonic modes, enabling real-time consciousness state decoding.
    """
    
    def __init__(self, n_modes: int, scales: List[int] = None):
        """
        Initialize multiscale encoder.
        
        Args:
            n_modes: Number of harmonic modes
            scales: List of temporal scales (in samples). If None, uses [1, 2, 4, 8]
        """
        self.n_modes = n_modes
        self.scales = scales if scales is not None else [1, 2, 4, 8]
        self.n_scales = len(self.scales)
        
        # Initialize encoder weights for each scale
        self.scale_weights = {}
        for scale in self.scales:
            # Each scale has different importance for different modes
            # Lower modes are captured better at larger scales (slower dynamics)
            # Higher modes are captured better at smaller scales (faster dynamics)
            weights = np.exp(-np.arange(n_modes) / (n_modes / scale))
            weights = weights / np.sum(weights)
            self.scale_weights[scale] = weights
    
    def encode(self, time_series: np.ndarray, mode_indices: np.ndarray = None) -> Dict[int, np.ndarray]:
        """
        Encode time series at multiple temporal scales.
        
        Args:
            time_series: Input time series (n_modes, n_timepoints)
            mode_indices: Optional indices of modes to encode. If None, encode all.
        
        Returns:
            Dictionary mapping scale -> encoded representation
        """
        if mode_indices is None:
            mode_indices = np.arange(self.n_modes)
        
        n_timepoints = time_series.shape[1] if time_series.ndim > 1 else len(time_series)
        
        encoded = {}
        for scale in self.scales:
            if time_series.ndim == 1:
                # Single mode
                downsampled = signal.decimate(time_series, scale, axis=0) if scale > 1 else time_series
            else:
                # Multiple modes
                downsampled = []
                for i in mode_indices:
                    mode_signal = time_series[i, :]
                    if scale > 1 and len(mode_signal) >= scale * 2:
                        ds = signal.decimate(mode_signal, scale, axis=0)
                    else:
                        ds = mode_signal[::scale]
                    downsampled.append(ds)
                downsampled = np.array(downsampled)
            
            encoded[scale] = downsampled
        
        return encoded
    
    def compute_multiscale_power(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute power spectrum at multiple scales and combine.
        
        Args:
            time_series: Input time series (n_modes, n_timepoints)
        
        Returns:
            Multiscale power distribution across modes
        """
        if time_series.ndim == 1:
            time_series = time_series.reshape(1, -1)
        
        n_modes = time_series.shape[0]
        power_combined = np.zeros(n_modes)
        
        for i, mode_signal in enumerate(time_series):
            mode_power = 0.0
            total_weight = 0.0
            
            for scale in self.scales:
                weight = self.scale_weights[scale][i]
                
                # Compute power at this scale
                if scale > 1 and len(mode_signal) >= scale * 2:
                    downsampled = signal.decimate(mode_signal, scale, axis=0)
                else:
                    downsampled = mode_signal[::scale]
                
                scale_power = np.mean(downsampled ** 2)
                mode_power += weight * scale_power
                total_weight += weight
            
            power_combined[i] = mode_power / total_weight if total_weight > 0 else 0.0
        
        return power_combined
    
    def handle_missing_data(self, data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct missing data using multiscale interpolation.
        
        Args:
            data: Input data with missing values (n_modes, n_timepoints)
            missing_mask: Boolean mask indicating missing data (True = missing)
        
        Returns:
            Reconstructed data with filled-in values
        """
        reconstructed = data.copy()
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            missing_mask = missing_mask.reshape(1, -1)
            single_mode = True
        else:
            single_mode = False
        
        for i in range(data.shape[0]):
            mode_data = data[i, :]
            mode_mask = missing_mask[i, :]
            
            if not np.any(mode_mask):
                continue
            
            # Use neighboring scales for interpolation
            valid_indices = np.where(~mode_mask)[0]
            missing_indices = np.where(mode_mask)[0]
            
            if len(valid_indices) == 0:
                # All data missing, use zeros
                reconstructed[i, missing_indices] = 0.0
                continue
            
            # Interpolate from valid data
            interpolated = np.interp(missing_indices, valid_indices, mode_data[valid_indices])
            reconstructed[i, missing_indices] = interpolated
        
        if single_mode:
            reconstructed = reconstructed.flatten()
        
        return reconstructed
    
    def extract_features(self, encoded_data: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Extract multiscale features from encoded data.
        
        Args:
            encoded_data: Dictionary of encoded representations at different scales
        
        Returns:
            Feature vector combining all scales
        """
        features = []
        
        for scale in sorted(encoded_data.keys()):
            scale_data = encoded_data[scale]
            
            if scale_data.ndim == 1:
                # Single mode case
                features.extend([
                    np.mean(scale_data),
                    np.std(scale_data),
                    np.max(np.abs(scale_data))
                ])
            else:
                # Multiple modes
                features.extend([
                    np.mean(scale_data),
                    np.std(scale_data),
                    np.mean(np.abs(scale_data), axis=1).mean(),
                    np.max(scale_data) - np.min(scale_data)
                ])
        
        return np.array(features)
