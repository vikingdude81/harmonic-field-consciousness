"""
Spike Encoder for Harmonic Modes

Converts harmonic mode amplitudes to spike trains and vice versa.
"""

import numpy as np
from typing import Tuple


class SpikeEncoder:
    """
    Encoder for converting between continuous harmonic amplitudes and spike trains.
    
    Supports both rate coding and temporal coding schemes.
    """
    
    def __init__(self, encoding: str = 'rate', dt: float = 0.1):
        """
        Initialize spike encoder.
        
        Args:
            encoding: Encoding scheme ('rate' or 'temporal')
            dt: Time step (ms)
        """
        self.encoding = encoding
        self.dt = dt
    
    def encode_rate(self, amplitudes: np.ndarray, duration: float = 100.0) -> np.ndarray:
        """
        Encode amplitudes as spike rates (Poisson process).
        
        Args:
            amplitudes: Harmonic mode amplitudes (n_modes,)
            duration: Duration of spike train (ms)
        
        Returns:
            Spike trains (n_modes, n_timesteps)
        """
        n_modes = len(amplitudes)
        n_steps = int(duration / self.dt)
        
        # Convert amplitudes to firing rates (scale to reasonable range)
        max_rate = 100.0  # Maximum firing rate in Hz
        rates = np.abs(amplitudes) / (np.max(np.abs(amplitudes)) + 1e-12) * max_rate
        
        # Generate Poisson spikes
        spike_trains = np.zeros((n_modes, n_steps), dtype=int)
        
        for i, rate in enumerate(rates):
            # Poisson probability
            p_spike = rate * (self.dt / 1000.0)  # Convert to probability per time step
            spike_trains[i, :] = np.random.rand(n_steps) < p_spike
        
        return spike_trains
    
    def encode_temporal(self, amplitudes: np.ndarray, duration: float = 100.0) -> np.ndarray:
        """
        Encode amplitudes as spike timing (time-to-first-spike).
        
        Args:
            amplitudes: Harmonic mode amplitudes (n_modes,)
            duration: Duration of spike train (ms)
        
        Returns:
            Spike trains (n_modes, n_timesteps)
        """
        n_modes = len(amplitudes)
        n_steps = int(duration / self.dt)
        
        spike_trains = np.zeros((n_modes, n_steps), dtype=int)
        
        # Larger amplitude -> earlier spike
        normalized_amps = np.abs(amplitudes) / (np.max(np.abs(amplitudes)) + 1e-12)
        
        for i, amp in enumerate(normalized_amps):
            if amp > 0.01:  # Threshold
                # Spike time inversely proportional to amplitude
                spike_time = int((1 - amp) * n_steps * 0.9)
                if spike_time < n_steps:
                    spike_trains[i, spike_time] = 1
        
        return spike_trains
    
    def decode_rate(self, spike_trains: np.ndarray, window: float = 100.0) -> np.ndarray:
        """
        Decode spike trains to amplitudes using rate coding.
        
        Args:
            spike_trains: Spike trains (n_modes, n_timesteps)
            window: Sliding window for rate estimation (ms)
        
        Returns:
            Decoded amplitudes (n_modes,)
        """
        n_modes = spike_trains.shape[0]
        n_steps = spike_trains.shape[1]
        
        # Compute firing rates
        window_steps = int(window / self.dt)
        rates = np.sum(spike_trains[:, -window_steps:], axis=1) / (window / 1000.0)
        
        # Convert rates to amplitudes (normalize)
        amplitudes = rates / (np.max(rates) + 1e-12)
        
        return amplitudes
    
    def decode_temporal(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Decode spike trains to amplitudes using temporal coding.
        
        Args:
            spike_trains: Spike trains (n_modes, n_timesteps)
        
        Returns:
            Decoded amplitudes (n_modes,)
        """
        n_modes = spike_trains.shape[0]
        n_steps = spike_trains.shape[1]
        
        amplitudes = np.zeros(n_modes)
        
        for i in range(n_modes):
            spike_indices = np.where(spike_trains[i, :] > 0)[0]
            
            if len(spike_indices) > 0:
                # Earlier spike -> larger amplitude
                first_spike = spike_indices[0]
                amplitudes[i] = 1.0 - (first_spike / n_steps)
            else:
                amplitudes[i] = 0.0
        
        return amplitudes
    
    def encode(self, amplitudes: np.ndarray, duration: float = 100.0) -> np.ndarray:
        """
        Encode amplitudes using configured encoding scheme.
        
        Args:
            amplitudes: Harmonic mode amplitudes
            duration: Duration of spike train (ms)
        
        Returns:
            Spike trains
        """
        if self.encoding == 'rate':
            return self.encode_rate(amplitudes, duration)
        elif self.encoding == 'temporal':
            return self.encode_temporal(amplitudes, duration)
        else:
            raise ValueError(f"Unknown encoding scheme: {self.encoding}")
    
    def decode(self, spike_trains: np.ndarray, window: float = 100.0) -> np.ndarray:
        """
        Decode spike trains using configured encoding scheme.
        
        Args:
            spike_trains: Spike trains
            window: Window for rate estimation (used for rate coding)
        
        Returns:
            Decoded amplitudes
        """
        if self.encoding == 'rate':
            return self.decode_rate(spike_trains, window)
        elif self.encoding == 'temporal':
            return self.decode_temporal(spike_trains)
        else:
            raise ValueError(f"Unknown encoding scheme: {self.encoding}")
