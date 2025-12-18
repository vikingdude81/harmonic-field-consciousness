"""
Harmonic Bridge - Connection between Neural Mass Models and Harmonic Field Theory

Provides conversion functions and integrated models that bridge:
- Push-pull oscillator dynamics (microscopic)
- Harmonic field modes (macroscopic)
- Consciousness state predictions

This module enables:
1. Converting E-I oscillations to harmonic mode representations
2. Predicting consciousness states from neural mass dynamics
3. Bidirectional translation between frameworks
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftfreq

from .push_pull_oscillator import PushPullOscillator, MultiScalePushPull


def oscillation_to_harmonic_mode(
    time_series: np.ndarray,
    dt: float,
    n_modes: int = 20,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Convert oscillatory time series to harmonic mode representation.
    
    Uses spectral decomposition to extract harmonic modes from neural mass
    model oscillations. Each frequency band corresponds to a different
    harmonic mode in the field theory.
    
    Args:
        time_series: Oscillatory signal (e.g., E population activity)
        dt: Time step (ms)
        n_modes: Number of harmonic modes to extract
        frequency_bands: Optional dictionary of band names to (low, high) Hz
    
    Returns:
        Dictionary containing:
            - 'modes': Harmonic mode amplitudes (n_modes,)
            - 'phases': Phase for each mode (n_modes,)
            - 'frequencies': Center frequency for each mode (Hz)
            - 'power_spectrum': Full power spectrum
    """
    # Default frequency bands (physiological ranges)
    if frequency_bands is None:
        frequency_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
    
    # Compute FFT
    n_samples = len(time_series)
    freqs = fftfreq(n_samples, d=dt / 1000.0)  # Convert dt to seconds
    fft_vals = fft(time_series)
    
    # Power spectrum (positive frequencies only)
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    power_spectrum = np.abs(fft_vals[positive_freq_mask]) ** 2
    
    # Extract modes by frequency bands
    mode_amplitudes = np.zeros(n_modes)
    mode_phases = np.zeros(n_modes)
    mode_frequencies = np.zeros(n_modes)
    
    # Distribute modes across frequency range
    freq_min = 0.1
    freq_max = min(100.0, freqs_pos[-1])
    mode_freq_centers = np.logspace(
        np.log10(freq_min),
        np.log10(freq_max),
        n_modes
    )
    
    for i, center_freq in enumerate(mode_freq_centers):
        # Define bandwidth around center frequency
        bandwidth = center_freq * 0.3  # 30% bandwidth
        freq_mask = (freqs_pos >= center_freq - bandwidth / 2) & \
                    (freqs_pos <= center_freq + bandwidth / 2)
        
        if np.any(freq_mask):
            # Amplitude is integrated power in this band
            band_power = power_spectrum[freq_mask]
            mode_amplitudes[i] = np.sqrt(np.sum(band_power))
            
            # Phase is average phase in this band
            band_fft = fft_vals[positive_freq_mask][freq_mask]
            mode_phases[i] = np.angle(np.mean(band_fft))
            
            mode_frequencies[i] = center_freq
    
    # Normalize amplitudes
    if np.sum(mode_amplitudes) > 0:
        mode_amplitudes /= np.sqrt(np.sum(mode_amplitudes ** 2))
    
    return {
        'modes': mode_amplitudes,
        'phases': mode_phases,
        'frequencies': mode_frequencies,
        'power_spectrum': power_spectrum,
        'freq_axis': freqs_pos
    }


def compute_harmonic_richness(mode_amplitudes: np.ndarray) -> float:
    """
    Compute harmonic richness from mode amplitudes.
    
    Harmonic richness H quantifies the diversity of active modes.
    Higher richness indicates more complex, conscious-like states.
    
    Args:
        mode_amplitudes: Array of mode amplitudes
    
    Returns:
        Harmonic richness H ∈ [0, 1]
    """
    # Normalize to probability distribution
    power = mode_amplitudes ** 2
    power_norm = power / (np.sum(power) + 1e-12)
    
    # Shannon entropy of mode distribution
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-12))
    
    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log(len(mode_amplitudes))
    
    if max_entropy > 0:
        richness = entropy / max_entropy
    else:
        richness = 0.0
    
    return float(richness)


def compute_participation_ratio(mode_amplitudes: np.ndarray) -> float:
    """
    Compute mode participation ratio.
    
    PR quantifies how many modes contribute significantly to the state.
    PR = (Σ |a_k|^2)^2 / Σ |a_k|^4
    
    Args:
        mode_amplitudes: Array of mode amplitudes
    
    Returns:
        Participation ratio (1 to n_modes)
    """
    power = mode_amplitudes ** 2
    sum_power_sq = np.sum(power) ** 2
    sum_power_fourth = np.sum(power ** 2)
    
    if sum_power_fourth > 1e-12:
        pr = sum_power_sq / sum_power_fourth
    else:
        pr = 1.0
    
    return float(pr)


class HarmonicNeuralMassModel:
    """
    Integrated model combining neural mass dynamics with harmonic field theory.
    
    This class provides bidirectional conversion between:
    - Microscopic: Push-pull oscillator dynamics
    - Macroscopic: Harmonic field mode representation
    
    It enables:
    1. Simulating neural population dynamics
    2. Converting to harmonic modes
    3. Predicting consciousness states from both representations
    4. Analyzing multi-scale brain rhythms
    """
    
    def __init__(
        self,
        n_modes: int = 20,
        n_scales: int = 3,
        dt: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize integrated harmonic-neural mass model.
        
        Args:
            n_modes: Number of harmonic modes
            n_scales: Number of hierarchical oscillator scales
            dt: Time step for integration (ms)
            seed: Random seed for reproducibility
        """
        self.n_modes = n_modes
        self.n_scales = n_scales
        self.dt = dt
        
        # Multi-scale oscillator
        self.oscillator = MultiScalePushPull(
            n_scales=n_scales,
            dt=dt,
            seed=seed
        )
        
        # Current harmonic representation
        self.harmonic_state: Optional[Dict[str, np.ndarray]] = None
        
        # History
        self.simulation_history: List[Dict] = []
    
    def simulate_and_convert(
        self,
        duration: float,
        external_input: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate neural mass dynamics and convert to harmonic modes.
        
        Args:
            duration: Simulation duration (ms)
            external_input: External input to finest scale
        
        Returns:
            Dictionary with both neural mass and harmonic representations
        """
        # Simulate neural mass dynamics
        nmm_activity = self.oscillator.simulate(duration, external_input)
        
        # Convert finest scale activity to harmonic modes
        finest_scale_activity = nmm_activity['scale_0_e']
        
        self.harmonic_state = oscillation_to_harmonic_mode(
            finest_scale_activity,
            dt=self.dt,
            n_modes=self.n_modes
        )
        
        # Combine results
        result = {
            'nmm_activity': nmm_activity,
            'harmonic_modes': self.harmonic_state['modes'],
            'harmonic_phases': self.harmonic_state['phases'],
            'mode_frequencies': self.harmonic_state['frequencies'],
            'power_spectrum': self.harmonic_state['power_spectrum'],
            'freq_axis': self.harmonic_state['freq_axis']
        }
        
        # Store in history
        self.simulation_history.append(result)
        
        return result
    
    def predict_consciousness_state(self) -> Dict[str, float]:
        """
        Predict consciousness state from current harmonic representation.
        
        Computes multiple consciousness metrics:
        - Harmonic richness
        - Mode participation ratio
        - Dominant frequency
        - Consciousness score
        
        Returns:
            Dictionary of consciousness metrics
        """
        if self.harmonic_state is None:
            raise ValueError("Must run simulate_and_convert first")
        
        modes = self.harmonic_state['modes']
        
        # Compute metrics
        richness = compute_harmonic_richness(modes)
        participation = compute_participation_ratio(modes)
        
        # Find dominant frequency
        power = modes ** 2
        dominant_idx = np.argmax(power)
        dominant_freq = self.harmonic_state['frequencies'][dominant_idx]
        
        # Consciousness score: combination of richness and participation
        # High values indicate wake-like states
        # Low values indicate sleep/anesthesia
        consciousness_score = 0.6 * richness + 0.4 * (participation / self.n_modes)
        
        return {
            'harmonic_richness': float(richness),
            'participation_ratio': float(participation),
            'dominant_frequency': float(dominant_freq),
            'consciousness_score': float(consciousness_score)
        }
    
    def classify_consciousness_state(self) -> str:
        """
        Classify current state into consciousness categories.
        
        Returns:
            State label: 'wake', 'nrem_sleep', 'rem_sleep', or 'anesthesia'
        """
        metrics = self.predict_consciousness_state()
        
        score = metrics['consciousness_score']
        richness = metrics['harmonic_richness']
        dominant_freq = metrics['dominant_frequency']
        
        # Classification rules based on metrics
        if score > 0.7 and richness > 0.6:
            return 'wake'
        elif score > 0.5 and dominant_freq > 4.0:
            return 'rem_sleep'
        elif score < 0.3:
            return 'anesthesia'
        else:
            return 'nrem_sleep'
    
    def get_scale_to_mode_mapping(self) -> Dict[int, np.ndarray]:
        """
        Map each oscillator scale to its corresponding harmonic modes.
        
        Returns:
            Dictionary mapping scale index to mode indices
        """
        scale_frequencies = self.oscillator.get_scale_frequencies()
        mode_frequencies = self.harmonic_state['frequencies']
        
        mapping = {}
        
        for scale_idx, scale_freq in enumerate(scale_frequencies):
            # Find modes close to this scale's frequency
            freq_diffs = np.abs(mode_frequencies - scale_freq)
            close_modes = np.where(freq_diffs < scale_freq * 0.5)[0]
            
            mapping[scale_idx] = close_modes
        
        return mapping
    
    def compute_consciousness_trajectory(
        self,
        duration: float,
        n_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute trajectory through consciousness state space.
        
        Args:
            duration: Total duration to simulate (ms)
            n_samples: Number of time points to sample
        
        Returns:
            Dictionary with consciousness metrics over time
        """
        segment_duration = duration / n_samples
        
        richness_trajectory = np.zeros(n_samples)
        participation_trajectory = np.zeros(n_samples)
        consciousness_trajectory = np.zeros(n_samples)
        states = []
        
        for i in range(n_samples):
            # Simulate segment
            self.simulate_and_convert(segment_duration)
            
            # Compute metrics
            metrics = self.predict_consciousness_state()
            richness_trajectory[i] = metrics['harmonic_richness']
            participation_trajectory[i] = metrics['participation_ratio']
            consciousness_trajectory[i] = metrics['consciousness_score']
            states.append(self.classify_consciousness_state())
        
        return {
            'time_points': np.arange(n_samples) * segment_duration,
            'harmonic_richness': richness_trajectory,
            'participation_ratio': participation_trajectory,
            'consciousness_score': consciousness_trajectory,
            'state_labels': states
        }
    
    def reset(self):
        """Reset model to initial state."""
        self.oscillator.reset()
        self.harmonic_state = None
        self.simulation_history = []
