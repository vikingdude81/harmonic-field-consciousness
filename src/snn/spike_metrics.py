"""
Consciousness Metrics from Spike Trains

Computes consciousness metrics (H_mode, PR, R, S_dot, kappa) from spiking neuron activity.
"""

import numpy as np
from typing import Dict, Tuple


def compute_spike_metrics(spike_trains: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """
    Compute all consciousness metrics from spike trains.
    
    Args:
        spike_trains: Binary spike matrix (n_neurons, n_timesteps)
        dt: Time step (ms)
    
    Returns:
        Dictionary of consciousness metrics
    """
    metrics = {}
    
    # Compute power distribution from firing rates
    power = compute_spike_power(spike_trains, dt)
    metrics['H_mode'] = compute_mode_entropy_spikes(power)
    metrics['PR'] = compute_participation_ratio_spikes(power)
    
    # Phase coherence from spike timing
    metrics['R'] = compute_phase_coherence_spikes(spike_trains, dt)
    
    # Entropy production from spike variability
    metrics['S_dot'] = compute_entropy_rate_spikes(spike_trains, dt)
    
    # Criticality from avalanche statistics
    metrics['kappa'] = compute_criticality_spikes(spike_trains)
    
    # Overall consciousness functional
    metrics['C_t'] = (0.3 * metrics['H_mode'] + 
                     0.25 * metrics['PR'] +
                     0.25 * metrics['R'] +
                     0.1 * min(metrics['S_dot'], 1.0) +
                     0.1 * metrics['kappa'] / 2.0)
    
    return metrics


def compute_spike_power(spike_trains: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute power distribution from spike trains.
    
    Args:
        spike_trains: Binary spike matrix (n_neurons, n_timesteps)
        dt: Time step (ms)
    
    Returns:
        Normalized power distribution across neurons
    """
    # Firing rates
    rates = np.sum(spike_trains, axis=1) / (spike_trains.shape[1] * dt / 1000.0)
    
    # Power proportional to rate squared
    power = rates ** 2
    
    # Normalize
    power = power / (np.sum(power) + 1e-12)
    
    return power


def compute_mode_entropy_spikes(power: np.ndarray) -> float:
    """
    Compute mode entropy from spike-based power distribution.
    
    Args:
        power: Normalized power distribution
    
    Returns:
        Normalized mode entropy [0, 1]
    """
    p = power[power > 1e-12]
    H = -np.sum(p * np.log(p + 1e-12))
    H_max = np.log(len(power))
    return H / (H_max + 1e-12)


def compute_participation_ratio_spikes(power: np.ndarray) -> float:
    """
    Compute participation ratio from spike-based power.
    
    Args:
        power: Normalized power distribution
    
    Returns:
        Normalized participation ratio [0, 1]
    """
    PR = 1.0 / (np.sum(power ** 2) + 1e-12)
    return PR / len(power)


def compute_phase_coherence_spikes(spike_trains: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute phase coherence from spike timing synchrony.
    
    Args:
        spike_trains: Binary spike matrix (n_neurons, n_timesteps)
        dt: Time step (ms)
    
    Returns:
        Phase coherence measure [0, 1]
    """
    n_neurons, n_steps = spike_trains.shape
    
    # Compute instantaneous population rate
    pop_rate = np.sum(spike_trains, axis=0)
    
    if np.max(pop_rate) == 0:
        return 0.0
    
    # Coherence measured as synchronization index
    # High values when many neurons spike together
    synchrony = np.std(pop_rate) / (np.mean(pop_rate) + 1e-12)
    
    # Normalize to [0, 1]
    R = np.tanh(synchrony / 5.0)
    
    return float(R)


def compute_entropy_rate_spikes(spike_trains: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute entropy production rate from spike variability.
    
    Args:
        spike_trains: Binary spike matrix (n_neurons, n_timesteps)
        dt: Time step (ms)
    
    Returns:
        Entropy production rate (normalized)
    """
    n_neurons, n_steps = spike_trains.shape
    
    # Compute inter-spike intervals for each neuron
    isi_entropy = 0.0
    count = 0
    
    for i in range(n_neurons):
        spike_times = np.where(spike_trains[i, :] > 0)[0] * dt
        
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            
            if len(isis) > 0:
                # Histogram of ISIs
                hist, _ = np.histogram(isis, bins=10, density=True)
                hist = hist[hist > 0]
                
                # Shannon entropy of ISI distribution
                H_isi = -np.sum(hist * np.log(hist + 1e-12))
                isi_entropy += H_isi
                count += 1
    
    if count > 0:
        # Average entropy across neurons
        S_dot = isi_entropy / count
        return min(S_dot / 3.0, 1.0)  # Normalize
    else:
        return 0.0


def compute_criticality_spikes(spike_trains: np.ndarray) -> float:
    """
    Compute criticality index from spike avalanche statistics.
    
    Args:
        spike_trains: Binary spike matrix (n_neurons, n_timesteps)
    
    Returns:
        Criticality index (0 = subcritical, 1 = critical, 2 = supercritical)
    """
    # Detect avalanches (contiguous periods of activity)
    pop_activity = np.sum(spike_trains, axis=0) > 0
    
    # Find avalanche boundaries
    avalanche_starts = np.where(np.diff(pop_activity.astype(int)) == 1)[0]
    avalanche_ends = np.where(np.diff(pop_activity.astype(int)) == -1)[0]
    
    if len(avalanche_starts) == 0 or len(avalanche_ends) == 0:
        return 0.5  # No clear avalanches
    
    # Match starts and ends
    if avalanche_ends[0] < avalanche_starts[0]:
        avalanche_ends = avalanche_ends[1:]
    
    n_avalanches = min(len(avalanche_starts), len(avalanche_ends))
    
    if n_avalanches == 0:
        return 0.5
    
    # Compute avalanche sizes
    avalanche_sizes = []
    for i in range(n_avalanches):
        start = avalanche_starts[i]
        end = avalanche_ends[i] if i < len(avalanche_ends) else spike_trains.shape[1]
        size = np.sum(spike_trains[:, start:end+1])
        avalanche_sizes.append(size)
    
    if len(avalanche_sizes) < 2:
        return 0.5
    
    avalanche_sizes = np.array(avalanche_sizes)
    
    # Power-law exponent estimation (simplified)
    # Critical systems have exponent ~ -1.5
    log_sizes = np.log(avalanche_sizes + 1)
    
    # Coefficient of variation
    cv = np.std(log_sizes) / (np.mean(log_sizes) + 1e-12)
    
    # Map to criticality index
    # cv ~ 1.0 indicates criticality
    kappa = 1.0 + (cv - 1.0) * 0.5
    kappa = np.clip(kappa, 0.0, 2.0)
    
    return float(kappa)
