"""
Leaky Integrate-and-Fire (LIF) Neuron Models

Implements spiking neurons for consciousness modeling with harmonic oscillators.
"""

import numpy as np
from typing import List, Tuple, Optional


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.
    
    Models individual spiking neurons with membrane potential dynamics.
    """
    
    def __init__(self, tau_m: float = 20.0, v_threshold: float = 1.0, 
                 v_reset: float = 0.0, v_rest: float = 0.0, dt: float = 0.1):
        """
        Initialize LIF neuron.
        
        Args:
            tau_m: Membrane time constant (ms)
            v_threshold: Spike threshold voltage
            v_reset: Reset voltage after spike
            v_rest: Resting potential
            dt: Time step (ms)
        """
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        
        # State
        self.v = v_rest
        self.spike_times = []
        self.t = 0.0
    
    def update(self, I_input: float) -> bool:
        """
        Update neuron state for one time step.
        
        Args:
            I_input: Input current
        
        Returns:
            True if neuron spiked, False otherwise
        """
        # Leak current
        dv = (-(self.v - self.v_rest) + I_input) / self.tau_m
        
        # Update voltage
        self.v += dv * self.dt
        
        # Check for spike
        spiked = False
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.spike_times.append(self.t)
            spiked = True
        
        self.t += self.dt
        
        return spiked
    
    def reset(self):
        """Reset neuron state."""
        self.v = self.v_rest
        self.spike_times = []
        self.t = 0.0


class LIFNetwork:
    """
    Network of LIF neurons with connectivity.
    
    Models population of spiking neurons with synaptic connections.
    """
    
    def __init__(self, n_neurons: int, connectivity: np.ndarray = None,
                 tau_m: float = 20.0, tau_s: float = 5.0, dt: float = 0.1):
        """
        Initialize LIF network.
        
        Args:
            n_neurons: Number of neurons in network
            connectivity: Adjacency matrix for synaptic connections (n_neurons x n_neurons)
            tau_m: Membrane time constant (ms)
            tau_s: Synaptic time constant (ms)
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.dt = dt
        
        # Create neurons
        self.neurons = [LIFNeuron(tau_m=tau_m, dt=dt) for _ in range(n_neurons)]
        
        # Connectivity matrix
        if connectivity is None:
            # Random connectivity with 10% connection probability
            connectivity = (np.random.rand(n_neurons, n_neurons) < 0.1).astype(float)
            connectivity *= np.random.randn(n_neurons, n_neurons) * 0.5
            np.fill_diagonal(connectivity, 0)
        
        self.connectivity = connectivity
        
        # Synaptic currents
        self.synaptic_currents = np.zeros(n_neurons)
        
        # Spike history
        self.spike_trains = [[] for _ in range(n_neurons)]
        self.t = 0.0
    
    def update(self, external_input: np.ndarray) -> np.ndarray:
        """
        Update network for one time step.
        
        Args:
            external_input: External input currents (n_neurons,)
        
        Returns:
            Binary spike array (1 = spike, 0 = no spike)
        """
        spikes = np.zeros(self.n_neurons, dtype=bool)
        
        # Update each neuron
        for i, neuron in enumerate(self.neurons):
            # Total input = external + synaptic
            total_input = external_input[i] + self.synaptic_currents[i]
            
            # Update neuron
            spiked = neuron.update(total_input)
            spikes[i] = spiked
            
            if spiked:
                self.spike_trains[i].append(self.t)
        
        # Update synaptic currents
        # Decay existing currents
        self.synaptic_currents -= self.synaptic_currents * (self.dt / self.tau_s)
        
        # Add new currents from spikes
        for i in range(self.n_neurons):
            if spikes[i]:
                # This neuron spiked, add currents to postsynaptic neurons
                self.synaptic_currents += self.connectivity[i, :]
        
        self.t += self.dt
        
        return spikes.astype(int)
    
    def simulate(self, input_sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Simulate network for multiple time steps.
        
        Args:
            input_sequence: External inputs (n_steps, n_neurons)
            n_steps: Number of simulation steps
        
        Returns:
            Tuple of (spike_matrix, spike_trains)
            - spike_matrix: Binary matrix (n_steps, n_neurons)
            - spike_trains: List of spike times for each neuron
        """
        spike_matrix = np.zeros((n_steps, self.n_neurons), dtype=int)
        
        for step in range(n_steps):
            external_input = input_sequence[step, :] if input_sequence.ndim > 1 else np.ones(self.n_neurons) * input_sequence[step]
            spikes = self.update(external_input)
            spike_matrix[step, :] = spikes
        
        return spike_matrix, self.spike_trains
    
    def reset(self):
        """Reset network state."""
        for neuron in self.neurons:
            neuron.reset()
        self.synaptic_currents = np.zeros(self.n_neurons)
        self.spike_trains = [[] for _ in range(self.n_neurons)]
        self.t = 0.0
    
    def get_firing_rates(self, window: float = 100.0) -> np.ndarray:
        """
        Compute firing rates for all neurons.
        
        Args:
            window: Time window for rate computation (ms)
        
        Returns:
            Firing rates (Hz) for each neuron
        """
        rates = np.zeros(self.n_neurons)
        
        for i, spike_times in enumerate(self.spike_trains):
            if len(spike_times) == 0:
                rates[i] = 0.0
            else:
                # Count spikes in recent window
                recent_spikes = [t for t in spike_times if t > self.t - window]
                rates[i] = (len(recent_spikes) / window) * 1000.0  # Convert to Hz
        
        return rates
