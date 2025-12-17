"""
Unit Tests for Spiking Neural Network Modules

Tests for:
- LIFNeuron: Leaky integrate-and-fire neuron
- LIFNetwork: Network of LIF neurons
- SpikeEncoder: Spike train encoding/decoding
- spike_metrics: Consciousness metrics from spikes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.snn import LIFNeuron, LIFNetwork, SpikeEncoder, compute_spike_metrics


class TestLIFNeuron:
    """Tests for LIFNeuron."""
    
    def test_initialization(self):
        """Test neuron initialization."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0)
        
        assert neuron.tau_m == 20.0
        assert neuron.v_threshold == 1.0
        assert neuron.v_reset == 0.0
        assert neuron.v_rest == 0.0
        assert neuron.v == 0.0
        assert len(neuron.spike_times) == 0
    
    def test_update_no_spike(self):
        """Test neuron update without spiking."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0)
        
        # Small input that doesn't cause spike
        spiked = neuron.update(I_input=0.1)
        
        assert not spiked
        assert neuron.v > 0  # Voltage increased but below threshold
        assert neuron.v < 1.0
    
    def test_update_with_spike(self):
        """Test neuron update with spiking."""
        neuron = LIFNeuron(tau_m=20.0, v_threshold=1.0, v_reset=0.0)
        
        # Large input that causes spike
        spiked = False
        for _ in range(100):  # Sufficient iterations for spike
            spiked = neuron.update(I_input=3.0)
            if spiked:
                break
        
        assert spiked
        assert neuron.v == 0.0  # Reset after spike
        assert len(neuron.spike_times) == 1
    
    def test_reset(self):
        """Test neuron reset."""
        neuron = LIFNeuron()
        
        # Simulate and spike
        for _ in range(20):
            neuron.update(I_input=2.0)
        
        neuron.reset()
        
        assert neuron.v == neuron.v_rest
        assert len(neuron.spike_times) == 0
        assert neuron.t == 0.0


class TestLIFNetwork:
    """Tests for LIFNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        n_neurons = 50
        network = LIFNetwork(n_neurons=n_neurons)
        
        assert network.n_neurons == n_neurons
        assert len(network.neurons) == n_neurons
        assert network.connectivity.shape == (n_neurons, n_neurons)
        assert np.all(np.diag(network.connectivity) == 0)  # No self-connections
    
    def test_update(self):
        """Test network update."""
        network = LIFNetwork(n_neurons=10)
        
        external_input = np.random.randn(10)
        spikes = network.update(external_input)
        
        assert spikes.shape == (10,)
        assert np.all((spikes == 0) | (spikes == 1))
    
    def test_simulate(self):
        """Test network simulation."""
        network = LIFNetwork(n_neurons=10)
        
        n_steps = 100
        input_sequence = np.random.randn(n_steps, 10)
        spike_matrix, spike_trains = network.simulate(input_sequence, n_steps)
        
        assert spike_matrix.shape == (n_steps, 10)
        assert len(spike_trains) == 10
        assert np.all((spike_matrix == 0) | (spike_matrix == 1))
    
    def test_firing_rates(self):
        """Test firing rate computation."""
        network = LIFNetwork(n_neurons=10)
        
        # Simulate with strong input
        input_sequence = np.ones((100, 10)) * 2.0
        spike_matrix, _ = network.simulate(input_sequence, 100)
        
        rates = network.get_firing_rates(window=100.0)
        
        assert rates.shape == (10,)
        assert np.all(rates >= 0)
    
    def test_reset(self):
        """Test network reset."""
        network = LIFNetwork(n_neurons=10)
        
        # Simulate
        input_sequence = np.random.randn(50, 10)
        network.simulate(input_sequence, 50)
        
        # Reset
        network.reset()
        
        assert network.t == 0.0
        assert all(len(st) == 0 for st in network.spike_trains)


class TestSpikeEncoder:
    """Tests for SpikeEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = SpikeEncoder(encoding='rate', dt=0.1)
        
        assert encoder.encoding == 'rate'
        assert encoder.dt == 0.1
    
    def test_rate_encoding(self):
        """Test rate coding."""
        encoder = SpikeEncoder(encoding='rate')
        
        amplitudes = np.random.rand(10)
        spike_trains = encoder.encode_rate(amplitudes, duration=100.0)
        
        assert spike_trains.shape[0] == 10
        assert spike_trains.shape[1] > 0
        assert np.all((spike_trains == 0) | (spike_trains == 1))
    
    def test_temporal_encoding(self):
        """Test temporal coding."""
        encoder = SpikeEncoder(encoding='temporal')
        
        amplitudes = np.array([1.0, 0.5, 0.1, 0.0])
        spike_trains = encoder.encode_temporal(amplitudes, duration=100.0)
        
        assert spike_trains.shape[0] == 4
        # Larger amplitude should spike earlier
        spike_times = [np.where(spike_trains[i, :] > 0)[0] for i in range(4)]
        if len(spike_times[0]) > 0 and len(spike_times[1]) > 0:
            assert spike_times[0][0] < spike_times[1][0]
    
    def test_rate_decoding(self):
        """Test rate decoding."""
        encoder = SpikeEncoder(encoding='rate')
        
        amplitudes = np.array([1.0, 0.5, 0.2])
        spike_trains = encoder.encode_rate(amplitudes, duration=100.0)
        decoded = encoder.decode_rate(spike_trains, window=100.0)
        
        assert decoded.shape == (3,)
        # Decoded should preserve relative ordering
        assert np.argmax(decoded) == np.argmax(amplitudes)
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode consistency."""
        encoder = SpikeEncoder(encoding='rate')
        
        amplitudes = np.array([1.0, 0.7, 0.4, 0.1])
        spike_trains = encoder.encode(amplitudes, duration=200.0)
        decoded = encoder.decode(spike_trains, window=200.0)
        
        # Should preserve relative ordering
        assert np.argsort(amplitudes)[-1] == np.argsort(decoded)[-1]


class TestSpikeMetrics:
    """Tests for spike-based consciousness metrics."""
    
    def test_compute_spike_metrics(self):
        """Test computing all metrics from spikes."""
        # Generate random spike trains
        n_neurons = 50
        n_steps = 500
        spike_trains = (np.random.rand(n_neurons, n_steps) < 0.1).astype(int)
        
        metrics = compute_spike_metrics(spike_trains, dt=0.1)
        
        assert 'H_mode' in metrics
        assert 'PR' in metrics
        assert 'R' in metrics
        assert 'S_dot' in metrics
        assert 'kappa' in metrics
        assert 'C_t' in metrics
        
        # Check ranges
        assert 0 <= metrics['H_mode'] <= 1
        assert 0 <= metrics['PR'] <= 1
        assert 0 <= metrics['R'] <= 1
        assert metrics['S_dot'] >= 0
        assert 0 <= metrics['kappa'] <= 2
        assert 0 <= metrics['C_t'] <= 1
    
    def test_metrics_with_no_spikes(self):
        """Test metrics with silent network."""
        spike_trains = np.zeros((30, 100), dtype=int)
        
        metrics = compute_spike_metrics(spike_trains, dt=0.1)
        
        # Should handle gracefully
        assert 'H_mode' in metrics
        assert 'C_t' in metrics
    
    def test_metrics_with_synchronized_spikes(self):
        """Test metrics with highly synchronized spikes."""
        n_neurons = 50
        n_steps = 100
        
        # All neurons spike together
        spike_trains = np.zeros((n_neurons, n_steps), dtype=int)
        spike_trains[:, ::10] = 1  # Spikes every 10 steps
        
        metrics = compute_spike_metrics(spike_trains, dt=0.1)
        
        # High synchrony should give high R
        assert metrics['R'] > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
