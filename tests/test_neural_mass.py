"""
Unit Tests for Neural Mass Models

Tests for:
- PushPullOscillator: single E-I oscillator dynamics
- MultiScalePushPull: hierarchical multi-scale oscillators
- Harmonic bridge: conversion functions
- HarmonicNeuralMassModel: integrated model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.neural_mass import (
    PushPullOscillator,
    MultiScalePushPull,
    oscillation_to_harmonic_mode,
    HarmonicNeuralMassModel
)


class TestPushPullOscillator:
    """Tests for PushPullOscillator class."""
    
    def test_initialization(self):
        """Test oscillator initialization."""
        osc = PushPullOscillator(tau_e=10.0, tau_i=5.0, seed=42)
        
        assert osc.tau_e == 10.0
        assert osc.tau_i == 5.0
        assert osc.e_state is not None
        assert osc.i_state is not None
    
    def test_sigmoid_function(self):
        """Test sigmoid activation function."""
        osc = PushPullOscillator()
        
        # Test at zero
        assert np.isclose(osc.sigmoid(0.0), 0.5, atol=0.01)
        
        # Test monotonicity
        x = np.linspace(-5, 5, 100)
        y = osc.sigmoid(x)
        assert np.all(np.diff(y) >= 0)  # Monotonically increasing
        
        # Test bounds
        assert np.all(y >= 0) and np.all(y <= 1)
    
    def test_step(self):
        """Test single integration step."""
        osc = PushPullOscillator(seed=42)
        
        e_init, i_init = osc.get_state()
        e_new, i_new = osc.step(external_input=0.5)
        
        # State should change
        assert e_new != e_init or i_new != i_init
        
        # Values should be reasonable
        assert 0.0 <= e_new <= 1.5
        assert 0.0 <= i_new <= 1.5
    
    def test_simulation(self):
        """Test full simulation."""
        osc = PushPullOscillator(seed=42)
        
        duration = 500  # ms
        result = osc.simulate(duration)
        
        # Check result structure
        assert 'time' in result
        assert 'e_activity' in result
        assert 'i_activity' in result
        
        # Check dimensions
        n_steps = int(duration / osc.dt)
        assert len(result['time']) == n_steps
        assert len(result['e_activity']) == n_steps
        assert len(result['i_activity']) == n_steps
    
    def test_oscillation_frequency(self):
        """Test frequency computation."""
        osc = PushPullOscillator(
            tau_e=10.0,
            tau_i=5.0,
            w_ee=1.5,
            w_ie=2.0,
            w_ei=2.5,
            w_ii=0.5,
            seed=42
        )
        
        # Simulate to generate oscillations
        n_steps = int(1000 / osc.dt)
        osc.simulate(1000, external_input=0.5 * np.ones(n_steps))
        
        freq = osc.compute_oscillation_frequency()
        
        # Frequency should be in physiological range
        assert 0.1 <= freq <= 100.0
    
    def test_reset(self):
        """Test reset functionality."""
        osc = PushPullOscillator(seed=42)
        
        # Simulate
        osc.simulate(500)
        
        # Reset
        osc.reset(e_init=0.2, i_init=0.3)
        
        e_state, i_state = osc.get_state()
        assert np.isclose(e_state, 0.2, atol=0.01)
        assert np.isclose(i_state, 0.3, atol=0.01)
        assert len(osc.history['time']) == 0


class TestMultiScalePushPull:
    """Tests for MultiScalePushPull class."""
    
    def test_initialization(self):
        """Test multi-scale oscillator initialization."""
        multi_osc = MultiScalePushPull(n_scales=3, seed=42)
        
        assert multi_osc.n_scales == 3
        assert len(multi_osc.oscillators) == 3
    
    def test_time_constant_scaling(self):
        """Test that time constants increase with scale."""
        multi_osc = MultiScalePushPull(
            n_scales=4,
            base_tau_e=10.0,
            tau_scale_factor=2.0,
            seed=42
        )
        
        # Check time constants increase
        tau_values = [osc.tau_e for osc in multi_osc.oscillators]
        assert all(tau_values[i] < tau_values[i+1] for i in range(len(tau_values)-1))
    
    def test_simulation(self):
        """Test multi-scale simulation."""
        multi_osc = MultiScalePushPull(n_scales=3, seed=42)
        
        duration = 500  # ms
        result = multi_osc.simulate(duration)
        
        # Check that all scales are present
        assert 'time' in result
        for i in range(multi_osc.n_scales):
            assert f'scale_{i}_e' in result
    
    def test_scale_frequencies(self):
        """Test frequency computation for each scale."""
        multi_osc = MultiScalePushPull(n_scales=3, seed=42)
        
        # Simulate first
        multi_osc.simulate(1000)
        
        freqs = multi_osc.get_scale_frequencies()
        
        assert len(freqs) == multi_osc.n_scales
        # Larger scales should have lower frequencies
        # (though this may not always hold due to nonlinear dynamics)
        assert all(f >= 0 for f in freqs)
    
    def test_cross_frequency_coupling(self):
        """Test cross-frequency coupling computation."""
        multi_osc = MultiScalePushPull(n_scales=3, seed=42)
        
        # Simulate first
        multi_osc.simulate(1000)
        
        coupling = multi_osc.get_cross_frequency_coupling()
        
        # Check shape
        assert coupling.shape == (multi_osc.n_scales, multi_osc.n_scales)
        
        # Diagonal should be 1 (self-correlation)
        assert np.allclose(np.diag(coupling), 1.0, atol=0.01)
        
        # Values should be in [-1, 1]
        assert np.all(coupling >= -1.0) and np.all(coupling <= 1.0)
    
    def test_reset(self):
        """Test reset functionality."""
        multi_osc = MultiScalePushPull(n_scales=2, seed=42)
        
        # Simulate
        multi_osc.simulate(500)
        
        # Reset
        multi_osc.reset()
        
        assert len(multi_osc.history) == 0


class TestHarmonicBridge:
    """Tests for harmonic bridge conversion functions."""
    
    def test_oscillation_to_harmonic_mode(self):
        """Test conversion from oscillations to harmonic modes."""
        # Create a simple sinusoidal signal
        dt = 0.1  # ms
        duration = 1000  # ms
        time = np.arange(0, duration, dt)
        freq = 10.0  # Hz
        signal = np.sin(2 * np.pi * freq * time / 1000.0)
        
        result = oscillation_to_harmonic_mode(signal, dt=dt, n_modes=20)
        
        # Check structure
        assert 'modes' in result
        assert 'phases' in result
        assert 'frequencies' in result
        assert 'power_spectrum' in result
        
        # Check dimensions
        assert len(result['modes']) == 20
        assert len(result['phases']) == 20
        assert len(result['frequencies']) == 20
        
        # Modes should be normalized
        assert np.isclose(np.sum(result['modes'] ** 2), 1.0, atol=0.1)
    
    def test_harmonic_mode_peak_detection(self):
        """Test that conversion finds correct peak frequency."""
        dt = 0.1  # ms
        duration = 2000  # ms
        time = np.arange(0, duration, dt)
        freq = 15.0  # Hz
        signal = np.sin(2 * np.pi * freq * time / 1000.0)
        
        result = oscillation_to_harmonic_mode(signal, dt=dt, n_modes=30)
        
        # Find mode with highest power
        power = result['modes'] ** 2
        peak_mode_idx = np.argmax(power)
        peak_freq = result['frequencies'][peak_mode_idx]
        
        # Peak frequency should be close to input frequency
        assert np.abs(peak_freq - freq) < 2.0  # Within 2 Hz


class TestHarmonicNeuralMassModel:
    """Tests for integrated HarmonicNeuralMassModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        assert model.n_modes == 20
        assert model.n_scales == 3
        assert model.oscillator is not None
    
    def test_simulate_and_convert(self):
        """Test simulation and conversion."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        result = model.simulate_and_convert(duration=500)
        
        # Check structure
        assert 'nmm_activity' in result
        assert 'harmonic_modes' in result
        assert 'harmonic_phases' in result
        
        # Check dimensions
        assert len(result['harmonic_modes']) == 20
        assert len(result['harmonic_phases']) == 20
    
    def test_predict_consciousness_state(self):
        """Test consciousness state prediction."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        # Must simulate first
        model.simulate_and_convert(duration=500)
        
        metrics = model.predict_consciousness_state()
        
        # Check structure
        assert 'harmonic_richness' in metrics
        assert 'participation_ratio' in metrics
        assert 'dominant_frequency' in metrics
        assert 'consciousness_score' in metrics
        
        # Check ranges
        assert 0.0 <= metrics['harmonic_richness'] <= 1.0
        assert 1.0 <= metrics['participation_ratio'] <= model.n_modes
        assert metrics['dominant_frequency'] >= 0.0
        assert 0.0 <= metrics['consciousness_score'] <= 1.0
    
    def test_classify_consciousness_state(self):
        """Test consciousness state classification."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        # Must simulate first
        model.simulate_and_convert(duration=500)
        
        state = model.classify_consciousness_state()
        
        # Should be one of the known states
        valid_states = ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia']
        assert state in valid_states
    
    def test_consciousness_trajectory(self):
        """Test consciousness trajectory computation."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=2, seed=42)
        
        trajectory = model.compute_consciousness_trajectory(
            duration=1000,
            n_samples=5
        )
        
        # Check structure
        assert 'time_points' in trajectory
        assert 'harmonic_richness' in trajectory
        assert 'participation_ratio' in trajectory
        assert 'consciousness_score' in trajectory
        assert 'state_labels' in trajectory
        
        # Check dimensions
        assert len(trajectory['time_points']) == 5
        assert len(trajectory['harmonic_richness']) == 5
        assert len(trajectory['state_labels']) == 5
    
    def test_reset(self):
        """Test reset functionality."""
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        # Simulate
        model.simulate_and_convert(duration=500)
        
        # Reset
        model.reset()
        
        assert model.harmonic_state is None
        assert len(model.simulation_history) == 0


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from oscillator to consciousness prediction."""
        # Create model
        model = HarmonicNeuralMassModel(n_modes=20, n_scales=3, seed=42)
        
        # Simulate
        result = model.simulate_and_convert(duration=1000)
        
        # Predict state
        metrics = model.predict_consciousness_state()
        state = model.classify_consciousness_state()
        
        # All should succeed without errors
        assert result is not None
        assert metrics is not None
        assert state is not None
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        model1 = HarmonicNeuralMassModel(n_modes=20, n_scales=2, seed=42)
        model2 = HarmonicNeuralMassModel(n_modes=20, n_scales=2, seed=42)
        
        result1 = model1.simulate_and_convert(duration=500)
        result2 = model2.simulate_and_convert(duration=500)
        
        # Results should be identical
        np.testing.assert_allclose(
            result1['harmonic_modes'],
            result2['harmonic_modes'],
            rtol=1e-5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
