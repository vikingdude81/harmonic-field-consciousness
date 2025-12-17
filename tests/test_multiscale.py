"""
Unit Tests for Multiscale Dynamics Modules

Tests for:
- MultiscaleEncoder: multiscale encoding and missing data handling
- RealtimeDecoder: real-time consciousness state prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.multiscale import MultiscaleEncoder, RealtimeDecoder


class TestMultiscaleEncoder:
    """Tests for MultiscaleEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = MultiscaleEncoder(n_modes=30, scales=[1, 2, 4, 8])
        
        assert encoder.n_modes == 30
        assert encoder.scales == [1, 2, 4, 8]
        assert encoder.n_scales == 4
        assert len(encoder.scale_weights) == 4
    
    def test_encode_single_mode(self):
        """Test encoding of single mode."""
        encoder = MultiscaleEncoder(n_modes=1, scales=[1, 2])
        
        # Single mode time series
        time_series = np.random.randn(100)
        encoded = encoder.encode(time_series)
        
        assert 1 in encoded
        assert 2 in encoded
        assert len(encoded[1]) <= len(time_series)
        assert len(encoded[2]) <= len(encoded[1])
    
    def test_encode_multiple_modes(self):
        """Test encoding of multiple modes."""
        encoder = MultiscaleEncoder(n_modes=10, scales=[1, 2, 4])
        
        # Multiple modes
        time_series = np.random.randn(10, 100)
        encoded = encoder.encode(time_series)
        
        assert 1 in encoded
        assert 2 in encoded
        assert 4 in encoded
        assert encoded[1].shape[0] == 10
    
    def test_multiscale_power(self):
        """Test multiscale power computation."""
        encoder = MultiscaleEncoder(n_modes=20, scales=[1, 2, 4])
        
        time_series = np.random.randn(20, 200)
        power = encoder.compute_multiscale_power(time_series)
        
        assert len(power) == 20
        assert np.all(power >= 0)
        assert not np.any(np.isnan(power))
    
    def test_missing_data_handling(self):
        """Test missing data reconstruction."""
        encoder = MultiscaleEncoder(n_modes=10, scales=[1, 2])
        
        # Create data with missing values
        data = np.random.randn(10, 100)
        missing_mask = np.random.rand(10, 100) < 0.3
        
        reconstructed = encoder.handle_missing_data(data, missing_mask)
        
        assert reconstructed.shape == data.shape
        assert not np.any(np.isnan(reconstructed))
        
        # Check that non-missing values are preserved
        assert np.allclose(reconstructed[~missing_mask], data[~missing_mask])
    
    def test_extract_features(self):
        """Test feature extraction from encoded data."""
        encoder = MultiscaleEncoder(n_modes=5, scales=[1, 2])
        
        time_series = np.random.randn(5, 50)
        encoded = encoder.encode(time_series)
        features = encoder.extract_features(encoded)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))


class TestRealtimeDecoder:
    """Tests for RealtimeDecoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = RealtimeDecoder(n_modes=30, buffer_size=100)
        
        assert decoder.n_modes == 30
        assert decoder.buffer_size == 100
        assert decoder.buffer.shape == (30, 100)
        assert decoder.buffer_index == 0
        assert not decoder.buffer_full
    
    def test_update(self):
        """Test decoder update with new sample."""
        decoder = RealtimeDecoder(n_modes=10, buffer_size=50)
        
        sample = np.random.randn(10)
        metrics = decoder.update(sample)
        
        assert 'H_mode' in metrics
        assert 'PR' in metrics
        assert 'R' in metrics
        assert 'S_dot' in metrics
        assert 'kappa' in metrics
        assert 'C_t' in metrics
        assert 'latency_ms' in metrics
        
        # Check that metrics are in valid ranges
        assert 0 <= metrics['H_mode'] <= 1
        assert 0 <= metrics['PR'] <= 1
        assert 0 <= metrics['R'] <= 1
        assert metrics['latency_ms'] >= 0
    
    def test_streaming(self):
        """Test streaming analysis over multiple samples."""
        decoder = RealtimeDecoder(n_modes=10, buffer_size=20)
        
        for i in range(30):
            sample = np.random.randn(10)
            metrics = decoder.update(sample)
            
            # After 20 samples, buffer should be full
            if i >= 20:
                assert decoder.buffer_full
    
    def test_state_prediction(self):
        """Test consciousness state prediction."""
        decoder = RealtimeDecoder(n_modes=20, buffer_size=50)
        
        # Generate samples
        for _ in range(60):
            sample = np.random.randn(20)
            metrics = decoder.update(sample)
        
        # Predict state
        state = decoder.predict_state(metrics)
        
        assert state in ['wake', 'nrem', 'rem', 'anesthesia']
    
    def test_reset(self):
        """Test decoder reset."""
        decoder = RealtimeDecoder(n_modes=10, buffer_size=50)
        
        # Generate some data
        for _ in range(10):
            sample = np.random.randn(10)
            decoder.update(sample)
        
        # Reset
        decoder.reset()
        
        assert decoder.buffer_index == 0
        assert not decoder.buffer_full
        assert decoder.n_samples == 0
        assert len(decoder.latencies) == 0
    
    def test_latency_stats(self):
        """Test latency statistics computation."""
        decoder = RealtimeDecoder(n_modes=10, buffer_size=50)
        
        # Generate samples
        for _ in range(20):
            sample = np.random.randn(10)
            decoder.update(sample)
        
        stats = decoder.get_latency_stats()
        
        assert 'mean_ms' in stats
        assert 'std_ms' in stats
        assert 'min_ms' in stats
        assert 'max_ms' in stats
        assert 'p50_ms' in stats
        assert 'p95_ms' in stats
        assert 'p99_ms' in stats
        
        # Check that stats are positive
        assert all(v >= 0 for v in stats.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
