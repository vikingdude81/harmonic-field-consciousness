"""
Unit Tests for Transformer Modules

Tests for:
- BaRISTAModel: Spatiotemporal transformer for consciousness
- AttentionVisualizer: Attention visualization tools
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from src.transformers import BaRISTAModel, AttentionVisualizer


class TestBaRISTAModel:
    """Tests for BaRISTAModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = BaRISTAModel(n_regions=8, n_features=32, n_heads=4)
        
        assert model.n_regions == 8
        assert model.n_features == 32
        assert model.n_heads == 4
        assert model.attention_weights.shape == (4, 8, 8)
    
    def test_encode_regions_single_timepoint(self):
        """Test region encoding with single timepoint."""
        model = BaRISTAModel(n_regions=5, n_features=16)
        
        # Single timepoint per region
        region_data = np.random.randn(5)
        tokens = model.encode_regions(region_data)
        
        assert tokens.shape == (5, 16)
    
    def test_encode_regions_time_series(self):
        """Test region encoding with time series."""
        model = BaRISTAModel(n_regions=5, n_features=16)
        
        # Time series per region
        region_data = np.random.randn(5, 100)
        tokens = model.encode_regions(region_data)
        
        assert tokens.shape == (5, 16)
    
    def test_compute_attention(self):
        """Test attention computation."""
        model = BaRISTAModel(n_regions=6, n_features=16, n_heads=2)
        
        tokens = np.random.randn(6, 16)
        attention = model.compute_attention(tokens, head=0)
        
        assert attention.shape == (6, 6)
        # Attention should be normalized (rows sum to 1)
        assert np.allclose(np.sum(attention, axis=1), 1.0)
        # Attention should be non-negative
        assert np.all(attention >= 0)
    
    def test_forward(self):
        """Test forward pass."""
        model = BaRISTAModel(n_regions=8, n_features=32, n_heads=4)
        
        region_data = np.random.randn(8, 100)
        outputs = model.forward(region_data)
        
        assert 'tokens' in outputs
        assert 'attention' in outputs
        assert 'output' in outputs
        
        assert outputs['tokens'].shape == (8, 32)
        assert outputs['attention'].shape == (4, 8, 8)
        assert outputs['output'].shape == (8, 32)
    
    def test_predict_consciousness_state(self):
        """Test consciousness state prediction."""
        model = BaRISTAModel(n_regions=8, n_features=32)
        
        region_data = np.random.randn(8, 100)
        prediction = model.predict_consciousness_state(region_data)
        
        assert 'state' in prediction
        assert 'C_score' in prediction
        assert 'H_mode' in prediction
        assert 'PR' in prediction
        assert 'attention' in prediction
        
        assert prediction['state'] in ['wake', 'nrem', 'rem', 'anesthesia']
        assert 0 <= prediction['C_score'] <= 1
        assert 0 <= prediction['H_mode'] <= 1
        assert 0 <= prediction['PR'] <= 1
    
    def test_masked_reconstruction(self):
        """Test masked reconstruction."""
        model = BaRISTAModel(n_regions=10, n_features=16)
        
        region_data = np.random.randn(10, 100)
        masked_data, reconstructed = model.masked_reconstruction(region_data, mask_ratio=0.3)
        
        assert masked_data.shape == region_data.shape
        assert reconstructed.shape == region_data.shape
        
        # Some regions should be masked (set to zero)
        n_masked = np.sum(np.all(masked_data == 0, axis=1))
        assert n_masked >= 2  # At least some masked with 30%


class TestAttentionVisualizer:
    """Tests for AttentionVisualizer."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        region_names = ['R1', 'R2', 'R3']
        viz = AttentionVisualizer(region_names=region_names)
        
        assert viz.region_names == region_names
    
    def test_plot_attention_matrix(self):
        """Test attention matrix plotting."""
        viz = AttentionVisualizer()
        
        # Single head attention
        attention = np.random.rand(5, 5)
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig = viz.plot_attention_matrix(attention, title="Test Attention")
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_attention_matrix_multihead(self):
        """Test multi-head attention matrix plotting."""
        viz = AttentionVisualizer()
        
        # Multi-head attention
        attention = np.random.rand(3, 5, 5)
        attention = attention / attention.sum(axis=2, keepdims=True)
        
        fig = viz.plot_attention_matrix(attention, head=1)
        
        assert fig is not None
    
    def test_plot_attention_flow(self):
        """Test attention flow plotting."""
        viz = AttentionVisualizer(region_names=['R1', 'R2', 'R3', 'R4'])
        
        attention = np.random.rand(4, 4)
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig = viz.plot_attention_flow(attention, top_k=2)
        
        assert fig is not None
    
    def test_plot_attention_entropy(self):
        """Test attention entropy plotting."""
        viz = AttentionVisualizer()
        
        attention = np.random.rand(6, 6)
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig = viz.plot_attention_entropy(attention)
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_multi_head_comparison(self):
        """Test multi-head comparison plotting."""
        viz = AttentionVisualizer()
        
        # Multi-head attention
        attention = np.random.rand(4, 5, 5)
        attention = attention / attention.sum(axis=2, keepdims=True)
        
        fig = viz.plot_multi_head_comparison(attention)
        
        assert fig is not None
    
    def test_plot_with_region_names(self):
        """Test plotting with region names."""
        region_names = ['V1', 'V2', 'MT', 'PFC']
        viz = AttentionVisualizer(region_names=region_names)
        
        attention = np.random.rand(4, 4)
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig = viz.plot_attention_matrix(attention)
        
        assert fig is not None
        # Check that labels are set
        ax = fig.axes[0]
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert any(name in ''.join(xticklabels) for name in region_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
