"""
Spiking Neural Network Module

Implements spiking neuron models for consciousness modeling with dynamic network
growth and pathway reusability.

Based on: "Spiking Neural Networks with Random Graph Architectures"
arXiv:2512.11743
"""

from .lif_neuron import LIFNeuron, LIFNetwork
from .spike_encoder import SpikeEncoder
from .spike_metrics import compute_spike_metrics

__all__ = [
    'LIFNeuron',
    'LIFNetwork',
    'SpikeEncoder',
    'compute_spike_metrics',
]
