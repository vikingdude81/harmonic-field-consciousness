"""
Spatiotemporal Transformer Module

Implements BaRISTA (Brain Scale Informed Spatiotemporal Representation) architecture
for consciousness state prediction and analysis.

Based on: "Brain Scale Informed Spatiotemporal Representation"
arXiv:2512.12135
"""

from .barista import BaRISTAModel
from .attention import AttentionVisualizer

__all__ = [
    'BaRISTAModel',
    'AttentionVisualizer',
]
