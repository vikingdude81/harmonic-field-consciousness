"""
Multiscale Dynamics Module

Implements multiscale encoder/decoder for handling different harmonic mode
frequencies and temporal resolutions in consciousness modeling.

Based on: "Dynamical modeling of nonlinear latent factors in multiscale neural activity"
arXiv:2512.12462
"""

from .encoder import MultiscaleEncoder
from .decoder import RealtimeDecoder

__all__ = [
    'MultiscaleEncoder',
    'RealtimeDecoder',
]
