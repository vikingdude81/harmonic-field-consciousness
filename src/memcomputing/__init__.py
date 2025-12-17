"""
Memcomputing Module

Implements meminductor-based RLC circuits for neuromorphic consciousness modeling
with memory-encoded dynamics.

Based on: "Beyond Memristor: Neuromorphic Computing Using Meminductor"
arXiv:2512.11002
"""

from .meminductor import Meminductor
from .rlc_dynamics import RLCOscillator

__all__ = [
    'Meminductor',
    'RLCOscillator',
]
