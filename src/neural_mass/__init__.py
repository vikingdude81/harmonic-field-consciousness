"""
Neural Mass Models Module

Implements neural mass models based on push-pull oscillator framework
from "A Rosetta Stone of Neural Mass Models" (arXiv:2512.10982).

This module provides:
- Push-pull oscillator dynamics (E-I populations)
- Multi-scale hierarchical oscillators
- Bridge between neural mass models and harmonic field theory
- Integration with consciousness modeling framework
"""

from .push_pull_oscillator import (
    PushPullOscillator,
    MultiScalePushPull
)

from .harmonic_bridge import (
    oscillation_to_harmonic_mode,
    HarmonicNeuralMassModel
)

__all__ = [
    'PushPullOscillator',
    'MultiScalePushPull',
    'oscillation_to_harmonic_mode',
    'HarmonicNeuralMassModel',
]
