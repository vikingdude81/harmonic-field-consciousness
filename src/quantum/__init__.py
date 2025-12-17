"""
Quantum Reality Steering Module

Implements quantum-inspired formalism for consciousness state transitions
based on "Steering Alternative Realities through Local Quantum Memory Operations"
(arXiv:2512.14377) by Xiongfeng Ma.

Key concepts:
- Quantum reality register for consciousness states
- Reality steering protocols between wake/sleep/anesthesia
- Local operations affecting global conscious state
- Quantum measurement and collapse
- Entanglement between brain regions
"""

from .reality_register import RealityRegister, QuantumConsciousnessState
from .steering_protocol import (
    SteeringProtocol,
    steer_consciousness_state,
    compute_steering_probability
)
from .quantum_measurement import (
    QuantumMeasurement,
    measure_consciousness_state,
    apply_measurement_collapse
)
from .entanglement import (
    compute_entanglement_entropy,
    compute_mutual_information,
    compute_regional_correlations,
    model_nonlocal_effects
)

__all__ = [
    'RealityRegister',
    'QuantumConsciousnessState',
    'SteeringProtocol',
    'steer_consciousness_state',
    'compute_steering_probability',
    'QuantumMeasurement',
    'measure_consciousness_state',
    'apply_measurement_collapse',
    'compute_entanglement_entropy',
    'compute_mutual_information',
    'compute_regional_correlations',
    'model_nonlocal_effects',
]
