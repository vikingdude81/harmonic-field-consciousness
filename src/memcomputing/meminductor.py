"""
Meminductor Model

Implements meminductor (inductor with memory) for neuromorphic consciousness modeling.
"""

import numpy as np
from typing import Tuple


class Meminductor:
    """
    Meminductor: inductor with memory-dependent inductance.
    
    The inductance L depends on the historical magnetic flux, creating
    memory effects in the circuit dynamics.
    """
    
    def __init__(self, L0: float = 1.0, alpha: float = 0.1, beta: float = 0.01):
        """
        Initialize meminductor.
        
        Args:
            L0: Base inductance (H)
            alpha: Memory strength parameter
            beta: Nonlinearity parameter
        """
        self.L0 = L0
        self.alpha = alpha
        self.beta = beta
        
        # State variables
        self.phi = 0.0  # Magnetic flux
        self.current = 0.0
        self.L = L0  # Current inductance value
    
    def update_inductance(self):
        """Update inductance based on flux history."""
        # Inductance depends on magnetic flux (memory)
        # L(φ) = L0 * (1 + α * tanh(β * φ))
        self.L = self.L0 * (1.0 + self.alpha * np.tanh(self.beta * self.phi))
    
    def update(self, voltage: float, dt: float) -> float:
        """
        Update meminductor state.
        
        Args:
            voltage: Applied voltage (V)
            dt: Time step (s)
        
        Returns:
            Current through meminductor (A)
        """
        # Update flux: dφ/dt = V
        self.phi += voltage * dt
        
        # Update inductance based on flux
        self.update_inductance()
        
        # Current-voltage relation: V = L * dI/dt
        # Therefore: dI/dt = V / L
        dI_dt = voltage / (self.L + 1e-12)
        self.current += dI_dt * dt
        
        return self.current
    
    def get_state(self) -> Tuple[float, float, float]:
        """
        Get meminductor state.
        
        Returns:
            Tuple of (current, flux, inductance)
        """
        return self.current, self.phi, self.L
    
    def reset(self):
        """Reset meminductor state."""
        self.phi = 0.0
        self.current = 0.0
        self.L = self.L0
    
    def encode_memory(self, amplitude: float):
        """
        Encode a harmonic amplitude in the meminductor's magnetic flux.
        
        Args:
            amplitude: Harmonic mode amplitude to encode
        """
        # Store amplitude in magnetic flux
        self.phi = amplitude * self.L0
        self.update_inductance()
    
    def read_memory(self) -> float:
        """
        Read stored memory from magnetic flux.
        
        Returns:
            Decoded amplitude
        """
        # Retrieve amplitude from flux
        return self.phi / (self.L0 + 1e-12)
    
    def memory_persistence(self, leak_rate: float, dt: float):
        """
        Apply memory decay/leak.
        
        Args:
            leak_rate: Rate of flux decay (1/s)
            dt: Time step (s)
        """
        # Exponential decay of magnetic flux
        self.phi *= np.exp(-leak_rate * dt)
        self.update_inductance()
