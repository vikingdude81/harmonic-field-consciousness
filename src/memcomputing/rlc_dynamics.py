"""
RLC Oscillator Dynamics with Meminductors

Implements RLC circuit oscillators for consciousness modeling with memory-dependent
inductance.
"""

import numpy as np
from typing import Tuple, Optional
from .meminductor import Meminductor


class RLCOscillator:
    """
    RLC oscillator with meminductor for memory effects.
    
    Models harmonic oscillations with resistance R, inductance L (memory-dependent),
    and capacitance C.
    """
    
    def __init__(self, R: float = 1.0, C: float = 1.0, L0: float = 1.0,
                 alpha: float = 0.1, beta: float = 0.01, dt: float = 0.01):
        """
        Initialize RLC oscillator.
        
        Args:
            R: Resistance (Ω)
            C: Capacitance (F)
            L0: Base inductance (H)
            alpha: Meminductor memory strength
            beta: Meminductor nonlinearity
            dt: Time step (s)
        """
        self.R = R
        self.C = C
        self.dt = dt
        
        # Meminductor
        self.meminductor = Meminductor(L0=L0, alpha=alpha, beta=beta)
        
        # State variables
        self.voltage = 0.0  # Capacitor voltage (V)
        self.current = 0.0  # Circuit current (A)
        self.t = 0.0
        
        # History
        self.voltage_history = []
        self.current_history = []
        self.time_history = []
    
    def update(self, external_voltage: float = 0.0) -> Tuple[float, float]:
        """
        Update RLC circuit for one time step.
        
        Args:
            external_voltage: External voltage source (V)
        
        Returns:
            Tuple of (voltage, current)
        """
        # Get current inductance
        _, _, L = self.meminductor.get_state()
        
        # RLC circuit equations:
        # L * dI/dt + R * I + V_C = V_external
        # C * dV_C/dt = I
        
        # Current through capacitor: I_C = C * dV_C/dt
        # Voltage across inductor: V_L = L * dI/dt
        
        # dI/dt = (V_external - R * I - V_C) / L
        dI_dt = (external_voltage - self.R * self.current - self.voltage) / (L + 1e-12)
        
        # dV_C/dt = I / C
        dV_dt = self.current / (self.C + 1e-12)
        
        # Update state
        self.current += dI_dt * self.dt
        self.voltage += dV_dt * self.dt
        
        # Update meminductor (using voltage across it)
        V_L = L * dI_dt
        self.meminductor.update(V_L, self.dt)
        
        # Record history
        self.voltage_history.append(self.voltage)
        self.current_history.append(self.current)
        self.time_history.append(self.t)
        
        self.t += self.dt
        
        return self.voltage, self.current
    
    def simulate(self, n_steps: int, external_voltage: float = 0.0,
                 external_signal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate RLC oscillator for multiple steps.
        
        Args:
            n_steps: Number of simulation steps
            external_voltage: Constant external voltage (used if external_signal is None)
            external_signal: Time-varying external voltage signal (n_steps,)
        
        Returns:
            Tuple of (voltage_array, current_array)
        """
        voltage_array = np.zeros(n_steps)
        current_array = np.zeros(n_steps)
        
        for i in range(n_steps):
            if external_signal is not None:
                V_ext = external_signal[i]
            else:
                V_ext = external_voltage
            
            v, i_curr = self.update(V_ext)
            voltage_array[i] = v
            current_array[i] = i_curr
        
        return voltage_array, current_array
    
    def get_natural_frequency(self) -> float:
        """
        Compute natural (resonant) frequency of RLC circuit.
        
        Returns:
            Natural frequency (Hz)
        """
        _, _, L = self.meminductor.get_state()
        omega_0 = 1.0 / np.sqrt((L + 1e-12) * (self.C + 1e-12))
        f_0 = omega_0 / (2 * np.pi)
        return f_0
    
    def get_damping_ratio(self) -> float:
        """
        Compute damping ratio of RLC circuit.
        
        Returns:
            Damping ratio (unitless)
        """
        _, _, L = self.meminductor.get_state()
        zeta = self.R / 2.0 * np.sqrt((self.C + 1e-12) / (L + 1e-12))
        return zeta
    
    def reset(self):
        """Reset oscillator state."""
        self.voltage = 0.0
        self.current = 0.0
        self.t = 0.0
        self.meminductor.reset()
        self.voltage_history = []
        self.current_history = []
        self.time_history = []
    
    def get_energy(self) -> float:
        """
        Compute total energy stored in circuit.
        
        Returns:
            Total energy (J)
        """
        _, _, L = self.meminductor.get_state()
        
        # Energy in capacitor: E_C = 0.5 * C * V^2
        E_C = 0.5 * self.C * self.voltage ** 2
        
        # Energy in inductor: E_L = 0.5 * L * I^2
        E_L = 0.5 * L * self.current ** 2
        
        return E_C + E_L


class RLCOscillatorBank:
    """
    Bank of RLC oscillators representing multiple harmonic modes.
    """
    
    def __init__(self, n_modes: int, R: float = 1.0, C: float = 1.0,
                 L0: float = 1.0, alpha: float = 0.1, beta: float = 0.01,
                 dt: float = 0.01):
        """
        Initialize oscillator bank.
        
        Args:
            n_modes: Number of oscillators (harmonic modes)
            R, C, L0, alpha, beta: Circuit parameters
            dt: Time step
        """
        self.n_modes = n_modes
        self.dt = dt
        
        # Create oscillators with different natural frequencies
        self.oscillators = []
        for k in range(n_modes):
            # Vary capacitance to create different frequencies
            C_k = C / (k + 1)
            osc = RLCOscillator(R=R, C=C_k, L0=L0, alpha=alpha, beta=beta, dt=dt)
            self.oscillators.append(osc)
    
    def update(self, external_inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update all oscillators.
        
        Args:
            external_inputs: External voltages for each oscillator (n_modes,)
        
        Returns:
            Tuple of (voltages, currents) arrays
        """
        voltages = np.zeros(self.n_modes)
        currents = np.zeros(self.n_modes)
        
        for i, osc in enumerate(self.oscillators):
            v, i_curr = osc.update(external_inputs[i])
            voltages[i] = v
            currents[i] = i_curr
        
        return voltages, currents
    
    def simulate(self, n_steps: int, external_inputs: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate oscillator bank.
        
        Args:
            n_steps: Number of steps
            external_inputs: External signals (n_steps, n_modes) or None for zero input
        
        Returns:
            Tuple of (voltages, currents) arrays of shape (n_steps, n_modes)
        """
        voltages = np.zeros((n_steps, self.n_modes))
        currents = np.zeros((n_steps, self.n_modes))
        
        for step in range(n_steps):
            if external_inputs is not None:
                inputs = external_inputs[step, :]
            else:
                inputs = np.zeros(self.n_modes)
            
            v, i_curr = self.update(inputs)
            voltages[step, :] = v
            currents[step, :] = i_curr
        
        return voltages, currents
    
    def get_amplitudes(self) -> np.ndarray:
        """
        Get current amplitudes from all oscillators.
        
        Returns:
            Amplitude array (n_modes,)
        """
        amplitudes = np.zeros(self.n_modes)
        for i, osc in enumerate(self.oscillators):
            # Amplitude from voltage
            amplitudes[i] = osc.voltage
        
        return amplitudes
    
    def reset(self):
        """Reset all oscillators."""
        for osc in self.oscillators:
            osc.reset()
