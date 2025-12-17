"""
Quantum Measurement - Measurement and State Collapse Operations

Implements quantum measurement operations for consciousness states including:
- Projective measurements in different bases
- State collapse following measurement
- Observable operators for consciousness metrics
- Continuous monitoring and weak measurements
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from .reality_register import RealityRegister, QuantumConsciousnessState


class QuantumMeasurement:
    """
    Quantum measurement operations for consciousness states.
    
    Provides various measurement schemes including:
    - Projective measurements in mode basis
    - Consciousness state measurements
    - Observable measurements (energy, entropy, etc.)
    - Weak measurements (partial state collapse)
    """
    
    def __init__(self, reality_register: RealityRegister):
        """
        Initialize quantum measurement system.
        
        Args:
            reality_register: RealityRegister to perform measurements on
        """
        self.register = reality_register
        self.n_modes = reality_register.n_modes
        self.measurement_history: List[Dict] = []
    
    def measure_mode_occupation(
        self,
        mode_index: int,
        collapse: bool = True
    ) -> Tuple[float, QuantumConsciousnessState]:
        """
        Measure the occupation of a specific harmonic mode.
        
        Args:
            mode_index: Index of mode to measure
            collapse: Whether to collapse state after measurement
        
        Returns:
            Tuple of (measured_power, collapsed_state)
        """
        state = self.register.get_state()
        
        # Measurement outcome is the power in that mode
        measured_power = state.power[mode_index]
        
        if collapse:
            # Collapse: project onto measured mode
            new_amplitudes = np.zeros(self.n_modes, dtype=complex)
            new_amplitudes[mode_index] = 1.0
            
            collapsed_state = QuantumConsciousnessState(
                amplitudes=new_amplitudes,
                phases=np.zeros(self.n_modes),
                power=np.abs(new_amplitudes) ** 2,
                label=f"collapsed_mode_{mode_index}"
            )
            
            self.register.set_state(collapsed_state)
        else:
            collapsed_state = state
        
        # Record measurement
        self.measurement_history.append({
            'type': 'mode_occupation',
            'mode': mode_index,
            'outcome': measured_power,
            'collapsed': collapse
        })
        
        return float(measured_power), collapsed_state
    
    def measure_consciousness_state(
        self,
        collapse: bool = True
    ) -> Tuple[str, float, QuantumConsciousnessState]:
        """
        Measure which consciousness state the system is in.
        
        Performs a projective measurement in the basis of consciousness states.
        
        Args:
            collapse: Whether to collapse state after measurement
        
        Returns:
            Tuple of (measured_state_name, probability, collapsed_state)
        """
        # Get overlap probabilities with all basis states
        decomposition = self.register.get_state_decomposition()
        
        # Sample from probability distribution
        state_names = list(decomposition.keys())
        probabilities = np.array([decomposition[name] for name in state_names])
        probabilities /= probabilities.sum()  # Normalize
        
        # Measure (sample)
        measured_idx = self.register.rng.choice(len(state_names), p=probabilities)
        measured_state_name = state_names[measured_idx]
        measured_prob = probabilities[measured_idx]
        
        if collapse:
            # Collapse onto measured basis state
            collapsed_state = self.register.get_basis_state(measured_state_name)
            self.register.set_state(collapsed_state)
        else:
            collapsed_state = self.register.get_state()
        
        # Record measurement
        self.measurement_history.append({
            'type': 'consciousness_state',
            'outcome': measured_state_name,
            'probability': float(measured_prob),
            'collapsed': collapse
        })
        
        return measured_state_name, float(measured_prob), collapsed_state
    
    def measure_power_distribution(
        self,
        n_samples: int = 1
    ) -> np.ndarray:
        """
        Measure the power distribution across modes.
        
        Non-collapsing measurement that samples the power distribution.
        
        Args:
            n_samples: Number of measurement samples
        
        Returns:
            Array of measured power distributions (n_samples × n_modes)
        """
        state = self.register.get_state()
        
        # Add measurement noise to simulate realistic measurements
        measurements = np.zeros((n_samples, self.n_modes))
        
        for i in range(n_samples):
            # True power with Gaussian noise
            noise = 0.05 * self.register.rng.randn(self.n_modes)
            measured = state.power + noise
            measured = np.maximum(measured, 0)  # Ensure non-negative
            measurements[i] = measured / measured.sum()
        
        return measurements
    
    def weak_measurement(
        self,
        observable: np.ndarray,
        strength: float = 0.1
    ) -> Tuple[float, QuantumConsciousnessState]:
        """
        Perform a weak measurement that only partially collapses the state.
        
        Args:
            observable: Observable operator (Hermitian matrix)
            strength: Measurement strength ∈ [0, 1]
                     (0 = no collapse, 1 = full projection)
        
        Returns:
            Tuple of (measurement_outcome, weakly_collapsed_state)
        """
        state = self.register.get_state()
        
        # Compute expectation value
        expectation = np.real(
            np.vdot(state.amplitudes, observable @ state.amplitudes)
        )
        
        # Weak measurement causes partial collapse
        # New state is a mixture of original and collapsed
        collapsed_amplitudes = observable @ state.amplitudes
        collapsed_amplitudes /= np.linalg.norm(collapsed_amplitudes)
        
        # Interpolate between original and collapsed
        new_amplitudes = (
            (1 - strength) * state.amplitudes + 
            strength * collapsed_amplitudes
        )
        new_amplitudes /= np.linalg.norm(new_amplitudes)
        
        new_state = QuantumConsciousnessState(
            amplitudes=new_amplitudes,
            phases=np.angle(new_amplitudes),
            power=np.abs(new_amplitudes) ** 2,
            label=f"weak_measured"
        )
        
        self.register.set_state(new_state)
        
        # Record measurement
        self.measurement_history.append({
            'type': 'weak_measurement',
            'outcome': float(expectation),
            'strength': strength
        })
        
        return float(expectation), new_state
    
    def measure_entropy(self) -> float:
        """
        Measure the mode entropy of the current state.
        
        This is a non-collapsing measurement of a consciousness metric.
        
        Returns:
            Mode entropy (Shannon entropy of power distribution)
        """
        state = self.register.get_state()
        power = state.power
        power = power / (power.sum() + 1e-12)
        
        # Filter out zero values
        p = power[power > 1e-12]
        
        # Compute Shannon entropy
        entropy = -np.sum(p * np.log(p + 1e-12))
        
        # Record measurement
        self.measurement_history.append({
            'type': 'entropy',
            'outcome': float(entropy)
        })
        
        return float(entropy)
    
    def measure_participation_ratio(self) -> float:
        """
        Measure the participation ratio of the current state.
        
        Non-collapsing measurement.
        
        Returns:
            Participation ratio (effective number of active modes)
        """
        state = self.register.get_state()
        power = state.power
        power = power / (power.sum() + 1e-12)
        
        pr = 1.0 / (np.sum(power ** 2) + 1e-12)
        
        # Record measurement
        self.measurement_history.append({
            'type': 'participation_ratio',
            'outcome': float(pr)
        })
        
        return float(pr)
    
    def measure_phase_coherence(self) -> float:
        """
        Measure the phase coherence (Kuramoto order parameter).
        
        Non-collapsing measurement.
        
        Returns:
            Phase coherence ∈ [0, 1]
        """
        state = self.register.get_state()
        
        # Compute Kuramoto order parameter
        phases = state.phases
        power = state.power
        power = power / (power.sum() + 1e-12)
        
        # R = |sum(p_k * exp(i*phase_k))|
        coherence = np.abs(np.sum(power * np.exp(1j * phases)))
        
        # Record measurement
        self.measurement_history.append({
            'type': 'phase_coherence',
            'outcome': float(coherence)
        })
        
        return float(coherence)
    
    def continuous_monitoring(
        self,
        n_timesteps: int,
        measurement_rate: float = 0.1,
        weak_measurement_strength: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Perform continuous weak measurements over time.
        
        Models continuous monitoring of consciousness state with weak measurements
        that gradually affect the quantum state.
        
        Args:
            n_timesteps: Number of time steps to monitor
            measurement_rate: Probability of measurement per timestep
            weak_measurement_strength: Strength of each weak measurement
        
        Returns:
            Dictionary with time series of measurements
        """
        entropies = []
        prs = []
        coherences = []
        measured_states = []
        
        for t in range(n_timesteps):
            # Measure with some probability
            if self.register.rng.rand() < measurement_rate:
                # Weak measurement of energy operator
                # (proportional to mode index)
                energy_operator = np.diag(np.arange(self.n_modes, dtype=float))
                _, _ = self.weak_measurement(
                    energy_operator, 
                    strength=weak_measurement_strength
                )
            
            # Record metrics
            entropies.append(self.measure_entropy())
            prs.append(self.measure_participation_ratio())
            coherences.append(self.measure_phase_coherence())
            
            state_name, prob, _ = self.measure_consciousness_state(collapse=False)
            measured_states.append((state_name, prob))
        
        return {
            'time': np.arange(n_timesteps),
            'entropies': np.array(entropies),
            'participation_ratios': np.array(prs),
            'coherences': np.array(coherences),
            'measured_states': measured_states
        }
    
    def get_measurement_history(self) -> List[Dict]:
        """Get the history of all measurements."""
        return self.measurement_history
    
    def clear_measurement_history(self):
        """Clear the measurement history."""
        self.measurement_history = []


def measure_consciousness_state(
    register: RealityRegister,
    collapse: bool = True
) -> Tuple[str, float]:
    """
    Convenience function to measure consciousness state.
    
    Args:
        register: RealityRegister to measure
        collapse: Whether to collapse state after measurement
    
    Returns:
        Tuple of (measured_state_name, probability)
    """
    measurement = QuantumMeasurement(register)
    state_name, prob, _ = measurement.measure_consciousness_state(collapse)
    return state_name, prob


def apply_measurement_collapse(
    state: QuantumConsciousnessState,
    measurement_basis: str = 'mode'
) -> QuantumConsciousnessState:
    """
    Apply measurement collapse to a quantum state.
    
    Args:
        state: Quantum consciousness state to collapse
        measurement_basis: Basis for measurement ('mode' or 'power')
    
    Returns:
        Collapsed quantum state
    """
    if measurement_basis == 'mode':
        # Collapse to highest power mode
        max_mode = np.argmax(state.power)
        amplitudes = np.zeros(len(state.amplitudes), dtype=complex)
        amplitudes[max_mode] = 1.0
        
    elif measurement_basis == 'power':
        # Sample from power distribution
        power_norm = state.power / state.power.sum()
        sampled_mode = np.random.choice(len(state.power), p=power_norm)
        amplitudes = np.zeros(len(state.amplitudes), dtype=complex)
        amplitudes[sampled_mode] = 1.0
    else:
        raise ValueError(f"Unknown measurement basis: {measurement_basis}")
    
    return QuantumConsciousnessState(
        amplitudes=amplitudes,
        phases=np.zeros(len(amplitudes)),
        power=np.abs(amplitudes) ** 2,
        label=f"collapsed_{measurement_basis}"
    )


def compute_measurement_probabilities(
    state: QuantumConsciousnessState,
    basis_states: List[QuantumConsciousnessState]
) -> np.ndarray:
    """
    Compute measurement probabilities in a given basis.
    
    Args:
        state: Quantum state to measure
        basis_states: List of basis states
    
    Returns:
        Array of measurement probabilities
    """
    probabilities = np.zeros(len(basis_states))
    
    for i, basis_state in enumerate(basis_states):
        probabilities[i] = state.overlap_probability(basis_state)
    
    # Normalize (should already be normalized, but ensure)
    probabilities /= probabilities.sum()
    
    return probabilities
