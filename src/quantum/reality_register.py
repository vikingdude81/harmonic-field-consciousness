"""
Reality Register - Quantum Memory for Consciousness States

Implements a quantum-inspired reality register that models consciousness states
as harmonic field configurations with quantum superposition properties.

The RealityRegister class represents the observer's "reality register" where:
- Harmonic modes serve as quantum basis states
- Consciousness states exist in superposition
- State amplitudes represent probability distributions
- Measurements collapse the state to a specific configuration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ConsciousnessState(Enum):
    """Enumeration of primary consciousness states."""
    WAKE = "wake"
    NREM_SLEEP = "nrem_sleep"
    REM_SLEEP = "rem_sleep"
    ANESTHESIA = "anesthesia"
    MEDITATION = "meditation"
    PSYCHEDELIC = "psychedelic"


@dataclass
class QuantumConsciousnessState:
    """
    Represents a quantum consciousness state as a superposition of harmonic modes.
    
    Attributes:
        amplitudes: Complex amplitudes for each harmonic mode (quantum state vector)
        phases: Phase angles for each mode (in radians)
        power: Power distribution across modes (|amplitude|^2)
        label: Optional label for the state (e.g., 'wake', 'sleep')
        timestamp: Optional timestamp for temporal evolution
    """
    amplitudes: np.ndarray
    phases: np.ndarray
    power: np.ndarray
    label: Optional[str] = None
    timestamp: Optional[float] = None
    
    @property
    def n_modes(self) -> int:
        """Number of harmonic modes."""
        return len(self.amplitudes)
    
    @property
    def norm(self) -> float:
        """Quantum state norm (should be 1 for normalized states)."""
        return float(np.sqrt(np.sum(np.abs(self.amplitudes) ** 2)))
    
    def normalize(self) -> 'QuantumConsciousnessState':
        """Return a normalized copy of this state."""
        norm = self.norm
        if norm < 1e-12:
            raise ValueError("Cannot normalize zero state")
        
        return QuantumConsciousnessState(
            amplitudes=self.amplitudes / norm,
            phases=self.phases,
            power=self.power / np.sum(self.power),
            label=self.label,
            timestamp=self.timestamp
        )
    
    def inner_product(self, other: 'QuantumConsciousnessState') -> complex:
        """
        Compute inner product with another quantum state.
        
        Returns:
            Complex inner product <self|other>
        """
        return np.vdot(self.amplitudes, other.amplitudes)
    
    def overlap_probability(self, other: 'QuantumConsciousnessState') -> float:
        """
        Compute overlap probability with another state.
        
        Returns:
            Probability |<self|other>|^2 ∈ [0, 1]
        """
        return float(np.abs(self.inner_product(other)) ** 2)


class RealityRegister:
    """
    Quantum reality register for consciousness states.
    
    Models the observer's reality register as a quantum system where consciousness
    states are represented as superpositions of harmonic field modes. The register
    supports:
    - Superposition of multiple consciousness states
    - Quantum evolution and steering operations
    - Measurement and state collapse
    - Memory of past states (quantum register history)
    """
    
    def __init__(
        self,
        n_modes: int,
        initial_state: Optional[QuantumConsciousnessState] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the reality register.
        
        Args:
            n_modes: Number of harmonic modes (dimension of quantum state space)
            initial_state: Initial quantum state (if None, defaults to vacuum state)
            seed: Random seed for reproducibility
        """
        self.n_modes = n_modes
        self.rng = np.random.RandomState(seed)
        
        # Initialize or set state
        if initial_state is None:
            self.current_state = self._create_vacuum_state()
        else:
            if initial_state.n_modes != n_modes:
                raise ValueError(
                    f"Initial state has {initial_state.n_modes} modes, "
                    f"expected {n_modes}"
                )
            self.current_state = initial_state.normalize()
        
        # History of states (quantum register memory)
        self.state_history: List[QuantumConsciousnessState] = [self.current_state]
        
        # Basis states for different consciousness types
        self._basis_states: Dict[str, QuantumConsciousnessState] = {}
        self._initialize_basis_states()
    
    def _create_vacuum_state(self) -> QuantumConsciousnessState:
        """Create the vacuum (ground) state with all power in lowest mode."""
        amplitudes = np.zeros(self.n_modes, dtype=complex)
        amplitudes[0] = 1.0
        phases = np.zeros(self.n_modes)
        power = np.abs(amplitudes) ** 2
        
        return QuantumConsciousnessState(
            amplitudes=amplitudes,
            phases=phases,
            power=power,
            label="vacuum"
        )
    
    def _initialize_basis_states(self):
        """Initialize basis states for different consciousness types."""
        # Wake state: broad distribution across modes
        wake_power = self._generate_wake_distribution()
        self._basis_states['wake'] = self._create_state_from_power(
            wake_power, label='wake'
        )
        
        # NREM sleep: concentrated in low modes
        nrem_power = self._generate_nrem_distribution()
        self._basis_states['nrem_sleep'] = self._create_state_from_power(
            nrem_power, label='nrem_sleep'
        )
        
        # REM sleep: mixed distribution
        rem_power = self._generate_rem_distribution()
        self._basis_states['rem_sleep'] = self._create_state_from_power(
            rem_power, label='rem_sleep'
        )
        
        # Anesthesia: extreme low-mode concentration
        anesthesia_power = self._generate_anesthesia_distribution()
        self._basis_states['anesthesia'] = self._create_state_from_power(
            anesthesia_power, label='anesthesia'
        )
        
        # Meditation: balanced, coherent state
        meditation_power = self._generate_meditation_distribution()
        self._basis_states['meditation'] = self._create_state_from_power(
            meditation_power, label='meditation'
        )
        
        # Psychedelic: enhanced high-mode activity
        psychedelic_power = self._generate_psychedelic_distribution()
        self._basis_states['psychedelic'] = self._create_state_from_power(
            psychedelic_power, label='psychedelic'
        )
    
    def _generate_wake_distribution(self) -> np.ndarray:
        """Generate power distribution for wake state."""
        k = np.arange(self.n_modes)
        power = 0.3 + 0.4 * np.exp(-k / 8) + 0.15 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _generate_nrem_distribution(self) -> np.ndarray:
        """Generate power distribution for NREM sleep."""
        k = np.arange(self.n_modes)
        power = np.exp(-k / 2) + 0.03 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _generate_rem_distribution(self) -> np.ndarray:
        """Generate power distribution for REM sleep."""
        k = np.arange(self.n_modes)
        power = 0.4 * np.exp(-k / 5) + 0.2 * np.exp(-k / 15)
        power += 0.1 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _generate_anesthesia_distribution(self) -> np.ndarray:
        """Generate power distribution for anesthesia."""
        k = np.arange(self.n_modes)
        power = np.exp(-k / 1.5) + 0.01 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _generate_meditation_distribution(self) -> np.ndarray:
        """Generate power distribution for meditation state."""
        k = np.arange(self.n_modes)
        # More uniform, balanced distribution
        power = 0.5 * np.exp(-k / 10) + 0.3 * np.ones(self.n_modes)
        power += 0.1 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _generate_psychedelic_distribution(self) -> np.ndarray:
        """Generate power distribution for psychedelic state."""
        k = np.arange(self.n_modes)
        # Enhanced high-mode activity
        power = 0.2 * np.exp(-k / 6) + 0.3 * np.exp(-k / 20)
        power += 0.2 * self.rng.rand(self.n_modes)
        return power / power.sum()
    
    def _create_state_from_power(
        self,
        power: np.ndarray,
        label: Optional[str] = None,
        phases: Optional[np.ndarray] = None
    ) -> QuantumConsciousnessState:
        """
        Create quantum state from power distribution.
        
        Args:
            power: Power distribution across modes
            label: Optional label for the state
            phases: Optional phase distribution (random if not provided)
        
        Returns:
            QuantumConsciousnessState
        """
        if phases is None:
            phases = 2 * np.pi * self.rng.rand(self.n_modes)
        
        amplitudes = np.sqrt(power) * np.exp(1j * phases)
        
        return QuantumConsciousnessState(
            amplitudes=amplitudes,
            phases=phases,
            power=power,
            label=label
        )
    
    def get_basis_state(self, state_name: str) -> QuantumConsciousnessState:
        """
        Get a basis state by name.
        
        Args:
            state_name: Name of the state ('wake', 'nrem_sleep', 'rem_sleep', 
                       'anesthesia', 'meditation', 'psychedelic')
        
        Returns:
            QuantumConsciousnessState
        """
        if state_name not in self._basis_states:
            raise ValueError(
                f"Unknown state '{state_name}'. "
                f"Available: {list(self._basis_states.keys())}"
            )
        return self._basis_states[state_name]
    
    def create_superposition(
        self,
        states: List[str],
        coefficients: Optional[np.ndarray] = None
    ) -> QuantumConsciousnessState:
        """
        Create a superposition of basis states.
        
        Args:
            states: List of state names to superpose
            coefficients: Complex coefficients for each state (equal if None)
        
        Returns:
            Superposed QuantumConsciousnessState
        """
        if len(states) == 0:
            raise ValueError("Must provide at least one state")
        
        if coefficients is None:
            coefficients = np.ones(len(states), dtype=complex) / np.sqrt(len(states))
        else:
            coefficients = np.asarray(coefficients, dtype=complex)
            if len(coefficients) != len(states):
                raise ValueError(
                    f"Number of coefficients ({len(coefficients)}) "
                    f"must match number of states ({len(states)})"
                )
        
        # Create superposition
        amplitudes = np.zeros(self.n_modes, dtype=complex)
        for state_name, coeff in zip(states, coefficients):
            basis_state = self.get_basis_state(state_name)
            amplitudes += coeff * basis_state.amplitudes
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        amplitudes /= norm
        
        power = np.abs(amplitudes) ** 2
        phases = np.angle(amplitudes)
        
        return QuantumConsciousnessState(
            amplitudes=amplitudes,
            phases=phases,
            power=power,
            label=f"superposition_{'_'.join(states)}"
        )
    
    def set_state(self, state: QuantumConsciousnessState):
        """
        Set the current state of the reality register.
        
        Args:
            state: New quantum state
        """
        if state.n_modes != self.n_modes:
            raise ValueError(
                f"State has {state.n_modes} modes, expected {self.n_modes}"
            )
        
        self.current_state = state.normalize()
        self.state_history.append(self.current_state)
    
    def get_state(self) -> QuantumConsciousnessState:
        """Get the current state of the reality register."""
        return self.current_state
    
    def compute_overlap_with_basis(self, state_name: str) -> float:
        """
        Compute overlap probability between current state and a basis state.
        
        Args:
            state_name: Name of the basis state
        
        Returns:
            Overlap probability ∈ [0, 1]
        """
        basis_state = self.get_basis_state(state_name)
        return self.current_state.overlap_probability(basis_state)
    
    def measure_consciousness_type(self) -> Tuple[str, float]:
        """
        Measure which consciousness type the current state most resembles.
        
        Returns:
            Tuple of (state_name, probability)
        """
        overlaps = {}
        for state_name in self._basis_states.keys():
            overlaps[state_name] = self.compute_overlap_with_basis(state_name)
        
        # Find maximum overlap
        best_state = max(overlaps.items(), key=lambda x: x[1])
        return best_state
    
    def get_state_decomposition(self) -> Dict[str, float]:
        """
        Decompose current state into basis state components.
        
        Returns:
            Dictionary mapping state names to overlap probabilities
        """
        return {
            state_name: self.compute_overlap_with_basis(state_name)
            for state_name in self._basis_states.keys()
        }
    
    def clear_history(self):
        """Clear the state history (keep current state)."""
        self.state_history = [self.current_state]
    
    def get_history_length(self) -> int:
        """Get the number of states in history."""
        return len(self.state_history)
