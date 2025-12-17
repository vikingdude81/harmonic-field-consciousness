"""
Steering Protocol - Reality Steering Between Consciousness States

Implements quantum steering operations for transitioning between alternative
consciousness states (wake ↔ sleep ↔ anesthesia) based on local quantum
memory operations.

The steering protocol models:
- Local operations on brain regions affecting global conscious state
- Probabilistic state transitions with quantum-inspired formalism
- Steering operators that smoothly interpolate between states
- Time-dependent evolution of consciousness
"""

import numpy as np
from typing import Optional, Tuple, Dict, Callable
from scipy.linalg import expm
from .reality_register import RealityRegister, QuantumConsciousnessState


class SteeringProtocol:
    """
    Quantum steering protocol for consciousness state transitions.
    
    Implements steering operations that can transition the reality register
    between different consciousness states using local quantum operations.
    """
    
    def __init__(self, reality_register: RealityRegister):
        """
        Initialize the steering protocol.
        
        Args:
            reality_register: RealityRegister instance to operate on
        """
        self.register = reality_register
        self.n_modes = reality_register.n_modes
    
    def compute_steering_operator(
        self,
        target_state_name: str,
        strength: float = 1.0,
        local_modes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the steering operator for transitioning toward a target state.
        
        Args:
            target_state_name: Name of the target consciousness state
            strength: Steering strength parameter (0 to 1)
            local_modes: Optional array of mode indices for local steering
                        (if None, operates globally on all modes)
        
        Returns:
            Steering operator matrix (n_modes × n_modes)
        """
        target_state = self.register.get_basis_state(target_state_name)
        current_state = self.register.get_state()
        
        # Build steering operator as a rotation in state space
        # U = exp(-i * strength * H_steering)
        # where H_steering guides current state toward target
        
        # Compute the "steering Hamiltonian" as outer product
        steering_hamiltonian = np.outer(
            target_state.amplitudes,
            np.conj(current_state.amplitudes)
        )
        
        # Make Hermitian
        steering_hamiltonian = (
            steering_hamiltonian + np.conj(steering_hamiltonian.T)
        ) / 2
        
        # Apply local masking if specified
        if local_modes is not None:
            mask = np.zeros((self.n_modes, self.n_modes))
            for i in local_modes:
                for j in local_modes:
                    mask[i, j] = 1.0
            steering_hamiltonian *= mask
        
        # Compute unitary evolution operator
        steering_operator = expm(-1j * strength * steering_hamiltonian)
        
        return steering_operator
    
    def apply_steering_operator(
        self,
        operator: np.ndarray,
        update_register: bool = True
    ) -> QuantumConsciousnessState:
        """
        Apply a steering operator to the current state.
        
        Args:
            operator: Steering operator matrix
            update_register: If True, update the register's current state
        
        Returns:
            New quantum consciousness state after steering
        """
        current_state = self.register.get_state()
        
        # Apply operator
        new_amplitudes = operator @ current_state.amplitudes
        
        # Normalize
        new_amplitudes /= np.sqrt(np.sum(np.abs(new_amplitudes) ** 2))
        
        new_power = np.abs(new_amplitudes) ** 2
        new_phases = np.angle(new_amplitudes)
        
        new_state = QuantumConsciousnessState(
            amplitudes=new_amplitudes,
            phases=new_phases,
            power=new_power,
            label=f"steered_from_{current_state.label}"
        )
        
        if update_register:
            self.register.set_state(new_state)
        
        return new_state
    
    def steer_to_state(
        self,
        target_state_name: str,
        strength: float = 0.1,
        local_modes: Optional[np.ndarray] = None,
        update_register: bool = True
    ) -> QuantumConsciousnessState:
        """
        Steer the current state toward a target consciousness state.
        
        Args:
            target_state_name: Name of target state
            strength: Steering strength (0 to 1, small values for gradual steering)
            local_modes: Optional mode indices for local steering
            update_register: If True, update the register's current state
        
        Returns:
            New state after steering
        """
        operator = self.compute_steering_operator(
            target_state_name, strength, local_modes
        )
        return self.apply_steering_operator(operator, update_register)
    
    def gradual_steering(
        self,
        target_state_name: str,
        n_steps: int = 10,
        total_strength: float = 1.0,
        local_modes: Optional[np.ndarray] = None
    ) -> list:
        """
        Perform gradual steering over multiple steps.
        
        Args:
            target_state_name: Name of target state
            n_steps: Number of steering steps
            total_strength: Total steering strength to apply
            local_modes: Optional mode indices for local steering
        
        Returns:
            List of states at each step
        """
        step_strength = total_strength / n_steps
        states = [self.register.get_state()]
        
        for _ in range(n_steps):
            new_state = self.steer_to_state(
                target_state_name,
                strength=step_strength,
                local_modes=local_modes,
                update_register=True
            )
            states.append(new_state)
        
        return states
    
    def oscillatory_steering(
        self,
        state_a: str,
        state_b: str,
        n_cycles: int = 5,
        steps_per_cycle: int = 20
    ) -> list:
        """
        Create oscillatory transitions between two states (e.g., wake/sleep cycles).
        
        Args:
            state_a: First state name
            state_b: Second state name
            n_cycles: Number of complete cycles
            steps_per_cycle: Time steps per cycle
        
        Returns:
            List of states showing oscillatory behavior
        """
        states = [self.register.get_state()]
        
        for cycle in range(n_cycles):
            # Phase of cycle
            for step in range(steps_per_cycle):
                t = step / steps_per_cycle
                phase = 2 * np.pi * t
                
                # Determine which direction to steer based on phase
                if phase < np.pi:
                    # First half: steer toward state_b
                    target = state_b
                    strength = 0.05
                else:
                    # Second half: steer toward state_a
                    target = state_a
                    strength = 0.05
                
                new_state = self.steer_to_state(
                    target, strength=strength, update_register=True
                )
                states.append(new_state)
        
        return states
    
    def local_to_global_steering(
        self,
        target_state_name: str,
        local_region_size: int = 5,
        n_steps: int = 10
    ) -> Dict[str, list]:
        """
        Demonstrate how local operations can affect global consciousness state.
        
        Applies steering to progressively larger regions, showing emergence
        of global state change from local operations.
        
        Args:
            target_state_name: Target consciousness state
            local_region_size: Size of initial local region
            n_steps: Number of expansion steps
        
        Returns:
            Dictionary with 'states' and 'region_sizes' lists
        """
        states = [self.register.get_state()]
        region_sizes = [0]
        
        for step in range(n_steps):
            # Expand local region
            region_size = min(
                local_region_size + step * (self.n_modes // n_steps),
                self.n_modes
            )
            local_modes = np.arange(region_size)
            
            # Apply local steering
            new_state = self.steer_to_state(
                target_state_name,
                strength=0.1,
                local_modes=local_modes,
                update_register=True
            )
            
            states.append(new_state)
            region_sizes.append(region_size)
        
        return {
            'states': states,
            'region_sizes': region_sizes
        }


def steer_consciousness_state(
    register: RealityRegister,
    target_state: str,
    strength: float = 0.1,
    local_modes: Optional[np.ndarray] = None
) -> QuantumConsciousnessState:
    """
    Convenience function to steer a reality register toward a target state.
    
    Args:
        register: RealityRegister to steer
        target_state: Name of target consciousness state
        strength: Steering strength parameter
        local_modes: Optional mode indices for local steering
    
    Returns:
        New quantum consciousness state after steering
    """
    protocol = SteeringProtocol(register)
    return protocol.steer_to_state(target_state, strength, local_modes)


def compute_steering_probability(
    current_state: QuantumConsciousnessState,
    target_state: QuantumConsciousnessState,
    steering_strength: float = 0.1
) -> float:
    """
    Compute the probability of successfully steering to a target state.
    
    Args:
        current_state: Current quantum consciousness state
        target_state: Target quantum consciousness state
        steering_strength: Steering strength parameter
    
    Returns:
        Transition probability ∈ [0, 1]
    """
    # Base transition probability from overlap
    base_prob = current_state.overlap_probability(target_state)
    
    # Modulate by steering strength
    # Higher strength increases probability of transition
    effective_prob = base_prob + (1 - base_prob) * steering_strength
    
    return float(np.clip(effective_prob, 0, 1))


def compute_steering_trajectory(
    register: RealityRegister,
    target_state: str,
    n_steps: int = 50,
    strength_per_step: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Compute the trajectory of consciousness metrics during steering.
    
    Args:
        register: RealityRegister instance
        target_state: Target consciousness state name
        n_steps: Number of steering steps
        strength_per_step: Steering strength per step
    
    Returns:
        Dictionary with arrays of metrics over time
    """
    protocol = SteeringProtocol(register)
    
    # Track metrics
    overlaps = []
    entropies = []
    norms = []
    
    for _ in range(n_steps):
        state = register.get_state()
        
        # Record metrics
        target = register.get_basis_state(target_state)
        overlaps.append(state.overlap_probability(target))
        
        # Compute mode entropy
        power = state.power
        power_norm = power / (power.sum() + 1e-12)
        p = power_norm[power_norm > 1e-12]
        entropy = -np.sum(p * np.log(p + 1e-12))
        entropies.append(entropy)
        
        norms.append(state.norm)
        
        # Apply steering
        protocol.steer_to_state(
            target_state, strength=strength_per_step, update_register=True
        )
    
    return {
        'overlaps': np.array(overlaps),
        'entropies': np.array(entropies),
        'norms': np.array(norms),
        'steps': np.arange(n_steps)
    }
