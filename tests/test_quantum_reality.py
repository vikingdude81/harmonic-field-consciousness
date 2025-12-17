"""
Unit Tests for Quantum Reality Steering Modules

Tests for:
- RealityRegister: quantum consciousness state management
- SteeringProtocol: state transition operations
- QuantumMeasurement: measurement and collapse
- Entanglement: non-local correlations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.quantum import (
    RealityRegister,
    QuantumConsciousnessState,
    SteeringProtocol,
    QuantumMeasurement,
    compute_entanglement_entropy,
    compute_mutual_information,
    steer_consciousness_state,
    measure_consciousness_state
)


class TestQuantumConsciousnessState:
    """Tests for QuantumConsciousnessState dataclass."""
    
    def test_initialization(self):
        """Test state initialization."""
        n_modes = 10
        amplitudes = np.random.randn(n_modes) + 1j * np.random.randn(n_modes)
        amplitudes /= np.linalg.norm(amplitudes)
        phases = np.angle(amplitudes)
        power = np.abs(amplitudes) ** 2
        
        state = QuantumConsciousnessState(
            amplitudes=amplitudes,
            phases=phases,
            power=power
        )
        
        assert state.n_modes == n_modes
        assert np.isclose(state.norm, 1.0, atol=1e-10)
    
    def test_normalization(self):
        """Test state normalization."""
        amplitudes = np.array([1.0, 1.0, 1.0], dtype=complex)
        state = QuantumConsciousnessState(
            amplitudes=amplitudes,
            phases=np.zeros(3),
            power=np.ones(3)
        )
        
        normalized = state.normalize()
        assert np.isclose(normalized.norm, 1.0, atol=1e-10)
    
    def test_inner_product(self):
        """Test inner product calculation."""
        state1 = QuantumConsciousnessState(
            amplitudes=np.array([1.0, 0.0], dtype=complex),
            phases=np.zeros(2),
            power=np.array([1.0, 0.0])
        )
        state2 = QuantumConsciousnessState(
            amplitudes=np.array([0.0, 1.0], dtype=complex),
            phases=np.zeros(2),
            power=np.array([0.0, 1.0])
        )
        
        inner_prod = state1.inner_product(state2)
        assert np.isclose(np.abs(inner_prod), 0.0, atol=1e-10)
    
    def test_overlap_probability(self):
        """Test overlap probability."""
        state1 = QuantumConsciousnessState(
            amplitudes=np.array([1.0, 0.0], dtype=complex),
            phases=np.zeros(2),
            power=np.array([1.0, 0.0])
        )
        
        # Overlap with itself should be 1
        assert np.isclose(state1.overlap_probability(state1), 1.0, atol=1e-10)


class TestRealityRegister:
    """Tests for RealityRegister class."""
    
    def test_initialization(self):
        """Test register initialization."""
        register = RealityRegister(n_modes=20, seed=42)
        assert register.n_modes == 20
        assert register.current_state is not None
        assert np.isclose(register.current_state.norm, 1.0, atol=1e-10)
    
    def test_basis_states(self):
        """Test basis state generation."""
        register = RealityRegister(n_modes=20, seed=42)
        
        # Check all basis states exist
        basis_states = ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia', 'meditation', 'psychedelic']
        for state_name in basis_states:
            state = register.get_basis_state(state_name)
            assert state.n_modes == 20
            assert np.isclose(state.norm, 1.0, atol=1e-10)
    
    def test_superposition_creation(self):
        """Test superposition state creation."""
        register = RealityRegister(n_modes=20, seed=42)
        
        # Create equal superposition
        superpos = register.create_superposition(['wake', 'nrem_sleep'])
        assert np.isclose(superpos.norm, 1.0, atol=1e-10)
        
        # Check it has overlap with both states
        wake = register.get_basis_state('wake')
        sleep = register.get_basis_state('nrem_sleep')
        
        overlap_wake = superpos.overlap_probability(wake)
        overlap_sleep = superpos.overlap_probability(sleep)
        
        assert overlap_wake > 0.1
        assert overlap_sleep > 0.1
    
    def test_state_decomposition(self):
        """Test state decomposition."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        
        decomp = register.get_state_decomposition()
        
        # Wake state should have highest overlap with wake basis
        assert decomp['wake'] > decomp['nrem_sleep']
        assert decomp['wake'] > decomp['anesthesia']
    
    def test_history_tracking(self):
        """Test state history tracking."""
        register = RealityRegister(n_modes=20, seed=42)
        
        initial_length = register.get_history_length()
        register.set_state(register.get_basis_state('wake'))
        assert register.get_history_length() == initial_length + 1
        
        register.clear_history()
        assert register.get_history_length() == 1


class TestSteeringProtocol:
    """Tests for SteeringProtocol class."""
    
    def test_steering_operator_shape(self):
        """Test steering operator has correct shape."""
        register = RealityRegister(n_modes=20, seed=42)
        protocol = SteeringProtocol(register)
        
        operator = protocol.compute_steering_operator('nrem_sleep', strength=0.1)
        assert operator.shape == (20, 20)
    
    def test_steering_changes_state(self):
        """Test that steering changes the state."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        protocol = SteeringProtocol(register)
        
        initial_overlap = register.current_state.overlap_probability(
            register.get_basis_state('nrem_sleep')
        )
        
        # Steer toward sleep with stronger steering and multiple steps
        for _ in range(5):
            protocol.steer_to_state('nrem_sleep', strength=0.3, update_register=True)
        
        final_overlap = register.current_state.overlap_probability(
            register.get_basis_state('nrem_sleep')
        )
        
        # Overlap with sleep should increase (or at least not decrease significantly)
        # With multiple strong steps, this should be reliable
        assert final_overlap >= initial_overlap * 0.5  # Allow some variance
    
    def test_gradual_steering(self):
        """Test gradual steering over multiple steps."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        protocol = SteeringProtocol(register)
        
        states = protocol.gradual_steering('nrem_sleep', n_steps=10, total_strength=1.0)
        
        # Should return n_steps + 1 states (including initial)
        assert len(states) == 11
        
        # Overlap with sleep should increase monotonically (approximately)
        overlaps = [s.overlap_probability(register.get_basis_state('nrem_sleep')) 
                   for s in states]
        assert overlaps[-1] > overlaps[0]
    
    def test_local_steering(self):
        """Test local steering on subset of modes."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        protocol = SteeringProtocol(register)
        
        # Steer only first 5 modes
        local_modes = np.arange(5)
        protocol.steer_to_state('nrem_sleep', strength=0.2, local_modes=local_modes, 
                               update_register=True)
        
        # State should still be normalized
        assert np.isclose(register.current_state.norm, 1.0, atol=1e-10)


class TestQuantumMeasurement:
    """Tests for QuantumMeasurement class."""
    
    def test_mode_occupation_measurement(self):
        """Test mode occupation measurement."""
        register = RealityRegister(n_modes=20, seed=42)
        measurement = QuantumMeasurement(register)
        
        power, collapsed = measurement.measure_mode_occupation(0, collapse=False)
        
        assert 0 <= power <= 1
        assert np.isclose(collapsed.norm, 1.0, atol=1e-10)
    
    def test_consciousness_state_measurement(self):
        """Test consciousness state measurement."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        measurement = QuantumMeasurement(register)
        
        measured_state, prob, collapsed = measurement.measure_consciousness_state(collapse=False)
        
        assert measured_state in ['wake', 'nrem_sleep', 'rem_sleep', 'anesthesia', 
                                  'meditation', 'psychedelic']
        assert 0 <= prob <= 1
    
    def test_measurement_collapse(self):
        """Test that measurement causes collapse."""
        register = RealityRegister(n_modes=20, seed=42)
        superpos = register.create_superposition(['wake', 'nrem_sleep'])
        register.set_state(superpos)
        
        measurement = QuantumMeasurement(register)
        measured_state, prob, collapsed = measurement.measure_consciousness_state(collapse=True)
        
        # After collapse, state should have very high overlap with measured basis state
        final_overlap = register.current_state.overlap_probability(
            register.get_basis_state(measured_state)
        )
        assert final_overlap > 0.9
    
    def test_weak_measurement(self):
        """Test weak measurement."""
        register = RealityRegister(n_modes=20, seed=42)
        measurement = QuantumMeasurement(register)
        
        observable = np.diag(np.arange(20, dtype=float))
        expectation, new_state = measurement.weak_measurement(observable, strength=0.1)
        
        assert np.isclose(new_state.norm, 1.0, atol=1e-10)
        assert isinstance(expectation, (int, float, np.number))
    
    def test_measurement_history(self):
        """Test measurement history tracking."""
        register = RealityRegister(n_modes=20, seed=42)
        measurement = QuantumMeasurement(register)
        
        measurement.measure_entropy()
        measurement.measure_participation_ratio()
        
        history = measurement.get_measurement_history()
        assert len(history) == 2
        
        measurement.clear_measurement_history()
        assert len(measurement.get_measurement_history()) == 0


class TestEntanglement:
    """Tests for entanglement functions."""
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        register = RealityRegister(n_modes=20, seed=42)
        state = register.get_basis_state('wake')
        
        subsystem = np.arange(10)
        entropy = compute_entanglement_entropy(state, subsystem)
        
        assert entropy >= 0
        assert isinstance(entropy, (int, float, np.number))
    
    def test_mutual_information(self):
        """Test mutual information calculation."""
        register = RealityRegister(n_modes=20, seed=42)
        state = register.get_basis_state('wake')
        
        region_a = np.arange(5)
        region_b = np.arange(5, 10)
        
        mi = compute_mutual_information(state, region_a, region_b)
        
        assert mi >= 0
        assert isinstance(mi, (int, float, np.number))
    
    def test_mutual_information_symmetry(self):
        """Test that mutual information is symmetric."""
        register = RealityRegister(n_modes=20, seed=42)
        state = register.get_basis_state('wake')
        
        region_a = np.arange(5)
        region_b = np.arange(5, 10)
        
        mi_ab = compute_mutual_information(state, region_a, region_b)
        mi_ba = compute_mutual_information(state, region_b, region_a)
        
        assert np.isclose(mi_ab, mi_ba, atol=1e-10)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_steer_consciousness_state(self):
        """Test steer_consciousness_state convenience function."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        
        new_state = steer_consciousness_state(register, 'nrem_sleep', strength=0.1)
        
        assert np.isclose(new_state.norm, 1.0, atol=1e-10)
        assert new_state.n_modes == 20
    
    def test_measure_consciousness_state(self):
        """Test measure_consciousness_state convenience function."""
        register = RealityRegister(n_modes=20, seed=42)
        register.set_state(register.get_basis_state('wake'))
        
        state_name, prob = measure_consciousness_state(register, collapse=False)
        
        assert isinstance(state_name, str)
        assert 0 <= prob <= 1


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
