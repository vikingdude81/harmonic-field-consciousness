"""
Unit Tests for Memcomputing Modules

Tests for:
- Meminductor: Memory-dependent inductor
- RLCOscillator: RLC circuit dynamics
- RLCOscillatorBank: Multiple coupled oscillators
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.memcomputing import Meminductor, RLCOscillator


class TestMeminductor:
    """Tests for Meminductor."""
    
    def test_initialization(self):
        """Test meminductor initialization."""
        mem = Meminductor(L0=1.0, alpha=0.1, beta=0.01)
        
        assert mem.L0 == 1.0
        assert mem.alpha == 0.1
        assert mem.beta == 0.01
        assert mem.phi == 0.0
        assert mem.current == 0.0
        assert mem.L == 1.0
    
    def test_update_inductance(self):
        """Test inductance update based on flux."""
        mem = Meminductor(L0=1.0, alpha=0.5, beta=0.1)
        
        # Set flux
        mem.phi = 10.0
        mem.update_inductance()
        
        # Inductance should change
        assert mem.L != mem.L0
        assert mem.L > 0
    
    def test_update(self):
        """Test meminductor state update."""
        mem = Meminductor(L0=1.0, alpha=0.1, beta=0.01)
        
        voltage = 1.0
        dt = 0.01
        current = mem.update(voltage, dt)
        
        assert current != 0.0
        assert mem.phi != 0.0
    
    def test_get_state(self):
        """Test getting meminductor state."""
        mem = Meminductor()
        mem.phi = 5.0
        mem.current = 0.5
        mem.update_inductance()
        
        current, flux, L = mem.get_state()
        
        assert current == 0.5
        assert flux == 5.0
        assert L > 0
    
    def test_encode_memory(self):
        """Test encoding amplitude in flux."""
        mem = Meminductor(L0=1.0)
        
        amplitude = 0.7
        mem.encode_memory(amplitude)
        
        assert mem.phi == 0.7
        assert mem.L != mem.L0  # Inductance changed
    
    def test_read_memory(self):
        """Test reading stored memory."""
        mem = Meminductor(L0=1.0)
        
        amplitude = 0.5
        mem.encode_memory(amplitude)
        retrieved = mem.read_memory()
        
        assert np.isclose(retrieved, amplitude)
    
    def test_memory_persistence(self):
        """Test memory decay."""
        mem = Meminductor(L0=1.0)
        
        mem.encode_memory(1.0)
        initial_flux = mem.phi
        
        # Apply decay
        mem.memory_persistence(leak_rate=0.1, dt=0.1)
        
        # Flux should decrease
        assert mem.phi < initial_flux
    
    def test_reset(self):
        """Test meminductor reset."""
        mem = Meminductor()
        
        mem.update(1.0, 0.01)
        mem.reset()
        
        assert mem.phi == 0.0
        assert mem.current == 0.0
        assert mem.L == mem.L0


class TestRLCOscillator:
    """Tests for RLCOscillator."""
    
    def test_initialization(self):
        """Test oscillator initialization."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0)
        
        assert osc.R == 1.0
        assert osc.C == 1.0
        assert osc.voltage == 0.0
        assert osc.current == 0.0
        assert osc.t == 0.0
    
    def test_update(self):
        """Test oscillator update."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0)
        
        external_voltage = 1.0
        v, i = osc.update(external_voltage)
        
        assert isinstance(v, float)
        assert isinstance(i, float)
        assert osc.t > 0
    
    def test_simulate(self):
        """Test oscillator simulation."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0, dt=0.01)
        
        n_steps = 100
        voltages, currents = osc.simulate(n_steps, external_voltage=1.0)
        
        assert voltages.shape == (n_steps,)
        assert currents.shape == (n_steps,)
        assert len(osc.voltage_history) == n_steps
    
    def test_natural_frequency(self):
        """Test natural frequency computation."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0)
        
        f0 = osc.get_natural_frequency()
        
        assert f0 > 0
        # f0 = 1/(2π√LC)
        expected = 1.0 / (2 * np.pi * np.sqrt(1.0 * 1.0))
        assert np.isclose(f0, expected)
    
    def test_damping_ratio(self):
        """Test damping ratio computation."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0)
        
        zeta = osc.get_damping_ratio()
        
        assert zeta > 0
    
    def test_energy(self):
        """Test energy computation."""
        osc = RLCOscillator(R=1.0, C=1.0, L0=1.0)
        
        # Apply voltage
        osc.simulate(100, external_voltage=1.0)
        
        energy = osc.get_energy()
        
        assert energy >= 0
    
    def test_reset(self):
        """Test oscillator reset."""
        osc = RLCOscillator()
        
        osc.simulate(50, external_voltage=1.0)
        osc.reset()
        
        assert osc.voltage == 0.0
        assert osc.current == 0.0
        assert osc.t == 0.0
        assert len(osc.voltage_history) == 0
    
    def test_oscillation(self):
        """Test that oscillator actually oscillates."""
        # Underdamped system should oscillate
        osc = RLCOscillator(R=0.1, C=1.0, L0=1.0, dt=0.01)
        
        # Impulse response
        external_signal = np.zeros(500)
        external_signal[0] = 10.0
        
        voltages, _ = osc.simulate(500, external_signal=external_signal)
        
        # Check for oscillations (multiple zero crossings)
        zero_crossings = np.where(np.diff(np.sign(voltages)))[0]
        assert len(zero_crossings) >= 2  # Should oscillate (at least 2 crossings)


class TestRLCOscillatorBank:
    """Tests for RLCOscillatorBank."""
    
    def test_initialization(self):
        """Test bank initialization."""
        from src.memcomputing.rlc_dynamics import RLCOscillatorBank
        
        n_modes = 10
        bank = RLCOscillatorBank(n_modes=n_modes)
        
        assert bank.n_modes == n_modes
        assert len(bank.oscillators) == n_modes
    
    def test_update(self):
        """Test bank update."""
        from src.memcomputing.rlc_dynamics import RLCOscillatorBank
        
        bank = RLCOscillatorBank(n_modes=5)
        
        external_inputs = np.random.randn(5)
        voltages, currents = bank.update(external_inputs)
        
        assert voltages.shape == (5,)
        assert currents.shape == (5,)
    
    def test_simulate(self):
        """Test bank simulation."""
        from src.memcomputing.rlc_dynamics import RLCOscillatorBank
        
        bank = RLCOscillatorBank(n_modes=5)
        
        n_steps = 50
        voltages, currents = bank.simulate(n_steps)
        
        assert voltages.shape == (n_steps, 5)
        assert currents.shape == (n_steps, 5)
    
    def test_get_amplitudes(self):
        """Test getting amplitudes."""
        from src.memcomputing.rlc_dynamics import RLCOscillatorBank
        
        bank = RLCOscillatorBank(n_modes=10)
        
        # Simulate
        bank.simulate(100)
        
        amplitudes = bank.get_amplitudes()
        
        assert amplitudes.shape == (10,)
    
    def test_reset(self):
        """Test bank reset."""
        from src.memcomputing.rlc_dynamics import RLCOscillatorBank
        
        bank = RLCOscillatorBank(n_modes=5)
        
        bank.simulate(50)
        bank.reset()
        
        # All oscillators should be reset
        for osc in bank.oscillators:
            assert osc.t == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
