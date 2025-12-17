"""
Push-Pull Oscillator Model

Implements the push-pull oscillator framework from arXiv:2512.10982
"A Rosetta Stone of Neural Mass Models".

The model represents brain oscillations as emerging from excitatory-inhibitory
(E-I) population interactions using coupled differential equations.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PopulationState:
    """
    State of a neural population (excitatory or inhibitory).
    
    Attributes:
        activity: Current firing rate or mean field activity
        adaptation: Adaptation variable (e.g., synaptic depression)
        input: External input to the population
    """
    activity: float
    adaptation: float = 0.0
    input: float = 0.0


class PushPullOscillator:
    """
    Push-pull oscillator model with excitatory and inhibitory populations.
    
    The model describes the dynamics of two coupled neural populations:
    - Excitatory (E) population: provides "push" drive
    - Inhibitory (I) population: provides "pull" feedback
    
    The interaction between E and I populations generates oscillations
    at frequencies determined by coupling strengths and time constants.
    
    Mathematical model:
        τ_E dE/dt = -E + S(w_EE*E - w_IE*I + I_ext)
        τ_I dI/dt = -I + S(w_EI*E - w_II*I)
    
    where:
        E, I: excitatory and inhibitory population activities
        τ_E, τ_I: time constants
        w_XY: synaptic weight from population Y to X
        S: sigmoid activation function
        I_ext: external input
    """
    
    def __init__(
        self,
        tau_e: float = 10.0,
        tau_i: float = 5.0,
        w_ee: float = 1.5,
        w_ie: float = 2.0,
        w_ei: float = 2.5,
        w_ii: float = 0.5,
        sigmoid_slope: float = 1.0,
        sigmoid_threshold: float = 0.0,
        dt: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize push-pull oscillator.
        
        Args:
            tau_e: Time constant for excitatory population (ms)
            tau_i: Time constant for inhibitory population (ms)
            w_ee: E→E coupling strength
            w_ie: I→E coupling strength (inhibitory to excitatory)
            w_ei: E→I coupling strength (excitatory to inhibitory)
            w_ii: I→I coupling strength
            sigmoid_slope: Slope of sigmoid activation function
            sigmoid_threshold: Threshold of sigmoid activation
            dt: Time step for integration (ms)
            seed: Random seed for reproducibility
        """
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = w_ee
        self.w_ie = w_ie
        self.w_ei = w_ei
        self.w_ii = w_ii
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_threshold = sigmoid_threshold
        self.dt = dt
        
        self.rng = np.random.RandomState(seed)
        
        # Current state
        self.e_state = PopulationState(activity=0.1, adaptation=0.0)
        self.i_state = PopulationState(activity=0.1, adaptation=0.0)
        
        # History for analysis
        self.history: Dict[str, List[float]] = {
            'time': [],
            'e_activity': [],
            'i_activity': [],
            'external_input': []
        }
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        Args:
            x: Input values
        
        Returns:
            Activated values S(x) = 1 / (1 + exp(-slope * (x - threshold)))
        """
        return 1.0 / (1.0 + np.exp(-self.sigmoid_slope * (x - self.sigmoid_threshold)))
    
    def dynamics(self, state: np.ndarray, t: float, external_input: float = 0.0) -> np.ndarray:
        """
        Compute derivatives for ODE integration.
        
        Args:
            state: Current state [E, I]
            t: Current time
            external_input: External input current
        
        Returns:
            Derivatives [dE/dt, dI/dt]
        """
        E, I = state
        
        # Compute inputs to each population
        input_e = self.w_ee * E - self.w_ie * I + external_input
        input_i = self.w_ei * E - self.w_ii * I
        
        # Apply activation function
        act_e = self.sigmoid(input_e)
        act_i = self.sigmoid(input_i)
        
        # Compute derivatives
        dE_dt = (-E + act_e) / self.tau_e
        dI_dt = (-I + act_i) / self.tau_i
        
        return np.array([dE_dt, dI_dt])
    
    def step(self, external_input: float = 0.0) -> Tuple[float, float]:
        """
        Take one integration step forward.
        
        Args:
            external_input: External input to excitatory population
        
        Returns:
            Tuple of (E_activity, I_activity)
        """
        state = np.array([self.e_state.activity, self.i_state.activity])
        
        # Integrate using RK4 or similar method
        t_span = [0, self.dt]
        sol = odeint(
            self.dynamics,
            state,
            t_span,
            args=(external_input,),
            tfirst=False
        )
        
        # Update state
        self.e_state.activity = sol[-1, 0]
        self.i_state.activity = sol[-1, 1]
        self.e_state.input = external_input
        
        return self.e_state.activity, self.i_state.activity
    
    def simulate(
        self,
        duration: float,
        external_input: Optional[np.ndarray] = None,
        initial_state: Optional[Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate the oscillator for a given duration.
        
        Args:
            duration: Simulation duration (ms)
            external_input: Time series of external input (if None, uses zeros)
            initial_state: Initial (E, I) state (if None, uses current state)
        
        Returns:
            Dictionary with 'time', 'e_activity', 'i_activity', 'external_input'
        """
        n_steps = int(duration / self.dt)
        
        if initial_state is not None:
            self.e_state.activity = initial_state[0]
            self.i_state.activity = initial_state[1]
        
        if external_input is None:
            external_input = np.zeros(n_steps)
        elif len(external_input) != n_steps:
            raise ValueError(
                f"External input length ({len(external_input)}) "
                f"must match number of steps ({n_steps})"
            )
        
        # Prepare storage
        time = np.arange(n_steps) * self.dt
        e_activity = np.zeros(n_steps)
        i_activity = np.zeros(n_steps)
        
        # Simulate
        for i in range(n_steps):
            e_act, i_act = self.step(external_input[i])
            e_activity[i] = e_act
            i_activity[i] = i_act
        
        # Store in history
        self.history = {
            'time': time,
            'e_activity': e_activity,
            'i_activity': i_activity,
            'external_input': external_input
        }
        
        return self.history
    
    def compute_oscillation_frequency(self) -> float:
        """
        Compute dominant oscillation frequency from recent activity.
        
        Returns:
            Dominant frequency in Hz (assuming dt in ms)
        """
        if len(self.history['e_activity']) == 0:
            return 0.0
        
        # Use FFT to find dominant frequency
        signal = self.history['e_activity']
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=self.dt / 1000.0)  # Convert to seconds
        
        # Find peak frequency (excluding DC component)
        power = np.abs(fft[1:]) ** 2
        peak_idx = np.argmax(power)
        
        return freqs[peak_idx + 1]
    
    def get_state(self) -> Tuple[float, float]:
        """Get current state (E, I)."""
        return self.e_state.activity, self.i_state.activity
    
    def reset(self, e_init: float = 0.1, i_init: float = 0.1):
        """Reset oscillator to initial state."""
        self.e_state = PopulationState(activity=e_init)
        self.i_state = PopulationState(activity=i_init)
        self.history = {
            'time': [],
            'e_activity': [],
            'i_activity': [],
            'external_input': []
        }


class MultiScalePushPull:
    """
    Multi-scale hierarchical push-pull oscillators.
    
    Implements a hierarchy of coupled push-pull oscillators at different
    spatial and temporal scales, modeling the multi-scale nature of brain
    oscillations from local circuits to global brain rhythms.
    
    Each scale has its own time constant and coupling parameters, with
    interactions between scales representing cross-frequency coupling
    and scale-to-scale information transfer.
    """
    
    def __init__(
        self,
        n_scales: int = 3,
        base_tau_e: float = 10.0,
        base_tau_i: float = 5.0,
        tau_scale_factor: float = 2.0,
        cross_scale_coupling: float = 0.3,
        dt: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize multi-scale oscillator hierarchy.
        
        Args:
            n_scales: Number of hierarchical scales
            base_tau_e: Base excitatory time constant (finest scale)
            base_tau_i: Base inhibitory time constant (finest scale)
            tau_scale_factor: Factor by which time constants increase per scale
            cross_scale_coupling: Coupling strength between adjacent scales
            dt: Time step for integration
            seed: Random seed for reproducibility
        """
        self.n_scales = n_scales
        self.cross_scale_coupling = cross_scale_coupling
        self.dt = dt
        
        # Create oscillators at different scales
        self.oscillators: List[PushPullOscillator] = []
        
        rng = np.random.RandomState(seed)
        
        for scale in range(n_scales):
            # Time constants increase with scale (slower dynamics at larger scales)
            tau_e = base_tau_e * (tau_scale_factor ** scale)
            tau_i = base_tau_i * (tau_scale_factor ** scale)
            
            # Create oscillator for this scale
            osc = PushPullOscillator(
                tau_e=tau_e,
                tau_i=tau_i,
                w_ee=1.5 + 0.1 * rng.randn(),
                w_ie=2.0 + 0.1 * rng.randn(),
                w_ei=2.5 + 0.1 * rng.randn(),
                w_ii=0.5 + 0.05 * rng.randn(),
                dt=dt,
                seed=rng.randint(0, 10000)
            )
            
            self.oscillators.append(osc)
        
        # History
        self.history: Dict[str, np.ndarray] = {}
    
    def simulate(
        self,
        duration: float,
        external_input: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate multi-scale oscillator hierarchy.
        
        Args:
            duration: Simulation duration (ms)
            external_input: External input to finest scale
        
        Returns:
            Dictionary with activity at each scale
        """
        n_steps = int(duration / self.dt)
        
        if external_input is None:
            external_input = np.zeros(n_steps)
        
        # Prepare storage for each scale
        time = np.arange(n_steps) * self.dt
        activities = {
            f'scale_{i}_e': np.zeros(n_steps)
            for i in range(self.n_scales)
        }
        activities['time'] = time
        
        # Simulate each time step
        for step in range(n_steps):
            # Compute cross-scale coupling
            scale_inputs = np.zeros(self.n_scales)
            scale_inputs[0] = external_input[step]
            
            # Bottom-up and top-down coupling
            for scale in range(1, self.n_scales):
                # Bottom-up: finer scale drives coarser scale
                e_fine, _ = self.oscillators[scale - 1].get_state()
                scale_inputs[scale] += self.cross_scale_coupling * e_fine
            
            for scale in range(self.n_scales - 1):
                # Top-down: coarser scale modulates finer scale
                e_coarse, _ = self.oscillators[scale + 1].get_state()
                scale_inputs[scale] += self.cross_scale_coupling * 0.5 * e_coarse
            
            # Step each oscillator
            for scale, osc in enumerate(self.oscillators):
                e_act, i_act = osc.step(scale_inputs[scale])
                activities[f'scale_{scale}_e'][step] = e_act
        
        self.history = activities
        return activities
    
    def get_scale_frequencies(self) -> np.ndarray:
        """
        Compute dominant frequency at each scale.
        
        Returns:
            Array of frequencies (Hz) for each scale
        """
        frequencies = np.zeros(self.n_scales)
        
        for i, osc in enumerate(self.oscillators):
            frequencies[i] = osc.compute_oscillation_frequency()
        
        return frequencies
    
    def get_cross_frequency_coupling(self) -> np.ndarray:
        """
        Compute cross-frequency coupling between scales.
        
        Returns:
            Coupling matrix (n_scales x n_scales)
        """
        coupling_matrix = np.zeros((self.n_scales, self.n_scales))
        
        # Compute phase-amplitude coupling between scales
        for i in range(self.n_scales):
            for j in range(self.n_scales):
                if i == j:
                    coupling_matrix[i, j] = 1.0
                else:
                    # Simplified coupling measure based on correlation
                    if f'scale_{i}_e' in self.history and f'scale_{j}_e' in self.history:
                        signal_i = self.history[f'scale_{i}_e']
                        signal_j = self.history[f'scale_{j}_e']
                        
                        # Compute correlation as a proxy for coupling
                        coupling_matrix[i, j] = np.corrcoef(signal_i, signal_j)[0, 1]
        
        return coupling_matrix
    
    def reset(self):
        """Reset all oscillators to initial states."""
        for osc in self.oscillators:
            osc.reset()
        self.history = {}
