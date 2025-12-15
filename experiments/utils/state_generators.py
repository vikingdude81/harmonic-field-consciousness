"""
Brain State Power Distribution Generators

Generates realistic mode power distributions for different brain states:
- Wake state (broad distribution)
- NREM unconscious (low-mode concentration)
- NREM dreaming (mixed distribution)
- Anesthesia (extreme low-mode concentration)
- Psychedelic states (enhanced high-mode activity)
"""

import numpy as np
from typing import Optional, Tuple


def generate_wake_state(
    n_modes: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate wake state power distribution.
    
    Characteristics:
    - Broad distribution across modes
    - High entropy
    - High participation ratio
    
    Args:
        n_modes: Number of modes
        seed: Random seed for reproducibility
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Broad exponential decay with noise
    power = 0.3 + 0.4 * np.exp(-k / 8) + 0.15 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_nrem_unconscious(
    n_modes: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate NREM unconscious state power distribution.
    
    Characteristics:
    - Concentrated in low modes (delta dominance)
    - Low entropy
    - Low participation ratio
    - High phase coherence (rigid)
    
    Args:
        n_modes: Number of modes
        seed: Random seed for reproducibility
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Sharp exponential decay (delta dominance)
    power = np.exp(-k / 2) + 0.03 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_nrem_dreaming(
    n_modes: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate NREM dreaming state power distribution.
    
    Characteristics:
    - Delta present but broader than unconscious
    - Intermediate entropy
    - Some high-mode activity
    - More flexible than unconscious
    
    Args:
        n_modes: Number of modes
        seed: Random seed for reproducibility
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Mixed distribution: delta + mid-range peak
    power = 0.35 * np.exp(-k / 3) + 0.25 * np.exp(-(k - 5)**2 / 10)
    power += 0.08 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_anesthesia_state(
    n_modes: int = 20,
    depth: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate anesthesia state power distribution.
    
    Characteristics:
    - Very concentrated in lowest modes
    - Minimal entropy
    - Minimal participation ratio
    - Very high phase coherence
    
    Args:
        n_modes: Number of modes
        depth: Anesthesia depth (0=light, 1=deep)
        seed: Random seed for reproducibility
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Very sharp decay, depends on depth
    decay_rate = 1.5 - 0.5 * depth  # Sharper for deeper anesthesia
    power = np.exp(-k / decay_rate) + 0.02 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_psychedelic_state(
    n_modes: int = 20,
    intensity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate psychedelic state power distribution.
    
    Characteristics:
    - Enhanced high-mode activity
    - Very high entropy
    - Reduced low-mode dominance
    - "Ego dissolution" as flattened distribution
    
    Args:
        n_modes: Number of modes
        intensity: Psychedelic intensity (0=baseline, 1=peak)
        seed: Random seed for reproducibility
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Start with wake-like distribution
    baseline = 0.3 + 0.4 * np.exp(-k / 8)
    
    # Add high-mode enhancement
    high_mode_boost = intensity * 0.4 * np.exp(-(k - n_modes * 0.6)**2 / 20)
    
    # Reduce low-mode dominance
    low_mode_suppression = 1.0 - intensity * 0.3 * np.exp(-k / 3)
    
    power = (baseline + high_mode_boost) * low_mode_suppression
    power += 0.15 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def interpolate_states(
    state1: np.ndarray,
    state2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Interpolate between two brain states.
    
    Useful for modeling state transitions.
    
    Args:
        state1: First state power distribution
        state2: Second state power distribution
        alpha: Interpolation parameter (0=state1, 1=state2)
    
    Returns:
        Interpolated power distribution
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Linear interpolation in log space (more natural for power)
    log_state1 = np.log(state1 + 1e-12)
    log_state2 = np.log(state2 + 1e-12)
    
    log_interp = (1 - alpha) * log_state1 + alpha * log_state2
    
    power = np.exp(log_interp)
    
    # Normalize
    power = power / power.sum()
    
    return power


def add_perturbation(
    power: np.ndarray,
    noise_level: float = 0.1,
    mode_indices: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add perturbation/noise to power distribution.
    
    Useful for modeling:
    - TMS stimulation
    - Sensory input
    - Random fluctuations
    
    Args:
        power: Original power distribution
        noise_level: Magnitude of perturbation (0-1)
        mode_indices: Optional specific modes to perturb (None=all)
        seed: Random seed
    
    Returns:
        Perturbed power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    power = power.copy()
    
    # Generate noise
    noise = np.random.normal(0, noise_level, len(power))
    
    if mode_indices is not None:
        # Only perturb specific modes
        mask = np.zeros(len(power))
        mask[mode_indices] = 1.0
        noise = noise * mask
    
    # Add noise
    power = power + noise
    
    # Ensure positivity
    power = np.clip(power, 0, None)
    
    # Renormalize
    power = power / (power.sum() + 1e-12)
    
    return power


def generate_state_transition_sequence(
    state_sequence: list,
    n_steps: int = 100,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, list]:
    """
    Generate a smooth sequence of state transitions.
    
    Args:
        state_sequence: List of state names ('wake', 'nrem', 'dream', 'anesthesia', 'psychedelic')
        n_steps: Total number of time steps
        seed: Random seed
    
    Returns:
        Tuple of (power array [n_steps x n_modes], state labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # State generators
    state_funcs = {
        'wake': lambda: generate_wake_state(20, seed=seed),
        'nrem': lambda: generate_nrem_unconscious(20, seed=seed),
        'dream': lambda: generate_nrem_dreaming(20, seed=seed),
        'anesthesia': lambda: generate_anesthesia_state(20, seed=seed),
        'psychedelic': lambda: generate_psychedelic_state(20, seed=seed),
    }
    
    n_states = len(state_sequence)
    steps_per_transition = n_steps // n_states
    
    power_sequence = []
    labels = []
    
    for i, state_name in enumerate(state_sequence):
        # Generate target state
        target = state_funcs[state_name]()
        
        if i == 0:
            # First state: just use it
            for _ in range(steps_per_transition):
                power_sequence.append(target)
                labels.append(state_name)
        else:
            # Interpolate from previous to current
            previous = power_sequence[-1]
            
            for j in range(steps_per_transition):
                alpha = j / steps_per_transition
                interp = interpolate_states(previous, target, alpha)
                power_sequence.append(interp)
                
                # Label as transition if alpha < 1
                if alpha < 0.9:
                    labels.append(f"{state_sequence[i-1]}→{state_name}")
                else:
                    labels.append(state_name)
    
    return np.array(power_sequence), labels


def generate_recovery_dynamics(
    baseline: np.ndarray,
    perturbation_magnitude: float = 0.3,
    recovery_steps: int = 50,
    recovery_rate: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate recovery dynamics after perturbation.
    
    Models exponential relaxation back to baseline.
    
    Args:
        baseline: Baseline power distribution
        perturbation_magnitude: Initial perturbation size
        recovery_steps: Number of time steps
        recovery_rate: Recovery rate constant
        seed: Random seed
    
    Returns:
        Array of power distributions [recovery_steps x n_modes]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Apply initial perturbation
    perturbed = add_perturbation(baseline, perturbation_magnitude, seed=seed)
    
    recovery_sequence = [perturbed]
    
    # Exponential recovery
    for t in range(1, recovery_steps):
        # Interpolate toward baseline
        alpha = 1 - np.exp(-recovery_rate * t)
        current = interpolate_states(perturbed, baseline, alpha)
        
        # Add small noise
        current = add_perturbation(current, 0.01, seed=seed + t)
        
        recovery_sequence.append(current)
    
    return np.array(recovery_sequence)


if __name__ == "__main__":
    # Test all state generators
    print("Testing brain state generators...")
    
    states = {
        'Wake': generate_wake_state(20, seed=42),
        'NREM Unconscious': generate_nrem_unconscious(20, seed=42),
        'NREM Dreaming': generate_nrem_dreaming(20, seed=42),
        'Anesthesia': generate_anesthesia_state(20, seed=42),
        'Psychedelic': generate_psychedelic_state(20, seed=42),
    }
    
    for name, power in states.items():
        H = -np.sum(power * np.log(power + 1e-12)) / np.log(20)
        PR = (1.0 / np.sum(power ** 2)) / 20
        print(f"{name:20s}: H={H:.3f}, PR={PR:.3f}, max_mode={np.argmax(power)}")
    
    # Test interpolation
    alpha = 0.5
    interp = interpolate_states(states['Wake'], states['Anesthesia'], alpha)
    print(f"\nInterpolated (α={alpha}): sum={interp.sum():.3f}")
    
    # Test perturbation
    perturbed = add_perturbation(states['Wake'], noise_level=0.2, seed=42)
    print(f"Perturbed: sum={perturbed.sum():.3f}")
    
    # Test state transition sequence
    sequence, labels = generate_state_transition_sequence(
        ['wake', 'nrem', 'dream', 'wake'],
        n_steps=80,
        seed=42
    )
    print(f"\nState transition sequence: {sequence.shape}")
    print(f"Unique labels: {set(labels)}")
    
    # Test recovery dynamics
    recovery = generate_recovery_dynamics(
        states['Wake'],
        perturbation_magnitude=0.3,
        recovery_steps=30,
        seed=42
    )
    print(f"Recovery dynamics: {recovery.shape}")
    
    print("\nAll state generators working correctly!")
