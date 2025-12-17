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


# ==============================================================================
# MEDITATION STATES
# ==============================================================================

def generate_focused_attention_meditation(
    n_modes: int = 20,
    depth: float = 0.7,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Focused Attention (FA) meditation state.
    
    Characteristics (based on EEG research):
    - Enhanced theta (4-8 Hz) - corresponds to mid-modes
    - Increased frontal alpha coherence
    - Reduced high-frequency noise
    - High concentration, narrow focus
    - Lower entropy than wake (focused)
    - Moderate participation ratio
    
    FA meditation: Concentration on single object (breath, mantra, etc.)
    Examples: Shamatha, Zen counting breaths, mantra meditation
    
    Args:
        n_modes: Number of modes
        depth: Meditation depth (0=beginner, 1=adept)
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Enhanced mid-range modes (theta-alpha equivalent)
    theta_peak = 0.3 + 0.4 * depth
    power = theta_peak * np.exp(-(k - n_modes * 0.25)**2 / (8 + 4 * depth))
    
    # Some low-mode presence (but not dominant like sleep)
    power += 0.15 * np.exp(-k / 5)
    
    # Reduced high-mode activity (calm, focused)
    high_mode_suppression = 1.0 - 0.6 * depth * (k / n_modes)
    power *= np.maximum(high_mode_suppression, 0.1)
    
    # Add small noise (less than wake - more stable)
    power += 0.03 * (1 - depth) * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_open_monitoring_meditation(
    n_modes: int = 20,
    depth: float = 0.7,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Open Monitoring (OM) meditation state.
    
    Characteristics (based on EEG research):
    - Enhanced theta AND gamma
    - Broader distribution than FA
    - High entropy (open awareness)
    - High participation ratio
    - "Witnessing" quality - aware of all arising experiences
    
    OM meditation: Non-reactive awareness of whatever arises
    Examples: Vipassana, Shikantaza, Dzogchen noting
    
    Args:
        n_modes: Number of modes
        depth: Meditation depth (0=beginner, 1=adept)
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Broad distribution (open awareness)
    base = 0.25 + 0.2 * np.exp(-k / 10)
    
    # Enhanced theta peak
    theta_enhance = 0.3 * depth * np.exp(-(k - n_modes * 0.2)**2 / 12)
    
    # Enhanced gamma (high modes) - unique to OM
    gamma_enhance = 0.25 * depth * np.exp(-(k - n_modes * 0.7)**2 / 15)
    
    power = base + theta_enhance + gamma_enhance
    
    # Minimal noise (stable attention)
    power += 0.04 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_nondual_awareness(
    n_modes: int = 20,
    depth: float = 0.8,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Non-dual Awareness state.
    
    Characteristics:
    - Very flat, uniform distribution
    - Extremely high entropy
    - Maximum participation ratio
    - Subject-object distinction dissolved
    - All modes participate equally
    - Similar to deep psychedelic but without chaos
    - Stable, peaceful (low Lyapunov)
    
    States: Rigpa (Dzogchen), Turiya (Vedanta), Satori, Moksha glimpse
    
    Args:
        n_modes: Number of modes
        depth: Realization depth (0=glimpse, 1=stable abiding)
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Near-uniform distribution (all modes equal)
    uniform = np.ones(n_modes)
    
    # Slight wave pattern (not perfectly flat - natural coherence)
    wave = 0.1 * np.sin(2 * np.pi * k / n_modes)
    
    # Blend based on depth
    power = (1 - 0.8 * depth) * (0.5 + 0.5 * np.exp(-k / 8))  # Wake-like baseline
    power += 0.8 * depth * uniform  # Approach uniformity
    power += wave * (1 - depth)  # Wave diminishes with depth
    
    # Very little noise (extremely stable)
    power += 0.02 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_jhana_state(
    n_modes: int = 20,
    jhana_level: int = 1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Jhana (absorption) meditation states.
    
    The 4 form jhanas have distinct characteristics:
    - Jhana 1: Applied/sustained attention, rapture, happiness
    - Jhana 2: Internal confidence, rapture, happiness (no thinking)
    - Jhana 3: Equanimity, happiness (rapture fades)
    - Jhana 4: Pure equanimity, neither pleasure nor pain
    
    Neuroimaging shows progressive:
    - Reduction in default mode network
    - Increased coherence
    - Simplified yet rich experience
    
    Args:
        n_modes: Number of modes
        jhana_level: 1-4 (or 5-8 for formless)
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    jhana_level = max(1, min(jhana_level, 8))
    
    if jhana_level <= 4:
        # Form jhanas: progressively simpler but still conscious
        
        # Concentration on fewer modes as jhana deepens
        concentration = 1 + (jhana_level - 1) * 0.5  # 1.0 to 2.5
        
        # Central peak (not low like sleep, not high like psychedelic)
        center = n_modes * 0.35 - (jhana_level - 1) * 0.05 * n_modes
        
        power = np.exp(-(k - center)**2 / (15 - jhana_level * 2))
        
        # Add bliss/rapture component (diminishes in higher jhanas)
        if jhana_level <= 2:
            bliss = 0.3 * (3 - jhana_level) * np.exp(-(k - n_modes * 0.5)**2 / 10)
            power += bliss
        
        # Very stable
        power += 0.02 * np.random.rand(n_modes)
        
    else:
        # Formless jhanas (5-8): progressively more uniform
        # Infinite space, infinite consciousness, nothingness, neither-perception
        
        formless_depth = (jhana_level - 4) / 4  # 0.25 to 1.0
        
        # Trend toward uniformity
        power = (1 - formless_depth) * np.exp(-(k - n_modes * 0.4)**2 / 20)
        power += formless_depth * np.ones(n_modes)
        
        # Extremely stable
        power += 0.01 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_loving_kindness_meditation(
    n_modes: int = 20,
    intensity: float = 0.7,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Loving-Kindness (Metta) meditation state.
    
    Characteristics:
    - Enhanced positive affect circuits
    - High gamma activity
    - Strong connectivity (compassion = connection)
    - Moderate entropy
    - Distinct from FA/OM - emotional component
    
    Args:
        n_modes: Number of modes
        intensity: Practice intensity (0=start, 1=deep metta)
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    k = np.arange(n_modes)
    
    # Baseline similar to positive wake state
    power = 0.25 + 0.3 * np.exp(-k / 7)
    
    # Enhanced mid-to-high modes (positive affect, gamma)
    affect_boost = 0.35 * intensity * np.exp(-(k - n_modes * 0.55)**2 / 12)
    power += affect_boost
    
    # Moderate low-mode activity (grounded, not drowsy)
    power += 0.15 * np.exp(-k / 4)
    
    # Some variability (emotional warmth)
    power += 0.06 * np.random.rand(n_modes)
    
    # Normalize
    power = power / power.sum()
    
    return power


def generate_meditation_state(
    state_type: str,
    n_modes: int = 20,
    depth: float = 0.7,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate any meditation state by name.
    
    Args:
        state_type: One of 'focused_attention', 'open_monitoring', 
                   'nondual', 'jhana_1' through 'jhana_8', 'metta'
        n_modes: Number of modes
        depth: Practice depth/intensity
        seed: Random seed
    
    Returns:
        Normalized power distribution
    """
    state_type = state_type.lower().replace(' ', '_').replace('-', '_')
    
    if state_type in ['fa', 'focused', 'focused_attention', 'concentration', 'shamatha']:
        return generate_focused_attention_meditation(n_modes, depth, seed)
    elif state_type in ['om', 'open', 'open_monitoring', 'vipassana', 'mindfulness']:
        return generate_open_monitoring_meditation(n_modes, depth, seed)
    elif state_type in ['nondual', 'non_dual', 'nondual_awareness', 'rigpa', 'turiya', 'satori']:
        return generate_nondual_awareness(n_modes, depth, seed)
    elif state_type.startswith('jhana'):
        level = int(state_type.split('_')[1]) if '_' in state_type else 1
        return generate_jhana_state(n_modes, level, seed)
    elif state_type in ['metta', 'loving_kindness', 'compassion']:
        return generate_loving_kindness_meditation(n_modes, depth, seed)
    else:
        raise ValueError(f"Unknown meditation state: {state_type}")


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
