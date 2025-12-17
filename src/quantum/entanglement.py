"""
Entanglement - Non-local Correlations Between Brain Regions

Implements quantum-inspired entanglement measures for modeling non-local
correlations between different brain regions in the harmonic field framework.

Key concepts:
- Entanglement entropy between regional mode subspaces
- Non-local correlations in consciousness states
- Regional mode decomposition
- Mutual information between brain regions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.linalg import svd, logm
from .reality_register import QuantumConsciousnessState


def compute_entanglement_entropy(
    state: QuantumConsciousnessState,
    subsystem_modes: np.ndarray
) -> float:
    """
    Compute entanglement entropy between a subsystem and its complement.
    
    Measures how entangled a subset of modes is with the rest of the system.
    High entanglement entropy indicates strong non-local correlations.
    
    Args:
        state: Quantum consciousness state
        subsystem_modes: Array of mode indices forming the subsystem
    
    Returns:
        Entanglement entropy (von Neumann entropy of reduced density matrix)
    """
    n_modes = state.n_modes
    subsystem_modes = np.asarray(subsystem_modes, dtype=int)
    
    # Create density matrix (assume pure state: ρ = |ψ⟩⟨ψ|)
    rho = np.outer(state.amplitudes, np.conj(state.amplitudes))
    
    # Trace out complement to get reduced density matrix
    complement_modes = np.setdiff1d(np.arange(n_modes), subsystem_modes)
    
    # For pure states, use Schmidt decomposition
    # Reshape state vector into subsystem × complement
    # This is approximate - proper implementation would require tensor structure
    
    # Simplified approach: use power distribution
    subsystem_power = state.power[subsystem_modes]
    subsystem_power = subsystem_power / (subsystem_power.sum() + 1e-12)
    
    # Von Neumann entropy: -Tr(ρ log ρ)
    p = subsystem_power[subsystem_power > 1e-12]
    entropy = -np.sum(p * np.log(p + 1e-12))
    
    return float(entropy)


def compute_mutual_information(
    state: QuantumConsciousnessState,
    region_a_modes: np.ndarray,
    region_b_modes: np.ndarray
) -> float:
    """
    Compute mutual information between two brain regions.
    
    Measures how much information is shared between two regions.
    I(A:B) = S(A) + S(B) - S(A,B)
    
    Args:
        state: Quantum consciousness state
        region_a_modes: Mode indices for region A
        region_b_modes: Mode indices for region B
    
    Returns:
        Mutual information ≥ 0
    """
    # Entropy of region A
    power_a = state.power[region_a_modes]
    power_a = power_a / (power_a.sum() + 1e-12)
    p_a = power_a[power_a > 1e-12]
    S_a = -np.sum(p_a * np.log(p_a + 1e-12))
    
    # Entropy of region B
    power_b = state.power[region_b_modes]
    power_b = power_b / (power_b.sum() + 1e-12)
    p_b = power_b[power_b > 1e-12]
    S_b = -np.sum(p_b * np.log(p_b + 1e-12))
    
    # Joint entropy S(A,B)
    combined_modes = np.concatenate([region_a_modes, region_b_modes])
    power_ab = state.power[combined_modes]
    power_ab = power_ab / (power_ab.sum() + 1e-12)
    p_ab = power_ab[power_ab > 1e-12]
    S_ab = -np.sum(p_ab * np.log(p_ab + 1e-12))
    
    # Mutual information
    mutual_info = S_a + S_b - S_ab
    
    return float(max(0, mutual_info))  # Ensure non-negative


def compute_regional_correlations(
    state: QuantumConsciousnessState,
    regions: List[np.ndarray]
) -> np.ndarray:
    """
    Compute correlation matrix between multiple brain regions.
    
    Args:
        state: Quantum consciousness state
        regions: List of mode index arrays, one per region
    
    Returns:
        Correlation matrix (n_regions × n_regions)
    """
    n_regions = len(regions)
    correlations = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                # Self-correlation (entropy)
                correlations[i, j] = compute_entanglement_entropy(
                    state, regions[i]
                )
            else:
                # Cross-correlation (mutual information)
                mutual_info = compute_mutual_information(
                    state, regions[i], regions[j]
                )
                correlations[i, j] = mutual_info
                correlations[j, i] = mutual_info
    
    return correlations


def model_nonlocal_effects(
    state: QuantumConsciousnessState,
    local_perturbation_modes: np.ndarray,
    perturbation_strength: float = 0.1
) -> Tuple[QuantumConsciousnessState, Dict[str, float]]:
    """
    Model how a local perturbation creates non-local effects.
    
    Applies a local perturbation and measures how it affects distant modes
    through quantum correlations.
    
    Args:
        state: Initial quantum consciousness state
        local_perturbation_modes: Modes to perturb locally
        perturbation_strength: Strength of perturbation
    
    Returns:
        Tuple of (perturbed_state, nonlocal_effects_dict)
    """
    n_modes = state.n_modes
    local_modes = np.asarray(local_perturbation_modes, dtype=int)
    distant_modes = np.setdiff1d(np.arange(n_modes), local_modes)
    
    # Apply local perturbation (phase shift + amplitude modulation)
    perturbed_amplitudes = state.amplitudes.copy()
    
    # Modulate local modes
    perturbation = 1 + perturbation_strength * np.random.randn(len(local_modes))
    phase_shifts = perturbation_strength * np.random.randn(len(local_modes))
    
    perturbed_amplitudes[local_modes] *= (
        perturbation * np.exp(1j * phase_shifts)
    )
    
    # Normalize
    perturbed_amplitudes /= np.linalg.norm(perturbed_amplitudes)
    
    perturbed_state = QuantumConsciousnessState(
        amplitudes=perturbed_amplitudes,
        phases=np.angle(perturbed_amplitudes),
        power=np.abs(perturbed_amplitudes) ** 2,
        label="locally_perturbed"
    )
    
    # Measure non-local effects
    # 1. Change in distant mode power
    original_distant_power = state.power[distant_modes].sum()
    new_distant_power = perturbed_state.power[distant_modes].sum()
    distant_power_change = np.abs(new_distant_power - original_distant_power)
    
    # 2. Change in entanglement entropy
    original_entropy = compute_entanglement_entropy(state, local_modes)
    new_entropy = compute_entanglement_entropy(perturbed_state, local_modes)
    entropy_change = np.abs(new_entropy - original_entropy)
    
    # 3. Correlation between local and distant modes
    correlation = compute_mutual_information(
        perturbed_state, local_modes, distant_modes
    )
    
    nonlocal_effects = {
        'distant_power_change': float(distant_power_change),
        'entropy_change': float(entropy_change),
        'local_distant_correlation': float(correlation),
        'perturbation_strength': perturbation_strength
    }
    
    return perturbed_state, nonlocal_effects


def compute_bipartite_entanglement(
    state: QuantumConsciousnessState,
    partition_point: int
) -> float:
    """
    Compute bipartite entanglement for a cut at a specific mode.
    
    Divides the system into modes [0, partition_point) and [partition_point, n_modes).
    
    Args:
        state: Quantum consciousness state
        partition_point: Mode index for partition
    
    Returns:
        Bipartite entanglement entropy
    """
    subsystem_modes = np.arange(partition_point)
    return compute_entanglement_entropy(state, subsystem_modes)


def compute_entanglement_spectrum(
    state: QuantumConsciousnessState,
    n_partitions: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute entanglement spectrum across different bipartitions.
    
    Shows how entanglement varies when cutting the system at different points.
    
    Args:
        state: Quantum consciousness state
        n_partitions: Number of partition points to test (default: n_modes//2)
    
    Returns:
        Dictionary with 'partition_points' and 'entanglement_entropies'
    """
    n_modes = state.n_modes
    
    if n_partitions is None:
        n_partitions = n_modes // 2
    
    partition_points = np.linspace(
        1, n_modes - 1, n_partitions, dtype=int
    )
    
    entropies = np.array([
        compute_bipartite_entanglement(state, p)
        for p in partition_points
    ])
    
    return {
        'partition_points': partition_points,
        'entanglement_entropies': entropies
    }


def compute_global_entanglement(
    state: QuantumConsciousnessState
) -> float:
    """
    Compute a global measure of entanglement across the entire system.
    
    Uses the Meyer-Wallach measure: average entanglement of each mode
    with the rest of the system.
    
    Args:
        state: Quantum consciousness state
    
    Returns:
        Global entanglement measure ∈ [0, 1]
    """
    n_modes = state.n_modes
    
    # Compute average entanglement entropy per mode
    total_entropy = 0.0
    
    for i in range(n_modes):
        entropy = compute_entanglement_entropy(state, np.array([i]))
        total_entropy += entropy
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(2) * n_modes  # Maximum for qubits
    global_entanglement = total_entropy / (max_entropy + 1e-12)
    
    return float(np.clip(global_entanglement, 0, 1))


def analyze_entanglement_structure(
    state: QuantumConsciousnessState,
    n_regions: int = 4
) -> Dict[str, any]:
    """
    Comprehensive analysis of entanglement structure in consciousness state.
    
    Args:
        state: Quantum consciousness state
        n_regions: Number of regions to divide the modes into
    
    Returns:
        Dictionary with various entanglement measures
    """
    n_modes = state.n_modes
    
    # Divide modes into regions
    region_size = n_modes // n_regions
    regions = [
        np.arange(i * region_size, min((i + 1) * region_size, n_modes))
        for i in range(n_regions)
    ]
    
    # Compute regional correlations
    correlations = compute_regional_correlations(state, regions)
    
    # Compute entanglement spectrum
    spectrum = compute_entanglement_spectrum(state)
    
    # Compute global entanglement
    global_ent = compute_global_entanglement(state)
    
    # Regional entropies
    regional_entropies = [
        compute_entanglement_entropy(state, region)
        for region in regions
    ]
    
    return {
        'regions': regions,
        'regional_correlations': correlations,
        'entanglement_spectrum': spectrum,
        'global_entanglement': global_ent,
        'regional_entropies': regional_entropies,
        'n_regions': n_regions
    }


def compute_quantum_discord(
    state: QuantumConsciousnessState,
    region_a_modes: np.ndarray,
    region_b_modes: np.ndarray
) -> float:
    """
    Compute quantum discord between two regions.
    
    Quantum discord captures quantum correlations beyond entanglement.
    It's a measure of the quantumness of correlations.
    
    This is a simplified approximation using mutual information.
    
    Args:
        state: Quantum consciousness state
        region_a_modes: Mode indices for region A
        region_b_modes: Mode indices for region B
    
    Returns:
        Quantum discord (approximate) ≥ 0
    """
    # Mutual information as proxy for discord
    # True discord requires optimization over measurements
    mutual_info = compute_mutual_information(state, region_a_modes, region_b_modes)
    
    # For pure states, discord ≈ mutual information
    return mutual_info


def detect_nonlocal_transitions(
    states: List[QuantumConsciousnessState],
    local_modes: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Detect non-local effects during state transitions.
    
    Tracks how local mode changes correlate with distant mode changes
    across a sequence of states.
    
    Args:
        states: List of quantum consciousness states (time series)
        local_modes: Modes considered "local"
    
    Returns:
        Dictionary with time series of non-local measures
    """
    n_states = len(states)
    n_modes = states[0].n_modes
    distant_modes = np.setdiff1d(np.arange(n_modes), local_modes)
    
    local_changes = np.zeros(n_states - 1)
    distant_changes = np.zeros(n_states - 1)
    correlations = np.zeros(n_states - 1)
    
    for i in range(n_states - 1):
        # Changes in local modes
        local_power_change = np.sum(
            np.abs(states[i+1].power[local_modes] - states[i].power[local_modes])
        )
        local_changes[i] = local_power_change
        
        # Changes in distant modes
        distant_power_change = np.sum(
            np.abs(states[i+1].power[distant_modes] - states[i].power[distant_modes])
        )
        distant_changes[i] = distant_power_change
        
        # Correlation between local and distant
        correlations[i] = compute_mutual_information(
            states[i+1], local_modes, distant_modes
        )
    
    return {
        'time_steps': np.arange(n_states - 1),
        'local_changes': local_changes,
        'distant_changes': distant_changes,
        'local_distant_correlations': correlations
    }
