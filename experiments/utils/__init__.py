"""
Utilities for the Harmonic Field Consciousness Experimental Framework

This package provides core functionality for:
- Network topology generation
- Consciousness metrics calculation
- Brain state generation
- Visualization utilities
"""

__version__ = "1.0.0"

from .graph_generators import (
    generate_small_world,
    generate_scale_free,
    generate_random,
    generate_lattice,
    generate_modular,
    generate_hub_network,
)

from .metrics import (
    compute_mode_entropy,
    compute_participation_ratio,
    compute_phase_coherence,
    compute_entropy_production,
    compute_criticality_index,
    compute_consciousness_functional,
    compute_lempel_ziv_complexity,
    compute_multiscale_entropy,
)

from .state_generators import (
    generate_wake_state,
    generate_nrem_unconscious,
    generate_nrem_dreaming,
    generate_anesthesia_state,
    generate_psychedelic_state,
    interpolate_states,
    add_perturbation,
)

from .visualization import (
    plot_network,
    plot_mode_distribution,
    plot_consciousness_radar,
    plot_time_series,
    plot_phase_space,
    plot_heatmap,
)

__all__ = [
    # Graph generators
    "generate_small_world",
    "generate_scale_free",
    "generate_random",
    "generate_lattice",
    "generate_modular",
    "generate_hub_network",
    # Metrics
    "compute_mode_entropy",
    "compute_participation_ratio",
    "compute_phase_coherence",
    "compute_entropy_production",
    "compute_criticality_index",
    "compute_consciousness_functional",
    "compute_lempel_ziv_complexity",
    "compute_multiscale_entropy",
    # State generators
    "generate_wake_state",
    "generate_nrem_unconscious",
    "generate_nrem_dreaming",
    "generate_anesthesia_state",
    "generate_psychedelic_state",
    "interpolate_states",
    "add_perturbation",
    # Visualization
    "plot_network",
    "plot_mode_distribution",
    "plot_consciousness_radar",
    "plot_time_series",
    "plot_phase_space",
    "plot_heatmap",
]
