# Experiments Changelog

## [2025-12-23] - Rotational Dynamics Integration

### Added
- **Rotational Dynamics Framework** based on Batabyal et al. (2025) JOCN research
  
  New utility modules:
  - `utils/rotational_dynamics.py`: Complete jittered PCA (jPCA) implementation
    - `jpca()`: Core jPCA algorithm with temporal jitter and skew-symmetric constraint
    - `compute_rotation_angle()`: Measures rotation in 2D plane
    - `compute_angular_velocity()`: Calculates rate of rotation
    - `compute_recovery_percentage()`: Quantifies recovery from perturbation
    - `analyze_rotational_dynamics()`: Comprehensive analysis wrapper
    - `project_to_jpca_plane()`, `reconstruct_from_jpca()`: Projection utilities
    - `compute_rotational_energy()`, `compute_phase_coherence()`: Additional metrics
    - `find_rotation_plane()`: Identifies optimal rotation plane
  
  - `utils/traveling_waves.py`: Spatial wave detection and analysis
    - `compute_optical_flow()`: Estimates wave propagation using optical flow
    - `detect_traveling_wave_correlation()`: Correlation-based wave detection
    - `analyze_wave_correspondence_to_rotation()`: Links waves to rotational dynamics
    - `comprehensive_wave_analysis()`: Full wave characterization
    - `compute_phase_gradient()`, `compute_wave_coherence()`: Supporting metrics
  
  - `utils/dynamic_stability.py`: Stability and recovery metrics
    - `compute_lyapunov_spectrum()`: Measures trajectory divergence
    - `measure_perturbation_recovery()`: Quantifies recovery dynamics
    - `compute_trajectory_stability()`: Assesses state space stability
    - `compute_convergence_time()`: Measures time to equilibrium

- **New Experiment**: `category2_dynamics/exp5_rotational_recovery.py`
  - Three comprehensive experiments testing rotational dynamics in consciousness
  - 300-node networks with 80 harmonic modes
  - Tests wake, NREM, and anesthesia states
  - Includes perturbation analysis, correlation studies, and wave-rotation correspondence
  - Progress tracking with tqdm, timing, and visual feedback
  - Generates CSV results and comprehensive visualizations

### Modified
- `utils/graph_generators.py`: Added `get_node_positions()` function
  - Supports multiple layout algorithms: spring, circular, kamada-kawai, spectral
  - Provides spatial coordinates for traveling wave analysis
  - Deterministic seeding for reproducibility

- `latex/refs.bib`: Added citation for Batabyal et al. (2025)
  - Full bibliographic entry for JOCN paper
  - Includes note about rotational dynamics and traveling waves

### Technical Details
- **Configuration**: 300 nodes, 80 modes, small-world topology (k=6, p=0.3)
- **Trials**: 50 per state (Exp 1), 20 per strength (Exp 2), 30 total (Exp 3)
- **Metrics**: Rotation angle (rad), angular velocity (rad/s), recovery %, consciousness C(t)
- **Runtime**: ~3-5 minutes for full experimental suite
- **Output**: CSV data files + PNG visualizations for each experiment

### Research Integration
Integrates findings from:
> Batabyal, T., Bahle, B., Tian, Y., Chong, J. Y., Joutsa, J., Grent-'t-Jong, T., & Kopell, N. J. (2025).
> Rotational dynamics in prefrontal cortex enable recovery from distraction during working memory.
> *Journal of Cognitive Neuroscience*, 37(1), 162-184.

Key insights applied:
1. Rotational dynamics facilitate recovery from perturbations
2. jPCA reveals rotational structure in neural trajectories
3. Traveling waves may manifest rotational dynamics spatially
4. Recovery percentage correlates with cognitive performance

### Dependencies
- numpy, scipy: Core numerical operations
- matplotlib, seaborn: Visualization
- pandas: Data management
- tqdm: Progress tracking
- networkx: Graph operations (existing)

### Next Steps
- [ ] Run full experimental suite and analyze results
- [ ] Compare findings with empirical neuroscience data
- [ ] Integrate rotational metrics into consciousness functional
- [ ] Extend to multi-scale networks and realistic brain topologies
- [ ] Test pharmacological perturbations (anesthesia, psychedelics)
