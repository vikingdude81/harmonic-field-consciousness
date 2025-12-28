# Experiments Changelog

## [2025-12-27] - RTX 5090 GPU Optimization & Scaling Analysis

### Added
- **RTX 5090 Setup Guide** (`category2_dynamics/RTX5090_SETUP_GUIDE.md`)
  - Complete installation instructions for PyTorch nightly with CUDA 12.8 (sm_120 support)
  - GPU selection code for multi-GPU systems
  - Eigendecomposition limits documentation
  - Troubleshooting guide for common issues
  - Memory usage estimates by configuration
  - Performance benchmarks vs previous generation

- **GPU Scaling Analysis** (`category2_dynamics/GPU_SCALING_ANALYSIS.md`)
  - Comprehensive benchmarks from small (961 nodes) to ultra (25,921 nodes)
  - Scaling laws for eigendecomposition and trajectory simulation
  - Scientific insights from large-scale experiments
  - Recommended protocols for different research goals
  - Future directions for beyond-cuSOLVER scales

### Fixed
- **GPU Selection**: Explicitly use GPU 0 (RTX 5090) instead of auto-detection
- **Python Environment**: Use system Python 3.11 with PyTorch nightly (not venv)
- **Unicode Encoding**: Replaced Unicode characters (✓, ✗, 🎉) with ASCII for Windows compatibility

### Optimized
- **Experiment Configurations** (`exp_gpu_massive_batched.py`)
  - Updated mega/giga/ultra configs with verified working parameters
  - Added 'max' config for maximum scale (25,921 nodes, 2,500 modes, 20,000 timesteps)
  - Documented cuSOLVER eigendecomp limit (~26,000 nodes)
  - Added configuration comments with timing and memory estimates

### Benchmarks (RTX 5090, 34 GB VRAM)

| Config | Nodes | Modes | Timesteps | Trials | Eigendecomp | Total Time |
|--------|-------|-------|-----------|--------|-------------|------------|
| small | 961 | 100 | 1,000 | 100 | 0.1s | 7s |
| medium | 2,401 | 300 | 2,000 | 500 | 0.3s | 10s |
| large | 4,900 | 800 | 5,000 | 200 | 0.8s | 20s |
| xlarge | 10,000 | 1,500 | 10,000 | 100 | 2.0s | 40s |
| mega | 24,964 | 2,000 | 10,000 | 50 | 13.6s | 55s |
| giga | 24,964 | 2,000 | 15,000 | 50 | 13.6s | 58s |
| ultra | 25,921 | 2,200 | 15,000 | 40 | 14.9s | 50s |
| **max** | **25,921** | **2,500** | **20,000** | **100** | **15.0s** | **227s** |

### Key Findings
- **cuSOLVER Limit**: Dense eigendecomposition fails at ~27,000 nodes
- **Maximum Scale**: 25,921 nodes (161×161 lattice) is the absolute maximum
- **Rotation Dynamics**: Mean rotation ~52,000° at 20,000 timesteps (max config)
- **Wave Detection**: Stabilizes at ~25% for large networks
- **Throughput**: Peak efficiency ~40M mode-timesteps/second at medium scale

---

## [2025-12-25] - Rotational Dynamics Refinements (v2)

### Refined
- **Rotational Dynamics Experiment** (`exp5_rotational_recovery_v2.py`)
  
  Key improvements over v1:
  - **Weaker perturbations**: Reduced from 0.1-0.5 to 0.01-0.1 range (10x weaker)
  - **Longer trajectories**: Extended from 100 to 200 time steps for better rotation detection
  - **Larger networks**: Increased from 300 to 500 nodes for enhanced wave propagation
  - **More jPCA components**: Increased from 6 to 10 for better rotation plane identification
  - **Improved recovery dynamics**: Added attractor forces that pull system back to baseline
  - **Better temporal resolution**: More time points for jPCA analysis
  - **2D lattice topology**: Added periodic boundary conditions for Experiment 3
  
  Addressing v1 limitations:
  - ❌ Binary rotation angles (0° or 180°) → ✓ Continuous angles expected
  - ❌ No recovery observed (0%) → ✓ Observable recovery with weaker perturbations
  - ❌ No traveling waves detected → ✓ Larger networks + lattice topology
  
  Expected outcomes:
  1. Continuous rotation angle distribution
  2. Measurable recovery percentages (10-80% range)
  3. Potential traveling wave detection in lattice topology
  4. Stronger correlation between rotation quality and recovery

### Context
Based on comprehensive empirical comparison analysis ([RESULTS_COMPARISON.md](RESULTS_COMPARISON.md)):
- Strong agreement with consciousness state differences
- Critical findings on synchronization and criticality
- Need for methodological refinements identified

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
