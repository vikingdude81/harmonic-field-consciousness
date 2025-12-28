# Category 2: Dynamical Systems Experiments

This category studies temporal evolution and state transitions in the consciousness model.

## Experiments

### exp1_state_transitions.py
**Animate transitions between conscious states**

Procedure:
- Implement smooth interpolation between states
- Create Wake → NREM → Dream → Wake cycle
- Track all metrics during transition
- Generate time series plots
- Create phase space trajectories (3D plots)
- Export animation as MP4/GIF
- Identify critical transition points

**Output**: `results/exp1_state_transitions/`

### exp2_perturbation_recovery.py
**Test resilience of conscious states**

Procedure:
- Start in wake state at equilibrium
- Add perturbations: noise to mode powers, phase disruption
- Implement relaxation dynamics
- Measure recovery time to baseline C(t)
- Test different perturbation magnitudes
- Compare resilience across states
- Model TMS or sensory stimulation

**Output**: `results/exp2_perturbation_recovery/`

### exp3_coupling_strength.py
**Explore mode coupling effects**

Procedure:
- Implement nonlinear coupling between modes
- Vary coupling strength parameter
- Test effect on synchronization (R)
- Find optimal coupling for consciousness
- Generate bifurcation diagrams
- Model pharmacological effects

**Output**: `results/exp3_coupling_strength/`

### exp4_criticality_tuning.py
**Find edge-of-chaos optimal point**

Procedure:
- Vary system parameters to tune criticality
- Measure avalanche size distributions
- Find optimal point for consciousness
- Test different network topologies

**Output**: `results/exp4_criticality_tuning/`

### exp5_rotational_recovery.py
**Rotational dynamics and recovery from perturbation**

Based on Batabyal et al. (2025) JOCN - "Rotational Dynamics in Prefrontal Cortex Enable Recovery from Distraction During Working Memory"

Implements jittered PCA (jPCA) to analyze rotational dynamics in neural state space and tests whether rotation facilitates recovery from perturbations.

**Three main experiments:**

1. **State Comparison**: Compare rotational dynamics (rotation angle, angular velocity, recovery percentage) across conscious states (Wake, NREM, Anesthesia)
   - Tests hypothesis: Rotational dynamics differ between conscious and unconscious states
   - 50 trials per state with perturbations
   - Measures: rotation angle, angular velocity, recovery %, consciousness metric C(t)

2. **Rotation-Recovery Correlation**: Test correlation between rotational dynamics and consciousness metrics
   - 20 trials across 10 perturbation strengths (0.1 to 1.0)
   - Correlates rotation angle and recovery % with consciousness functional C(t)
   - Tests hypothesis: Stronger rotation → better recovery → higher consciousness

3. **Wave-Rotation Correspondence**: Analyze relationship between traveling waves and rotational dynamics
   - 30 trials measuring both wave properties and rotational dynamics
   - Correlates wave speed, direction, coherence with rotation metrics
   - Tests hypothesis: Traveling waves manifest rotational dynamics spatially

**Key features:**
- jPCA implementation for rotation analysis
- Traveling wave detection using optical flow and correlation methods
- Dynamic stability metrics (Lyapunov spectrum, convergence time)
- Comprehensive visualizations (trajectories, correlations, spatial waves)
- Progress tracking with detailed timing and checkmarks

**Output**: `results/exp5_rotational_recovery/`
- CSV files: state_comparison.csv, rotation_recovery_correlation.csv, wave_rotation_correspondence.csv
- Visualizations: state_trajectories.png, recovery_correlation.png, wave_patterns.png

**Dependencies**: New utility modules
- `utils/rotational_dynamics.py`: jPCA, rotation metrics
- `utils/traveling_waves.py`: Wave detection and analysis
- `utils/dynamic_stability.py`: Stability and convergence metrics

Procedure:
- Vary system parameters to control κ
- Test C(t) as function of κ
- Identify critical regime (κ ≈ 1)
- Compare subcritical, critical, supercritical
- Generate phase transition plots
- Relate to empirical brain data

**Output**: `results/exp4_criticality_tuning/`

## Running the Experiments

Run all experiments in this category:
```bash
python exp1_state_transitions.py
python exp2_perturbation_recovery.py
python exp3_coupling_strength.py
python exp4_criticality_tuning.py
```

## Results

All results are saved in the `results/` subdirectory with visualizations and time series data.
