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
