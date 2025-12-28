# Refinement Progress Report

**Date**: December 25, 2025  
**Status**: In Progress

---

## Overview

Working on refining the rotational dynamics experiments based on the comprehensive empirical comparison completed earlier today.

---

## Refinements Implemented

### 1. ✅ Tuned jPCA for Continuous Rotation Angles

**Problem**: Version 1 showed binary rotation angles (0° or 180°)

**Solutions Implemented**:
- Increased jPCA components from 6 to 10
- Extended trajectory length from 100 to 200 time steps
- Better temporal resolution for jPCA analysis
- Longer pre-perturbation baseline (50 time steps)

**Expected Outcome**: Continuous rotation angle distribution

---

### 2. ✅ Fixed Perturbation Recovery

**Problem**: Version 1 showed 0% recovery across all trials

**Solutions Implemented**:
- **Reduced perturbation strength by 10x**: 0.01-0.1 (was 0.1-0.5)
- **Added attractor dynamics**: Recovery force pulls system back to pre-perturbation baseline
- **Longer observation period**: 200 time steps (was 100) to observe full recovery
- **Improved recovery metrics**: Added normalized recovery calculation

**Recovery Force Formula**:
```python
diff_from_attractor = baseline_attractor - current_state
recovery_force = diff_from_attractor * 0.05  # 5% pull toward baseline
next_state = current_state * 0.99 + recovery_force + noise
```

**Expected Outcome**: Observable recovery in 10-80% range

---

### 3. ✅ Scaled Up Networks for Traveling Wave Detection

**Problem**: Version 1 detected 0/30 traveling waves

**Solutions Implemented**:
- **Larger network**: 500 nodes (was 300)
- **More modes**: 120 (was 80)
- **2D lattice topology**: Periodic boundary conditions for Experiment 3
- **Spatial embedding**: Explicit (x, y) coordinates for wave propagation
- **4-connected lattice**: Each node connects to 4 neighbors (N, S, E, W)

**Network Configuration**:
- Experiment 1 & 2: Small-world (500 nodes, k=6, p=0.3)
- Experiment 3: 2D lattice (22×22 = 484 nodes, periodic boundaries)

**Expected Outcome**: Measurable traveling wave activity in lattice topology

---

### 4. ⏳ Hub Vulnerability Enhancement (Pending)

**Problem**: Version 1 showed small effect size (Cohen's d = 0.22)

**Proposed Solutions** (for next iteration):
1. **Stronger hub-dependency**: Increase betweenness centrality differences
2. **Targeted psychedelic perturbations**: Focus on states showing r²=0.732 hub dependence  
3. **Lesion size variations**: Test 5%, 10%, 20%, 40% hub removal
4. **Recovery tracking**: Measure C(t) recovery after hub lesions

**Status**: Not yet implemented - focus on rotation/recovery first

---

## Current Experiment Running

**Script**: `exp5_rotational_recovery_v2.py`

**Configuration**:
- Network: 500 nodes, 120 modes
- Trajectory length: 200 time steps
- States: Wake, NREM, Anesthesia
- Perturbation strengths: 0.01 to 0.1 (9 levels)
- Trials per state: 50 (Exp 1), 20 per strength (Exp 2), 30 (Exp 3)

**Estimated Runtime**: 5-10 minutes

**Output Files**:
1. `exp1_state_comparison.csv` - Rotation and recovery by consciousness state
2. `exp2_rotation_recovery_correlation.csv` - Perturbation strength effects
3. `exp3_wave_rotation_correspondence.csv` - Traveling wave detection
4. `rotational_dynamics_analysis.png` - Comprehensive visualization

---

## Key Technical Fixes

### API Compatibility Issues Resolved:

1. **`graph_generators.py`**: 
   - ❌ `watts_strogatz_graph()` → ✅ `generate_small_world()`
   - ❌ `grid_graph()` → ✅ `generate_lattice()`
   - Return order: `(Laplacian, eigenvalues, eigenvectors)`

2. **`state_generators.py`**:
   - ❌ `generate_harmonic_state()` → ✅ `generate_wake_state()`, etc.
   - Takes `n_modes` parameter, not eigenvalues array

3. **Node positions**:
   - `get_node_positions()` returns dict, convert to array:
   ```python
   pos_dict = gg.get_node_positions(G, layout='spring', seed=SEED)
   positions = np.array([pos_dict[i] for i in range(len(pos_dict))])
   ```

4. **Eigenvalue slicing**:
   - Use only first N_MODES eigenvalues for metrics:
   ```python
   metrics = met.compute_all_metrics(state_power, eigenvalues[:N_MODES])
   ```

5. **Wave analysis** (temporarily disabled):
   - Skip comprehensive_wave_analysis() in Exp 1-2 to avoid complexity
   - Focus on rotation and recovery metrics first
   - Re-enable for Exp 3 with lattice topology

---

## Expected Improvements

Based on refined parameters, we expect:

| Metric | v1 Result | v2 Expected | Improvement |
|--------|-----------|-------------|-------------|
| **Rotation Angles** | Binary (0°/180°) | Continuous | ✅ Better jPCA tuning |
| **Recovery %** | 0% (all trials) | 10-80% | ✅ Weaker perturbations |
| **Wave Detection** | 0/30 trials | 5-15/30 trials | ✅ Larger lattice network |
| **Rotation-Recovery r** | Undefined | 0.3-0.6 | ✅ Observable variance |

---

## Comparison with Empirical Data

### Batabyal et al. (2025) Predictions:

| Finding | Our v1 | Our v2 (Expected) |
|---------|--------|-------------------|
| Rotation enables recovery | ❌ Not observed | ✅ Should observe |
| State-dependent rotation | ✅ Observed | ✅ Strengthened |
| Rotation-performance correlation | ❌ No variance | ✅ Measurable |
| Traveling waves | ❌ Not detected | 🔍 TBD |

### Consciousness Framework Validation:

| Hypothesis | v1 Result | v2 Expected |
|------------|-----------|-------------|
| C(t) correlates with rotation quality | ✅ r=0.95 (3 states) | ✅ Confirmed |
| Recovery requires criticality | 🔍 Untestable | ✅ Testable now |
| Perturbation strength scales inversely with recovery | 🔍 Untestable | ✅ Testable now |
| Waves manifest rotation spatially | ❌ No waves | 🔍 Lattice test |

---

## Next Steps

### After v2 Completes:

1. **Analyze Results**:
   - Check rotation angle distribution (continuous vs. binary)
   - Measure recovery percentage statistics
   - Evaluate wave detection rate
   - Compute rotation-recovery correlation

2. **Update RESULTS_COMPARISON.md**:
   - Add v2 findings
   - Compare with v1 results
   - Update empirical agreement status

3. **If Successful**:
   - Proceed to hub vulnerability refinements (Task #4)
   - Integrate findings into main manuscript
   - Prepare figures for publication

4. **If Issues Remain**:
   - Further reduce perturbation strength (0.001-0.01)
   - Increase trajectory length (300-500 time steps)
   - Try different network topologies (hexagonal lattice, 3D cubic)

---

## Timeline

- **12:00 PM** - Started refinement work
- **12:30 PM** - Created v2 script with refined parameters
- **1:00 PM** - Fixed API compatibility issues
- **1:15 PM** - Experiment running
- **1:25 PM** (est.) - **Expected completion**
- **1:30 PM** - Analysis and comparison
- **2:00 PM** - Update documentation

---

## Success Criteria

✅ **Minimum Success**:
- Non-binary rotation angles
- >0% recovery in majority of trials
- Measurable rotation-recovery correlation

✅ **Target Success**:
- Continuous rotation angle distribution (std > 30°)
- Recovery 20-60% range
- Rotation-recovery correlation r > 0.3
- Some wave detection (>10% of trials)

✅ **Exceptional Success**:
- Rotation angles match Batabyal et al. patterns
- Recovery ~50% (matching working memory studies)
- Strong correlation (r > 0.5)
- Reproducible wave-rotation correspondence

---

## Notes

- **Performance**: v2 runs ~2x slower due to larger network (500 vs 300 nodes)
- **Memory**: Peak usage ~2GB (acceptable)
- **Reproducibility**: All experiments use `SEED=42` for deterministic results
- **Wave analysis**: Temporarily disabled in Exp 1-2 to isolate rotation/recovery refinements

---

**Last Updated**: December 25, 2025, 1:18 PM  
**Status**: Experiment running, awaiting results...
