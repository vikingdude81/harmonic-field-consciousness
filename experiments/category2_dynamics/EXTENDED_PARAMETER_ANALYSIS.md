# Extended Parameter Analysis - In Progress

**Date:** December 25, 2024  
**Status:** 🔄 RUNNING

## Analysis 1: Full 180-Trial Statistical Analysis ✅ COMPLETE

**Results:** [`exp2_full_analysis.png`](results/exp_rotational_recovery_v2/exp2_full_analysis.png)

### Key Findings

#### Critical Discovery: Binary Angles Persist
- **ALL 180 trials show binary angles** (0° or 180°)
- **0% continuous angles** despite v2 refinements
- This suggests the rotation detection may need further refinement

#### Strong Perturbation-Recovery Correlation  
- **r = 0.829, p < 0.001*** (highly significant)
- Normalized recovery increases linearly with perturbation strength
- Range: 0.525 (weak) → 0.687 (strong perturbations)

#### Detailed Statistics by Perturbation Strength

| Strength | Trials | Rotation Angle | Raw Recovery | Normalized Recovery | Quality |
|----------|--------|----------------|--------------|---------------------|---------|
| 0.01 | 20 | 117.0° ± 88.1° | 0.0% | 0.525 ± 0.037 | 0.750 |
| 0.021 | 20 | 81.0° ± 91.9° | 0.0% | 0.545 ± 0.035 | 0.741 |
| 0.033 | 20 | 117.0° ± 88.1° | 0.0% | 0.557 ± 0.042 | 0.737 |
| 0.044 | 20 | 81.0° ± 91.9° | 0.0% | 0.562 ± 0.037 | 0.737 |
| 0.055 | 20 | 81.0° ± 91.9° | 0.0% | 0.594 ± 0.041 | 0.741 |
| 0.066 | 20 | 81.0° ± 91.9° | 0.0% | 0.626 ± 0.032 | 0.749 |
| 0.078 | 20 | 108.0° ± 90.5° | 0.0% | 0.636 ± 0.025 | 0.742 |
| 0.089 | 20 | 99.0° ± 91.9° | 0.0% | 0.652 ± 0.028 | 0.749 |
| 0.10 | 20 | 126.0° ± 84.6° | 0.0% | 0.687 ± 0.031 | 0.743 |

#### Interpretations

**Binary Angle Problem:**
- Despite v2's improvements (Exp 1 showed continuous angles), Exp 2 reverted to binary
- Possible causes:
  1. Different attractor dynamics between state comparison (Exp 1) vs perturbation-recovery (Exp 2)
  2. Perturbation methodology may be forcing system to specific attractors
  3. jPCA plane selection may be missing intermediate rotations

**Normalized Recovery Success:**
- Strong linear relationship: stronger perturbations → better normalized recovery (30% increase)
- Suggests system IS responding proportionally to perturbations
- "Recovery" may mean reaching alternative stable state, not original state

**Raw Recovery = 0% Universally:**
- System never returns to exact initial conditions
- All recovery is to alternative attractors
- Supports multi-stability hypothesis in harmonic field theory

---

## Analysis 2: Dedicated Traveling Waves Experiment 🔄 RUNNING

**Script:** `exp_traveling_waves_dedicated.py`  
**Output:** `results/exp_traveling_waves_dedicated/`

### Extended Parameter Space

**Goal:** Test if more parameters and longer trajectories capture traveling waves

#### Network Sizes (3 levels)
- **N = 100** nodes (10×10 lattice)
- **N = 300** nodes (17×17 lattice)  
- **N = 500** nodes (22×22 lattice)

#### Lattice Connectivity (2 types)
- **4-connected:** Von Neumann neighborhood (up/down/left/right)
- **8-connected:** Moore neighborhood (includes diagonals)

#### Initial Conditions (4 types per trial)
1. **Gaussian bump:** Circular wave expanding from center
2. **Plane wave:** Traveling in linear direction (sin wave)
3. **Spiral wave:** Rotating pattern from center
4. **Random patch:** Localized excitation

#### Extended Parameters
- **50 trials per configuration** (vs 30 in v2)
- **300 time steps** (vs 200 in v2)
- **Nonlinear dynamics:** Added cubic coupling term
- **Dual detection methods:**
  - Comprehensive analysis (full wave characterization)
  - Simple detection (fallback for edge cases)

### Total Scope
- **3 sizes × 2 connectivities × 50 trials = 300 experiments**
- **4 wave types tested per trial**
- **90,000 time steps total** (300 trials × 300 steps)

### Expected Outcomes

**If waves are detectable:**
- Different detection rates by network size (expect higher in larger networks)
- Different rates by connectivity (8-connected should support waves better)
- Different rates by initial condition (plane/spiral should be easiest)
- Measurable wave speeds (spatial propagation velocity)
- Correlation between wave speed and rotational velocity

**If no waves detected:**
- May indicate:
  1. Harmonic modes naturally suppress traveling waves (smooth out spatial patterns)
  2. Time scales incompatible (waves too fast/slow for 300 steps)
  3. Network topology prevents wave propagation (even in lattices)
  4. Analysis methods insufficient for harmonic decomposition context

### Wave-Rotation Correspondence Test

For trials where waves ARE detected:
- **Correspondence score:** `|wave_speed - rotation_vel| / (wave_speed + rotation_vel)`
- Lower score = better correspondence
- Expected: If theory holds, should see scores < 0.3 indicating similar dynamics

---

## Comparison: V2 vs Extended Analysis

| Aspect | V2 (Original) | Extended Analysis |
|--------|---------------|-------------------|
| **Exp 2 trials** | 180 | Same data, full stats |
| **Statistical depth** | Summary only | Full correlation matrix, by-perturbation breakdown |
| **Visualization** | 6 basic panels | 6 detailed panels + distributions |
| **Wave experiment** | Crashed (API issues) | New dedicated experiment |
| **Network sizes** | 500 only | 100, 300, 500 |
| **Connectivity** | N/A (crashed) | 4-conn, 8-conn |
| **Wave init conditions** | 1 type | 4 types |
| **Trials per config** | 30 (crashed) | 50 |
| **Trajectory length** | 200 | 300 |
| **Detection methods** | 1 | 2 (comprehensive + simple) |

---

## Next Steps (Once Wave Experiment Completes)

### Immediate Analysis
1. **Wave detection rates:** Overall and by configuration
2. **Network size effects:** Does larger = more waves?
3. **Connectivity effects:** 8-connected vs 4-connected
4. **Initial condition effects:** Which wave type most detectable?
5. **Wave-rotation correspondence:** Is there alignment?

### If Waves Detected
1. Create detailed correspondence analysis
2. Map wave speed distributions
3. Correlate with consciousness metrics from Exp 1
4. Test hub vulnerability in wave-supporting networks

### If No Waves Detected
1. Analyze why harmonic decomposition suppresses waves
2. Test alternative spatial representations (raw activity vs modes)
3. Explore different time scales (longer/shorter)
4. Consider discrete wave detection (avalanches vs continuous waves)

### Theoretical Implications

**Scenario A: Waves detected at reasonable rates (>10%)**
- Supports dual dynamics: rotational + wave propagation
- Validates spatial component of harmonic field theory
- Opens path to integration-differentiation analysis

**Scenario B: Low wave detection (<10%)**
- Suggests harmonic modes may smooth out traveling waves
- May indicate consciousness more about rotational than propagation dynamics
- Requires rethinking spatial aspects of theory

**Scenario C: No waves detected (0%)**
- Fundamental question: Are traveling waves compatible with harmonic decomposition?
- May need raw spatial analysis separate from mode analysis
- Could indicate waves emerge at different organizational level

---

## Files Generated

### Analysis 1 ✅
- `analyze_exp2_full.py` - Statistical analysis script
- `results/exp_rotational_recovery_v2/exp2_full_analysis.png` - 6-panel visualization

### Analysis 2 🔄 (In progress)
- `exp_traveling_waves_dedicated.py` - Wave experiment script (running)
- `results/exp_traveling_waves_dedicated/wave_analysis_results.csv` (pending)
- `results/exp_traveling_waves_dedicated/wave_analysis_visualization.png` (pending)

---

## Computational Notes

**Analysis 1 Runtime:** ~5 seconds (pure analysis, no simulation)  
**Analysis 2 Estimated Runtime:** ~10-20 minutes (300 configs × 300 timesteps with jPCA)  
**Memory Usage:** Moderate (~2-4 GB for largest networks)

**Bottlenecks:**
- jPCA computation (O(n³) for eigendecomposition)
- Optical flow computation in wave detection
- Mode projection/reconstruction for spatial analysis

---

## Research Questions Addressed

1. ✅ **Do stronger perturbations show measurable effects?**
   - YES: Strong correlation (r=0.829) with normalized recovery

2. ✅ **Are rotation angles truly continuous or artifact of v2?**
   - NO: Exp 2 shows 100% binary angles despite v2 claiming continuous

3. 🔄 **Do traveling waves exist in harmonic mode dynamics?**
   - TESTING: 300 trials across multiple configurations

4. 🔄 **Does network size affect wave propagation?**
   - TESTING: 100 vs 300 vs 500 nodes

5. 🔄 **Does connectivity affect wave detection?**
   - TESTING: 4-connected vs 8-connected lattices

6. 🔄 **Which initial conditions best generate waves?**
   - TESTING: Gaussian, plane, spiral, random

7. 🔄 **Is there wave-rotation correspondence?**
   - TESTING: Correlation analysis for detected waves

---

## Open Questions for Future Work

1. **Why binary angles in Exp 2 but continuous in Exp 1?**
2. **Can we design perturbations that maintain continuous rotations?**
3. **If no waves, what spatial dynamics DO exist in harmonic modes?**
4. **How do these findings relate to empirical fMRI/EEG data?**
5. **Can hub vulnerability restore continuous rotations?**

---

*Last Updated: December 25, 2024 - Wave experiment in progress*
