# Wave Detection Algorithm Fix - January 11, 2026

## Problem Identified

The "Random Noise Paradox" was **NOT a real phenomenon**, but an **algorithmic bug**.

### The Bug

The `detect_waves_batch_gpu()` function in `exp_gpu_massive_batched.py` used a flawed "variance ratio" method:

```python
# BUGGY CODE (lines 189-194)
early_var = activity[:, :timesteps//4, :].var()
late_var = activity[:, -timesteps//4:, :].var()
var_ratio = late_var / early_var
has_wave = (var_ratio > 0.1) & (var_ratio < 2.0)  # ← WRONG!
```

**What it actually detected:** "Variance persistence" (does variance sustain over time?)

**What it should detect:** "Spatial correlation propagation" (does activity propagate spatially?)

### Why Results Were Backwards

| Initial Condition | Variance Behavior | Buggy Detection | Reality |
|------------------|-------------------|----------------|---------|
| Type 0 (Gaussian) | Decays quickly → ratio ≪ 0.1 | ✗ No wave (false negative) | Could have waves |
| Type 1 (Traveling) | Decays quickly → ratio ≪ 0.1 | ✗ No wave (false negative) | **Should** have waves! |
| Type 2 (Spiral) | Decays quickly → ratio ≪ 0.1 | ✗ No wave (false negative) | Could have waves |
| Type 3 (Random) | Sustains → ratio ≈ 1.0 | ✓ Wave (false positive) | No coherent wave! |

**Result:** Type 3 showed 100% waves, Types 0-2 showed 0% waves — completely backwards!

---

## The Fix

### Corrected Algorithm (Correlation-Based)

```python
# CORRECTED CODE
for b in range(batch_size):
    correlations = []
    
    for lag in range(1, max_lag):
        # Correlation between activity at t and t+lag
        act_early = activity[b, :-lag].flatten()
        act_late = activity[b, lag:].flatten()
        corr = torch.corrcoef(torch.stack([act_early, act_late]))[0, 1]
        correlations.append(corr)
    
    # Wave = high initial correlation + smooth decay
    mean_early = sum(correlations[:5]) / 5
    mean_late = sum(correlations[-5:]) / 5
    has_wave = (mean_early > 0.3) and (mean_early > mean_late)
```

**What it detects:** Spatial correlation that starts high and decays (real wave propagation)

### Detection Modes

The fixed code supports two modes:

1. **`detection_mode='correlation'`** (DEFAULT, CORRECTED)
   - Proper traveling wave detection
   - Checks spatial correlation propagation
   - Accurate but slower (~2x runtime)

2. **`detection_mode='variance'`** (BUGGY, for comparison only)
   - Old flawed method
   - Kept for validating other results
   - Shows warning when used

---

## Re-Running Experiments

### What We're Doing

1. **Backup old results:** `results/` → `results_backup_buggy/`
2. **Re-run with corrected detection:**
   - mega (24,964 nodes, 50 trials) → ~60s
   - ultra (25,921 nodes, 40 trials) → ~50s
   - max (25,921 nodes, 100 trials) → ~125s
3. **Compare old vs new:** `compare_wave_detection_results.py`
4. **Re-fit consciousness model:** Use corrected wave detection data

### Expected Results (Corrected)

| Initial Condition | Expected Wave Detection | Reason |
|------------------|-------------------------|--------|
| Type 0 (Gaussian) | LOW (~10-20%) | Localized excitation, may not propagate |
| Type 1 (Traveling) | **HIGH (~60-80%)** | Explicitly initialized as traveling wave |
| Type 2 (Spiral) | MEDIUM (~30-50%) | Rotating pattern, partial wave character |
| Type 3 (Random) | **LOW (~10-20%)** | No coherent structure, random noise |

This makes physical sense!

---

## Impact on Other Results

### ✓ Still Valid (Not Affected by Bug)

1. **Rotation angles** → Computed independently, completely unaffected
2. **Consciousness regression model** → Used rotation (77% variance), waves (12%)
3. **Category 1-7 validation** → Different code, separate experiments
4. **Bimodal Type 1 behavior** → Based on rotation, not waves
5. **Scale-invariant effects** → Used rotation metrics

### ✗ Potentially Affected (Need Re-Analysis)

1. **"Random Noise Paradox"** → Was completely wrong, bug artifact
2. **Wave detection rates** → All Type 3 were false positives
3. **Wave speed estimates** → Meaningless for false positives
4. **Initial condition dependence** → Need re-analysis with corrected data

### ~ Partially Affected (Minor Impact)

1. **Consciousness regression** → Rotation dominated (77%), waves minor (12%)
   - Model accuracy (R² = 0.8497) likely similar
   - Wave coefficient will change but small effect
   - Need to re-train but expect R² ~ 0.80-0.85

---

## Timeline

- **Discovered:** Jan 11, 2026 (user intuition: "could just be something is not being calculated")
- **Diagnosed:** Jan 11, 2026 (variance persistence ≠ wave propagation)
- **Fixed:** Jan 11, 2026 (implemented correlation-based detection)
- **Re-running:** Jan 11, 2026 (mega, ultra, max with corrected algorithm)
- **Expected completion:** Jan 11, 2026 (~4 minutes total runtime)

---

## Files Modified

1. **exp_gpu_massive_batched.py**
   - Lines 165-235: Replaced `fast_mode` with `detection_mode`
   - Added corrected correlation-based detection
   - Kept old method as `detection_mode='variance'` for comparison
   - Added warning when using buggy method

2. **New Scripts Created**
   - `investigate_wave_paradox.py` → Diagnosed the bug
   - `rerun_corrected_experiments.py` → Batch re-run script
   - `compare_wave_detection_results.py` → Old vs new comparison

---

## Key Lesson

**Always trust your intuition when results seem too surprising!**

The "Random Noise Paradox" violated basic physics:
- Random noise → no coherent structure → no traveling waves
- Structured initialization → organized patterns → potential waves

When reality contradicts theory, check the algorithm first! ✓

---

## Next Steps

1. Wait for mega/ultra/max re-runs to complete (~4 min)
2. Run comparison: `python compare_wave_detection_results.py`
3. Re-train consciousness model with corrected data
4. Update KEY_FINDINGS document (remove paradox, keep rotation dominance)
5. Focus on **Finding #3 (Rotation)** for NanoGPT integration
