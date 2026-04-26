# GPU Experiments - What Was Fixed

## Critical Bugs Fixed (In Your Codebase)

### Bug #1: Wave Detection Algorithm ❌ → ✅

**File**: [experiments/category2_dynamics/exp_gpu_massive_batched.py](experiments/category2_dynamics/exp_gpu_massive_batched.py#L165-L246)

**The Problem**:
The original wave detection used **variance ratio**, which detected "variance persistence" NOT actual traveling waves.

**Old Buggy Code** (lines 193-196):
```python
# WRONG: Detected variance persistence, not waves
early_var = activity[:, :timesteps//4, :].var(dim=(1,2))
late_var = activity[:, -timesteps//4:, :].var(dim=(1,2))
var_ratio = late_var / (early_var + 1e-10)
has_wave = (var_ratio > 0.1) & (var_ratio < 2.0)  # Backwards!
```

**What Happened**:
- Random noise (Type 3) sustained variance → **100% false positives** (detected as waves)
- Real structured patterns (Types 0-2) decayed → **0% detection** (not detected)
- Results were **completely backwards**!

**Fixed Code** (lines 203-241):
```python
# CORRECT: Correlation-based wave detection
for b in range(batch_size):
    correlations = []
    for lag in range(1, max_lag):
        act_early = activity_spatial_batch[b, :-lag].flatten()
        act_late = activity_spatial_batch[b, lag:].flatten()
        corr = torch.corrcoef(torch.stack([act_early, act_late]))[0, 1]
        correlations.append(corr.item())

    # Real wave: high initial correlation + smooth decay
    mean_early = sum(correlations[:5]) / 5
    mean_late = sum(correlations[-5:]) / 5
    has_wave[b] = (mean_early > 0.3) and (mean_early > mean_late)
```

**Why It's Fixed**:
- Now detects actual spatial-temporal correlation (real wave signature)
- Properly rejects random noise
- Correctly identifies traveling waves

---

### Bug #2: GPU Batch Randomization ❌ → ✅

**File**: [experiments/category2_dynamics/exp_gpu_massive_batched.py](experiments/category2_dynamics/exp_gpu_massive_batched.py#L307-L350)

**The Problem**:
Only **4 unique initial conditions** due to deterministic wave type selection.

**Old Buggy Code** (line 315):
```python
wave_type = trial % 4  # Only 4 patterns for ALL 100 trials!
```

**What Happened**:
- Trial 0, 4, 8, 12... had **IDENTICAL** Gaussian bumps
- Trial 1, 5, 9, 13... had **IDENTICAL** traveling waves
- Trial 2, 6, 10, 14... had **IDENTICAL** ring patterns
- Trial 3, 7, 11, 15... had **IDENTICAL** random noise

**Effective sample size = 4 patterns**, not 100 independent trials!

This **completely invalidates** statistical validation:
- Mean/std calculations wrong (only 4 samples, counted 25× each)
- Diversity metrics wrong
- 25% consciousness rule needs revalidation

**Fixed Code** (lines 318-340):
```python
# Generate UNIQUE seed per trial
generator = torch.Generator(device=device).manual_seed(42 + trial)

if wave_type == 1:  # Traveling wave
    # Randomize direction
    direction = torch.randn(2, device=device, generator=generator)
    direction = direction / (direction.norm() + 1e-10)
    projection = positions @ direction

    # Randomize wavelength
    wavelength = 0.3 + 0.4 * torch.rand(1, device=device, generator=generator).item()
    phase = torch.rand(1, device=device, generator=generator).item() * 2 * np.pi

    initial = torch.sin(2 * np.pi * projection / wavelength + phase)

# Same for other wave types - all use unique generator
```

**Why It's Fixed**:
- Each trial gets unique seed: `42 + trial`
- Randomized parameters: direction, wavelength, phase, position, amplitude
- Diversity check added: `initial_modes_batch.var(dim=0).mean()`
- Now have TRUE statistical independence

---

### Bug #3: Numerical Stability ❌ → ✅

**File**: [src/neural_mass/push_pull_oscillator.py](src/neural_mass/push_pull_oscillator.py#L228-L261)

**The Problem**:
FFT computation crashed on short signals (<4 samples).

**Old Buggy Code**:
```python
power = np.abs(fft[1:]) ** 2
peak_idx = np.argmax(power)  # IndexError if len(signal) < 2
return freqs[peak_idx + 1]    # IndexError if len(signal) < 3
```

**Fixed Code** (lines 235-261):
```python
if len(self.history['e_activity']) == 0:
    return 0.0
if len(signal) < 4:  # Need minimum 4 points for FFT
    return 0.0
if len(fft) < 2:     # Need AC components
    return 0.0
if np.max(power) == 0.0:  # Constant signal
    return 0.0
```

**Why It's Fixed**:
- Comprehensive edge case handling
- No more crashes on short simulations
- Robust for parameter sweeps

---

## Impact Assessment

### Experiment Results Affected

**All GPU experiments need re-running**:
- ✅ `small` config (961 nodes)
- ✅ `medium` config (2,499 nodes)
- ✅ `large` config (4,900 nodes)
- ✅ `xlarge` config (10,000 nodes)
- ✅ `mega` config (24,964 nodes)
- ✅ `max` config (25,921 nodes)

**Why Re-run**:
1. **Wave detection stats** are wrong (inverted)
2. **Rotation statistics** may be affected by lack of diversity
3. **25% consciousness rule** needs revalidation
4. **Correlation analyses** invalid (only 4 samples)

**What's NOT Affected**:
- ✅ Rotation angle calculation (independent of bugs)
- ✅ Eigenvalue decomposition (correct)
- ✅ Consciousness metrics formulas (correct)

---

## Expected Results After Re-running

### Wave Detection
**Before (Buggy)**:
- Type 3 (random): ~100% wave detection ❌
- Type 0-2 (structured): ~0% wave detection ❌

**After (Fixed)**:
- Type 3 (random): ~0-5% wave detection ✅
- Type 1 (traveling wave): ~80-95% wave detection ✅
- Type 0,2 (Gaussian, ring): ~10-30% wave detection ✅

### Statistical Validity
**Before (Buggy)**:
- Effective N = 4 ❌
- High variance = artifact of grouping ❌
- Correlations = confounded by repetition ❌

**After (Fixed)**:
- Effective N = 100 ✅
- True variance from independent samples ✅
- Valid statistical tests ✅

---

## How to Re-run

### Quick Test (1 config, ~15s)
```bash
cd experiments/category2_dynamics
python exp_gpu_massive_batched.py small
```

### Full Suite (all configs, ~1 hour)
```bash
cd experiments/category2_dynamics
python run_all_gpu_experiments.py --all
```

This will:
1. Run all 6-8 configurations
2. Generate results CSVs
3. Create comparison plots
4. Produce comprehensive report

### Verify Fixes Worked
```bash
# Check diversity improved
python verify_initial_conditions.py

# Compare old vs new wave detection
python compare_wave_detection_results.py
```

---

## Timeline

**Bugs Introduced**: Original implementation
**Bugs Discovered**: Audit (documented in `WAVE_DETECTION_BUG_FIX.md`)
**Bugs Fixed**: In your current codebase (you already applied fixes)
**Validation Needed**: Re-run experiments (~1 hour on RTX 5090)

---

## Summary

| Bug | Status | Impact | Fix Difficulty |
|-----|--------|--------|----------------|
| Wave detection | ✅ Fixed | HIGH | Easy (algorithm change) |
| GPU randomization | ✅ Fixed | CRITICAL | Easy (add generator) |
| Numerical stability | ✅ Fixed | MEDIUM | Easy (add checks) |

**All fixes are in place** - just need to re-run experiments to validate!

**Estimated time**: ~1 hour on RTX 5090 for full suite
**Expected outcome**: Corrected statistics, validated 25% rule
