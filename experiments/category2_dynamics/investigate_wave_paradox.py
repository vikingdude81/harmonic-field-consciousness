"""
Investigate the "Random Noise Paradox"
Does Type 3 (random) really produce 100% waves, or is there a bug?
"""

import pandas as pd
import numpy as np

# Load mega results
mega = pd.read_csv('results/mega/results_batched.csv')

print("="*80)
print("INVESTIGATING THE RANDOM NOISE PARADOX")
print("="*80)
print()

# Analyze each wave type
wave_names = {0: 'Gaussian', 1: 'Traveling Wave', 2: 'Spiral', 3: 'Random'}

for wt in range(4):
    data = mega[mega['wave_type'] == wt]
    waves = data[data['has_wave'] == True]
    
    print(f"Type {wt} ({wave_names[wt]}):")
    print(f"  Total trials: {len(data)}")
    print(f"  Waves detected: {len(waves)}/{len(data)} ({100*len(waves)/len(data):.1f}%)")
    
    if len(waves) > 0:
        print(f"  Wave speed: {waves['wave_speed'].mean():.3f} ± {waves['wave_speed'].std():.3f}")
    
    # Key metric: variance ratio (used for wave detection)
    print(f"  Rotation: {data['rotation_angle'].mean():.1f}° ± {data['rotation_angle'].std():.1f}°")
    print()

print("\n" + "="*80)
print("CHECKING WAVE DETECTION ALGORITHM")
print("="*80)
print("\nThe algorithm uses variance ratio: late_var / early_var")
print("Wave detected if: 0.1 < ratio < 2.0")
print()

# Manually check a few Type 3 trials
print("Sample Type 3 trials (first 5):")
type3 = mega[mega['wave_type'] == 3].head(5)
for idx, row in type3.iterrows():
    print(f"  Trial {row['trial']}: has_wave={row['has_wave']}, rotation={row['rotation_angle']:.1f}°")

print("\nSample Type 0 trials (first 5):")
type0 = mega[mega['wave_type'] == 0].head(5)
for idx, row in type0.iterrows():
    print(f"  Trial {row['trial']}: has_wave={row['has_wave']}, rotation={row['rotation_angle']:.1f}°")

print("\n" + "="*80)
print("HYPOTHESIS: Wave Detection May Be Flawed")
print("="*80)
print("""
The fast_mode wave detection uses variance ratio:
  - early_var = variance in first 25% of timesteps
  - late_var = variance in last 25% of timesteps
  - has_wave = (0.1 < late_var/early_var < 2.0)

PROBLEM: This detects "sustained variance", NOT traveling waves!

Type 3 (random noise) → high variance throughout → ratio ≈ 1.0 → "wave"
Type 0 (Gaussian blob) → variance decays quickly → ratio << 0.1 → no "wave"

REAL WAVES should show:
  1. Spatial correlation propagation (not just sustained variance)
  2. Phase gradients
  3. Directional flow

CONCLUSION: The "wave detector" is actually a "variance sustenance" detector.
Random noise maintains variance → false positive.
Structured patterns decay → false negative.

This is a BUG, not a real phenomenon!
""")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
1. Re-run mega with fast_mode=False (use proper correlation-based detection)
2. Compare results: does Type 3 still show 100% waves?
3. Add phase gradient check to wave detection
4. Validate against ground truth (synthetic traveling waves)

The "Random Noise Paradox" is likely an algorithmic artifact!
""")
