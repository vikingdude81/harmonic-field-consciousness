"""
Compare Old (BUGGY) vs New (CORRECTED) Wave Detection Results
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("COMPARING BUGGY vs CORRECTED WAVE DETECTION")
print("="*80)
print()

configs = ['mega', 'ultra', 'max']
wave_names = {0: 'Gaussian', 1: 'Traveling Wave', 2: 'Spiral', 3: 'Random'}

for config in configs:
    print(f"\n{'='*80}")
    print(f"{config.upper()}")
    print(f"{'='*80}\n")
    
    # Load old and new results
    old_path = Path(f'results_backup_buggy/{config}/results_batched.csv')
    new_path = Path(f'results/{config}/results_batched.csv')
    
    if not old_path.exists():
        print(f"[SKIP] Old results not found: {old_path}")
        continue
    
    if not new_path.exists():
        print(f"[SKIP] New results not found: {new_path}")
        continue
    
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    
    print(f"Total trials: {len(old)}")
    print()
    
    # Overall wave detection comparison
    print("Overall Wave Detection:")
    print(f"  OLD (buggy):     {old['has_wave'].sum()}/{len(old)} ({100*old['has_wave'].mean():.1f}%)")
    print(f"  NEW (corrected): {new['has_wave'].sum()}/{len(new)} ({100*new['has_wave'].mean():.1f}%)")
    print(f"  DIFFERENCE:      {new['has_wave'].sum() - old['has_wave'].sum()} trials")
    print()
    
    # By wave type
    print("Wave Detection by Initial Condition Type:")
    print(f"{'Type':<18} {'OLD (buggy)':<15} {'NEW (corrected)':<15} {'Change':<10}")
    print("-" * 60)
    
    for wt in range(4):
        old_wt = old[old['wave_type'] == wt]
        new_wt = new[new['wave_type'] == wt]
        
        old_rate = 100 * old_wt['has_wave'].mean() if len(old_wt) > 0 else 0
        new_rate = 100 * new_wt['has_wave'].mean() if len(new_wt) > 0 else 0
        change = new_rate - old_rate
        
        print(f"{wave_names[wt]:<18} {old_rate:>6.1f}%         {new_rate:>6.1f}%         {change:>+6.1f}%")
    
    print()
    
    # Rotation angles (should be unchanged)
    print("Rotation Angles (should be identical):")
    print(f"  OLD: {old['rotation_angle'].mean():8.1f}° ± {old['rotation_angle'].std():5.1f}°")
    print(f"  NEW: {new['rotation_angle'].mean():8.1f}° ± {new['rotation_angle'].std():5.1f}°")
    print(f"  Match: {np.allclose(old['rotation_angle'].values, new['rotation_angle'].values, rtol=0.01)}")
    print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The CORRECTED algorithm should show:
  - Type 0 (Gaussian): LOW wave detection (structured blob, not wave)
  - Type 1 (Traveling): MEDIUM-HIGH (actual traveling wave initialization)
  - Type 2 (Spiral): MEDIUM (coherent structure, may have wave properties)
  - Type 3 (Random): LOW-MEDIUM (random noise, no coherent propagation)

The OLD buggy algorithm showed:
  - Type 0-2: 0% (false negatives - structured patterns decay variance)
  - Type 3: 100% (false positives - random noise sustains variance)

This was BACKWARDS!
""")
