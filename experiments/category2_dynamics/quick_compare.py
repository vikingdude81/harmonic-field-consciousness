import pandas as pd
import numpy as np

# Load mega results
old = pd.read_csv('results_backup_buggy/mega/results_batched.csv')
new = pd.read_csv('results/mega/results_batched.csv')

print('MEGA Results Comparison')
print('='*60)
print(f'Total trials: {len(old)}')
print()
print('Overall Wave Detection:')
print(f'  OLD (buggy):     {old["has_wave"].sum()}/{len(old)} ({100*old["has_wave"].mean():.1f}%)')
print(f'  NEW (corrected): {new["has_wave"].sum()}/{len(new)} ({100*new["has_wave"].mean():.1f}%)')
print()

wave_names = {0: 'Gaussian', 1: 'Traveling', 2: 'Spiral', 3: 'Random'}
print('By Initial Condition Type:')
for wt in range(4):
    old_wt = old[old['wave_type'] == wt]
    new_wt = new[new['wave_type'] == wt]
    old_rate = 100 * old_wt['has_wave'].mean() if len(old_wt) > 0 else 0
    new_rate = 100 * new_wt['has_wave'].mean() if len(new_wt) > 0 else 0
    print(f'  Type {wt} ({wave_names[wt]:<10}): OLD={old_rate:5.1f}%, NEW={new_rate:5.1f}%')
