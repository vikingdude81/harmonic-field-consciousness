import pandas as pd
import numpy as np

mega = pd.read_csv('results/mega/results_batched.csv')
print('Analyzing CORRECTED mega results...')
print()
print('Wave detection by type:')
for wt in range(4):
    data = mega[mega['wave_type'] == wt]
    waves = data['has_wave'].sum()
    total = len(data)
    print(f'  Type {wt}: {waves}/{total} ({100*waves/total:.1f}%)')
    if waves > 0:
        speeds = data[data["has_wave"]]["wave_speed"].values
        print(f'    Wave speeds: {speeds}')
print()
print('Sample rows (first 20):')
print(mega[['trial', 'wave_type', 'has_wave', 'wave_speed', 'rotation_angle']].head(20).to_string())
