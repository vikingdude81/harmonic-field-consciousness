#!/usr/bin/env python3
"""Compute summary stats from ultra/max GPU experiments"""
import pandas as pd
import numpy as np
from pathlib import Path

ultra = pd.read_csv('results/ultra/results_batched.csv')
max_df = pd.read_csv('results/max/results_batched.csv')

print("=" * 80)
print("ULTRA CONFIG (25,921 nodes, 2,200 modes, 15,000 steps, 40 trials)")
print("=" * 80)
print(f"Rotation angle: {ultra['rotation_angle'].mean():.2f}° ± {ultra['rotation_angle'].std():.2f}°")
print(f"Wave detection: {ultra['has_wave'].sum()}/{len(ultra)} ({ultra['has_wave'].mean()*100:.1f}%)")
if ultra['has_wave'].any():
    print(f"Mean wave speed: {ultra[ultra['has_wave']]['wave_speed'].mean():.2f}")
print(f"Rotation by wave_type:")
for wt in ultra['wave_type'].unique():
    wt_data = ultra[ultra['wave_type'] == wt]['rotation_angle']
    print(f"  Type {wt}: {wt_data.mean():.2f}° ± {wt_data.std():.2f}° (n={len(wt_data)})")

print("\n" + "=" * 80)
print("MAX CONFIG (25,921 nodes, 2,500 modes, 20,000 steps, 100 trials)")
print("=" * 80)
print(f"Rotation angle: {max_df['rotation_angle'].mean():.2f}° ± {max_df['rotation_angle'].std():.2f}°")
print(f"Wave detection: {max_df['has_wave'].sum()}/{len(max_df)} ({max_df['has_wave'].mean()*100:.1f}%)")
if max_df['has_wave'].any():
    print(f"Mean wave speed: {max_df[max_df['has_wave']]['wave_speed'].mean():.2f}")
print(f"Rotation by wave_type:")
for wt in max_df['wave_type'].unique():
    wt_data = max_df[max_df['wave_type'] == wt]['rotation_angle']
    print(f"  Type {wt}: {wt_data.mean():.2f}° ± {wt_data.std():.2f}° (n={len(wt_data)})")
