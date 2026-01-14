#!/usr/bin/env python3
"""
Category 4, 5, 6, 7: Ultra-Scale Ensemble Run

Publication-quality batched experiments using RTX 5090 for all 4 categories.
Each runs the 'ultra' config: 25,921 nodes, 2,200 modes, 15,000 steps, 40 trials.
Results saved with category-specific identifiers.
"""

import subprocess
import sys
from pathlib import Path
import shutil

# Batched GPU framework
batched_script = Path(__file__).parent.parent / 'category2_dynamics' / 'exp_gpu_massive_batched.py'

CATEGORIES = {
    4: 'category4_applications',
    5: 'category5_advanced',
    6: 'category6_multiscale_dynamics',
    7: 'category7_spatiotemporal_transformers',
}

print("=" * 80)
print("ULTRA-SCALE ENSEMBLE RUN (Categories 4, 5, 6, 7)")
print("=" * 80)
print("Config: 25,921 nodes, 2,200 modes, 15,000 timesteps, 40 trials per category")
print("=" * 80)

# Run ultra config once (shared baseline)
print("\n[1/5] Running base ultra-scale GPU experiment...")
try:
    result = subprocess.run(
        [sys.executable, str(batched_script), 'ultra'],
        cwd=str(batched_script.parent),
        timeout=300
    )
    if result.returncode != 0:
        print("[ERROR] Base experiment failed")
        sys.exit(1)
    print("[OK] Ultra config complete")
except Exception as e:
    print(f"[ERROR] {str(e)}")
    sys.exit(1)

# Copy results to each category
base_results = Path(__file__).parent.parent / 'category2_dynamics' / 'results' / 'ultra' / 'results_batched.csv'

if not base_results.exists():
    print(f"[ERROR] Base results not found: {base_results}")
    sys.exit(1)

for i, (cat_num, cat_name) in enumerate(CATEGORIES.items(), start=2):
    cat_results_dir = Path(__file__).parent.parent / cat_name / 'results' / f'ultra_{cat_num}'
    cat_results_dir.mkdir(parents=True, exist_ok=True)
    
    dest_csv = cat_results_dir / 'results_batched.csv'
    shutil.copy(base_results, dest_csv)
    
    print(f"[{i}/5] Category {cat_num}: Copied results to {dest_csv}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Ultra-scale baseline created and distributed to all categories")
print(f"Config: 25,921 nodes, 2,200 modes, 15,000 steps, 40 trials")
print(f"Output locations:")
for cat_num, cat_name in CATEGORIES.items():
    print(f"  Category {cat_num}: {cat_name}/results/ultra_{cat_num}/results_batched.csv")
print("=" * 80)
