"""
Re-run Large Scale Experiments with CORRECTED Wave Detection
Runs: mega, ultra, max with proper correlation-based detection
"""

import subprocess
import sys
from pathlib import Path
import time

EXPERIMENTS = ['mega', 'ultra', 'max']

print("="*80)
print("RE-RUNNING EXPERIMENTS WITH CORRECTED WAVE DETECTION")
print("="*80)
print("\nOld (BUGGY) results backed up to: results_backup_buggy/")
print("New (CORRECTED) results will be saved to: results/\n")
print("Using correlation-based detection (proper traveling wave detection)")
print("="*80)
print()

for exp_name in EXPERIMENTS:
    print(f"\n{'='*80}")
    print(f"RUNNING: {exp_name.upper()}")
    print(f"{'='*80}\n")
    
    start = time.time()
    
    # Run experiment
    result = subprocess.run(
        [sys.executable, 'exp_gpu_massive_batched.py', exp_name],
        capture_output=False,
        text=True
    )
    
    runtime = time.time() - start
    
    if result.returncode == 0:
        print(f"\n[OK] {exp_name} completed in {runtime:.1f}s")
    else:
        print(f"\n[ERROR] {exp_name} failed with code {result.returncode}")
        sys.exit(1)
    
    print()

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED!")
print("="*80)
print("\nNext steps:")
print("  1. Compare old vs new results: python compare_wave_detection_results.py")
print("  2. Re-run consciousness regression: python ../../consciousness_regression_module.py")
print("  3. Analyze findings: python analyze_mega_results.py")
