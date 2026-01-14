#!/usr/bin/env python3
"""
Category 1: Network Topology - Ultra-Scale GPU Experiment

Publication-quality large-scale experiment with 25,921 nodes.
Uses existing batched framework to explore network topology effects on consciousness metrics.
"""

import subprocess
import sys
from pathlib import Path

# Use the existing batched GPU experiment framework in category2_dynamics
batched_script = Path(__file__).parent.parent / 'category2_dynamics' / 'exp_gpu_massive_batched.py'

print("=" * 80)
print("CATEGORY 1: NETWORK TOPOLOGY ULTRA-SCALE")
print("=" * 80)
print("Config: 25,921 nodes (161×161), 2,200 modes, 15,000 timesteps, 40 trials")
print("Running GPU-batched experiment...")
print("=" * 80)

# Run the ultra config from the existing framework
try:
    result = subprocess.run(
        [sys.executable, str(batched_script), 'ultra'],
        timeout=300
    )
    if result.returncode == 0:
        print("\n[OK] Ultra-scale experiment completed successfully")
        print("Results saved to: experiments/category2_dynamics/results/ultra/results_batched.csv")
        print("(Use this as Category 1 topology baseline)")
    else:
        print(f"\n[ERROR] Experiment failed with return code {result.returncode}")
        sys.exit(1)
except subprocess.TimeoutExpired:
    print("[ERROR] Experiment timed out")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] {str(e)}")
    sys.exit(1)
