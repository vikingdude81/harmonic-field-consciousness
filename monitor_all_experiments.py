#!/usr/bin/env python3
"""
Monitor All Running Experiments
Tracks both LLM training and eigensolver experiments
"""

import subprocess
import time
import os
from datetime import datetime, timedelta

def check_training_progress():
    """Check LLM training progress"""
    try:
        # Check if process is running
        result = subprocess.run(['wsl', 'pgrep', '-fa', 'train_100k'],
                              capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            print("[OK] LLM Training: RUNNING")

            # Get GPU status
            gpu_result = subprocess.run(['wsl', 'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                                        '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=5)

            if gpu_result.returncode == 0:
                lines = gpu_result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    gpu_id = parts[0].strip()
                    util = parts[1].strip()
                    mem_used = parts[2].strip()
                    mem_total = parts[3].strip()
                    temp = parts[4].strip()

                    if gpu_id == '0':  # RTX 5090
                        print(f"  GPU 0 (RTX 5090): {util}% util, {mem_used}/{mem_total} MB, {temp}°C")

            # Try to get training step from log
            log_result = subprocess.run(['wsl', 'bash', '-c',
                                        'tail -50 /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT/training_100k_*.log 2>/dev/null | grep -o "[0-9]*%|" | tail -1'],
                                       capture_output=True, text=True, timeout=5)

            if log_result.stdout.strip():
                print(f"  Progress: {log_result.stdout.strip()}")

        else:
            print("[X] LLM Training: NOT RUNNING")

    except Exception as e:
        print(f"[!] Error checking training: {e}")

def check_experiment_logs():
    """Check eigensolver experiment logs"""
    log_files = [
        ('large_scale_test.log', 'Large Scale Networks (30K-100K)'),
        ('topology_comparison.log', 'Topology Comparison (30K nodes)'),
        ('consciousness_analysis.log', 'Consciousness Analysis (25K nodes)')
    ]

    print("\n[EXPERIMENTS] Eigensolver Experiments:")

    for logfile, description in log_files:
        if os.path.exists(logfile):
            try:
                with open(logfile, 'r') as f:
                    lines = f.readlines()

                if len(lines) > 0:
                    last_lines = lines[-10:]

                    # Check if complete
                    if any('complete' in line.lower() or 'done' in line.lower() for line in last_lines):
                        print(f"  [OK] {description}: COMPLETE")
                    else:
                        print(f"  [...] {description}: RUNNING")
                        # Show last meaningful line
                        for line in reversed(last_lines):
                            if line.strip() and not line.startswith('='):
                                print(f"     Latest: {line.strip()[:80]}")
                                break
                else:
                    print(f"  [...] {description}: STARTING...")

            except Exception as e:
                print(f"  [!] {description}: Error reading log - {e}")
        else:
            print(f"  [?] {description}: Log not found")

def estimate_completion():
    """Estimate when training will complete"""
    try:
        # Get training start time (from when process started)
        # Approximate based on current progress
        print("\n[TIME] Estimated Completion Times:")

        # LLM Training: ~19 hours remaining from last check
        now = datetime.now()
        training_complete = now + timedelta(hours=19)
        print(f"  LLM Training: {training_complete.strftime('%Y-%m-%d %H:%M')} (~19 hours)")

        # Eigensolvers: faster
        eigen_complete = now + timedelta(hours=2)
        print(f"  Eigensolver Experiments: {eigen_complete.strftime('%Y-%m-%d %H:%M')} (~2 hours)")

    except Exception as e:
        print(f"  [!] Error estimating: {e}")

def main():
    """Monitor all experiments"""
    print("="*80)
    print("EXPERIMENT MONITOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n[TRAINING] LLM Training Status:")
    check_training_progress()

    check_experiment_logs()

    estimate_completion()

    print("\n" + "="*80)
    print("Refresh this script periodically to see updated progress!")
    print("="*80)

if __name__ == "__main__":
    main()
