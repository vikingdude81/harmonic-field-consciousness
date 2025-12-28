#!/usr/bin/env python3
"""
RTX 5090 Experiment Runner

Runs all Category 2 dynamics experiments with expanded parameters
optimized for RTX 5090 (34 GB VRAM, sm_120 architecture).

IMPORTANT: Run with system Python 3.11 that has PyTorch nightly:
    C:\\Users\\akbon\\AppData\\Local\\Programs\\Python\\Python311\\python.exe run_all_rtx5090.py

Requirements:
- PyTorch 2.11+ nightly with CUDA 12.8 (for sm_120 support)
- RTX 5090 as GPU 0
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Configuration
PYTHON_EXE = r"C:\Users\akbon\AppData\Local\Programs\Python\Python311\python.exe"
SCRIPT_DIR = Path(__file__).parent

# Define experiments with their enhanced parameters
EXPERIMENTS = {
    'exp1_state_transitions': {
        'description': 'State Transitions (Wake/NREM/Dream cycles)',
        'params': {
            'N_NODES': 2500,      # 50x increase from 100
            'N_MODES': 200,       # 10x increase from 20
            'N_STEPS': 1000,      # 5x increase from 200
        },
        'estimated_time': '2 min'
    },
    'exp2_perturbation_recovery': {
        'description': 'Perturbation Recovery Dynamics',
        'params': {
            'N_NODES': 2500,      # 25x increase from 100
            'N_MODES': 200,       # 6.7x increase from 30
            'RECOVERY_STEPS': 200,  # 4x increase from 50
            'N_TRIALS': 100,      # Multiple trials per level
        },
        'estimated_time': '3 min'
    },
    'exp3_coupling_strength': {
        'description': 'Coupling Strength & Synchronization',
        'params': {
            'N_NODES': 5000,      # 16.7x increase from 300
            'N_MODES': 400,       # 5x increase from 80
            'N_COUPLING_STEPS': 100,  # 2x increase from 50
        },
        'estimated_time': '5 min'
    },
    'exp4_criticality_tuning': {
        'description': 'Criticality Tuning (Edge of Chaos)',
        'params': {
            'N_NODES': 5000,      # 16.7x increase from 300
            'N_MODES': 400,       # 5x increase from 80
            'N_ALPHA_STEPS': 100, # 1.7x increase from 60
        },
        'estimated_time': '5 min'
    },
    'exp5_rotational_recovery': {
        'description': 'Rotational Dynamics & Traveling Waves',
        'params': {
            'N_NODES': 5000,      # 16.7x increase from 300
            'N_MODES': 400,       # 5x increase from 80
            'N_TRIALS': 100,      # 2x increase from 50
            'POST_PERTURBATION_TIME': 500,  # 5x increase from 100
        },
        'estimated_time': '8 min'
    },
    'exp_gpu_massive_batched': {
        'description': 'GPU Massive Batched (giga scale)',
        'params': {
            'config': 'giga',  # 25K nodes, 2K modes, 15K steps
        },
        'estimated_time': '1 min'
    }
}


def check_pytorch_cuda():
    """Verify PyTorch CUDA setup for RTX 5090."""
    print("=" * 70)
    print("CHECKING PYTORCH CUDA SETUP")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute: sm_{props.major}{props.minor}")
                
            # Verify we can use GPU 0
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x)
            print(f"\n[OK] GPU 0 test passed")
            return True
        else:
            print("[ERROR] CUDA not available!")
            return False
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def run_experiment(name: str, config: dict):
    """Run a single experiment with enhanced parameters."""
    script_path = SCRIPT_DIR / f"{name}.py"
    
    if not script_path.exists():
        print(f"[SKIP] {script_path} not found")
        return None
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"Description: {config['description']}")
    print(f"Estimated time: {config['estimated_time']}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Build command with parameters
    if name == 'exp_gpu_massive_batched':
        # Special case: pass config as argument
        cmd = [PYTHON_EXE, str(script_path), config['params'].get('config', 'giga')]
    else:
        # Regular experiments: set environment variables for parameters
        env = os.environ.copy()
        for key, value in config['params'].items():
            env[f'EXP_{key}'] = str(value)
        
        cmd = [PYTHON_EXE, str(script_path)]
    
    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=False,
            text=True,
            env=env if name != 'exp_gpu_massive_batched' else None
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n[OK] {name} completed in {elapsed:.1f}s")
            return elapsed
        else:
            print(f"\n[ERROR] {name} failed with code {result.returncode}")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] {name}: {e}")
        return None


def main():
    """Run all experiments sequentially."""
    print("=" * 70)
    print("RTX 5090 EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check CUDA setup
    if not check_pytorch_cuda():
        print("\n[ABORT] CUDA setup failed. Please ensure:")
        print("  1. Run with system Python 3.11 (not venv)")
        print(f"  2. Use: {PYTHON_EXE}")
        print("  3. PyTorch nightly is installed with CUDA 12.8")
        sys.exit(1)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for name, config in EXPERIMENTS.items():
        elapsed = run_experiment(name, config)
        results[name] = elapsed
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    succeeded = 0
    failed = 0
    
    for name, elapsed in results.items():
        if elapsed is not None:
            print(f"[OK] {name}: {elapsed:.1f}s")
            succeeded += 1
        else:
            print(f"[FAILED] {name}")
            failed += 1
    
    print(f"\nTotal: {succeeded} succeeded, {failed} failed")
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
