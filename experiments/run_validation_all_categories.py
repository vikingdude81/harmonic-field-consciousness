#!/usr/bin/env python3
"""
Validation-Scale Multi-Category GPU Runner

Runs validation-scale experiments (large config: 4,900 nodes, 800 modes, 5,000 steps, 200 trials)
across all categories (1, 2, 3, 4, 5, 6, 7) to generate initial large datasets.

Config: Large (validation scale)
  - Nodes: 4,900 (70×70 lattice)
  - Modes: 800
  - Timesteps: 5,000
  - Trials: 200
  - Batch size: 16 (parallel)
  - Expected runtime: ~100 total per category

Usage:
  python run_validation_all_categories.py [category_number]
  python run_validation_all_categories.py 1       # Category 1 only
  python run_validation_all_categories.py all     # All categories (1-7)
"""

import subprocess
import sys
from pathlib import Path

CATEGORIES = {
    1: 'category1_network_topology',
    2: 'category2_dynamics',
    3: 'category3_functional_modifications',
    4: 'category4_applications',
    5: 'category5_advanced',
    6: 'category6_multiscale_dynamics',
    7: 'category7_spatiotemporal_transformers',
}

def run_category_gpu_experiment(category_num):
    """Run GPU-batched experiment for a category."""
    category_dir = Path(__file__).parent.parent / CATEGORIES[category_num]
    
    if not category_dir.exists():
        print(f"[SKIP] Category {category_num} directory not found: {category_dir}")
        return False
    
    # Look for exp_gpu_*.py or create one
    gpu_exp_script = category_dir / 'exp_gpu_validation_large.py'
    
    if not gpu_exp_script.exists():
        print(f"[SKIP] Category {category_num}: No exp_gpu_validation_large.py script found")
        return False
    
    print(f"\n{'='*80}")
    print(f"CATEGORY {category_num}: {CATEGORIES[category_num]}")
    print(f"Running: {gpu_exp_script}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(gpu_exp_script)],
            cwd=category_dir,
            timeout=300  # 5 minute timeout
        )
        success = result.returncode == 0
        if success:
            print(f"[OK] Category {category_num} completed successfully")
        else:
            print(f"[ERROR] Category {category_num} failed with return code {result.returncode}")
        return success
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Category {category_num} timed out (>5 min)")
        return False
    except Exception as e:
        print(f"[ERROR] Category {category_num}: {str(e)}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_validation_all_categories.py [category_number|all]")
        print(f"Available categories: {list(CATEGORIES.keys())}")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg.lower() == 'all':
        categories_to_run = list(CATEGORIES.keys())
    else:
        try:
            cat_num = int(arg)
            if cat_num not in CATEGORIES:
                print(f"Invalid category: {cat_num}")
                sys.exit(1)
            categories_to_run = [cat_num]
        except ValueError:
            print(f"Invalid argument: {arg}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("VALIDATION-SCALE GPU EXPERIMENTS (LARGE CONFIG)")
    print("="*80)
    print(f"Config: 4,900 nodes (70×70), 800 modes, 5,000 steps, 200 trials")
    print(f"Batch size: 16 (parallel)")
    print(f"Expected runtime: ~100s per category")
    print(f"Categories to run: {categories_to_run}")
    print("="*80)
    
    results = {}
    for cat_num in categories_to_run:
        results[cat_num] = run_category_gpu_experiment(cat_num)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    for cat_num in categories_to_run:
        status = "OK" if results[cat_num] else "FAILED"
        print(f"Category {cat_num}: [{status}]")
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    print("="*80)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
