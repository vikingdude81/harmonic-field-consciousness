#!/usr/bin/env python3
"""
Master Experiment Runner

Run all experiments in the framework with a single command.
Provides options for parallel execution, progress tracking, and result aggregation.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import json

# Find all experiment scripts
EXPERIMENTS_DIR = Path(__file__).parent
CATEGORIES = [
    'category1_network_topology',
    'category2_dynamics',
    'category3_functional_modifications',
    'category4_applications',
]

def find_experiments():
    """Find all experiment scripts."""
    experiments = []
    for category in CATEGORIES:
        category_dir = EXPERIMENTS_DIR / category
        if not category_dir.exists():
            continue
        
        for exp_file in sorted(category_dir.glob('exp*.py')):
            experiments.append({
                'category': category,
                'name': exp_file.stem,
                'path': exp_file,
            })
    
    return experiments


def run_experiment(exp_info, verbose=True):
    """Run a single experiment."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {exp_info['category']}/{exp_info['name']}")
        print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(exp_info['path'])],
            cwd=exp_info['path'].parent,
            capture_output=not verbose,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        success = result.returncode == 0
        
        if not success and verbose:
            print(f"ERROR: Experiment failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
        
        return {
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout if not verbose else None,
            'stderr': result.stderr if not verbose else None,
        }
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment timed out")
        return {
            'success': False,
            'error': 'timeout',
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            'success': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run experimental framework')
    parser.add_argument(
        '--category',
        type=str,
        help='Run only experiments in this category'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Run only this specific experiment (e.g., exp1_topology_comparison)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress experiment output'
    )
    
    args = parser.parse_args()
    
    # Find all experiments
    experiments = find_experiments()
    
    if args.list:
        print("\nAvailable experiments:")
        print("="*60)
        for exp in experiments:
            print(f"  {exp['category']}/{exp['name']}")
        print(f"\nTotal: {len(experiments)} experiments")
        return
    
    # Filter experiments
    if args.category:
        experiments = [e for e in experiments if e['category'] == args.category]
    
    if args.experiment:
        experiments = [e for e in experiments if e['name'] == args.experiment]
    
    if not experiments:
        print("No experiments found matching criteria.")
        return
    
    # Run experiments
    print("\n" + "="*60)
    print(f"Experimental Framework Runner")
    print(f"Running {len(experiments)} experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    for exp_info in experiments:
        result = run_experiment(exp_info, verbose=not args.quiet)
        results.append({
            'category': exp_info['category'],
            'name': exp_info['name'],
            **result,
        })
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['category']}/{r['name']}")
    
    # Save results
    results_file = EXPERIMENTS_DIR / 'run_all_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
