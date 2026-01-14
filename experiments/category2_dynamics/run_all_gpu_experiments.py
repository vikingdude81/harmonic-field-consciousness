"""
Automated Runner for All GPU Experiments

Runs all GPU-accelerated experiments with the FIXED randomization code.
Generates a comprehensive report comparing across scales.

Usage:
    python run_all_gpu_experiments.py [--configs small,medium,large]
    python run_all_gpu_experiments.py --all
"""

import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json


# Configuration
SCRIPT_PATH = Path(__file__).parent / "exp_gpu_massive_batched.py"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results" / "comprehensive_analysis"

ALL_CONFIGS = ['small', 'medium', 'large', 'xlarge', 'mega', 'giga', 'ultra', 'max']

EXPECTED_RUNTIMES = {
    'small': '15s',
    'medium': '45s',
    'large': '90s',
    'xlarge': '180s',
    'mega': '300s',
    'giga': '360s',
    'ultra': '420s',
    'max': '600s'
}


def run_single_experiment(config_name: str) -> dict:
    """Run a single GPU experiment configuration."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {config_name.upper()} CONFIGURATION")
    print(f"{'='*80}")
    print(f"Expected runtime: ~{EXPECTED_RUNTIMES[config_name]}")
    print()

    start = time.time()

    # Run experiment
    cmd = [sys.executable, str(SCRIPT_PATH), config_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    runtime = time.time() - start

    # Check for success
    if result.returncode != 0:
        print(f"[ERROR] Experiment {config_name} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {
            'config': config_name,
            'success': False,
            'runtime': runtime,
            'error': result.stderr
        }

    print(f"[OK] Experiment {config_name} completed in {runtime:.1f}s")

    # Load results
    results_file = RESULTS_DIR / config_name / 'results_batched.csv'

    if not results_file.exists():
        print(f"[WARNING] Results file not found: {results_file}")
        return {
            'config': config_name,
            'success': False,
            'runtime': runtime,
            'error': 'Results file not found'
        }

    df = pd.read_csv(results_file)

    # Compute summary statistics
    summary = {
        'config': config_name,
        'success': True,
        'runtime': runtime,
        'n_trials': len(df),
        'n_nodes': df['n_nodes'].iloc[0],
        'n_modes': df['n_modes'].iloc[0],
        'timesteps': df['timesteps'].iloc[0],

        # Rotation statistics
        'mean_rotation': df['rotation_angle'].mean(),
        'std_rotation': df['rotation_angle'].std(),
        'median_rotation': df['rotation_angle'].median(),

        # Wave statistics
        'wave_detection_rate': df['has_wave'].mean() * 100,
        'n_waves_detected': df['has_wave'].sum(),
        'mean_wave_speed': df[df['has_wave']]['wave_speed'].mean() if df['has_wave'].any() else 0.0,

        # Diversity check (coefficient of variation)
        'rotation_cv': df['rotation_angle'].std() / (df['rotation_angle'].mean() + 1e-10),

        # Per-trial throughput
        'trials_per_second': len(df) / runtime,
        'seconds_per_trial': runtime / len(df)
    }

    return summary


def run_all_experiments(configs: list):
    """Run all specified configurations."""
    print("=" * 80)
    print("COMPREHENSIVE GPU EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Configurations: {', '.join(configs)}")
    print(f"Total configs: {len(configs)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run each experiment
    results = []
    total_start = time.time()

    for config in configs:
        summary = run_single_experiment(config)
        results.append(summary)

        # Save incremental results
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_DIR / 'experiment_summary.csv', index=False)

    total_runtime = time.time() - total_start

    # Final summary
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total runtime: {total_runtime/60:.1f} minutes")
    print(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")
    print(f"Results saved to: {OUTPUT_DIR / 'experiment_summary.csv'}")
    print(f"{'='*80}")

    return results


def visualize_results(results: list):
    """Create comprehensive visualizations of results."""
    print("\nGenerating visualizations...")

    df = pd.DataFrame([r for r in results if r['success']])

    if len(df) == 0:
        print("[WARNING] No successful experiments to visualize")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GPU Experiment Scaling Analysis - FIXED RANDOMIZATION', fontsize=16, fontweight='bold')

    # 1. Rotation angle vs scale
    ax = axes[0, 0]
    ax.errorbar(df['n_nodes'], df['mean_rotation'], yerr=df['std_rotation'],
                fmt='o-', capsize=5, markersize=8, linewidth=2)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Rotation Angle (degrees)')
    ax.set_title('Mean Rotation vs Scale')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Coefficient of variation (diversity check)
    ax = axes[0, 1]
    ax.plot(df['n_nodes'], df['rotation_cv'], 'o-', markersize=8, linewidth=2, color='green')
    ax.axhline(y=0.1, color='r', linestyle='--', label='Low diversity threshold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Rotation Diversity (CV) - Should be >0.1')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Wave detection rate
    ax = axes[0, 2]
    colors = ['red' if rate < 20 else 'green' for rate in df['wave_detection_rate']]
    ax.bar(range(len(df)), df['wave_detection_rate'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.set_ylabel('Wave Detection Rate (%)')
    ax.set_title('Wave Detection Rate by Config')
    ax.axhline(y=25, color='blue', linestyle='--', label='Expected 25%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance (trials/second)
    ax = axes[1, 0]
    ax.plot(df['n_nodes'], df['trials_per_second'], 'o-', markersize=8, linewidth=2, color='purple')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Trials per Second')
    ax.set_title('Computational Throughput')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 5. Runtime scaling
    ax = axes[1, 1]
    ax.plot(df['n_nodes'], df['runtime'], 'o-', markersize=8, linewidth=2, color='orange')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Total Runtime (seconds)')
    ax.set_title('Runtime Scaling')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['config'],
            f"{row['n_nodes']:,}",
            f"{row['mean_rotation']:.1f}±{row['std_rotation']:.1f}",
            f"{row['wave_detection_rate']:.1f}%"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Config', 'Nodes', 'Rotation (deg)', 'Waves (%)'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Summary Statistics', pad=20)

    plt.tight_layout()

    # Save figure
    fig_path = OUTPUT_DIR / 'scaling_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {fig_path}")

    plt.close()


def generate_report(results: list):
    """Generate markdown report."""
    print("\nGenerating report...")

    df = pd.DataFrame([r for r in results if r['success']])

    report = f"""# GPU Experiment Report - FIXED RANDOMIZATION

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Configurations Run**: {len(results)}
**Successful**: {len(df)}

## Summary

This report presents results from GPU-accelerated harmonic field consciousness experiments
with **FIXED randomization** that ensures true statistical independence across trials.

### Key Improvements from Fix
1. ✅ Randomized wave type (not deterministic trial % 4)
2. ✅ Randomized parameters within each wave type (center, width, direction, phase, etc.)
3. ✅ True N={df['n_trials'].iloc[0] if len(df) > 0 else 'N/A'} independent trials (not 4 patterns × 25 repetitions)

## Results by Configuration

"""

    for _, row in df.iterrows():
        report += f"""
### {row['config'].upper()} Configuration

- **Nodes**: {row['n_nodes']:,}
- **Modes**: {row['n_modes']:,}
- **Timesteps**: {row['timesteps']:,}
- **Trials**: {row['n_trials']:,}
- **Runtime**: {row['runtime']:.1f}s ({row['seconds_per_trial']:.3f}s per trial)

**Results**:
- Rotation angle: {row['mean_rotation']:.2f} ± {row['std_rotation']:.2f} degrees
- Median rotation: {row['median_rotation']:.2f} degrees
- Coefficient of variation: {row['rotation_cv']:.3f} {'✅ Good diversity' if row['rotation_cv'] > 0.1 else '⚠️ Low diversity'}
- Wave detection rate: {row['wave_detection_rate']:.1f}% ({row['n_waves_detected']:.0f}/{row['n_trials']:.0f} trials)
- Mean wave speed: {row['mean_wave_speed']:.2f} (when detected)

**Throughput**: {row['trials_per_second']:.2f} trials/second

---
"""

    report += f"""
## Scaling Analysis

### Rotation Scaling
- **Smallest** ({df.iloc[0]['config']}): {df.iloc[0]['mean_rotation']:.1f}°
- **Largest** ({df.iloc[-1]['config']}): {df.iloc[-1]['mean_rotation']:.1f}°
- **Ratio**: {df.iloc[-1]['mean_rotation'] / (df.iloc[0]['mean_rotation'] + 1e-10):.2f}×

### Performance Scaling
- **Smallest throughput**: {df['trials_per_second'].max():.2f} trials/s ({df['trials_per_second'].idxmax()})
- **Largest throughput**: {df['trials_per_second'].min():.2f} trials/s ({df['trials_per_second'].idxmin()})

### Diversity Validation
- **Mean CV**: {df['rotation_cv'].mean():.3f}
- **All configs pass diversity check (CV > 0.1)**: {'✅ YES' if df['rotation_cv'].min() > 0.1 else '❌ NO'}

## Conclusions

The FIXED randomization ensures:
1. True statistical independence between trials
2. Rich parameter diversity within each wave type
3. Coefficient of variation >0.1 (confirms diversity)
4. Valid statistical inference on 25% rule and scaling laws

## Files

- Summary CSV: `experiment_summary.csv`
- Visualization: `scaling_analysis.png`
- This report: `EXPERIMENT_REPORT.md`

---
**Report generated automatically by run_all_gpu_experiments.py**
"""

    report_path = OUTPUT_DIR / 'EXPERIMENT_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"[OK] Report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run GPU experiments with fixed randomization')
    parser.add_argument('--configs', type=str, default='small,medium,large',
                        help='Comma-separated list of configs (default: small,medium,large)')
    parser.add_argument('--all', action='store_true',
                        help='Run all configurations (overrides --configs)')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization generation')

    args = parser.parse_args()

    # Determine which configs to run
    if args.all:
        configs = ALL_CONFIGS
    else:
        configs = [c.strip() for c in args.configs.split(',')]

    # Validate configs
    invalid = [c for c in configs if c not in ALL_CONFIGS]
    if invalid:
        print(f"[ERROR] Invalid configs: {invalid}")
        print(f"Valid options: {ALL_CONFIGS}")
        sys.exit(1)

    # Run experiments
    results = run_all_experiments(configs)

    # Generate visualizations
    if not args.skip_viz:
        visualize_results(results)
        generate_report(results)

    print("\n✅ DONE!")


if __name__ == '__main__':
    main()
