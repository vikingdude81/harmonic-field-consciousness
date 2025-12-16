#!/usr/bin/env python3
"""
Comparative Analysis Utilities

Side-by-side experiment comparison tools for:
- Comparing results across experiments
- Statistical comparison of methods
- Ranking and scoring systems
- Multi-criteria decision analysis
- Publication-ready comparison tables
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field
import json
import warnings

# Optional imports
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ExperimentRecord:
    """Record of an experiment for comparison."""
    name: str
    results: pd.DataFrame
    metrics: Dict[str, float]
    parameters: Dict
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentComparator:
    """
    Compare results across multiple experiments.
    
    Features:
    - Side-by-side metric comparison
    - Statistical significance testing
    - Ranking across metrics
    - Multi-criteria scoring
    - Visualization and table generation
    
    Example:
        comparator = ExperimentComparator('network_methods')
        
        # Add experiments to compare
        comparator.add_experiment('method_a', results_a, {'C': 0.85, 'H': 0.92})
        comparator.add_experiment('method_b', results_b, {'C': 0.78, 'H': 0.95})
        
        # Compare
        comparison = comparator.compare_all()
        ranking = comparator.rank_experiments('C', maximize=True)
        
        # Generate outputs
        comparator.to_latex_table()
        comparator.plot_comparison()
    """
    
    def __init__(self, name: str = 'comparison',
                 output_dir: Optional[Path] = None):
        """
        Initialize experiment comparator.
        
        Args:
            name: Name for this comparison
            output_dir: Output directory for results
        """
        self.name = name
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'comparisons'
        
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentRecord] = {}
        self._load_existing()
    
    def _load_existing(self):
        """Load existing comparison data."""
        state_file = self.output_dir / 'comparison_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                for exp_data in data.get('experiments', []):
                    # Reconstruct experiment records
                    results_file = self.output_dir / f"{exp_data['name']}_results.csv"
                    if results_file.exists():
                        results = pd.read_csv(results_file)
                    else:
                        results = pd.DataFrame()
                    
                    self.experiments[exp_data['name']] = ExperimentRecord(
                        name=exp_data['name'],
                        results=results,
                        metrics=exp_data.get('metrics', {}),
                        parameters=exp_data.get('parameters', {}),
                        metadata=exp_data.get('metadata', {}),
                        timestamp=exp_data.get('timestamp', '')
                    )
    
    def _save_state(self):
        """Save comparison state."""
        state_file = self.output_dir / 'comparison_state.json'
        
        exp_data = []
        for name, record in self.experiments.items():
            # Save results DataFrame
            if not record.results.empty:
                record.results.to_csv(self.output_dir / f"{name}_results.csv", index=False)
            
            exp_data.append({
                'name': record.name,
                'metrics': record.metrics,
                'parameters': record.parameters,
                'metadata': record.metadata,
                'timestamp': record.timestamp
            })
        
        with open(state_file, 'w') as f:
            json.dump({'experiments': exp_data}, f, indent=2, default=str)
    
    def add_experiment(self, name: str,
                       results: Optional[pd.DataFrame] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       parameters: Optional[Dict] = None,
                       metadata: Optional[Dict] = None):
        """
        Add an experiment for comparison.
        
        Args:
            name: Experiment identifier
            results: Full results DataFrame
            metrics: Summary metrics dict
            parameters: Experiment parameters
            metadata: Additional metadata
        """
        self.experiments[name] = ExperimentRecord(
            name=name,
            results=results if results is not None else pd.DataFrame(),
            metrics=metrics or {},
            parameters=parameters or {},
            metadata=metadata or {}
        )
        self._save_state()
    
    def add_from_csv(self, name: str, filepath: Union[str, Path],
                     metric_columns: Optional[List[str]] = None,
                     parameters: Optional[Dict] = None):
        """
        Add experiment from CSV file.
        
        Args:
            name: Experiment identifier
            filepath: Path to CSV file
            metric_columns: Columns to use as summary metrics (uses mean)
            parameters: Experiment parameters
        """
        df = pd.read_csv(filepath)
        
        metrics = {}
        if metric_columns:
            for col in metric_columns:
                if col in df.columns:
                    metrics[col] = float(df[col].mean())
        else:
            # Use all numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                metrics[col] = float(df[col].mean())
        
        self.add_experiment(name, results=df, metrics=metrics, parameters=parameters)
    
    def remove_experiment(self, name: str):
        """Remove an experiment from comparison."""
        if name in self.experiments:
            del self.experiments[name]
            self._save_state()
    
    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        return list(self.experiments.keys())
    
    def get_metrics_table(self) -> pd.DataFrame:
        """
        Get table of all metrics across experiments.
        
        Returns:
            DataFrame with experiments as rows, metrics as columns
        """
        rows = []
        for name, record in self.experiments.items():
            row = {'experiment': name, **record.metrics}
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('experiment')
    
    def compare_all(self, normalize: bool = False) -> pd.DataFrame:
        """
        Compare all experiments across all metrics.
        
        Args:
            normalize: Normalize metrics to 0-1 range
            
        Returns:
            Comparison DataFrame
        """
        df = self.get_metrics_table()
        
        if normalize and not df.empty:
            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    def compare_pair(self, exp1: str, exp2: str,
                     metrics: Optional[List[str]] = None) -> Dict:
        """
        Detailed comparison of two experiments.
        
        Args:
            exp1: First experiment name
            exp2: Second experiment name
            metrics: Specific metrics to compare (default: all)
            
        Returns:
            Comparison results dict
        """
        if exp1 not in self.experiments or exp2 not in self.experiments:
            raise ValueError("Both experiments must exist")
        
        rec1 = self.experiments[exp1]
        rec2 = self.experiments[exp2]
        
        if metrics is None:
            metrics = list(set(rec1.metrics.keys()) & set(rec2.metrics.keys()))
        
        comparison = {
            'experiment_1': exp1,
            'experiment_2': exp2,
            'metrics': {}
        }
        
        for metric in metrics:
            val1 = rec1.metrics.get(metric, np.nan)
            val2 = rec2.metrics.get(metric, np.nan)
            
            metric_comp = {
                'value_1': val1,
                'value_2': val2,
                'difference': val1 - val2,
                'relative_diff': (val1 - val2) / (abs(val2) + 1e-10) * 100,
                'winner': exp1 if val1 > val2 else (exp2 if val2 > val1 else 'tie')
            }
            
            # Statistical test if we have full results
            if not rec1.results.empty and not rec2.results.empty:
                if metric in rec1.results.columns and metric in rec2.results.columns:
                    vals1 = rec1.results[metric].dropna().values
                    vals2 = rec2.results[metric].dropna().values
                    
                    if HAS_SCIPY and len(vals1) > 1 and len(vals2) > 1:
                        stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                        metric_comp['p_value'] = pval
                        metric_comp['significant'] = pval < 0.05
            
            comparison['metrics'][metric] = metric_comp
        
        return comparison
    
    def rank_experiments(self, metric: str, maximize: bool = True) -> List[Tuple[str, float, int]]:
        """
        Rank experiments by a specific metric.
        
        Args:
            metric: Metric to rank by
            maximize: True if higher is better
            
        Returns:
            List of (experiment_name, value, rank) tuples
        """
        values = []
        for name, record in self.experiments.items():
            if metric in record.metrics:
                values.append((name, record.metrics[metric]))
        
        sorted_values = sorted(values, key=lambda x: x[1], reverse=maximize)
        
        return [(name, val, rank + 1) for rank, (name, val) in enumerate(sorted_values)]
    
    def compute_overall_ranking(self, metrics: Optional[List[str]] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 maximize: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """
        Compute overall ranking using multiple metrics.
        
        Args:
            metrics: Metrics to use (default: all)
            weights: Weight for each metric (default: equal)
            maximize: Dict specifying if higher is better for each metric
            
        Returns:
            DataFrame with rankings and scores
        """
        df = self.get_metrics_table()
        
        if df.empty:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = list(df.columns)
        
        if weights is None:
            weights = {m: 1.0 for m in metrics}
        
        if maximize is None:
            maximize = {m: True for m in metrics}
        
        # Normalize and weight
        scores = pd.DataFrame(index=df.index)
        
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                min_val, max_val = values.min(), values.max()
                
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.ones_like(values) * 0.5
                
                # Invert if lower is better
                if not maximize.get(metric, True):
                    normalized = 1 - normalized
                
                scores[metric] = normalized * weights.get(metric, 1.0)
        
        # Compute total score
        scores['total_score'] = scores.sum(axis=1)
        scores['rank'] = scores['total_score'].rank(ascending=False).astype(int)
        
        return scores.sort_values('rank')
    
    def statistical_comparison(self, baseline: str,
                               metric: str,
                               test: str = 'mannwhitney') -> pd.DataFrame:
        """
        Statistical comparison of all experiments against a baseline.
        
        Args:
            baseline: Name of baseline experiment
            metric: Metric to compare
            test: Statistical test ('mannwhitney', 'ttest', 'wilcoxon')
            
        Returns:
            DataFrame with p-values and effect sizes
        """
        if baseline not in self.experiments:
            raise ValueError(f"Baseline '{baseline}' not found")
        
        baseline_rec = self.experiments[baseline]
        
        if baseline_rec.results.empty or metric not in baseline_rec.results.columns:
            raise ValueError(f"Baseline has no data for metric '{metric}'")
        
        baseline_values = baseline_rec.results[metric].dropna().values
        
        results = []
        
        for name, record in self.experiments.items():
            if name == baseline:
                continue
            
            if record.results.empty or metric not in record.results.columns:
                continue
            
            exp_values = record.results[metric].dropna().values
            
            if len(exp_values) < 2:
                continue
            
            row = {
                'experiment': name,
                'mean': float(exp_values.mean()),
                'std': float(exp_values.std()),
                'baseline_mean': float(baseline_values.mean()),
                'diff': float(exp_values.mean() - baseline_values.mean())
            }
            
            if HAS_SCIPY:
                # Statistical test
                if test == 'mannwhitney':
                    stat, pval = stats.mannwhitneyu(exp_values, baseline_values, alternative='two-sided')
                elif test == 'ttest':
                    stat, pval = stats.ttest_ind(exp_values, baseline_values)
                elif test == 'wilcoxon' and len(exp_values) == len(baseline_values):
                    stat, pval = stats.wilcoxon(exp_values, baseline_values)
                else:
                    stat, pval = np.nan, np.nan
                
                row['statistic'] = float(stat)
                row['p_value'] = float(pval)
                row['significant_0.05'] = pval < 0.05
                row['significant_0.01'] = pval < 0.01
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(exp_values) + np.var(baseline_values)) / 2)
                row['effect_size'] = float((exp_values.mean() - baseline_values.mean()) / (pooled_std + 1e-10))
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def to_latex_table(self, metrics: Optional[List[str]] = None,
                       caption: str = 'Experiment Comparison',
                       label: str = 'tab:comparison',
                       highlight_best: bool = True,
                       precision: int = 4) -> str:
        """
        Generate LaTeX table for publication.
        
        Args:
            metrics: Metrics to include
            caption: Table caption
            label: LaTeX label
            highlight_best: Bold the best value in each column
            precision: Decimal precision
            
        Returns:
            LaTeX table string
        """
        df = self.get_metrics_table()
        
        if metrics:
            df = df[metrics]
        
        # Find best values
        best_idx = {}
        if highlight_best:
            for col in df.columns:
                best_idx[col] = df[col].idxmax()
        
        # Build LaTeX
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        col_spec = 'l' + 'c' * len(df.columns)
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append('\\toprule')
        
        # Header
        header = 'Experiment & ' + ' & '.join(df.columns) + ' \\\\'
        lines.append(header)
        lines.append('\\midrule')
        
        # Data rows
        for idx, row in df.iterrows():
            cells = [str(idx)]
            for col in df.columns:
                val = row[col]
                formatted = f'{val:.{precision}f}'
                if highlight_best and best_idx.get(col) == idx:
                    formatted = f'\\textbf{{{formatted}}}'
                cells.append(formatted)
            lines.append(' & '.join(cells) + ' \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex = '\n'.join(lines)
        
        # Save to file
        with open(self.output_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        return latex
    
    def to_markdown_table(self, metrics: Optional[List[str]] = None,
                          precision: int = 4) -> str:
        """
        Generate Markdown table.
        
        Args:
            metrics: Metrics to include
            precision: Decimal precision
            
        Returns:
            Markdown table string
        """
        df = self.get_metrics_table()
        
        if metrics:
            df = df[metrics]
        
        lines = []
        
        # Header
        header = '| Experiment | ' + ' | '.join(df.columns) + ' |'
        separator = '|' + '---|' * (len(df.columns) + 1)
        lines.append(header)
        lines.append(separator)
        
        # Data rows
        for idx, row in df.iterrows():
            cells = [str(idx)]
            for col in df.columns:
                cells.append(f'{row[col]:.{precision}f}')
            lines.append('| ' + ' | '.join(cells) + ' |')
        
        md = '\n'.join(lines)
        
        # Save to file
        with open(self.output_dir / 'comparison_table.md', 'w') as f:
            f.write(md)
        
        return md
    
    def plot_comparison(self, metrics: Optional[List[str]] = None,
                        plot_type: str = 'bar',
                        save: bool = True):
        """
        Plot comparison across experiments.
        
        Args:
            metrics: Metrics to plot
            plot_type: 'bar', 'radar', 'heatmap', or 'box'
            save: Save figure
        """
        import matplotlib.pyplot as plt
        
        df = self.get_metrics_table()
        
        if metrics:
            df = df[metrics]
        
        if df.empty:
            print("No data to plot")
            return
        
        if plot_type == 'bar':
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(df.index))
            width = 0.8 / len(df.columns)
            
            for i, col in enumerate(df.columns):
                ax.bar(x + i * width, df[col], width, label=col)
            
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Experiment Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * (len(df.columns) - 1) / 2)
            ax.set_xticklabels(df.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        elif plot_type == 'radar':
            # Normalize for radar plot
            df_norm = (df - df.min()) / (df.max() - df.min() + 1e-10)
            
            angles = np.linspace(0, 2 * np.pi, len(df.columns), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            for idx in df_norm.index:
                values = df_norm.loc[idx].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=idx)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(df.columns)
            ax.set_title('Experiment Comparison (Normalized)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        elif plot_type == 'heatmap':
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize
            df_norm = (df - df.min()) / (df.max() - df.min() + 1e-10)
            
            im = ax.imshow(df_norm.values, cmap='RdYlGn', aspect='auto')
            
            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(df.index)))
            ax.set_yticklabels(df.index)
            
            plt.colorbar(im, label='Normalized Value')
            ax.set_title('Experiment Comparison Heatmap', fontsize=14, fontweight='bold')
            
            # Add value annotations
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    ax.text(j, i, f'{df.iloc[i, j]:.3f}', ha='center', va='center', fontsize=8)
        
        elif plot_type == 'box':
            # Need full results for box plot
            fig, axes = plt.subplots(1, len(df.columns), figsize=(4 * len(df.columns), 5))
            if len(df.columns) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, df.columns):
                data = []
                labels = []
                
                for name, record in self.experiments.items():
                    if not record.results.empty and metric in record.results.columns:
                        data.append(record.results[metric].dropna().values)
                        labels.append(name)
                
                if data:
                    ax.boxplot(data, labels=labels)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel(metric, fontsize=12)
                    ax.set_title(metric, fontsize=14)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'comparison_{plot_type}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: comparison_{plot_type}.png")
        
        plt.close()
    
    def summary(self) -> Dict:
        """Get summary of comparison."""
        df = self.get_metrics_table()
        
        return {
            'n_experiments': len(self.experiments),
            'experiments': list(self.experiments.keys()),
            'metrics': list(df.columns) if not df.empty else [],
            'best_per_metric': {
                col: df[col].idxmax() for col in df.columns
            } if not df.empty else {}
        }


class BenchmarkSuite:
    """
    Run standardized benchmarks across experiments.
    
    Defines a set of standard tests and metrics for consistent comparison.
    
    Example:
        suite = BenchmarkSuite('consciousness_benchmarks')
        
        suite.add_benchmark('small_world', generate_small_world_data)
        suite.add_benchmark('scale_free', generate_scale_free_data)
        
        results = suite.run_all(compute_metrics_fn)
        suite.generate_report()
    """
    
    def __init__(self, name: str, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite.
        
        Args:
            name: Suite name
            output_dir: Output directory
        """
        self.name = name
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'benchmarks'
        
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks: Dict[str, Callable] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_benchmark(self, name: str, data_generator: Callable,
                      description: str = ''):
        """
        Add a benchmark test.
        
        Args:
            name: Benchmark name
            data_generator: Function that generates test data
            description: Benchmark description
        """
        self.benchmarks[name] = {
            'generator': data_generator,
            'description': description
        }
    
    def run_benchmark(self, benchmark_name: str,
                      metric_fn: Callable,
                      n_runs: int = 10) -> Dict:
        """
        Run a single benchmark.
        
        Args:
            benchmark_name: Name of benchmark to run
            metric_fn: Function to compute metrics
            n_runs: Number of runs for statistics
            
        Returns:
            Benchmark results dict
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        import time
        
        benchmark = self.benchmarks[benchmark_name]
        generator = benchmark['generator']
        
        all_metrics = []
        runtimes = []
        
        for run in range(n_runs):
            data = generator()
            
            start = time.time()
            metrics = metric_fn(data)
            runtime = time.time() - start
            
            all_metrics.append(metrics)
            runtimes.append(runtime)
        
        # Aggregate results
        metric_names = list(all_metrics[0].keys())
        
        results = {
            'benchmark': benchmark_name,
            'n_runs': n_runs,
            'avg_runtime': np.mean(runtimes),
            'metrics': {}
        }
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            results['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        self.results[benchmark_name] = results
        return results
    
    def run_all(self, metric_fn: Callable, n_runs: int = 10,
                show_progress: bool = True) -> Dict[str, Dict]:
        """
        Run all benchmarks.
        
        Args:
            metric_fn: Function to compute metrics
            n_runs: Number of runs per benchmark
            show_progress: Show progress bar
            
        Returns:
            Dict of all results
        """
        from tqdm import tqdm
        
        iterator = tqdm(self.benchmarks.keys(), desc="Benchmarks") if show_progress else self.benchmarks.keys()
        
        for name in iterator:
            self.run_benchmark(name, metric_fn, n_runs)
        
        self._save_results()
        return self.results
    
    def _save_results(self):
        """Save benchmark results."""
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate benchmark report."""
        lines = [
            f"# Benchmark Report: {self.name}",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\n## Summary",
            f"\nTotal benchmarks: {len(self.results)}",
            ""
        ]
        
        for name, result in self.results.items():
            lines.append(f"\n### {name}")
            lines.append(f"- Runs: {result['n_runs']}")
            lines.append(f"- Avg runtime: {result['avg_runtime']:.4f}s")
            lines.append("\n| Metric | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            
            for metric, stats in result['metrics'].items():
                lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
        
        report = '\n'.join(lines)
        
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.write(report)
        
        return report


# Convenience functions
def quick_compare(results_dict: Dict[str, Dict[str, float]],
                  output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Quick comparison of experiment results.
    
    Args:
        results_dict: Dict mapping experiment names to metric dicts
        output_dir: Optional output directory
        
    Returns:
        Comparison DataFrame
        
    Example:
        df = quick_compare({
            'method_a': {'C': 0.85, 'H': 0.92},
            'method_b': {'C': 0.78, 'H': 0.95}
        })
    """
    comparator = ExperimentComparator('quick_compare', output_dir)
    
    for name, metrics in results_dict.items():
        comparator.add_experiment(name, metrics=metrics)
    
    return comparator.compare_all()


def rank_methods(results_dict: Dict[str, Dict[str, float]],
                 primary_metric: str,
                 maximize: bool = True) -> List[Tuple[str, float, int]]:
    """
    Quick ranking of methods by a metric.
    
    Args:
        results_dict: Dict mapping method names to metric dicts
        primary_metric: Metric to rank by
        maximize: Higher is better
        
    Returns:
        List of (method, value, rank) tuples
    """
    comparator = ExperimentComparator('quick_rank')
    
    for name, metrics in results_dict.items():
        comparator.add_experiment(name, metrics=metrics)
    
    return comparator.rank_experiments(primary_metric, maximize)
