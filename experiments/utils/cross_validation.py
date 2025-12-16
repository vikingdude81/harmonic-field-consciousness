#!/usr/bin/env python3
"""
Cross-Validation Utilities

K-fold and other validation strategies for:
- Metric robustness testing
- Model generalization assessment
- Confidence interval estimation
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from itertools import combinations
from tqdm import tqdm
import warnings

# Optional imports
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class CVResult:
    """Container for cross-validation results."""
    fold: int
    train_indices: List[int]
    test_indices: List[int]
    metrics: Dict
    runtime: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CrossValidator:
    """
    Cross-validation framework for consciousness experiments.
    
    Supports:
    - K-fold cross-validation
    - Stratified K-fold (for labeled data)
    - Leave-one-out cross-validation
    - Bootstrap validation
    - Monte Carlo cross-validation
    
    Example:
        def evaluate(train_data, test_data):
            model = fit_model(train_data)
            predictions = model.predict(test_data)
            return {'accuracy': compute_accuracy(predictions, test_data)}
        
        cv = CrossValidator(data, evaluate_fn=evaluate)
        results = cv.k_fold(k=5)
        summary = cv.summary()
    """
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, List],
                 evaluate_fn: Callable,
                 labels: Optional[np.ndarray] = None,
                 name: str = 'cv',
                 output_dir: Optional[Path] = None,
                 seed: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            data: Dataset (array, DataFrame, or list)
            evaluate_fn: Function taking (train_data, test_data) -> metrics dict
            labels: Optional labels for stratified CV
            name: Name for saving results
            output_dir: Output directory
            seed: Random seed
        """
        self.data = data
        self.evaluate_fn = evaluate_fn
        self.labels = labels
        self.name = name
        self.seed = seed
        self.n_samples = len(data)
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'cv_results'
        
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[CVResult] = []
        
        np.random.seed(seed)
    
    def _get_subset(self, indices: np.ndarray) -> Any:
        """Get data subset by indices."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[indices]
        elif isinstance(self.data, np.ndarray):
            return self.data[indices]
        else:
            return [self.data[i] for i in indices]
    
    def _run_fold(self, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> CVResult:
        """Run a single fold evaluation."""
        import time
        
        train_data = self._get_subset(train_idx)
        test_data = self._get_subset(test_idx)
        
        start = time.time()
        try:
            metrics = self.evaluate_fn(train_data, test_data)
            if not isinstance(metrics, dict):
                metrics = {'result': metrics}
        except Exception as e:
            metrics = {'error': str(e)}
        
        runtime = time.time() - start
        
        return CVResult(
            fold=fold,
            train_indices=train_idx.tolist(),
            test_indices=test_idx.tolist(),
            metrics=metrics,
            runtime=runtime
        )
    
    def k_fold(self, k: int = 5, stratified: bool = False, 
               shuffle: bool = True, show_progress: bool = True) -> List[CVResult]:
        """
        Run K-fold cross-validation.
        
        Args:
            k: Number of folds
            stratified: Use stratified folds (requires labels)
            shuffle: Shuffle data before splitting
            show_progress: Show progress bar
            
        Returns:
            List of CVResult objects
        """
        self.results = []
        
        if HAS_SKLEARN:
            if stratified and self.labels is not None:
                splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=self.seed)
                splits = list(splitter.split(np.zeros(self.n_samples), self.labels))
            else:
                splitter = KFold(n_splits=k, shuffle=shuffle, random_state=self.seed)
                splits = list(splitter.split(np.zeros(self.n_samples)))
        else:
            # Manual implementation
            indices = np.arange(self.n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            fold_size = self.n_samples // k
            splits = []
            for i in range(k):
                test_start = i * fold_size
                test_end = test_start + fold_size if i < k - 1 else self.n_samples
                test_idx = indices[test_start:test_end]
                train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
                splits.append((train_idx, test_idx))
        
        iterator = tqdm(enumerate(splits), total=k, desc=f"{k}-Fold CV") if show_progress else enumerate(splits)
        
        for fold, (train_idx, test_idx) in iterator:
            result = self._run_fold(fold, train_idx, test_idx)
            self.results.append(result)
        
        self._save_results()
        return self.results
    
    def leave_one_out(self, show_progress: bool = True) -> List[CVResult]:
        """
        Run Leave-One-Out cross-validation.
        
        Useful for small datasets where every sample matters.
        
        Returns:
            List of CVResult objects
        """
        self.results = []
        
        iterator = tqdm(range(self.n_samples), desc="LOO CV") if show_progress else range(self.n_samples)
        
        for i in iterator:
            test_idx = np.array([i])
            train_idx = np.array([j for j in range(self.n_samples) if j != i])
            
            result = self._run_fold(i, train_idx, test_idx)
            self.results.append(result)
        
        self._save_results()
        return self.results
    
    def bootstrap(self, n_iterations: int = 100, sample_size: Optional[int] = None,
                  show_progress: bool = True) -> List[CVResult]:
        """
        Run bootstrap validation.
        
        Args:
            n_iterations: Number of bootstrap iterations
            sample_size: Size of bootstrap sample (default: same as data)
            show_progress: Show progress bar
            
        Returns:
            List of CVResult objects
        """
        self.results = []
        
        if sample_size is None:
            sample_size = self.n_samples
        
        iterator = tqdm(range(n_iterations), desc="Bootstrap") if show_progress else range(n_iterations)
        
        for i in iterator:
            # Sample with replacement for training
            train_idx = np.random.choice(self.n_samples, size=sample_size, replace=True)
            # Out-of-bag samples for testing
            test_idx = np.array([j for j in range(self.n_samples) if j not in train_idx])
            
            if len(test_idx) == 0:
                # If all samples used, use random 20% for testing
                test_idx = np.random.choice(train_idx, size=max(1, len(train_idx) // 5), replace=False)
            
            result = self._run_fold(i, train_idx, test_idx)
            self.results.append(result)
        
        self._save_results()
        return self.results
    
    def monte_carlo(self, n_iterations: int = 100, test_fraction: float = 0.2,
                    show_progress: bool = True) -> List[CVResult]:
        """
        Run Monte Carlo cross-validation (random splits).
        
        Args:
            n_iterations: Number of random splits
            test_fraction: Fraction of data for testing
            show_progress: Show progress bar
            
        Returns:
            List of CVResult objects
        """
        self.results = []
        
        test_size = int(self.n_samples * test_fraction)
        
        iterator = tqdm(range(n_iterations), desc="Monte Carlo CV") if show_progress else range(n_iterations)
        
        for i in iterator:
            indices = np.random.permutation(self.n_samples)
            test_idx = indices[:test_size]
            train_idx = indices[test_size:]
            
            result = self._run_fold(i, train_idx, test_idx)
            self.results.append(result)
        
        self._save_results()
        return self.results
    
    def _save_results(self):
        """Save results to file."""
        import json
        
        results_file = self.output_dir / 'cv_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'name': self.name,
                'n_samples': self.n_samples,
                'n_folds': len(self.results),
                'results': [r.to_dict() for r in self.results]
            }, f, indent=2, default=str)
        
        # Also save summary as CSV
        df = self.to_dataframe()
        df.to_csv(self.output_dir / 'cv_summary.csv', index=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for result in self.results:
            row = {'fold': result.fold, 'runtime': result.runtime, **result.metrics}
            rows.append(row)
        return pd.DataFrame(rows)
    
    def summary(self) -> Dict:
        """
        Get summary statistics across folds.
        
        Returns:
            Dict with mean, std, CI for each metric
        """
        df = self.to_dataframe()
        
        # Identify metric columns
        metric_cols = [c for c in df.columns if c not in ['fold', 'runtime', 'error']]
        
        summary = {
            'n_folds': len(self.results),
            'n_errors': sum(1 for r in self.results if 'error' in r.metrics),
            'total_runtime': df['runtime'].sum(),
            'metrics': {}
        }
        
        for metric in metric_cols:
            if metric in df.columns and df[metric].dtype in [float, int, np.float64, np.int64]:
                values = df[metric].dropna()
                
                metric_summary = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median())
                }
                
                # Confidence intervals
                if HAS_SCIPY and len(values) > 2:
                    ci_95 = stats.t.interval(0.95, len(values)-1, 
                                             loc=values.mean(), 
                                             scale=stats.sem(values))
                    metric_summary['ci_95_lower'] = float(ci_95[0])
                    metric_summary['ci_95_upper'] = float(ci_95[1])
                
                summary['metrics'][metric] = metric_summary
        
        return summary
    
    def get_confidence_interval(self, metric: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for a metric.
        
        Args:
            metric: Metric name
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        df = self.to_dataframe()
        values = df[metric].dropna()
        
        if not HAS_SCIPY:
            # Simple percentile-based CI
            alpha = 1 - confidence
            return (np.percentile(values, 100 * alpha / 2),
                    np.percentile(values, 100 * (1 - alpha / 2)))
        
        return stats.t.interval(confidence, len(values)-1,
                               loc=values.mean(),
                               scale=stats.sem(values))
    
    def plot_results(self, metric: str, save: bool = True):
        """
        Plot cross-validation results.
        
        Args:
            metric: Metric to plot
            save: Save figure
        """
        import matplotlib.pyplot as plt
        
        df = self.to_dataframe()
        values = df[metric].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        ax = axes[0]
        ax.boxplot(values, labels=[metric])
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Distribution Across Folds', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Bar plot with error
        ax = axes[1]
        summary = self.summary()['metrics'][metric]
        ax.bar([metric], [summary['mean']], yerr=[summary['std']], 
               capsize=5, color='steelblue', edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}: {summary["mean"]:.4f} ± {summary["std"]:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'cv_{metric}.png', dpi=150)
            print(f"Saved: cv_{metric}.png")
        
        plt.close()


class MetricRobustnessTester:
    """
    Test robustness of consciousness metrics across perturbations.
    
    Assesses:
    - Noise sensitivity
    - Sample size stability
    - Parameter sensitivity
    - Temporal stability
    
    Example:
        tester = MetricRobustnessTester(compute_metrics_fn)
        results = tester.test_noise_sensitivity(data, noise_levels=[0.01, 0.05, 0.1])
        tester.plot_robustness()
    """
    
    def __init__(self, metric_fn: Callable,
                 metric_names: Optional[List[str]] = None,
                 name: str = 'robustness',
                 output_dir: Optional[Path] = None,
                 seed: int = 42):
        """
        Initialize robustness tester.
        
        Args:
            metric_fn: Function that computes metrics from data
            metric_names: Names of metrics returned by metric_fn
            name: Name for saving results
            output_dir: Output directory
            seed: Random seed
        """
        self.metric_fn = metric_fn
        self.metric_names = metric_names
        self.name = name
        self.seed = seed
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'robustness'
        
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, pd.DataFrame] = {}
        
        np.random.seed(seed)
    
    def test_noise_sensitivity(self, data: np.ndarray, 
                               noise_levels: List[float] = None,
                               n_trials: int = 20,
                               noise_type: str = 'gaussian') -> pd.DataFrame:
        """
        Test metric sensitivity to noise.
        
        Args:
            data: Input data array
            noise_levels: List of noise standard deviations
            n_trials: Number of trials per noise level
            noise_type: 'gaussian', 'uniform', or 'salt_pepper'
            
        Returns:
            DataFrame with results
        """
        if noise_levels is None:
            noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        
        results = []
        
        for noise_level in tqdm(noise_levels, desc="Noise sensitivity"):
            for trial in range(n_trials):
                # Add noise
                if noise_level == 0:
                    noisy_data = data.copy()
                elif noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_level, data.shape)
                    noisy_data = data + noise
                elif noise_type == 'uniform':
                    noise = np.random.uniform(-noise_level, noise_level, data.shape)
                    noisy_data = data + noise
                else:  # salt_pepper
                    noisy_data = data.copy()
                    mask = np.random.random(data.shape) < noise_level
                    noisy_data[mask] = np.random.choice([data.min(), data.max()], mask.sum())
                
                # Compute metrics
                metrics = self.metric_fn(noisy_data)
                
                results.append({
                    'noise_level': noise_level,
                    'trial': trial,
                    'noise_type': noise_type,
                    **metrics
                })
        
        df = pd.DataFrame(results)
        self.results['noise_sensitivity'] = df
        df.to_csv(self.output_dir / 'noise_sensitivity.csv', index=False)
        
        return df
    
    def test_sample_stability(self, data: np.ndarray,
                              sample_fractions: List[float] = None,
                              n_trials: int = 20) -> pd.DataFrame:
        """
        Test metric stability with different sample sizes.
        
        Args:
            data: Input data array
            sample_fractions: Fractions of data to use
            n_trials: Number of trials per fraction
            
        Returns:
            DataFrame with results
        """
        if sample_fractions is None:
            sample_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        n_samples = len(data)
        results = []
        
        for fraction in tqdm(sample_fractions, desc="Sample stability"):
            sample_size = max(2, int(n_samples * fraction))
            
            for trial in range(n_trials):
                if fraction < 1.0:
                    indices = np.random.choice(n_samples, size=sample_size, replace=False)
                    sample_data = data[indices]
                else:
                    sample_data = data
                
                metrics = self.metric_fn(sample_data)
                
                results.append({
                    'fraction': fraction,
                    'sample_size': sample_size,
                    'trial': trial,
                    **metrics
                })
        
        df = pd.DataFrame(results)
        self.results['sample_stability'] = df
        df.to_csv(self.output_dir / 'sample_stability.csv', index=False)
        
        return df
    
    def test_temporal_stability(self, data_sequence: List[np.ndarray],
                                window_sizes: List[int] = None) -> pd.DataFrame:
        """
        Test metric stability over time windows.
        
        Args:
            data_sequence: List of data arrays over time
            window_sizes: Sizes of sliding windows
            
        Returns:
            DataFrame with results
        """
        if window_sizes is None:
            window_sizes = [1, 5, 10, 20]
        
        n_timepoints = len(data_sequence)
        results = []
        
        for window_size in tqdm(window_sizes, desc="Temporal stability"):
            for t in range(n_timepoints - window_size + 1):
                window_data = data_sequence[t:t + window_size]
                
                # Aggregate window data (e.g., mean)
                aggregated = np.mean(window_data, axis=0)
                metrics = self.metric_fn(aggregated)
                
                results.append({
                    'window_size': window_size,
                    'time_start': t,
                    'time_end': t + window_size,
                    **metrics
                })
        
        df = pd.DataFrame(results)
        self.results['temporal_stability'] = df
        df.to_csv(self.output_dir / 'temporal_stability.csv', index=False)
        
        return df
    
    def compute_robustness_scores(self) -> Dict:
        """
        Compute overall robustness scores for each metric.
        
        Returns:
            Dict with robustness scores (0-1, higher = more robust)
        """
        scores = {}
        
        if 'noise_sensitivity' in self.results:
            df = self.results['noise_sensitivity']
            
            # Get metric columns
            metric_cols = [c for c in df.columns 
                          if c not in ['noise_level', 'trial', 'noise_type']]
            
            for metric in metric_cols:
                # Compute coefficient of variation at different noise levels
                grouped = df.groupby('noise_level')[metric].agg(['mean', 'std'])
                
                # Robustness = 1 - normalized change from baseline
                baseline = grouped.loc[0, 'mean'] if 0 in grouped.index else grouped['mean'].iloc[0]
                max_deviation = (grouped['mean'] - baseline).abs().max()
                
                noise_robustness = 1 - min(1, max_deviation / (abs(baseline) + 1e-10))
                
                if metric not in scores:
                    scores[metric] = {}
                scores[metric]['noise_robustness'] = noise_robustness
        
        if 'sample_stability' in self.results:
            df = self.results['sample_stability']
            
            metric_cols = [c for c in df.columns 
                          if c not in ['fraction', 'sample_size', 'trial']]
            
            for metric in metric_cols:
                # Robustness = how stable is metric across sample sizes
                grouped = df.groupby('fraction')[metric].agg(['mean', 'std'])
                cv = (grouped['std'] / (grouped['mean'].abs() + 1e-10)).mean()
                
                sample_robustness = 1 - min(1, cv)
                
                if metric not in scores:
                    scores[metric] = {}
                scores[metric]['sample_robustness'] = sample_robustness
        
        # Compute overall score
        for metric in scores:
            individual_scores = list(scores[metric].values())
            scores[metric]['overall'] = np.mean(individual_scores)
        
        return scores
    
    def plot_robustness(self, save: bool = True):
        """Plot all robustness test results."""
        import matplotlib.pyplot as plt
        
        n_plots = len(self.results)
        if n_plots == 0:
            print("No results to plot. Run tests first.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        if 'noise_sensitivity' in self.results:
            ax = axes[plot_idx]
            df = self.results['noise_sensitivity']
            
            metric_cols = [c for c in df.columns 
                          if c not in ['noise_level', 'trial', 'noise_type']]
            
            for metric in metric_cols[:4]:  # Limit to 4 metrics
                grouped = df.groupby('noise_level')[metric].agg(['mean', 'std'])
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           marker='o', capsize=3, label=metric)
            
            ax.set_xlabel('Noise Level', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_title('Noise Sensitivity', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        if 'sample_stability' in self.results:
            ax = axes[plot_idx]
            df = self.results['sample_stability']
            
            metric_cols = [c for c in df.columns 
                          if c not in ['fraction', 'sample_size', 'trial']]
            
            for metric in metric_cols[:4]:
                grouped = df.groupby('fraction')[metric].agg(['mean', 'std'])
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           marker='o', capsize=3, label=metric)
            
            ax.set_xlabel('Sample Fraction', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_title('Sample Stability', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'robustness_plots.png', dpi=150)
            print(f"Saved: robustness_plots.png")
        
        plt.close()


# Statistical significance testing
def paired_significance_test(values1: np.ndarray, values2: np.ndarray,
                             test: str = 'wilcoxon') -> Dict:
    """
    Test statistical significance between paired samples.
    
    Args:
        values1: First set of values
        values2: Second set of values
        test: 'wilcoxon', 'ttest', or 'sign'
        
    Returns:
        Dict with statistic and p-value
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not available'}
    
    if test == 'wilcoxon':
        stat, pval = stats.wilcoxon(values1, values2)
    elif test == 'ttest':
        stat, pval = stats.ttest_rel(values1, values2)
    elif test == 'sign':
        # Sign test
        diff = values1 - values2
        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        stat = min(n_pos, n_neg)
        n = n_pos + n_neg
        pval = 2 * stats.binom.cdf(stat, n, 0.5)
    else:
        return {'error': f'Unknown test: {test}'}
    
    return {
        'test': test,
        'statistic': float(stat),
        'p_value': float(pval),
        'significant_0.05': pval < 0.05,
        'significant_0.01': pval < 0.01
    }


def effect_size(values1: np.ndarray, values2: np.ndarray, 
                method: str = 'cohens_d') -> float:
    """
    Compute effect size between two groups.
    
    Args:
        values1: First group values
        values2: Second group values
        method: 'cohens_d' or 'hedges_g'
        
    Returns:
        Effect size value
    """
    mean_diff = np.mean(values1) - np.mean(values2)
    
    if method == 'cohens_d':
        pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
        return mean_diff / (pooled_std + 1e-10)
    
    elif method == 'hedges_g':
        n1, n2 = len(values1), len(values2)
        pooled_std = np.sqrt(((n1-1)*np.var(values1) + (n2-1)*np.var(values2)) / (n1+n2-2))
        d = mean_diff / (pooled_std + 1e-10)
        # Hedges correction
        correction = 1 - 3 / (4*(n1+n2) - 9)
        return d * correction
    
    return 0.0
