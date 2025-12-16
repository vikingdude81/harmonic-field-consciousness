#!/usr/bin/env python3
"""
Parameter Sweep Utilities

Systematic hyperparameter exploration with:
- Grid search
- Random search
- Latin hypercube sampling
- Bayesian optimization (optional)
- Parallel execution support
- Result tracking and visualization
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from itertools import product
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Optional imports
try:
    from scipy.stats import qmc
    HAS_SCIPY_QMC = True
except ImportError:
    HAS_SCIPY_QMC = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPTIM = True
except ImportError:
    HAS_SCIPY_OPTIM = False


@dataclass
class ParameterSpace:
    """
    Define a parameter search space.
    
    Example:
        space = ParameterSpace()
        space.add_continuous('learning_rate', 1e-4, 1e-1, log_scale=True)
        space.add_discrete('hidden_size', [64, 128, 256, 512])
        space.add_integer('n_layers', 1, 5)
        space.add_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    """
    params: Dict = field(default_factory=dict)
    
    def add_continuous(self, name: str, low: float, high: float, 
                       log_scale: bool = False):
        """Add continuous parameter."""
        self.params[name] = {
            'type': 'continuous',
            'low': low,
            'high': high,
            'log_scale': log_scale
        }
        return self
    
    def add_discrete(self, name: str, values: List):
        """Add discrete parameter with specific values."""
        self.params[name] = {
            'type': 'discrete',
            'values': values
        }
        return self
    
    def add_integer(self, name: str, low: int, high: int):
        """Add integer parameter within range."""
        self.params[name] = {
            'type': 'integer',
            'low': low,
            'high': high
        }
        return self
    
    def add_categorical(self, name: str, categories: List[str]):
        """Add categorical parameter."""
        self.params[name] = {
            'type': 'categorical',
            'categories': categories
        }
        return self
    
    def sample_random(self, n_samples: int = 1, seed: Optional[int] = None) -> List[Dict]:
        """Sample random parameter combinations."""
        if seed is not None:
            np.random.seed(seed)
        
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, spec in self.params.items():
                if spec['type'] == 'continuous':
                    if spec['log_scale']:
                        log_val = np.random.uniform(np.log(spec['low']), np.log(spec['high']))
                        sample[name] = np.exp(log_val)
                    else:
                        sample[name] = np.random.uniform(spec['low'], spec['high'])
                elif spec['type'] == 'discrete':
                    sample[name] = np.random.choice(spec['values'])
                elif spec['type'] == 'integer':
                    sample[name] = np.random.randint(spec['low'], spec['high'] + 1)
                elif spec['type'] == 'categorical':
                    sample[name] = np.random.choice(spec['categories'])
            samples.append(sample)
        
        return samples
    
    def sample_grid(self, n_points_per_param: Union[int, Dict[str, int]] = 5) -> List[Dict]:
        """Generate grid of parameter combinations."""
        param_values = {}
        
        for name, spec in self.params.items():
            if isinstance(n_points_per_param, dict):
                n_points = n_points_per_param.get(name, 5)
            else:
                n_points = n_points_per_param
            
            if spec['type'] == 'continuous':
                if spec['log_scale']:
                    param_values[name] = np.exp(np.linspace(
                        np.log(spec['low']), np.log(spec['high']), n_points
                    ))
                else:
                    param_values[name] = np.linspace(spec['low'], spec['high'], n_points)
            elif spec['type'] == 'discrete':
                param_values[name] = spec['values']
            elif spec['type'] == 'integer':
                param_values[name] = list(range(spec['low'], spec['high'] + 1))
            elif spec['type'] == 'categorical':
                param_values[name] = spec['categories']
        
        # Generate all combinations
        keys = list(param_values.keys())
        combinations = list(product(*[param_values[k] for k in keys]))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def sample_latin_hypercube(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """Sample using Latin Hypercube Sampling for better coverage."""
        if not HAS_SCIPY_QMC:
            warnings.warn("scipy.stats.qmc not available, falling back to random sampling")
            return self.sample_random(n_samples, seed)
        
        # Get continuous/integer params for LHS
        continuous_params = []
        for name, spec in self.params.items():
            if spec['type'] in ['continuous', 'integer']:
                continuous_params.append(name)
        
        n_dims = len(continuous_params)
        
        if n_dims == 0:
            return self.sample_random(n_samples, seed)
        
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        lhs_samples = sampler.random(n=n_samples)
        
        samples = []
        for i in range(n_samples):
            sample = {}
            
            # Map LHS values to parameter ranges
            for j, name in enumerate(continuous_params):
                spec = self.params[name]
                unit_val = lhs_samples[i, j]
                
                if spec['type'] == 'continuous':
                    if spec['log_scale']:
                        log_range = np.log(spec['high']) - np.log(spec['low'])
                        sample[name] = np.exp(np.log(spec['low']) + unit_val * log_range)
                    else:
                        sample[name] = spec['low'] + unit_val * (spec['high'] - spec['low'])
                elif spec['type'] == 'integer':
                    sample[name] = int(spec['low'] + unit_val * (spec['high'] - spec['low'] + 1))
                    sample[name] = min(sample[name], spec['high'])
            
            # Random sample for discrete/categorical
            for name, spec in self.params.items():
                if name not in sample:
                    if spec['type'] == 'discrete':
                        sample[name] = np.random.choice(spec['values'])
                    elif spec['type'] == 'categorical':
                        sample[name] = np.random.choice(spec['categories'])
            
            samples.append(sample)
        
        return samples


@dataclass
class SweepResult:
    """Container for parameter sweep results."""
    params: Dict
    metrics: Dict
    runtime: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ParameterSweep:
    """
    Run parameter sweeps with multiple search strategies.
    
    Example:
        # Define parameter space
        space = ParameterSpace()
        space.add_continuous('coupling', 0.1, 5.0)
        space.add_discrete('n_nodes', [50, 100, 200])
        
        # Define objective function
        def run_experiment(coupling, n_nodes):
            result = compute_consciousness(coupling, n_nodes)
            return {'C': result['C'], 'H_mode': result['H_mode']}
        
        # Run sweep
        sweep = ParameterSweep(space, run_experiment, 'my_sweep')
        results = sweep.run_grid(n_points=5)
        
        # Analyze
        best = sweep.get_best('C', maximize=True)
        sweep.plot_results()
    """
    
    def __init__(self, parameter_space: ParameterSpace, 
                 objective_fn: Callable,
                 name: str = 'sweep',
                 output_dir: Optional[Path] = None,
                 n_jobs: int = 1,
                 use_cache: bool = True):
        """
        Initialize parameter sweep.
        
        Args:
            parameter_space: ParameterSpace defining search space
            objective_fn: Function that takes params and returns metrics dict
            name: Name of the sweep (for saving results)
            output_dir: Directory for saving results
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            use_cache: Whether to cache results
        """
        self.space = parameter_space
        self.objective_fn = objective_fn
        self.name = name
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.use_cache = use_cache
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'sweeps'
        
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[SweepResult] = []
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load any existing results from previous runs."""
        results_file = self.output_dir / 'results.json'
        if results_file.exists() and self.use_cache:
            with open(results_file, 'r') as f:
                data = json.load(f)
                self.results = [SweepResult(**r) for r in data.get('results', [])]
    
    def _save_results(self):
        """Save results to file."""
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'name': self.name,
                'space': self.space.params,
                'results': [r.to_dict() for r in self.results]
            }, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        if self.results:
            df = self.to_dataframe()
            df.to_csv(self.output_dir / 'results.csv', index=False)
    
    def _run_single(self, params: Dict) -> SweepResult:
        """Run objective function with given parameters."""
        import time
        start = time.time()
        
        try:
            metrics = self.objective_fn(**params)
            if not isinstance(metrics, dict):
                metrics = {'result': metrics}
        except Exception as e:
            metrics = {'error': str(e)}
        
        runtime = time.time() - start
        
        return SweepResult(
            params=params,
            metrics=metrics,
            runtime=runtime
        )
    
    def _params_hash(self, params: Dict) -> str:
        """Create hash of parameters for caching."""
        import hashlib
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _was_computed(self, params: Dict) -> bool:
        """Check if parameters were already computed."""
        if not self.use_cache:
            return False
        
        param_hash = self._params_hash(params)
        for result in self.results:
            if self._params_hash(result.params) == param_hash:
                return True
        return False
    
    def run_grid(self, n_points: Union[int, Dict[str, int]] = 5,
                 show_progress: bool = True) -> List[SweepResult]:
        """
        Run grid search over parameter space.
        
        Args:
            n_points: Number of points per parameter (or dict per param)
            show_progress: Show progress bar
            
        Returns:
            List of SweepResult objects
        """
        param_configs = self.space.sample_grid(n_points)
        return self._run_configs(param_configs, "Grid Search", show_progress)
    
    def run_random(self, n_samples: int = 50, seed: Optional[int] = None,
                   show_progress: bool = True) -> List[SweepResult]:
        """
        Run random search over parameter space.
        
        Args:
            n_samples: Number of random samples
            seed: Random seed
            show_progress: Show progress bar
            
        Returns:
            List of SweepResult objects
        """
        param_configs = self.space.sample_random(n_samples, seed)
        return self._run_configs(param_configs, "Random Search", show_progress)
    
    def run_lhs(self, n_samples: int = 50, seed: Optional[int] = None,
                show_progress: bool = True) -> List[SweepResult]:
        """
        Run Latin Hypercube Sampling search.
        
        Args:
            n_samples: Number of samples
            seed: Random seed
            show_progress: Show progress bar
            
        Returns:
            List of SweepResult objects
        """
        param_configs = self.space.sample_latin_hypercube(n_samples, seed)
        return self._run_configs(param_configs, "LHS Search", show_progress)
    
    def _run_configs(self, param_configs: List[Dict], 
                     desc: str, show_progress: bool) -> List[SweepResult]:
        """Run a list of parameter configurations."""
        # Filter already computed
        to_compute = [p for p in param_configs if not self._was_computed(p)]
        
        if not to_compute:
            print(f"All {len(param_configs)} configurations already cached")
            return self.results
        
        print(f"Running {len(to_compute)} configurations ({len(param_configs) - len(to_compute)} cached)")
        
        new_results = []
        
        if self.n_jobs == 1:
            # Sequential execution
            iterator = tqdm(to_compute, desc=desc) if show_progress else to_compute
            for params in iterator:
                result = self._run_single(params)
                new_results.append(result)
                self.results.append(result)
                self._save_results()
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(self._run_single, p): p for p in to_compute}
                
                iterator = tqdm(as_completed(futures), total=len(futures), desc=desc) if show_progress else as_completed(futures)
                
                for future in iterator:
                    result = future.result()
                    new_results.append(result)
                    self.results.append(result)
                    self._save_results()
        
        return self.results
    
    def get_best(self, metric: str, maximize: bool = True) -> Optional[SweepResult]:
        """Get best result for a given metric."""
        valid_results = [r for r in self.results if metric in r.metrics and 'error' not in r.metrics]
        
        if not valid_results:
            return None
        
        if maximize:
            return max(valid_results, key=lambda r: r.metrics[metric])
        else:
            return min(valid_results, key=lambda r: r.metrics[metric])
    
    def get_top_k(self, metric: str, k: int = 10, maximize: bool = True) -> List[SweepResult]:
        """Get top-k results for a metric."""
        valid_results = [r for r in self.results if metric in r.metrics and 'error' not in r.metrics]
        sorted_results = sorted(valid_results, key=lambda r: r.metrics[metric], reverse=maximize)
        return sorted_results[:k]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        for result in self.results:
            row = {**result.params, **result.metrics, 'runtime': result.runtime}
            rows.append(row)
        return pd.DataFrame(rows)
    
    def plot_results(self, metric: str, 
                     param1: Optional[str] = None, 
                     param2: Optional[str] = None,
                     save: bool = True):
        """
        Plot sweep results.
        
        Args:
            metric: Metric to plot
            param1: First parameter for x-axis
            param2: Second parameter for color/hue
            save: Whether to save figure
        """
        import matplotlib.pyplot as plt
        
        df = self.to_dataframe()
        
        if param1 is None:
            # Plot metric distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[metric].dropna(), bins=30, edgecolor='black')
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{metric} Distribution', fontsize=14)
        
        elif param2 is None:
            # 2D scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[param1], df[metric], alpha=0.6)
            ax.set_xlabel(param1, fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} vs {param1}', fontsize=14)
        
        else:
            # 3D-style plot with color
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df[param1], df[metric], c=df[param2], cmap='viridis', alpha=0.6)
            ax.set_xlabel(param1, fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} vs {param1} (colored by {param2})', fontsize=14)
            plt.colorbar(scatter, label=param2)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'sweep_{metric}.png', dpi=150)
            print(f"Saved: sweep_{metric}.png")
        
        plt.close()
    
    def plot_heatmap(self, param1: str, param2: str, metric: str, save: bool = True):
        """Plot heatmap of metric across two parameters."""
        import matplotlib.pyplot as plt
        
        df = self.to_dataframe()
        
        # Pivot for heatmap
        pivot = df.pivot_table(values=metric, index=param2, columns=param1, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.3g}' for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.3g}' for y in pivot.index])
        
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_title(f'{metric} Heatmap', fontsize=14)
        
        plt.colorbar(im, label=metric)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'heatmap_{metric}_{param1}_{param2}.png', dpi=150)
            print(f"Saved: heatmap_{metric}_{param1}_{param2}.png")
        
        plt.close()
    
    def summary(self) -> Dict:
        """Get summary statistics of sweep."""
        df = self.to_dataframe()
        
        # Identify metric columns
        metric_cols = [c for c in df.columns if c not in self.space.params and c != 'runtime']
        
        summary = {
            'n_runs': len(self.results),
            'n_errors': sum(1 for r in self.results if 'error' in r.metrics),
            'total_runtime': df['runtime'].sum(),
            'metrics': {}
        }
        
        for metric in metric_cols:
            if metric in df.columns and df[metric].dtype in [float, int, np.float64, np.int64]:
                summary['metrics'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max())
                }
        
        return summary


# Convenience functions
def quick_sweep(objective_fn: Callable, 
                param_ranges: Dict[str, Tuple],
                n_samples: int = 50,
                method: str = 'random') -> pd.DataFrame:
    """
    Quick parameter sweep with minimal setup.
    
    Args:
        objective_fn: Function taking params as kwargs, returning metrics dict
        param_ranges: Dict mapping param names to (low, high) tuples
        n_samples: Number of samples
        method: 'random', 'grid', or 'lhs'
        
    Returns:
        DataFrame with results
        
    Example:
        def run(x, y):
            return {'result': x**2 + y**2}
        
        df = quick_sweep(run, {'x': (0, 1), 'y': (0, 1)}, n_samples=100)
    """
    space = ParameterSpace()
    for name, (low, high) in param_ranges.items():
        if isinstance(low, int) and isinstance(high, int):
            space.add_integer(name, low, high)
        else:
            space.add_continuous(name, low, high)
    
    sweep = ParameterSweep(space, objective_fn, name='quick_sweep')
    
    if method == 'grid':
        sweep.run_grid(n_points=int(np.sqrt(n_samples)))
    elif method == 'lhs':
        sweep.run_lhs(n_samples=n_samples)
    else:
        sweep.run_random(n_samples=n_samples)
    
    return sweep.to_dataframe()
