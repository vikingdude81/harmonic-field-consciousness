# Experimental Framework Utilities

This package provides core utilities for the Harmonic Field Consciousness experimental framework.

## Table of Contents

- [Core Utilities](#core-utilities)
  - [Graph Generators](#graph-generators)
  - [Metrics](#metrics)
  - [State Generators](#state-generators)
  - [Visualization](#visualization)
- [Advanced Utilities](#advanced-utilities)
  - [GPU Acceleration](#gpu-acceleration)
  - [Result Caching](#result-caching)
  - [Parameter Sweeps](#parameter-sweeps)
  - [Cross-Validation](#cross-validation)
  - [Comparative Analysis](#comparative-analysis)

---

## Core Utilities

### Graph Generators (`graph_generators.py`)

Generate network topologies for consciousness simulations.

```python
from utils import generate_small_world, generate_scale_free, generate_modular

# Small-world network (Watts-Strogatz)
G = generate_small_world(n_nodes=100, k=6, p=0.3)

# Scale-free network (Barabási-Albert)
G = generate_scale_free(n_nodes=100, m=3)

# Modular network with communities
G = generate_modular(n_nodes=100, n_modules=4, p_in=0.3, p_out=0.02)
```

**Available functions:**
- `generate_small_world(n_nodes, k, p)` - Watts-Strogatz small-world
- `generate_scale_free(n_nodes, m)` - Barabási-Albert scale-free
- `generate_random(n_nodes, p)` - Erdős-Rényi random graph
- `generate_lattice(n_nodes, dim)` - Regular lattice
- `generate_modular(n_nodes, n_modules, p_in, p_out)` - Modular community structure
- `generate_hub_network(n_nodes, n_hubs)` - Hub-and-spoke topology

### Metrics (`metrics.py`)

Compute consciousness-related metrics from network states.

```python
from utils import compute_all_metrics, compute_consciousness_functional

# Compute all metrics from modal power distribution
metrics = compute_all_metrics(power_spectrum, eigenvalues)
print(f"Consciousness: {metrics['C']:.4f}")
print(f"Mode Entropy: {metrics['H_mode']:.4f}")
print(f"Participation Ratio: {metrics['PR']:.4f}")

# Or compute individual metrics
C = compute_consciousness_functional(power, eigenvalues)
```

**Available functions:**
- `compute_mode_entropy(power)` - Shannon entropy of mode distribution
- `compute_participation_ratio(power)` - Inverse participation ratio
- `compute_phase_coherence(phases)` - Phase synchronization measure
- `compute_entropy_production(power)` - Non-equilibrium entropy rate
- `compute_criticality_index(power, eigenvalues)` - Distance from criticality
- `compute_consciousness_functional(power, eigenvalues)` - Main C(t) metric
- `compute_lempel_ziv_complexity(signal)` - Algorithmic complexity
- `compute_multiscale_entropy(signal, scales)` - Multiscale sample entropy

### State Generators (`state_generators.py`)

Generate brain state power distributions.

```python
from utils import generate_wake_state, generate_anesthesia_state, interpolate_states

# Generate different brain states
wake = generate_wake_state(n_modes=50)
anesthesia = generate_anesthesia_state(n_modes=50, depth=0.8)

# Interpolate between states
transition = interpolate_states(wake, anesthesia, alpha=0.5)
```

**Available functions:**
- `generate_wake_state(n_modes)` - Normal waking consciousness
- `generate_nrem_unconscious(n_modes)` - Deep sleep (low consciousness)
- `generate_nrem_dreaming(n_modes)` - REM-like dreaming state
- `generate_anesthesia_state(n_modes, depth)` - Anesthetic unconsciousness
- `generate_psychedelic_state(n_modes, intensity)` - Expanded consciousness
- `interpolate_states(state1, state2, alpha)` - Blend between states
- `add_perturbation(state, strength)` - Add noise/perturbation

### Visualization (`visualization.py`)

Publication-quality plotting functions.

```python
from utils import plot_consciousness_radar, plot_mode_distribution

# Radar plot comparing multiple conditions
plot_consciousness_radar(metrics_dict, save_path='radar.png')

# Mode power distribution
plot_mode_distribution(power, eigenvalues, save_path='modes.png')
```

**Available functions:**
- `plot_network(G, layout)` - Network topology visualization
- `plot_mode_distribution(power, eigenvalues)` - Spectral power plot
- `plot_consciousness_radar(metrics_dict)` - Multi-metric radar chart
- `plot_time_series(data, labels)` - Temporal evolution plots
- `plot_phase_space(x, y, z)` - 3D state space trajectory
- `plot_heatmap(matrix, labels)` - Correlation/similarity matrices

---

## Advanced Utilities

### GPU Acceleration (`gpu_utils.py`)

CUDA/GPU support for computationally intensive operations.

```python
from utils import get_device_info, gpu_eigendecomposition, GPUAccelerator

# Check GPU availability
info = get_device_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"CuPy available: {info['cupy_available']}")

# GPU-accelerated eigendecomposition
eigenvalues, eigenvectors = gpu_eigendecomposition(laplacian_matrix, use_gpu=True)

# Context manager for GPU operations
with GPUAccelerator() as gpu:
    xp = gpu.xp  # numpy or cupy depending on availability
    result = xp.linalg.eigh(matrix)
```

**Available functions:**
- `get_device_info()` - Get GPU/CUDA information
- `get_array_module(use_gpu)` - Get numpy or cupy module
- `gpu_eigendecomposition(matrix, use_gpu)` - GPU-accelerated eigh
- `batch_compute_metrics_gpu(data_list)` - Batch GPU metric computation
- `print_gpu_status()` - Print GPU availability info
- `GPUAccelerator` - Context manager for GPU operations

**Installation:**
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# For PyTorch GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Result Caching (`caching.py`)

Save and resume experiment results.

```python
from utils import ExperimentCache, CheckpointManager, quick_save, quick_load

# Full cache manager
cache = ExperimentCache('my_experiment', version='1.0')

# Save results (supports dict, array, DataFrame)
cache.save('results', {'C': 0.85, 'data': np.array([1,2,3])})
cache.save('dataframe', df, format='csv')
cache.save('large_data', big_array, format='hdf5')

# Load results
results = cache.load('results')
df = cache.load('dataframe')

# Check if cached
if cache.exists('expensive_computation'):
    data = cache.load('expensive_computation')
else:
    data = run_expensive_computation()
    cache.save('expensive_computation', data)

# Decorator for automatic caching
@cache.cached()
def expensive_function(x, y):
    # ... expensive computation ...
    return result

# Checkpoint manager for resumable experiments
ckpt = CheckpointManager('long_experiment')

for i in range(1000):
    if ckpt.should_skip(i):
        continue  # Already computed
    
    result = run_step(i)
    ckpt.save_checkpoint(i, result)

# Get all completed results
all_results = ckpt.get_all_results()

# Quick save/load convenience functions
quick_save(data, 'my_data', experiment='quick')
data = quick_load('my_data', experiment='quick')
```

**Classes:**
- `ExperimentCache` - Full-featured cache with metadata, versioning
- `CheckpointManager` - Resumable experiment checkpoints

**Supported formats:** pickle, JSON, HDF5, CSV, NumPy

### Parameter Sweeps (`parameter_sweep.py`)

Systematic hyperparameter exploration.

```python
from utils import ParameterSpace, ParameterSweep, quick_sweep

# Define parameter space
space = ParameterSpace()
space.add_continuous('coupling', 0.1, 5.0, log_scale=False)
space.add_discrete('n_nodes', [50, 100, 200, 500])
space.add_integer('n_modules', 2, 8)
space.add_categorical('topology', ['small_world', 'scale_free', 'modular'])

# Define objective function (returns metrics dict)
def run_experiment(coupling, n_nodes, n_modules, topology):
    G = generate_network(topology, n_nodes, n_modules)
    metrics = compute_metrics(G, coupling)
    return {'C': metrics['C'], 'H_mode': metrics['H_mode']}

# Create sweep
sweep = ParameterSweep(space, run_experiment, name='coupling_study')

# Run with different strategies
results = sweep.run_grid(n_points=5)           # Grid search
results = sweep.run_random(n_samples=100)      # Random search
results = sweep.run_lhs(n_samples=50)          # Latin Hypercube

# Analyze results
best = sweep.get_best('C', maximize=True)
print(f"Best C: {best.metrics['C']:.4f} with params: {best.params}")

top_10 = sweep.get_top_k('C', k=10)
df = sweep.to_dataframe()

# Visualize
sweep.plot_results('C', param1='coupling')
sweep.plot_heatmap('coupling', 'n_nodes', 'C')

# Quick one-liner sweep
df = quick_sweep(
    run_experiment,
    {'x': (0, 1), 'y': (0, 10)},
    n_samples=100,
    method='random'
)
```

**Classes:**
- `ParameterSpace` - Define search spaces
- `ParameterSweep` - Run and analyze sweeps
- `SweepResult` - Container for individual results

**Search methods:** Grid, Random, Latin Hypercube Sampling

### Cross-Validation (`cross_validation.py`)

Validate metric robustness and statistical significance.

```python
from utils import CrossValidator, MetricRobustnessTester, paired_significance_test

# Cross-validation
def evaluate(train_data, test_data):
    model = fit_model(train_data)
    return {'accuracy': model.score(test_data)}

cv = CrossValidator(data, evaluate_fn=evaluate)

# Different CV strategies
results = cv.k_fold(k=5)                    # Standard K-fold
results = cv.k_fold(k=5, stratified=True)   # Stratified (needs labels)
results = cv.leave_one_out()                # LOO for small datasets
results = cv.bootstrap(n_iterations=100)   # Bootstrap validation
results = cv.monte_carlo(n_iterations=50)  # Random splits

# Get summary with confidence intervals
summary = cv.summary()
print(f"Accuracy: {summary['metrics']['accuracy']['mean']:.3f} ± {summary['metrics']['accuracy']['std']:.3f}")
print(f"95% CI: [{summary['metrics']['accuracy']['ci_95_lower']:.3f}, {summary['metrics']['accuracy']['ci_95_upper']:.3f}]")

# Robustness testing
def compute_metrics(data):
    return {'C': calculate_C(data), 'H': calculate_H(data)}

tester = MetricRobustnessTester(compute_metrics)

# Test noise sensitivity
df_noise = tester.test_noise_sensitivity(data, noise_levels=[0.01, 0.05, 0.1])

# Test sample size stability
df_sample = tester.test_sample_stability(data, sample_fractions=[0.1, 0.5, 0.9])

# Get robustness scores (0-1, higher = more robust)
scores = tester.compute_robustness_scores()
print(f"C noise robustness: {scores['C']['noise_robustness']:.3f}")

tester.plot_robustness()

# Statistical significance testing
from utils import paired_significance_test, effect_size

result = paired_significance_test(values1, values2, test='wilcoxon')
print(f"p-value: {result['p_value']:.4f}, significant: {result['significant_0.05']}")

d = effect_size(values1, values2, method='cohens_d')
print(f"Effect size (Cohen's d): {d:.3f}")
```

**Classes:**
- `CrossValidator` - K-fold and other CV strategies
- `CVResult` - Container for fold results
- `MetricRobustnessTester` - Robustness analysis

**CV methods:** K-fold, Stratified, LOO, Bootstrap, Monte Carlo

### Comparative Analysis (`comparative_analysis.py`)

Compare experiments and generate publication-ready outputs.

```python
from utils import ExperimentComparator, BenchmarkSuite, quick_compare

# Create comparator
comparator = ExperimentComparator('network_methods')

# Add experiments
comparator.add_experiment('small_world', results_sw, 
                          metrics={'C': 0.85, 'H': 0.92, 'PR': 0.45})
comparator.add_experiment('scale_free', results_sf,
                          metrics={'C': 0.78, 'H': 0.88, 'PR': 0.52})
comparator.add_experiment('modular', results_mod,
                          metrics={'C': 0.91, 'H': 0.95, 'PR': 0.38})

# Or add from CSV files
comparator.add_from_csv('experiment_1', 'results/exp1.csv', 
                        metric_columns=['C', 'H', 'PR'])

# Compare all
comparison_df = comparator.compare_all(normalize=True)

# Pairwise comparison with statistical tests
pair_result = comparator.compare_pair('small_world', 'modular', metrics=['C', 'H'])
print(f"C difference: {pair_result['metrics']['C']['difference']:.4f}")
print(f"p-value: {pair_result['metrics']['C']['p_value']:.4f}")

# Ranking
ranking = comparator.rank_experiments('C', maximize=True)
for name, value, rank in ranking:
    print(f"#{rank}: {name} (C = {value:.4f})")

# Multi-criteria ranking with weights
overall = comparator.compute_overall_ranking(
    metrics=['C', 'H', 'PR'],
    weights={'C': 2.0, 'H': 1.0, 'PR': 1.0},
    maximize={'C': True, 'H': True, 'PR': True}
)

# Statistical comparison against baseline
stats_df = comparator.statistical_comparison('small_world', metric='C')

# Generate publication tables
latex = comparator.to_latex_table(
    caption='Comparison of Network Methods',
    label='tab:comparison',
    highlight_best=True
)

markdown = comparator.to_markdown_table()

# Visualize
comparator.plot_comparison(plot_type='bar')    # Bar chart
comparator.plot_comparison(plot_type='radar')  # Radar plot
comparator.plot_comparison(plot_type='heatmap') # Heatmap
comparator.plot_comparison(plot_type='box')    # Box plots (needs full results)

# Benchmark suite for standardized testing
suite = BenchmarkSuite('consciousness_benchmarks')

suite.add_benchmark('small_world', lambda: generate_small_world(100, 6, 0.3))
suite.add_benchmark('scale_free', lambda: generate_scale_free(100, 3))

results = suite.run_all(compute_metrics, n_runs=10)
report = suite.generate_report()

# Quick comparison one-liner
df = quick_compare({
    'method_a': {'C': 0.85, 'H': 0.92},
    'method_b': {'C': 0.78, 'H': 0.95}
})
```

**Classes:**
- `ExperimentComparator` - Full comparison framework
- `ExperimentRecord` - Container for experiment data
- `BenchmarkSuite` - Standardized benchmark runner

**Output formats:** LaTeX tables, Markdown tables, PNG plots

---

## Quick Reference

```python
# Import everything
from utils import *

# Or import specific modules
from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg
from utils import visualization as viz
from utils.gpu_utils import GPUAccelerator
from utils.caching import ExperimentCache
from utils.parameter_sweep import ParameterSweep
from utils.cross_validation import CrossValidator
from utils.comparative_analysis import ExperimentComparator
```

## Dependencies

**Required:**
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- networkx >= 2.6.0
- pandas >= 1.3.0
- tqdm >= 4.62.0

**Optional:**
- cupy-cuda11x or cupy-cuda12x (GPU acceleration)
- torch (neural network experiments)
- h5py (HDF5 storage)
- scikit-learn (advanced CV)

## File Structure

```
utils/
├── __init__.py              # Package exports
├── README.md                # This documentation
├── graph_generators.py      # Network topology generation
├── metrics.py               # Consciousness metrics
├── state_generators.py      # Brain state simulation
├── visualization.py         # Plotting utilities
├── gpu_utils.py             # CUDA/GPU acceleration
├── caching.py               # Result caching & checkpoints
├── parameter_sweep.py       # Hyperparameter exploration
├── cross_validation.py      # Validation & robustness
└── comparative_analysis.py  # Experiment comparison
```
