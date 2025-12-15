# Category 3: Consciousness Functional Modifications

This category optimizes and extends the consciousness metrics themselves.

## Experiments

### exp1_weighted_components.py
**Optimize component weightings**

Procedure:
- Test all combinations of weights for 5 components
- Grid search or optimization approach
- Maximize separation between Wake and Anesthesia
- Find minimal effective set
- Generate weight sensitivity heatmaps
- Propose simplified functionals

**Output**: `results/exp1_weighted_components/`

### exp2_new_metrics.py
**Add novel complexity measures**

Procedure:
- Implement Lempel-Ziv complexity
- Add multiscale entropy
- Test neural complexity measure
- Compare to existing 5 components
- Correlation analysis
- Feature importance ranking

**Output**: `results/exp2_new_metrics/`

### exp3_threshold_detection.py
**Find clinical consciousness threshold**

Procedure:
- Calculate C(t) for many simulated states
- Vary parameters continuously from unconscious to conscious
- Use ROC analysis to find optimal threshold
- Test sensitivity and specificity
- Model clinical monitoring scenarios
- Generate decision boundary plots

**Output**: `results/exp3_threshold_detection/`

### exp4_component_correlation.py
**Analyze metric relationships**

Procedure:
- Calculate correlations between all metrics
- PCA/factor analysis on components
- Test for redundancy
- Identify independent dimensions
- Generate correlation matrices
- Dimensionality reduction visualizations

**Output**: `results/exp4_component_correlation/`

## Running the Experiments

Run all experiments in this category:
```bash
python exp1_weighted_components.py
python exp2_new_metrics.py
python exp3_threshold_detection.py
python exp4_component_correlation.py
```

## Results

All results are saved in the `results/` subdirectory.
