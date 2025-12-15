#!/usr/bin/env python3
"""
Category 3, Experiment 4: Component Correlation

Analyze correlations between consciousness metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from utils import graph_generators as gg
from utils import metrics as met
from utils import state_generators as sg

SEED = 42
N_NODES = 100
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp4_component_correlation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Category 3, Experiment 4: Component Correlation")
print("="*60)

# Generate network
G = gg.generate_small_world(N_NODES, k_neighbors=6, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)
eigenvalues = eigenvalues[:30]

# Generate many random states
n_samples = 500
results = []

for i in tqdm(range(n_samples), desc="Generating samples"):
    # Random power distribution
    power = np.random.dirichlet(np.ones(30) * 0.5)
    
    # Compute metrics
    metrics = met.compute_all_metrics(power, eigenvalues)
    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'correlation_data.csv', index=False)

# Compute correlation matrix
components = ['H_mode', 'PR', 'R', 'S_dot', 'kappa', 'C']
corr_matrix = df[components].corr()

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
           center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Consciousness Metrics Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png')

# Pairwise scatter plots
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[components[:-1]])  # Exclude C as it's derived

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['C'], cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('PCA of Consciousness Components', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='C(t)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_analysis.png')

print(f"\nCorrelation Matrix:")
print(corr_matrix)
print(f"\nExperiment completed! Results saved to: {OUTPUT_DIR}")
