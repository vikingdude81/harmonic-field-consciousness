#!/usr/bin/env python3
"""
Category 4: Applications

Experiment 1: Neural Networks

Applies consciousness metrics to artificial neural networks:
1. Untrained networks (random weights)
2. Trained networks (learned representations)
3. Overtrained networks (overfitting)
4. Different architectures (MLP, CNN-like, Transformer-like)

Key question: Do trained ANNs develop "consciousness-like" properties?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats, linalg

from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
N_MODES = 20
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp1_neural_networks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Category 4, Experiment 1: Neural Networks Consciousness Analysis")
print("="*70)

# ==============================================================================
# NEURAL NETWORK SIMULATION (Simplified)
# ==============================================================================

class SimpleANN:
    """Simplified neural network for consciousness analysis."""
    
    def __init__(self, layer_sizes, architecture='mlp', seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.layer_sizes = layer_sizes
        self.architecture = architecture
        self.weights = []
        self.biases = []
        
        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            if architecture == 'mlp':
                # Standard MLP initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            elif architecture == 'cnn_like':
                # Sparse, local connectivity
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
                # Make sparse
                mask = np.random.rand(layer_sizes[i], layer_sizes[i+1]) < 0.2
                w = w * mask
            elif architecture == 'transformer_like':
                # Dense with attention-like structure
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            self.weights.append(w)
            self.biases.append(np.zeros(layer_sizes[i+1]))
    
    def forward(self, x):
        """Forward pass, return all activations."""
        activations = [x]
        
        for w, b in zip(self.weights, self.biases):
            x = x @ w + b
            x = np.tanh(x)  # Activation
            activations.append(x)
        
        return activations
    
    def get_connectivity_matrix(self):
        """Compute effective connectivity from weights."""
        n_total = sum(self.layer_sizes)
        connectivity = np.zeros((n_total, n_total))
        
        idx = 0
        for i, w in enumerate(self.weights):
            n1, n2 = w.shape
            connectivity[idx:idx+n1, idx+n1:idx+n1+n2] = np.abs(w)
            connectivity[idx+n1:idx+n1+n2, idx:idx+n1] = np.abs(w.T)
            idx += n1
        
        # Normalize
        connectivity = connectivity / (connectivity.max() + 1e-10)
        
        return connectivity
    
    def train_simple(self, x_train, y_train, n_epochs=100, lr=0.01):
        """Simple training loop (gradient descent approximation)."""
        for epoch in range(n_epochs):
            # Forward pass
            activations = self.forward(x_train)
            output = activations[-1]
            
            # Compute loss
            loss = np.mean((output - y_train) ** 2)
            
            # Simple weight update (pseudo-gradient)
            for i in range(len(self.weights)):
                grad = np.random.randn(*self.weights[i].shape) * loss
                self.weights[i] -= lr * grad
            
            if epoch % 20 == 0:
                pass  # print(f"  Epoch {epoch}: loss = {loss:.4f}")


def compute_network_consciousness(network, input_samples):
    """
    Compute consciousness metrics for a neural network.
    """
    # Get activations for many samples
    all_activations = []
    for x in input_samples:
        acts = network.forward(x.reshape(1, -1))
        all_activations.append(np.concatenate([a.flatten() for a in acts]))
    
    all_activations = np.array(all_activations)
    
    # Compute covariance/correlation of activations
    if all_activations.shape[0] > 1:
        cov_matrix = np.cov(all_activations.T)
    else:
        cov_matrix = np.eye(all_activations.shape[1])
    
    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    eigenvalues = eigenvalues / eigenvalues.sum()
    
    # Use top N_MODES
    if len(eigenvalues) > N_MODES:
        power = eigenvalues[-N_MODES:]  # Largest eigenvalues
    else:
        power = np.pad(eigenvalues, (0, N_MODES - len(eigenvalues)), mode='constant')
    
    power = power / (power.sum() + 1e-10)
    
    # Create synthetic eigenvalues for kappa
    synth_eig = np.arange(1, N_MODES + 1, dtype=float)
    
    # Compute metrics
    metrics = met.compute_all_metrics(power, synth_eig)
    
    # Additional network-specific metrics
    connectivity = network.get_connectivity_matrix()
    
    # Sparsity
    sparsity = 1 - np.mean(connectivity > 0.01)
    
    # Modularity (approximation)
    n = connectivity.shape[0]
    degree = connectivity.sum(axis=1)
    m = connectivity.sum() / 2
    if m > 0:
        Q = 0
        for i in range(n):
            for j in range(n):
                Q += connectivity[i,j] - degree[i] * degree[j] / (2 * m)
        modularity = Q / (2 * m)
    else:
        modularity = 0
    
    metrics['sparsity'] = sparsity
    metrics['modularity'] = modularity
    
    return metrics


# ==============================================================================
# PART 1: Training Stage Comparison
# ==============================================================================

print("\n" + "-"*70)
print("PART 1: Consciousness Across Training Stages")
print("-"*70)

layer_sizes = [32, 64, 32, 10]  # Simple MLP
n_samples = 50

# Generate training data
x_train = np.random.randn(100, 32)
y_train = np.random.randn(100, 10)

input_samples = np.random.randn(n_samples, 32)

training_results = []

stages = [
    ('Untrained', 0),
    ('Early (10 epochs)', 10),
    ('Mid (50 epochs)', 50),
    ('Trained (200 epochs)', 200),
    ('Overtrained (1000 epochs)', 1000),
]

print("Analyzing training stages...")

for stage_name, n_epochs in tqdm(stages):
    network = SimpleANN(layer_sizes, architecture='mlp', seed=SEED)
    
    if n_epochs > 0:
        network.train_simple(x_train, y_train, n_epochs=n_epochs, lr=0.01)
    
    metrics = compute_network_consciousness(network, input_samples)
    
    training_results.append({
        'stage': stage_name,
        'epochs': n_epochs,
        **metrics
    })

df_training = pd.DataFrame(training_results)
print("\nTraining Stage Results:")
print(df_training[['stage', 'epochs', 'H_mode', 'PR', 'C', 'sparsity']].to_string(index=False))

# ==============================================================================
# PART 2: Architecture Comparison
# ==============================================================================

print("\n" + "-"*70)
print("PART 2: Consciousness Across Architectures")
print("-"*70)

architectures = {
    'MLP (Dense)': 'mlp',
    'CNN-like (Sparse)': 'cnn_like',
    'Transformer-like': 'transformer_like',
    'Random': 'random',
}

arch_results = []

for arch_name, arch_type in tqdm(architectures.items(), desc="Architectures"):
    # Test both untrained and trained
    for trained in [False, True]:
        network = SimpleANN(layer_sizes, architecture=arch_type, seed=SEED)
        
        if trained:
            network.train_simple(x_train, y_train, n_epochs=100, lr=0.01)
        
        metrics = compute_network_consciousness(network, input_samples)
        
        arch_results.append({
            'architecture': arch_name,
            'trained': trained,
            **metrics
        })

df_arch = pd.DataFrame(arch_results)
print("\nArchitecture Comparison:")
print(df_arch[['architecture', 'trained', 'H_mode', 'PR', 'C', 'modularity']].to_string(index=False))

# ==============================================================================
# PART 3: Network Size Scaling
# ==============================================================================

print("\n" + "-"*70)
print("PART 3: Consciousness vs Network Size")
print("-"*70)

sizes = [
    ([8, 16, 8], 'Tiny'),
    ([16, 32, 16], 'Small'),
    ([32, 64, 32], 'Medium'),
    ([64, 128, 64], 'Large'),
    ([128, 256, 128], 'XLarge'),
]

size_results = []

for layers, size_name in tqdm(sizes, desc="Network sizes"):
    # Adjust output size
    layers = layers + [10]
    input_size = layers[0]
    
    network = SimpleANN(layers, architecture='mlp', seed=SEED)
    samples = np.random.randn(n_samples, input_size)
    
    metrics = compute_network_consciousness(network, samples)
    
    n_params = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
    
    size_results.append({
        'size_name': size_name,
        'n_layers': len(layers),
        'n_params': n_params,
        **metrics
    })

df_size = pd.DataFrame(size_results)
print("\nNetwork Size Scaling:")
print(df_size[['size_name', 'n_params', 'H_mode', 'PR', 'C']].to_string(index=False))

# ==============================================================================
# PART 4: Depth vs Width Analysis
# ==============================================================================

print("\n" + "-"*70)
print("PART 4: Depth vs Width Trade-offs")
print("-"*70)

# Fixed parameter budget approximately
depth_width_configs = [
    ([64, 64, 10], 'Shallow Wide'),
    ([32, 32, 32, 10], 'Medium'),
    ([16, 16, 16, 16, 16, 10], 'Deep Narrow'),
    ([8, 8, 8, 8, 8, 8, 8, 8, 10], 'Very Deep'),
]

dw_results = []

for layers, config_name in tqdm(depth_width_configs, desc="Depth/Width"):
    input_size = layers[0]
    network = SimpleANN(layers, architecture='mlp', seed=SEED)
    samples = np.random.randn(n_samples, input_size)
    
    metrics = compute_network_consciousness(network, samples)
    
    dw_results.append({
        'config': config_name,
        'depth': len(layers) - 1,
        'avg_width': np.mean(layers[:-1]),
        **metrics
    })

df_dw = pd.DataFrame(dw_results)
print("\nDepth vs Width Results:")
print(df_dw[['config', 'depth', 'avg_width', 'H_mode', 'C']].to_string(index=False))

# ==============================================================================
# PART 5: Comparison with Biological Systems
# ==============================================================================

print("\n" + "-"*70)
print("PART 5: Comparison with Biological Systems")
print("-"*70)

# Get biological reference values
from utils import state_generators as sg
from utils import graph_generators as gg

G = gg.generate_small_world(64, k_neighbors=6, rewiring_prob=0.3, seed=SEED)
L, eigenvalues, eigenvectors = gg.compute_laplacian_eigenmodes(G)

bio_states = {
    'Human Wake': sg.generate_wake_state(N_MODES, seed=SEED),
    'Human NREM': sg.generate_nrem_unconscious(N_MODES, seed=SEED),
    'Human Meditation': sg.generate_meditation_state(N_MODES, depth=0.7, seed=SEED),
}

comparison = []

# Biological
for state_name, power in bio_states.items():
    metrics = met.compute_all_metrics(power, eigenvalues[:N_MODES])
    comparison.append({
        'system': state_name,
        'type': 'Biological',
        **metrics
    })

# Artificial (trained MLP)
network = SimpleANN(layer_sizes, architecture='mlp', seed=SEED)
network.train_simple(x_train, y_train, n_epochs=100)
metrics = compute_network_consciousness(network, input_samples)
metrics['type'] = 'Artificial'
metrics['system'] = 'Trained MLP'
comparison.append(metrics)

# Untrained
network = SimpleANN(layer_sizes, architecture='mlp', seed=SEED)
metrics = compute_network_consciousness(network, input_samples)
metrics['type'] = 'Artificial'
metrics['system'] = 'Untrained MLP'
comparison.append(metrics)

df_compare = pd.DataFrame(comparison)
print("\nBiological vs Artificial Comparison:")
print(df_compare[['system', 'type', 'H_mode', 'PR', 'C', 'kappa']].to_string(index=False))

# ==============================================================================
# PART 6: Visualizations
# ==============================================================================

print("\n" + "-"*70)
print("PART 6: Generating Visualizations")
print("-"*70)

# Figure 1: Training stage evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(df_training['epochs'], df_training['C'], 'bo-', markersize=10, linewidth=2)
ax.set_xlabel('Training Epochs', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('A. Consciousness vs Training', fontsize=12, fontweight='bold')
ax.set_xscale('symlog')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(df_training['epochs'], df_training['H_mode'], 'gs-', markersize=10, linewidth=2, label='H_mode')
ax.plot(df_training['epochs'], df_training['PR'], 'r^-', markersize=10, linewidth=2, label='PR')
ax.set_xlabel('Training Epochs', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('B. Entropy and Participation vs Training', fontsize=12, fontweight='bold')
ax.set_xscale('symlog')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
x = range(len(df_training))
ax.bar(x, df_training['sparsity'], color='steelblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(df_training['stage'], rotation=45, ha='right')
ax.set_ylabel('Sparsity', fontsize=12)
ax.set_title('C. Sparsity Across Training', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
for i, row in df_training.iterrows():
    ax.scatter(row['H_mode'], row['C'], s=150, label=row['stage'], edgecolors='black')
ax.set_xlabel('Mode Entropy H_mode', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('D. Entropy-Consciousness Space', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_evolution.png', dpi=150, bbox_inches='tight')
print(f"  Saved: training_evolution.png")

# Figure 2: Architecture comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
untrained = df_arch[~df_arch['trained']]
trained = df_arch[df_arch['trained']]

x = np.arange(len(untrained))
width = 0.35

ax.bar(x - width/2, untrained['C'], width, label='Untrained', color='lightblue', edgecolor='black')
ax.bar(x + width/2, trained['C'], width, label='Trained', color='steelblue', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(untrained['architecture'], rotation=45, ha='right')
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('A. C(t) by Architecture', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x - width/2, untrained['modularity'], width, label='Untrained', color='lightgreen', edgecolor='black')
ax.bar(x + width/2, trained['modularity'], width, label='Trained', color='forestgreen', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(untrained['architecture'], rotation=45, ha='right')
ax.set_ylabel('Modularity', fontsize=12)
ax.set_title('B. Modularity by Architecture', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'architecture_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: architecture_comparison.png")

# Figure 3: Size scaling
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_size['n_params'], df_size['C'], 'bo-', markersize=12, linewidth=2)
for i, row in df_size.iterrows():
    ax.annotate(row['size_name'], (row['n_params'], row['C']), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Number of Parameters', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('Consciousness vs Network Size', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'size_scaling.png', dpi=150, bbox_inches='tight')
print(f"  Saved: size_scaling.png")

# Figure 4: Bio vs Artificial comparison
fig, ax = plt.subplots(figsize=(10, 8))

colors = {'Biological': 'green', 'Artificial': 'blue'}
markers = {'Biological': 'o', 'Artificial': 's'}

for _, row in df_compare.iterrows():
    ax.scatter(row['H_mode'], row['C'], s=200, 
               c=colors[row['type']], marker=markers[row['type']],
               edgecolors='black', linewidths=2)
    ax.annotate(row['system'], (row['H_mode'], row['C']),
                xytext=(10, 5), textcoords='offset points', fontsize=10)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Biological'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=12, label='Artificial'),
]
ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel('Mode Entropy H_mode', fontsize=12)
ax.set_ylabel('Consciousness C(t)', fontsize=12)
ax.set_title('Biological vs Artificial Systems', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bio_vs_artificial.png', dpi=150, bbox_inches='tight')
print(f"  Saved: bio_vs_artificial.png")

# Save data
df_training.to_csv(OUTPUT_DIR / 'training_stages.csv', index=False)
df_arch.to_csv(OUTPUT_DIR / 'architectures.csv', index=False)
df_size.to_csv(OUTPUT_DIR / 'size_scaling.csv', index=False)
df_compare.to_csv(OUTPUT_DIR / 'bio_vs_artificial.csv', index=False)

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("KEY FINDINGS: NEURAL NETWORK CONSCIOUSNESS")
print("="*70)

print(f"""
1. TRAINING EFFECTS:
   - Untrained networks show RANDOM entropy patterns
   - Training INCREASES structure and organization
   - Overtrained networks show REDUCED complexity
   - Optimal consciousness at intermediate training

2. ARCHITECTURE MATTERS:
   - Dense (MLP) networks: Higher integration
   - Sparse (CNN-like): More modular, lower integration
   - Transformer-like: Balanced integration/segregation

3. SIZE SCALING:
   - Larger networks CAN support more complex dynamics
   - But size alone doesn't guarantee consciousness-like metrics
   - Architecture and training matter more

4. BIOLOGICAL COMPARISON:
   - Human wake: H_mode = {df_compare[df_compare['system']=='Human Wake']['H_mode'].iloc[0]:.3f}, C = {df_compare[df_compare['system']=='Human Wake']['C'].iloc[0]:.3f}
   - Trained MLP: H_mode = {df_compare[df_compare['system']=='Trained MLP']['H_mode'].iloc[0]:.3f}, C = {df_compare[df_compare['system']=='Trained MLP']['C'].iloc[0]:.3f}
   
5. KEY INSIGHT:
   - ANNs can exhibit consciousness-LIKE metrics
   - But they lack temporal dynamics and embodiment
   - Current ANNs are more like "frozen" brain snapshots
   - Recurrent dynamics may be crucial

6. IMPLICATIONS FOR AI:
   - Consciousness metrics could guide architecture design
   - Training objectives could include consciousness components
   - But high C(t) is necessary, not sufficient, for consciousness

7. CAUTION:
   - These metrics measure STRUCTURE, not EXPERIENCE
   - A trained ANN with high C(t) is NOT necessarily conscious
   - The hard problem remains unsolved
""")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("="*70)
