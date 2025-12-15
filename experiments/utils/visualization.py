"""
Visualization Utilities

Comprehensive plotting functions for experimental results:
- Network topology plots
- Mode power distribution plots
- Consciousness component radar charts
- Time series evolution plots
- Phase space trajectories
- Heatmaps for sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List, Dict, Tuple
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D


# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def plot_network(
    G: nx.Graph,
    node_colors: Optional[np.ndarray] = None,
    title: str = "Network Topology",
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    layout: str = 'spring',
    seed: int = 42
) -> plt.Figure:
    """
    Plot network topology.
    
    Args:
        G: NetworkX graph
        node_colors: Optional array of node colors/values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        seed: Random seed for layout
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed, k=1.5)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray', width=0.5)
    
    # Draw nodes
    if node_colors is not None:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=100,
            node_color=node_colors,
            cmap='RdBu_r',
            vmin=-np.max(np.abs(node_colors)),
            vmax=np.max(np.abs(node_colors))
        )
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, node_color='steelblue')
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_mode_distribution(
    power: np.ndarray,
    title: str = "Mode Power Distribution",
    color: str = 'steelblue',
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
    show_metrics: bool = True
) -> plt.Figure:
    """
    Plot mode power distribution as bar chart.
    
    Args:
        power: Mode power distribution
        title: Plot title
        color: Bar color
        figsize: Figure size
        save_path: Optional path to save figure
        show_metrics: Whether to show H and PR on plot
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    k = np.arange(len(power))
    ax.bar(k, power, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Mode index $k$')
    ax.set_ylabel('Normalized power $p_k$')
    ax.set_title(title)
    ax.set_xlim(-0.5, len(power) - 0.5)
    
    if show_metrics:
        # Compute metrics
        H = -np.sum(power * np.log(power + 1e-12)) / np.log(len(power))
        PR = (1.0 / np.sum(power ** 2)) / len(power)
        
        ax.text(0.95, 0.95, f'$H_{{mode}} = {H:.2f}$\n$PR = {PR:.2f}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_consciousness_radar(
    metrics: Dict[str, float],
    title: str = "Consciousness Components",
    figsize: Tuple[float, float] = (7, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot consciousness metrics as radar chart.
    
    Args:
        metrics: Dictionary with keys: H_mode, PR, R, S_dot, kappa
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Extract values
    categories = ['$H_{mode}$', '$PR$', '$R$', '$\\dot{S}$', '$\\kappa$']
    values = [
        metrics.get('H_mode', 0),
        metrics.get('PR', 0),
        metrics.get('R', 0),
        metrics.get('S_dot', 0),
        metrics.get('kappa', 0),
    ]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    # Set ylim
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_title(title, y=1.08, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_time_series(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Time Series Evolution",
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    legend: bool = True
) -> plt.Figure:
    """
    Plot multiple time series.
    
    Args:
        data: Array of shape (n_timesteps, n_series) or (n_timesteps,)
        labels: Optional labels for each series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure
        legend: Whether to show legend
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    t = np.arange(data.shape[0])
    
    for i in range(data.shape[1]):
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        ax.plot(t, data[:, i], label=label, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if legend and data.shape[1] > 1:
        ax.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_phase_space(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    title: str = "Phase Space Trajectory",
    labels: Optional[Tuple[str, str, str]] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    colormap: str = 'viridis'
) -> plt.Figure:
    """
    Plot phase space trajectory (2D or 3D).
    
    Args:
        x: X coordinates
        y: Y coordinates
        z: Optional Z coordinates for 3D plot
        title: Plot title
        labels: Optional axis labels (xlabel, ylabel, zlabel)
        figsize: Figure size
        save_path: Optional path to save figure
        colormap: Colormap for trajectory
    
    Returns:
        Matplotlib figure
    """
    if z is not None:
        # 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by time
        colors = np.arange(len(x))
        scatter = ax.scatter(x, y, z, c=colors, cmap=colormap, s=10)
        ax.plot(x, y, z, alpha=0.3, linewidth=0.5, color='gray')
        
        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
        
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by time
        colors = np.arange(len(x))
        scatter = ax.scatter(x, y, c=colors, cmap=colormap, s=20, alpha=0.6)
        ax.plot(x, y, alpha=0.3, linewidth=0.5, color='gray')
        
        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        
        plt.colorbar(scatter, ax=ax, label='Time')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_heatmap(
    data: np.ndarray,
    title: str = "Heatmap",
    xlabel: str = "X",
    ylabel: str = "Y",
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'YlOrRd',
    save_path: Optional[str] = None,
    annot: bool = False,
    fmt: str = '.2f'
) -> plt.Figure:
    """
    Plot heatmap for sensitivity analysis.
    
    Args:
        data: 2D array to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xticklabels: Optional x-axis tick labels
        yticklabels: Optional y-axis tick labels
        figsize: Figure size
        cmap: Colormap
        save_path: Optional path to save figure
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': 'Value'}
    )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_multi_state_comparison(
    states_data: Dict[str, np.ndarray],
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple brain states side-by-side for comparison.
    
    Args:
        states_data: Dictionary mapping state names to power distributions
        figsize: Figure size
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    n_states = len(states_data)
    ncols = min(3, n_states)
    nrows = (n_states + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if n_states == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_states))
    
    for idx, (state_name, power) in enumerate(states_data.items()):
        ax = axes[idx]
        k = np.arange(len(power))
        
        ax.bar(k, power, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Mode index $k$')
        ax.set_ylabel('Power $p_k$')
        ax.set_title(state_name)
        ax.set_xlim(-0.5, len(power) - 0.5)
        
        # Add metrics
        H = -np.sum(power * np.log(power + 1e-12)) / np.log(len(power))
        PR = (1.0 / np.sum(power ** 2)) / len(power)
        ax.text(0.95, 0.95, f'H={H:.2f}\nPR={PR:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide extra subplots
    for idx in range(n_states, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_correlation_matrix(
    correlation_matrix: np.ndarray,
    labels: List[str],
    title: str = "Correlation Matrix",
    figsize: Tuple[float, float] = (8, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix between metrics.
    
    Args:
        correlation_matrix: NxN correlation matrix
        labels: Labels for each variable
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        correlation_matrix,
        ax=ax,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test network plot
    G = nx.watts_strogatz_graph(30, 4, 0.3, seed=42)
    plot_network(G, title="Test Network", save_path="/tmp/test_network.png")
    print("  Network plot: OK")
    
    # Test mode distribution
    power = np.exp(-np.arange(20) / 5)
    power /= power.sum()
    plot_mode_distribution(power, title="Test Distribution", save_path="/tmp/test_dist.png")
    print("  Mode distribution: OK")
    
    # Test radar chart
    metrics = {'H_mode': 0.8, 'PR': 0.7, 'R': 0.5, 'S_dot': 0.6, 'kappa': 0.75}
    plot_consciousness_radar(metrics, save_path="/tmp/test_radar.png")
    print("  Radar chart: OK")
    
    # Test time series
    data = np.random.randn(100, 3).cumsum(axis=0)
    plot_time_series(data, labels=['A', 'B', 'C'], save_path="/tmp/test_timeseries.png")
    print("  Time series: OK")
    
    # Test phase space
    t = np.linspace(0, 10, 200)
    x = np.sin(t)
    y = np.cos(t)
    z = t / 10
    plot_phase_space(x, y, z, save_path="/tmp/test_phase.png")
    print("  Phase space: OK")
    
    # Test heatmap
    data = np.random.rand(5, 5)
    plot_heatmap(data, save_path="/tmp/test_heatmap.png")
    print("  Heatmap: OK")
    
    plt.close('all')
    print("\nAll visualization functions working correctly!")
