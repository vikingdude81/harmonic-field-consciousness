"""
Attention Visualization Utilities

Tools for visualizing attention patterns in transformer models.
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AttentionVisualizer:
    """
    Visualizer for attention weights and patterns.
    """
    
    def __init__(self, region_names: Optional[list] = None):
        """
        Initialize attention visualizer.
        
        Args:
            region_names: Optional list of region names for labeling
        """
        self.region_names = region_names
    
    def plot_attention_matrix(self, attention: np.ndarray, title: str = "Attention Weights",
                             save_path: Optional[str] = None, head: int = 0) -> plt.Figure:
        """
        Plot attention weight matrix as heatmap.
        
        Args:
            attention: Attention weights (n_heads, n_regions, n_regions) or (n_regions, n_regions)
            title: Plot title
            save_path: Optional path to save figure
            head: Which attention head to plot (if multi-head)
        
        Returns:
            Matplotlib figure
        """
        # Extract attention for specified head
        if attention.ndim == 3:
            attn_matrix = attention[head, :, :]
        else:
            attn_matrix = attention
        
        n_regions = attn_matrix.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Regions', fontsize=12)
        ax.set_ylabel('Query Regions', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12)
        
        # Add region labels if available
        if self.region_names is not None and len(self.region_names) == n_regions:
            ax.set_xticks(range(n_regions))
            ax.set_yticks(range(n_regions))
            ax.set_xticklabels(self.region_names, rotation=45, ha='right')
            ax.set_yticklabels(self.region_names)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_attention_flow(self, attention: np.ndarray, top_k: int = 5,
                           title: str = "Top Attention Connections",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot top-k attention connections as flow diagram.
        
        Args:
            attention: Attention weights (n_regions, n_regions)
            top_k: Number of top connections to show per region
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        if attention.ndim == 3:
            attention = np.mean(attention, axis=0)  # Average over heads
        
        n_regions = attention.shape[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Arrange regions in circle
        angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Plot regions
        ax.scatter(x, y, s=200, c='blue', alpha=0.6, zorder=3)
        
        # Add region labels
        for i in range(n_regions):
            label = self.region_names[i] if self.region_names else f"R{i}"
            offset = 1.15
            ax.text(x[i] * offset, y[i] * offset, label, 
                   ha='center', va='center', fontsize=10)
        
        # Plot top-k connections for each region
        for i in range(n_regions):
            # Get top-k strongest connections
            attn_scores = attention[i, :]
            top_indices = np.argsort(attn_scores)[-top_k-1:-1]  # Exclude self
            
            for j in top_indices:
                if i != j:
                    weight = attn_scores[j]
                    # Draw arrow
                    ax.annotate('', xy=(x[j], y[j]), xytext=(x[i], y[i]),
                              arrowprops=dict(arrowstyle='->', lw=weight*3, 
                                            alpha=0.5, color='red'))
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_attention_entropy(self, attention: np.ndarray, 
                              title: str = "Attention Entropy per Region",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot entropy of attention distribution for each region.
        
        High entropy = attention spread across many regions
        Low entropy = attention focused on few regions
        
        Args:
            attention: Attention weights (n_heads, n_regions, n_regions) or (n_regions, n_regions)
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        if attention.ndim == 3:
            attention = np.mean(attention, axis=0)  # Average over heads
        
        n_regions = attention.shape[0]
        
        # Compute entropy for each region's attention distribution
        entropies = []
        for i in range(n_regions):
            attn_dist = attention[i, :]
            attn_dist = attn_dist / (np.sum(attn_dist) + 1e-12)
            
            # Shannon entropy
            p = attn_dist[attn_dist > 1e-12]
            H = -np.sum(p * np.log(p + 1e-12))
            entropies.append(H)
        
        entropies = np.array(entropies)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(n_regions)
        ax.bar(x, entropies, color='steelblue', alpha=0.7)
        ax.set_xlabel('Region Index', fontsize=12)
        ax.set_ylabel('Attention Entropy (nats)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        if self.region_names is not None and len(self.region_names) == n_regions:
            ax.set_xticks(x)
            ax.set_xticklabels(self.region_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_multi_head_comparison(self, attention: np.ndarray,
                                  title: str = "Multi-Head Attention Comparison",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare attention patterns across different heads.
        
        Args:
            attention: Attention weights (n_heads, n_regions, n_regions)
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        if attention.ndim != 3:
            raise ValueError("Expected 3D attention tensor (n_heads, n_regions, n_regions)")
        
        n_heads = attention.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        if n_heads == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for head in range(n_heads):
            row = head // n_cols
            col = head % n_cols
            ax = axes[row, col]
            
            im = ax.imshow(attention[head, :, :], cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head}', fontsize=10)
            ax.set_xlabel('Key', fontsize=8)
            ax.set_ylabel('Query', fontsize=8)
            plt.colorbar(im, ax=ax)
        
        # Hide empty subplots
        for head in range(n_heads, n_rows * n_cols):
            row = head // n_cols
            col = head % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
