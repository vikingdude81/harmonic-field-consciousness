"""
Graph/Network Topology Generators

Provides various network architectures for testing consciousness metrics:
- Small-world networks (Watts-Strogatz)
- Scale-free networks (Barabási-Albert)
- Random graphs (Erdős-Rényi)
- Lattice graphs (2D/3D grids)
- Modular networks with communities
- Hub-based networks
"""

import numpy as np
import networkx as nx
from typing import Optional, List, Tuple


def generate_small_world(
    n_nodes: int = 100,
    k_neighbors: int = 6,
    rewiring_prob: float = 0.3,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a small-world network using the Watts-Strogatz model.
    
    Args:
        n_nodes: Number of nodes
        k_neighbors: Each node is connected to k nearest neighbors in ring topology
        rewiring_prob: Probability of rewiring each edge
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewiring_prob, seed=seed)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G


def generate_scale_free(
    n_nodes: int = 100,
    m_edges: int = 3,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a scale-free network using the Barabási-Albert model.
    
    Args:
        n_nodes: Number of nodes
        m_edges: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX graph with hub structure
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G


def generate_random(
    n_nodes: int = 100,
    edge_prob: float = 0.1,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a random graph using the Erdős-Rényi model.
    
    Args:
        n_nodes: Number of nodes
        edge_prob: Probability for edge creation
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX random graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        # Add edges to connect components
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G


def generate_lattice(
    n_nodes: int = 100,
    dimension: int = 2,
    periodic: bool = False,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a lattice graph (2D or 3D grid).
    
    Args:
        n_nodes: Approximate number of nodes (will be adjusted to fit grid)
        dimension: Dimension of lattice (2 or 3)
        periodic: Whether to use periodic boundary conditions
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX lattice graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    if dimension == 2:
        # Create square lattice
        side_length = int(np.sqrt(n_nodes))
        if periodic:
            G = nx.grid_2d_graph(side_length, side_length, periodic=True)
        else:
            G = nx.grid_2d_graph(side_length, side_length)
    elif dimension == 3:
        # Create cubic lattice
        side_length = int(np.cbrt(n_nodes))
        G = nx.grid_graph(dim=[side_length, side_length, side_length], periodic=periodic)
    else:
        raise ValueError("Dimension must be 2 or 3")
    
    # Convert to simple graph with integer node labels
    G = nx.convert_node_labels_to_integers(G)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G


def generate_modular(
    n_nodes: int = 100,
    n_modules: int = 4,
    intra_prob: float = 0.3,
    inter_prob: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, List[List[int]]]:
    """
    Generate a modular network with communities.
    
    Args:
        n_nodes: Number of nodes
        n_modules: Number of modules/communities
        intra_prob: Probability of edges within modules
        inter_prob: Probability of edges between modules
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (NetworkX graph, list of node lists per module)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Divide nodes into modules
    nodes_per_module = n_nodes // n_modules
    modules = []
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Create modules
    for i in range(n_modules):
        start_idx = i * nodes_per_module
        if i == n_modules - 1:
            # Last module gets remaining nodes
            end_idx = n_nodes
        else:
            end_idx = (i + 1) * nodes_per_module
        
        module_nodes = list(range(start_idx, end_idx))
        modules.append(module_nodes)
        
        # Add intra-module edges
        for u in module_nodes:
            for v in module_nodes:
                if u < v and np.random.rand() < intra_prob:
                    G.add_edge(u, v)
    
    # Add inter-module edges
    for i in range(n_modules):
        for j in range(i + 1, n_modules):
            for u in modules[i]:
                for v in modules[j]:
                    if np.random.rand() < inter_prob:
                        G.add_edge(u, v)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G, modules


def generate_hub_network(
    n_nodes: int = 100,
    n_hubs: int = 5,
    hub_degree_fraction: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a network with explicit hub nodes.
    
    Args:
        n_nodes: Number of nodes
        n_hubs: Number of hub nodes
        hub_degree_fraction: Fraction of nodes each hub connects to
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (NetworkX graph, list of hub node indices)
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Designate hub nodes
    hub_nodes = list(np.random.choice(n_nodes, n_hubs, replace=False))
    
    # Connect hubs to many nodes
    hub_connections = int(n_nodes * hub_degree_fraction)
    for hub in hub_nodes:
        # Connect to random nodes
        targets = np.random.choice(
            [n for n in range(n_nodes) if n != hub],
            hub_connections,
            replace=False
        )
        for target in targets:
            G.add_edge(hub, target)
    
    # Add random background edges
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if not G.has_edge(u, v) and np.random.rand() < 0.02:
                G.add_edge(u, v)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    return G, hub_nodes


def compute_laplacian_eigenmodes(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the graph Laplacian and its eigenmodes.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Tuple of (Laplacian matrix, eigenvalues, eigenvectors)
    """
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray().astype(float)
    
    # Compute degree matrix
    D = np.diag(A.sum(axis=1))
    
    # Compute Laplacian
    L = D - A
    
    # Compute eigenmodes
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return L, eigenvalues, eigenvectors


def get_node_positions(G: nx.Graph, 
                      layout: str = 'spring',
                      seed: Optional[int] = None,
                      **kwargs) -> dict:
    """
    Get 2D positions for network nodes for visualization and spatial analysis.
    
    Args:
        G: NetworkX graph
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        seed: Random seed for reproducible layouts
        **kwargs: Additional arguments for layout algorithm
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed, **kwargs)
    elif layout == 'circular':
        pos = nx.circular_layout(G, **kwargs)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, **kwargs)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G, **kwargs)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    return pos


if __name__ == "__main__":
    # Test all generators
    print("Testing graph generators...")
    
    graphs = {
        "Small-world": generate_small_world(50, seed=42),
        "Scale-free": generate_scale_free(50, seed=42),
        "Random": generate_random(50, seed=42),
        "Lattice 2D": generate_lattice(49, dimension=2, seed=42),
        "Modular": generate_modular(50, n_modules=4, seed=42)[0],
        "Hub network": generate_hub_network(50, n_hubs=3, seed=42)[0],
    }
    
    for name, G in graphs.items():
        L, eigenvalues, eigenvectors = compute_laplacian_eigenmodes(G)
        print(f"{name:15s}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"λ_max={eigenvalues[-1]:.2f}")
    
    print("\nAll graph generators working correctly!")
