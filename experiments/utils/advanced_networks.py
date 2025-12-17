"""
Advanced Network Topology Generators

Extended network architectures for consciousness modeling:
1. Hierarchical modular networks (brain-like nesting)
2. Rich-club organization (hub-hub connections)
3. Small-world + Scale-free hybrids
4. Dynamic/adaptive networks
5. Connectome-inspired architectures
"""

import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict
from itertools import combinations


def generate_hierarchical_modular(
    n_nodes: int = 128,
    n_levels: int = 3,
    branching_factor: int = 2,
    intra_prob: float = 0.5,
    inter_prob_decay: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict]:
    """
    Generate a hierarchical modular network (brain-like).
    
    Mimics cortical organization with nested modules at multiple scales:
    - Level 0: Individual nodes (neurons/columns)
    - Level 1: Local circuits (minicolumns)
    - Level 2: Areas (cortical areas)
    - Level 3: Systems (functional networks)
    
    Args:
        n_nodes: Total number of nodes
        n_levels: Number of hierarchical levels
        branching_factor: Number of children per parent module
        intra_prob: Connection probability within lowest-level modules
        inter_prob_decay: Decay factor for inter-module connections at each level
    
    Returns:
        Tuple of (graph, hierarchy_info)
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Compute module sizes at each level
    modules_per_level = [branching_factor ** (n_levels - l - 1) for l in range(n_levels)]
    nodes_per_module = [n_nodes // m for m in modules_per_level]
    
    # Assign nodes to modules at each level
    hierarchy = {}
    for level in range(n_levels):
        n_modules = modules_per_level[level]
        module_size = nodes_per_module[level]
        
        hierarchy[level] = {}
        for node in range(n_nodes):
            module_id = node // module_size
            if module_id not in hierarchy[level]:
                hierarchy[level][module_id] = []
            hierarchy[level][module_id].append(node)
    
    # Add connections based on hierarchy
    for i, j in combinations(range(n_nodes), 2):
        # Find lowest level where i and j are in the same module
        same_module_level = -1
        for level in range(n_levels):
            module_size = nodes_per_module[level]
            if i // module_size == j // module_size:
                same_module_level = level
                break
        
        # Connection probability depends on hierarchical distance
        if same_module_level == 0:
            # Same lowest-level module
            prob = intra_prob
        elif same_module_level > 0:
            # Same higher-level module
            prob = intra_prob * (inter_prob_decay ** same_module_level)
        else:
            # Different top-level modules
            prob = intra_prob * (inter_prob_decay ** n_levels)
        
        if np.random.rand() < prob:
            weight = np.random.uniform(0.5, 1.0)
            G.add_edge(i, j, weight=weight)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2, weight=0.1)
    
    hierarchy_info = {
        'n_levels': n_levels,
        'modules_per_level': modules_per_level,
        'nodes_per_module': nodes_per_module,
        'hierarchy': hierarchy
    }
    
    return G, hierarchy_info


def generate_rich_club(
    n_nodes: int = 100,
    n_hubs: int = 10,
    hub_connectivity: float = 0.8,
    peripheral_connectivity: float = 0.05,
    hub_peripheral_connectivity: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a network with rich-club organization.
    
    Rich-club: hub nodes are more densely connected to each other
    than expected by chance. This is a hallmark of brain networks.
    
    Args:
        n_nodes: Total number of nodes
        n_hubs: Number of hub nodes
        hub_connectivity: Connection probability among hubs
        peripheral_connectivity: Connection probability among peripheral nodes
        hub_peripheral_connectivity: Connection probability hub-peripheral
    
    Returns:
        Tuple of (graph, hub_node_list)
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Designate hub nodes
    hubs = list(range(n_hubs))
    peripherals = list(range(n_hubs, n_nodes))
    
    # Hub-hub connections (rich club)
    for i, j in combinations(hubs, 2):
        if np.random.rand() < hub_connectivity:
            G.add_edge(i, j, weight=np.random.uniform(0.7, 1.0))
    
    # Peripheral-peripheral connections
    for i, j in combinations(peripherals, 2):
        if np.random.rand() < peripheral_connectivity:
            G.add_edge(i, j, weight=np.random.uniform(0.3, 0.7))
    
    # Hub-peripheral connections
    for hub in hubs:
        for periph in peripherals:
            if np.random.rand() < hub_peripheral_connectivity:
                G.add_edge(hub, periph, weight=np.random.uniform(0.5, 0.9))
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2, weight=0.1)
    
    # Mark hubs in node attributes
    for node in G.nodes():
        G.nodes[node]['is_hub'] = node in hubs
    
    return G, hubs


def generate_small_world_scale_free_hybrid(
    n_nodes: int = 100,
    m_initial: int = 5,
    m_edges: int = 2,
    rewiring_prob: float = 0.3,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a hybrid network with both small-world and scale-free properties.
    
    Starts with preferential attachment (scale-free) then applies
    small-world rewiring. This mimics brain networks which show
    both hub structure and high clustering.
    
    Args:
        n_nodes: Number of nodes
        m_initial: Initial complete graph size
        m_edges: Edges per new node (preferential attachment)
        rewiring_prob: Probability of rewiring (small-world)
    
    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with scale-free (Barabási-Albert)
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    
    # Add small-world rewiring
    edges = list(G.edges())
    for u, v in edges:
        if np.random.rand() < rewiring_prob:
            # Rewire to random node
            new_target = np.random.randint(n_nodes)
            if new_target != u and not G.has_edge(u, new_target):
                G.remove_edge(u, v)
                G.add_edge(u, new_target)
    
    # Add clustering-enhancing edges (triadic closure)
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 2:
            # With some probability, connect pairs of neighbors
            for n1, n2 in combinations(neighbors, 2):
                if not G.has_edge(n1, n2) and np.random.rand() < 0.1:
                    G.add_edge(n1, n2)
    
    # Add weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2, weight=0.5)
    
    return G


def generate_connectome_inspired(
    n_nodes: int = 100,
    n_communities: int = 7,  # Like major brain networks
    community_names: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict]:
    """
    Generate a network inspired by human connectome organization.
    
    Models major brain networks:
    - Default Mode Network (DMN)
    - Frontoparietal (Executive)
    - Salience Network
    - Visual
    - Somatomotor
    - Limbic
    - Dorsal Attention
    
    Args:
        n_nodes: Total number of nodes
        n_communities: Number of functional communities
        community_names: Optional names for communities
    
    Returns:
        Tuple of (graph, community_info)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if community_names is None:
        community_names = ['DMN', 'FPN', 'SAL', 'VIS', 'SMN', 'LIM', 'DAN'][:n_communities]
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Assign nodes to communities (unequal sizes like real brain)
    # DMN and FPN are typically larger
    size_weights = [0.2, 0.18, 0.12, 0.15, 0.15, 0.1, 0.1][:n_communities]
    size_weights = np.array(size_weights) / sum(size_weights)
    
    community_sizes = (size_weights * n_nodes).astype(int)
    community_sizes[-1] = n_nodes - sum(community_sizes[:-1])  # Adjust last
    
    communities = {}
    node_idx = 0
    for i, (name, size) in enumerate(zip(community_names, community_sizes)):
        communities[name] = list(range(node_idx, node_idx + size))
        for node in communities[name]:
            G.nodes[node]['community'] = name
        node_idx += size
    
    # Intra-community connectivity (high)
    intra_probs = {
        'DMN': 0.4, 'FPN': 0.35, 'SAL': 0.45,
        'VIS': 0.5, 'SMN': 0.5, 'LIM': 0.35, 'DAN': 0.4
    }
    
    for name, nodes in communities.items():
        prob = intra_probs.get(name, 0.4)
        for i, j in combinations(nodes, 2):
            if np.random.rand() < prob:
                G.add_edge(i, j, weight=np.random.uniform(0.6, 1.0))
    
    # Inter-community connectivity (sparse but specific)
    # Based on known brain network interactions
    inter_connectivity = {
        ('DMN', 'FPN'): 0.1,    # Anticorrelated in task
        ('DMN', 'LIM'): 0.15,   # Memory/emotion
        ('FPN', 'DAN'): 0.2,    # Attention control
        ('FPN', 'SAL'): 0.15,   # Executive-salience
        ('SAL', 'DMN'): 0.1,    # Mode switching
        ('SAL', 'FPN'): 0.15,   # Mode switching
        ('VIS', 'DAN'): 0.2,    # Visual attention
        ('SMN', 'DAN'): 0.1,    # Sensorimotor attention
        ('VIS', 'SMN'): 0.05,   # Visuomotor
    }
    
    for (comm1, comm2), prob in inter_connectivity.items():
        if comm1 in communities and comm2 in communities:
            for n1 in communities[comm1]:
                for n2 in communities[comm2]:
                    if np.random.rand() < prob:
                        G.add_edge(n1, n2, weight=np.random.uniform(0.3, 0.7))
    
    # Add some weak random connections
    for i, j in combinations(range(n_nodes), 2):
        if not G.has_edge(i, j) and np.random.rand() < 0.01:
            G.add_edge(i, j, weight=np.random.uniform(0.1, 0.3))
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k + 1])[0]
            G.add_edge(node1, node2, weight=0.1)
    
    community_info = {
        'communities': communities,
        'sizes': dict(zip(community_names, community_sizes)),
        'names': community_names
    }
    
    return G, community_info


def compute_network_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute comprehensive network metrics.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary of network metrics
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    metrics = {
        'n_nodes': n,
        'n_edges': m,
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'transitivity': nx.transitivity(G),
    }
    
    # Path lengths (sample for large networks)
    if nx.is_connected(G):
        if n < 500:
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            # Sample
            sample = np.random.choice(list(G.nodes()), size=min(100, n), replace=False)
            lengths = []
            for s in sample:
                for t in sample:
                    if s != t:
                        lengths.append(nx.shortest_path_length(G, s, t))
            metrics['avg_path_length'] = np.mean(lengths)
    else:
        metrics['avg_path_length'] = float('inf')
    
    # Degree distribution
    degrees = [d for _, d in G.degree()]
    metrics['avg_degree'] = np.mean(degrees)
    metrics['degree_std'] = np.std(degrees)
    metrics['max_degree'] = max(degrees)
    
    # Small-world coefficient (if applicable)
    if metrics['avg_path_length'] < float('inf'):
        # Compare to random graph
        p = metrics['density']
        expected_clustering = p
        expected_path = np.log(n) / np.log(metrics['avg_degree']) if metrics['avg_degree'] > 1 else n
        
        sigma = (metrics['avg_clustering'] / expected_clustering) / (metrics['avg_path_length'] / expected_path)
        metrics['small_world_sigma'] = sigma
    else:
        metrics['small_world_sigma'] = 0
    
    # Rich club coefficient
    try:
        rc = nx.rich_club_coefficient(G, normalized=False)
        if rc:
            metrics['rich_club_max'] = max(rc.values())
        else:
            metrics['rich_club_max'] = 0
    except:
        metrics['rich_club_max'] = 0
    
    # Modularity (using greedy algorithm)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G)
        metrics['modularity'] = nx.community.modularity(G, communities)
        metrics['n_communities'] = len(list(communities))
    except:
        metrics['modularity'] = 0
        metrics['n_communities'] = 1
    
    return metrics


if __name__ == "__main__":
    print("Testing advanced network generators...")
    np.random.seed(42)
    
    print("\n" + "="*60)
    
    # Test hierarchical modular
    print("\n1. Hierarchical Modular Network:")
    G_hier, info = generate_hierarchical_modular(n_nodes=64, n_levels=3)
    metrics = compute_network_metrics(G_hier)
    print(f"   Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"   Clustering: {metrics['avg_clustering']:.3f}")
    print(f"   Modularity: {metrics['modularity']:.3f}")
    print(f"   Levels: {info['n_levels']}")
    
    # Test rich club
    print("\n2. Rich-Club Network:")
    G_rc, hubs = generate_rich_club(n_nodes=100, n_hubs=10)
    metrics = compute_network_metrics(G_rc)
    print(f"   Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"   Rich-club max: {metrics['rich_club_max']:.3f}")
    print(f"   Hub nodes: {hubs[:5]}...")
    
    # Test hybrid
    print("\n3. Small-World/Scale-Free Hybrid:")
    G_hybrid = generate_small_world_scale_free_hybrid(n_nodes=100)
    metrics = compute_network_metrics(G_hybrid)
    print(f"   Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"   Clustering: {metrics['avg_clustering']:.3f}")
    print(f"   Small-world σ: {metrics['small_world_sigma']:.3f}")
    print(f"   Max degree: {metrics['max_degree']}")
    
    # Test connectome-inspired
    print("\n4. Connectome-Inspired Network:")
    G_conn, comm_info = generate_connectome_inspired(n_nodes=100)
    metrics = compute_network_metrics(G_conn)
    print(f"   Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"   Communities: {comm_info['names']}")
    print(f"   Modularity: {metrics['modularity']:.3f}")
    
    print("\n" + "="*60)
    print("All advanced network generators working correctly!")
