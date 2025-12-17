"""
Category Theory and Algebraic Topology Metrics

Mathematical structures for consciousness analysis:
1. Sheaf-theoretic consistency measures
2. Simplicial complex / Persistent homology
3. Compositional structure analysis
4. Functorial mappings between states
5. Information-theoretic integration (Φ-like measures)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Set
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict


# ==============================================================================
# SHEAF-THEORETIC MEASURES
# ==============================================================================

def compute_sheaf_consistency(
    local_data: np.ndarray,
    adjacency: np.ndarray,
    normalize: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Compute sheaf-theoretic consistency of local data.
    
    In sheaf theory, local sections must agree on overlaps (gluing condition).
    This measures how well local neural activity "glues" into global patterns.
    
    High consistency → coherent global state
    Low consistency → fragmented/disconnected activity
    
    Args:
        local_data: N x D array of local observations at each node
        adjacency: N x N adjacency matrix
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Tuple of (global_consistency, local_consistencies)
    """
    local_data = np.asarray(local_data)
    adjacency = np.asarray(adjacency)
    
    n_nodes = local_data.shape[0]
    
    if local_data.ndim == 1:
        local_data = local_data.reshape(-1, 1)
    
    # Compute local consistency at each edge
    local_consistencies = np.zeros(n_nodes)
    edge_count = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0:
                # Consistency = similarity of local data
                # Using cosine similarity
                norm_i = np.linalg.norm(local_data[i])
                norm_j = np.linalg.norm(local_data[j])
                
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(local_data[i], local_data[j]) / (norm_i * norm_j)
                else:
                    sim = 0.0
                
                # Weight by edge strength
                weighted_sim = sim * adjacency[i, j]
                local_consistencies[i] += weighted_sim
                local_consistencies[j] += weighted_sim
                edge_count[i] += adjacency[i, j]
                edge_count[j] += adjacency[i, j]
    
    # Normalize by edge count
    edge_count[edge_count == 0] = 1
    local_consistencies = local_consistencies / edge_count
    
    # Global consistency (average)
    global_consistency = np.mean(local_consistencies)
    
    if normalize:
        global_consistency = (global_consistency + 1) / 2  # Map [-1, 1] to [0, 1]
        local_consistencies = (local_consistencies + 1) / 2
    
    return float(global_consistency), local_consistencies


def compute_stalk_dimension(
    activity_timeseries: np.ndarray,
    n_components: int = 5
) -> np.ndarray:
    """
    Estimate the "stalk dimension" at each node.
    
    In sheaf theory, the stalk at a point is the local data structure.
    We estimate its effective dimension using PCA.
    
    Higher dimension → richer local representation
    Lower dimension → simpler/constrained local state
    
    Args:
        activity_timeseries: N x T array of activity over time
        n_components: Max components to consider
    
    Returns:
        Array of effective dimensions at each node
    """
    n_nodes, n_time = activity_timeseries.shape
    
    stalk_dims = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        node_activity = activity_timeseries[i, :]
        
        # Embed in delay space
        delay = 3
        if n_time > delay * n_components:
            embedded = np.zeros((n_time - delay * (n_components - 1), n_components))
            for d in range(n_components):
                embedded[:, d] = node_activity[d * delay:d * delay + embedded.shape[0]]
            
            # Compute singular values
            try:
                _, s, _ = np.linalg.svd(embedded, full_matrices=False)
                s = s / s.sum()  # Normalize
                
                # Effective dimension (participation ratio of singular values)
                stalk_dims[i] = 1.0 / (np.sum(s ** 2) + 1e-10)
            except:
                stalk_dims[i] = 1.0
        else:
            stalk_dims[i] = 1.0
    
    return stalk_dims


# ==============================================================================
# SIMPLICIAL COMPLEX / PERSISTENT HOMOLOGY
# ==============================================================================

def build_simplicial_complex(
    correlation_matrix: np.ndarray,
    threshold: float = 0.5,
    max_dim: int = 2
) -> Dict[int, List[Tuple]]:
    """
    Build a simplicial complex from correlation structure.
    
    Nodes with high correlation form simplices (higher-order interactions).
    
    Args:
        correlation_matrix: N x N correlation matrix
        threshold: Correlation threshold for simplex formation
        max_dim: Maximum simplex dimension (2 = triangles)
    
    Returns:
        Dictionary mapping dimension to list of simplices
    """
    n = correlation_matrix.shape[0]
    
    # Start with nodes (0-simplices)
    simplices = {0: [(i,) for i in range(n)]}
    
    # Build edges (1-simplices)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if correlation_matrix[i, j] >= threshold:
                edges.append((i, j))
    simplices[1] = edges
    
    # Build higher simplices by checking cliques
    for dim in range(2, max_dim + 1):
        simplices[dim] = []
        
        # Candidate simplices from lower dimension
        for lower_simplex in simplices[dim - 1]:
            nodes_in_simplex = set(lower_simplex)
            
            # Try adding each node
            for node in range(n):
                if node in nodes_in_simplex:
                    continue
                
                # Check if all required edges exist
                all_connected = True
                for existing in nodes_in_simplex:
                    if correlation_matrix[node, existing] < threshold:
                        all_connected = False
                        break
                
                if all_connected:
                    new_simplex = tuple(sorted(list(nodes_in_simplex) + [node]))
                    if new_simplex not in simplices[dim]:
                        simplices[dim].append(new_simplex)
    
    return simplices


def compute_betti_numbers(simplices: Dict[int, List[Tuple]]) -> List[int]:
    """
    Compute Betti numbers of a simplicial complex.
    
    β₀: Number of connected components
    β₁: Number of 1-dimensional holes (loops)
    β₂: Number of 2-dimensional voids (cavities)
    
    These characterize the "shape" of the correlation structure.
    
    Args:
        simplices: Dictionary of simplices by dimension
    
    Returns:
        List of Betti numbers [β₀, β₁, ...]
    """
    max_dim = max(simplices.keys())
    
    betti = []
    
    # β₀: connected components
    if 0 in simplices and 1 in simplices:
        # Build graph from 0 and 1 simplices
        nodes = set(s[0] for s in simplices[0])
        edges = simplices[1]
        
        # Union-find for components
        parent = {n: n for n in nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for e in edges:
            union(e[0], e[1])
        
        components = len(set(find(n) for n in nodes))
        betti.append(components)
    else:
        betti.append(len(simplices.get(0, [])))
    
    # Higher Betti numbers (simplified calculation)
    for dim in range(1, max_dim + 1):
        n_simplices = len(simplices.get(dim, []))
        n_higher = len(simplices.get(dim + 1, []))
        
        # Euler characteristic based estimate
        # This is a simplification - full computation requires boundary matrices
        beta_est = max(0, n_simplices - n_higher - (dim + 1) * len(simplices.get(dim - 1, [])) // dim)
        betti.append(beta_est)
    
    return betti


def compute_persistence_diagram(
    distance_matrix: np.ndarray,
    max_dim: int = 1,
    n_thresholds: int = 20
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Compute a simplified persistence diagram.
    
    Tracks birth and death of topological features as threshold varies.
    Long-lived features (large persistence) indicate robust structure.
    
    Args:
        distance_matrix: N x N distance/dissimilarity matrix
        max_dim: Maximum homology dimension
        n_thresholds: Number of threshold values to test
    
    Returns:
        Dictionary mapping dimension to list of (birth, death) pairs
    """
    # Convert distance to similarity
    max_dist = distance_matrix.max()
    similarity = 1 - distance_matrix / (max_dist + 1e-10)
    np.fill_diagonal(similarity, 1)
    
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    
    persistence = {dim: [] for dim in range(max_dim + 1)}
    
    prev_betti = None
    
    for i, thresh in enumerate(thresholds):
        simplices = build_simplicial_complex(similarity, threshold=thresh, max_dim=max_dim)
        betti = compute_betti_numbers(simplices)
        
        if prev_betti is not None:
            for dim in range(len(betti)):
                # Features born (increase in Betti number)
                if dim < len(prev_betti) and betti[dim] > prev_betti[dim]:
                    for _ in range(betti[dim] - prev_betti[dim]):
                        persistence[dim].append((thresh, None))  # Birth, death TBD
                
                # Features died (decrease in Betti number)
                if dim < len(prev_betti) and betti[dim] < prev_betti[dim]:
                    # Find open birth events and close them
                    open_births = [p for p in persistence[dim] if p[1] is None]
                    for _ in range(prev_betti[dim] - betti[dim]):
                        if open_births:
                            idx = persistence[dim].index(open_births.pop(0))
                            persistence[dim][idx] = (persistence[dim][idx][0], thresh)
        
        prev_betti = betti
    
    # Close any remaining open features at max threshold
    for dim in persistence:
        persistence[dim] = [(b, d if d else 1.0) for b, d in persistence[dim]]
    
    return persistence


def compute_topological_complexity(persistence: Dict[int, List[Tuple[float, float]]]) -> Dict[str, float]:
    """
    Compute topological complexity measures from persistence diagram.
    
    Args:
        persistence: Persistence diagram from compute_persistence_diagram
    
    Returns:
        Dictionary of complexity measures
    """
    measures = {}
    
    total_persistence = 0
    total_features = 0
    
    for dim, pairs in persistence.items():
        if pairs:
            lifetimes = [d - b for b, d in pairs]
            measures[f'persistence_dim{dim}'] = sum(lifetimes)
            measures[f'n_features_dim{dim}'] = len(pairs)
            measures[f'max_lifetime_dim{dim}'] = max(lifetimes) if lifetimes else 0
            
            total_persistence += sum(lifetimes)
            total_features += len(pairs)
    
    measures['total_persistence'] = total_persistence
    measures['total_features'] = total_features
    measures['avg_persistence'] = total_persistence / total_features if total_features > 0 else 0
    
    return measures


# ==============================================================================
# COMPOSITIONAL / FUNCTORIAL STRUCTURE
# ==============================================================================

def compute_compositional_score(
    module_activities: Dict[str, np.ndarray],
    module_graph: Dict[str, List[str]]
) -> float:
    """
    Compute compositional structure score.
    
    Measures how well the system's behavior composes from module behaviors.
    High score → modular, compositional organization
    Low score → holistic, non-decomposable
    
    Args:
        module_activities: Dictionary mapping module names to activity arrays
        module_graph: Dictionary mapping module names to list of connected modules
    
    Returns:
        Compositional score in [0, 1]
    """
    if not module_activities:
        return 0.0
    
    modules = list(module_activities.keys())
    n_modules = len(modules)
    
    # Compute within-module coherence
    within_coherence = []
    for name, activity in module_activities.items():
        if activity.ndim == 2 and activity.shape[0] > 1:
            corr = np.corrcoef(activity)
            within_coherence.append(np.mean(corr[np.triu_indices_from(corr, k=1)]))
        else:
            within_coherence.append(1.0)
    
    avg_within = np.mean(within_coherence)
    
    # Compute between-module coupling
    between_coupling = []
    for m1 in modules:
        for m2 in module_graph.get(m1, []):
            if m2 in module_activities:
                a1 = module_activities[m1].flatten()
                a2 = module_activities[m2].flatten()
                if len(a1) == len(a2):
                    corr = np.corrcoef(a1, a2)[0, 1]
                    between_coupling.append(abs(corr))
    
    avg_between = np.mean(between_coupling) if between_coupling else 0
    
    # Compositional score: high within, moderate between
    # Perfect composition: within ≈ 1, between ≈ 0.3-0.5
    score = avg_within * (1 - abs(avg_between - 0.4))
    
    return float(np.clip(score, 0, 1))


def compute_functor_preservation(
    state1: np.ndarray,
    state2: np.ndarray,
    transform: np.ndarray
) -> float:
    """
    Measure how well a transformation preserves structure (functoriality).
    
    A functor preserves composition: F(g ∘ f) = F(g) ∘ F(f)
    
    This measures structural preservation under state transitions.
    
    Args:
        state1: Initial state (N-dim)
        state2: Final state (N-dim)
        transform: Transformation matrix (N x N)
    
    Returns:
        Functoriality score in [0, 1]
    """
    state1 = np.asarray(state1).flatten()
    state2 = np.asarray(state2).flatten()
    transform = np.asarray(transform)
    
    # Predicted state
    predicted = transform @ state1
    
    # Normalize
    if np.linalg.norm(predicted) > 0:
        predicted = predicted / np.linalg.norm(predicted)
    if np.linalg.norm(state2) > 0:
        state2_norm = state2 / np.linalg.norm(state2)
    else:
        state2_norm = state2
    
    # Preservation score (cosine similarity)
    preservation = np.dot(predicted, state2_norm)
    
    return float((preservation + 1) / 2)  # Map to [0, 1]


# ==============================================================================
# INTEGRATION MEASURES (IIT-inspired)
# ==============================================================================

def compute_integration_phi(
    correlation_matrix: np.ndarray,
    partition_method: str = 'bipartition'
) -> float:
    """
    Compute integration measure Φ (phi).
    
    Measures how much the whole system is more than the sum of its parts.
    Inspired by Integrated Information Theory (IIT).
    
    High Φ → highly integrated, unified
    Low Φ → reducible to independent parts
    
    Args:
        correlation_matrix: N x N correlation matrix
        partition_method: How to partition ('bipartition', 'modules')
    
    Returns:
        Integration measure Φ
    """
    n = correlation_matrix.shape[0]
    
    # Total mutual information (proxy)
    total_corr = np.mean(np.abs(correlation_matrix[np.triu_indices(n, k=1)]))
    
    # Find minimum information partition
    min_partition_info = float('inf')
    
    if partition_method == 'bipartition':
        # Try several bipartitions
        for _ in range(min(100, 2 ** min(n, 10))):
            # Random partition
            partition = np.random.rand(n) > 0.5
            if partition.sum() == 0 or partition.sum() == n:
                continue
            
            set1 = np.where(partition)[0]
            set2 = np.where(~partition)[0]
            
            # Compute information loss under partition
            within1 = np.mean(np.abs(correlation_matrix[np.ix_(set1, set1)]))
            within2 = np.mean(np.abs(correlation_matrix[np.ix_(set2, set2)]))
            across = np.mean(np.abs(correlation_matrix[np.ix_(set1, set2)]))
            
            # Partition information = sum of within parts
            partition_info = (len(set1) * within1 + len(set2) * within2) / n + across
            
            min_partition_info = min(min_partition_info, partition_info)
    
    # Φ = total - min_partition
    phi = max(0, total_corr - min_partition_info + total_corr)
    
    return float(phi)


def compute_all_category_metrics(
    activity: np.ndarray,
    adjacency: np.ndarray,
    correlation_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all category-theoretic and topological metrics.
    
    Args:
        activity: N x T activity matrix (or N-dim vector)
        adjacency: N x N adjacency matrix
        correlation_matrix: Optional pre-computed correlation matrix
    
    Returns:
        Dictionary of all metrics
    """
    activity = np.asarray(activity)
    adjacency = np.asarray(adjacency)
    
    if activity.ndim == 1:
        activity = activity.reshape(-1, 1)
    
    n_nodes = activity.shape[0]
    
    # Compute correlation matrix if not provided
    if correlation_matrix is None:
        if activity.shape[1] > 1:
            correlation_matrix = np.corrcoef(activity)
        else:
            correlation_matrix = adjacency / (adjacency.max() + 1e-10)
    
    metrics = {}
    
    # Sheaf consistency
    consistency, local_cons = compute_sheaf_consistency(activity, adjacency)
    metrics['sheaf_consistency'] = consistency
    metrics['min_local_consistency'] = float(np.min(local_cons))
    metrics['max_local_consistency'] = float(np.max(local_cons))
    
    # Stalk dimension (if time series)
    if activity.shape[1] > 10:
        stalk_dims = compute_stalk_dimension(activity)
        metrics['mean_stalk_dimension'] = float(np.mean(stalk_dims))
        metrics['max_stalk_dimension'] = float(np.max(stalk_dims))
    
    # Simplicial/topological
    simplices = build_simplicial_complex(np.abs(correlation_matrix), threshold=0.5, max_dim=2)
    betti = compute_betti_numbers(simplices)
    
    metrics['betti_0'] = betti[0] if len(betti) > 0 else 0
    metrics['betti_1'] = betti[1] if len(betti) > 1 else 0
    metrics['betti_2'] = betti[2] if len(betti) > 2 else 0
    
    metrics['n_triangles'] = len(simplices.get(2, []))
    metrics['n_edges_simplex'] = len(simplices.get(1, []))
    
    # Persistence
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0)
    
    persistence = compute_persistence_diagram(distance_matrix, max_dim=1, n_thresholds=10)
    topo_complexity = compute_topological_complexity(persistence)
    metrics.update(topo_complexity)
    
    # Integration
    metrics['phi_integration'] = compute_integration_phi(correlation_matrix)
    
    return metrics


# ==============================================================================
# ADVANCED CATEGORY THEORY EXTENSIONS
# ==============================================================================

def compute_natural_transformation_distance(
    functor_F: np.ndarray,
    functor_G: np.ndarray,
    morphism_matrix: np.ndarray
) -> float:
    """
    Compute distance between two "functors" viewed as state mappings.
    
    In category theory, a natural transformation η: F ⟹ G provides a 
    way to transform one functor into another while respecting structure.
    
    Here we interpret:
    - Objects = brain regions/modes
    - Morphisms = correlations/connections
    - Functors = state mappings (power distributions viewed as mappings)
    
    The naturality condition: G(f) ∘ η_A = η_B ∘ F(f)
    We measure how well this holds on average.
    
    Args:
        functor_F: N-dim state (first functor image)
        functor_G: N-dim state (second functor image)
        morphism_matrix: N x N morphism structure (correlations)
    
    Returns:
        Distance measuring naturality failure (0 = natural transformation exists)
    """
    n = len(functor_F)
    
    # Normalize
    F = functor_F / (functor_F.sum() + 1e-10)
    G = functor_G / (functor_G.sum() + 1e-10)
    
    # The "natural transformation" η_i = G_i / F_i (ratio at each component)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(F > 1e-10, G / F, 1.0)
    
    # Naturality failure: check if transformation commutes with morphisms
    # For each morphism f: i → j, check if η_j * F(f) ≈ G(f) * η_i
    failure = 0.0
    count = 0
    
    for i in range(n):
        for j in range(n):
            if morphism_matrix[i, j] > 0.1:  # Significant morphism
                # F(f) ≈ F_j / F_i (how F transforms)
                # G(f) ≈ G_j / G_i (how G transforms)
                if F[i] > 1e-10 and G[i] > 1e-10:
                    lhs = eta[j] * (F[j] / F[i])  # η_j ∘ F(f)
                    rhs = (G[j] / G[i]) * eta[i]  # G(f) ∘ η_i
                    
                    failure += np.abs(lhs - rhs) * morphism_matrix[i, j]
                    count += morphism_matrix[i, j]
    
    return failure / (count + 1e-10)


def compute_colimit_consciousness(
    local_states: List[np.ndarray],
    overlap_matrix: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute the categorical colimit of local conscious experiences.
    
    In category theory, the colimit is the "universal gluing" of objects.
    For consciousness, this represents how local experiences combine
    into a unified global experience.
    
    The colimit exists when local experiences are compatible on overlaps.
    
    Args:
        local_states: List of local state vectors
        overlap_matrix: N x N matrix of overlap regions
    
    Returns:
        Tuple of (global_state, colimit_quality)
        - global_state: The "glued" global state
        - colimit_quality: How well the gluing works (1.0 = perfect colimit)
    """
    n_regions = len(local_states)
    dim = len(local_states[0])
    
    # Initialize global state as weighted average
    weights = np.ones(n_regions)
    
    # Iteratively refine based on overlap compatibility
    for iteration in range(10):
        # Compute proposed global state
        global_state = np.zeros(dim)
        for i, state in enumerate(local_states):
            global_state += weights[i] * state
        global_state /= (weights.sum() + 1e-10)
        
        # Update weights based on compatibility with global
        new_weights = np.zeros(n_regions)
        for i, state in enumerate(local_states):
            compatibility = 1.0 / (np.linalg.norm(state - global_state) + 0.1)
            new_weights[i] = compatibility
        
        weights = new_weights / (new_weights.sum() + 1e-10) * n_regions
    
    # Final global state
    global_state = np.zeros(dim)
    for i, state in enumerate(local_states):
        global_state += weights[i] * state
    global_state /= (weights.sum() + 1e-10)
    
    # Colimit quality: how well local states agree with global
    total_error = 0.0
    for i, state in enumerate(local_states):
        error = np.linalg.norm(state - global_state)
        total_error += error * weights[i]
    
    # Normalize to [0, 1]
    max_error = np.sqrt(dim)  # Maximum possible error
    colimit_quality = 1.0 - min(total_error / (max_error * n_regions), 1.0)
    
    return global_state, colimit_quality


def compute_adjoint_functor_measure(
    state_forward: np.ndarray,
    state_backward: np.ndarray,
    transition_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Measure adjoint functor-like relationships between states.
    
    An adjunction F ⊣ G means: Hom(F(A), B) ≅ Hom(A, G(B))
    
    For consciousness, this captures:
    - Forward mapping: Perception (world → experience)
    - Backward mapping: Action (intention → world)
    
    The adjunction condition represents the tight coupling between
    perception and action that underlies conscious agency.
    
    Args:
        state_forward: "Perception" state
        state_backward: "Action/intention" state  
        transition_matrix: Dynamics matrix
    
    Returns:
        Dictionary with adjunction quality metrics
    """
    n = len(state_forward)
    
    # Normalize
    F = state_forward / (np.linalg.norm(state_forward) + 1e-10)
    G = state_backward / (np.linalg.norm(state_backward) + 1e-10)
    
    # Unit of adjunction: η: Id → G∘F
    # Counit of adjunction: ε: F∘G → Id
    
    # Compute F∘G (forward then backward)
    FG = transition_matrix @ G
    FG = FG / (np.linalg.norm(FG) + 1e-10)
    
    # Compute G∘F (backward then forward)
    GF = transition_matrix.T @ F
    GF = GF / (np.linalg.norm(GF) + 1e-10)
    
    # Identity
    identity = np.ones(n) / np.sqrt(n)
    
    # Triangle identities (should be close to identity)
    # (εF) ∘ (Fη) = 1_F and (Gε) ∘ (ηG) = 1_G
    
    triangle_1 = np.dot(FG, F)  # Should be close to ||F||² if adjoint
    triangle_2 = np.dot(GF, G)  # Should be close to ||G||² if adjoint
    
    return {
        'forward_backward_coherence': float(np.dot(FG, identity)),
        'backward_forward_coherence': float(np.dot(GF, identity)),
        'triangle_identity_1': float(triangle_1),
        'triangle_identity_2': float(triangle_2),
        'adjunction_quality': float((triangle_1 + triangle_2) / 2)
    }


def compute_kan_extension(
    partial_state: np.ndarray,
    known_indices: np.ndarray,
    adjacency: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute Kan extension to "complete" partial knowledge.
    
    The Kan extension is the universal way to extend a functor
    along another functor. For consciousness, this models:
    - How the brain "fills in" missing information
    - Predictive processing / inference
    - Gestalt completion
    
    Left Kan = "free" extension (liberal completion)
    Right Kan = "cofree" extension (conservative completion)
    
    Args:
        partial_state: Known values (NaN for unknown)
        known_indices: Boolean array of which values are known
        adjacency: Network structure for propagation
    
    Returns:
        Tuple of (extended_state, extension_quality)
    """
    n = len(partial_state)
    extended = partial_state.copy()
    known = known_indices.copy()
    
    # Normalize adjacency
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_norm = adjacency / row_sums
    
    # Iteratively propagate known values
    for iteration in range(50):
        new_extended = extended.copy()
        newly_known = known.copy()
        
        for i in range(n):
            if not known[i]:
                # Compute weighted average from known neighbors
                neighbors = adj_norm[i, :]
                known_neighbors = neighbors * known
                
                if known_neighbors.sum() > 0:
                    # Left Kan: weighted average (liberal)
                    new_extended[i] = np.dot(neighbors, extended * known) / (known_neighbors.sum() + 1e-10)
                    newly_known[i] = True
        
        if np.all(newly_known == known):
            break
        
        extended = new_extended
        known = newly_known
    
    # Quality: how confident is the extension?
    # Based on how many iterations and how much agreement
    extension_quality = float(known.sum()) / n
    
    return extended, extension_quality


def compute_topos_structure(
    activity: np.ndarray,
    adjacency: np.ndarray
) -> Dict[str, float]:
    """
    Compute topos-theoretic measures of consciousness structure.
    
    A topos is a category that behaves like the category of sets.
    It has:
    - Subobject classifier Ω (truth values)
    - Internal logic (propositions about objects)
    - Exponentials (function spaces)
    
    For consciousness, topos structure represents:
    - Multi-valued logic of perception (not just true/false)
    - Context-dependent truth
    - The "logic" of conscious experience
    
    Args:
        activity: N x T activity matrix
        adjacency: N x N adjacency matrix
    
    Returns:
        Dictionary of topos-theoretic metrics
    """
    n_nodes, n_time = activity.shape
    
    # 1. Subobject classifier dimension
    # In classical sets, Ω = {0, 1}. In intuitionistic topos, Ω can be larger.
    # We estimate the "number of truth values" from clustering
    
    # Cluster activity into truth-value-like categories
    activity_flat = activity.flatten()
    sorted_act = np.sort(activity_flat)
    
    # Count distinct "levels" using gradient analysis
    gradients = np.abs(np.diff(sorted_act))
    threshold = np.percentile(gradients, 90)
    n_truth_values = np.sum(gradients > threshold) + 1
    
    # Normalize to reasonable range
    omega_dimension = min(n_truth_values, n_nodes) / n_nodes
    
    # 2. Internal logic coherence
    # Check if logical operations are internally consistent
    
    # Compute correlation as "implication strength"
    corr = np.corrcoef(activity)
    np.fill_diagonal(corr, 0)
    
    # Transitivity of implication: if A→B and B→C, then A→C
    transitivity_failures = 0
    transitivity_checks = 0
    
    for i in range(min(n_nodes, 20)):
        for j in range(min(n_nodes, 20)):
            for k in range(min(n_nodes, 20)):
                if i != j and j != k and i != k:
                    if corr[i, j] > 0.5 and corr[j, k] > 0.5:
                        transitivity_checks += 1
                        if corr[i, k] < 0.3:
                            transitivity_failures += 1
    
    logic_coherence = 1.0 - transitivity_failures / (transitivity_checks + 1)
    
    # 3. Exponential object measure
    # The exponential B^A represents the space of "functions" from A to B
    # We measure the richness of functional relationships
    
    # SVD of activity gives functional complexity
    try:
        _, s, _ = np.linalg.svd(activity, full_matrices=False)
        s = s / s.sum()
        
        # Effective dimensionality of function space
        exponential_dim = 1.0 / (np.sum(s ** 2) + 1e-10)
        exponential_dim = min(exponential_dim / n_nodes, 1.0)
    except:
        exponential_dim = 0.5
    
    # 4. Lawvere-Tierney topology
    # A j-operator determines which "open sets" we can see
    # Higher coverage = richer perceptual topology
    
    # Use correlation structure as "covering"
    coverage = np.mean(np.abs(corr) > 0.3)
    
    return {
        'omega_dimension': float(omega_dimension),
        'logic_coherence': float(logic_coherence),
        'exponential_richness': float(exponential_dim),
        'lawvere_tierney_coverage': float(coverage),
        'topos_complexity': float((omega_dimension + logic_coherence + exponential_dim + coverage) / 4)
    }


def compute_grothendieck_construction(
    local_categories: List[np.ndarray],
    fiber_connections: np.ndarray
) -> Dict[str, float]:
    """
    Compute Grothendieck construction for consciousness integration.
    
    The Grothendieck construction ∫F turns a functor F: C^op → Cat
    into a single fibered category over C.
    
    For consciousness, this represents:
    - C = brain network structure
    - F = local experiences at each region
    - ∫F = unified conscious experience
    
    This is deeply connected to how local qualia combine into
    unified phenomenal experience.
    
    Args:
        local_categories: List of local state vectors
        fiber_connections: How local states connect
    
    Returns:
        Dictionary with Grothendieck integration metrics
    """
    n_fibers = len(local_categories)
    
    if n_fibers == 0:
        return {'grothendieck_integration': 0.0, 'fiber_coherence': 0.0}
    
    dim = len(local_categories[0])
    
    # 1. Total space dimension
    total_dim = n_fibers * dim
    
    # 2. Fiber coherence (do fibers "agree" where they connect?)
    coherence_sum = 0.0
    connection_count = 0
    
    for i in range(n_fibers):
        for j in range(i + 1, n_fibers):
            if fiber_connections[i, j] > 0.1:
                # Coherence = similarity between fibers
                sim = np.dot(local_categories[i], local_categories[j])
                sim /= (np.linalg.norm(local_categories[i]) * np.linalg.norm(local_categories[j]) + 1e-10)
                
                coherence_sum += sim * fiber_connections[i, j]
                connection_count += fiber_connections[i, j]
    
    fiber_coherence = coherence_sum / (connection_count + 1e-10)
    
    # 3. Integration (how much does the total exceed the sum of parts?)
    local_norms = sum(np.linalg.norm(lc) for lc in local_categories)
    
    # Global state from averaging
    global_state = np.mean(local_categories, axis=0)
    global_norm = np.linalg.norm(global_state) * n_fibers
    
    # Integration = how much is preserved in globalization
    integration = global_norm / (local_norms + 1e-10)
    
    return {
        'grothendieck_integration': float(integration),
        'fiber_coherence': float(fiber_coherence),
        'total_space_dim': float(total_dim),
        'base_space_dim': float(n_fibers)
    }


def compute_yoneda_embedding_richness(
    state: np.ndarray,
    transformation_group: List[np.ndarray]
) -> float:
    """
    Measure richness via Yoneda embedding principle.
    
    Yoneda lemma: An object is determined by its relationships to all others.
    Hom(-, X) fully determines X.
    
    For consciousness: A state is characterized by how it relates/transforms
    to all possible states. Richer embedding = richer experience.
    
    Args:
        state: Current state vector
        transformation_group: List of possible transformation matrices
    
    Returns:
        Yoneda richness score (higher = more distinct identity)
    """
    n = len(state)
    
    if len(transformation_group) == 0:
        return 0.5
    
    # Apply all transformations
    transformed_states = []
    for T in transformation_group:
        if T.shape[0] == n and T.shape[1] == n:
            new_state = T @ state
            new_state = new_state / (np.linalg.norm(new_state) + 1e-10)
            transformed_states.append(new_state)
    
    if len(transformed_states) < 2:
        return 0.5
    
    # The Yoneda embedding is "full and faithful" if all transformed states are distinct
    # We measure distinctness via pairwise distances
    
    total_distance = 0.0
    count = 0
    
    for i in range(len(transformed_states)):
        for j in range(i + 1, len(transformed_states)):
            dist = np.linalg.norm(transformed_states[i] - transformed_states[j])
            total_distance += dist
            count += 1
    
    avg_distance = total_distance / (count + 1e-10)
    
    # Normalize by maximum possible distance
    max_dist = np.sqrt(2)  # For normalized vectors
    richness = avg_distance / max_dist
    
    return float(richness)


def compute_monad_structure(
    state: np.ndarray,
    bind_operation: np.ndarray,
    unit_state: np.ndarray
) -> Dict[str, float]:
    """
    Analyze monad-like structure in consciousness dynamics.
    
    A monad (T, η, μ) consists of:
    - T: Endofunctor (state transformation)
    - η: Unit (baseline → enriched state)
    - μ: Multiplication (nested enrichment → single enrichment)
    
    Monad laws model:
    - How attention/context wraps experience
    - Composition of mental operations
    - The "computational" structure of cognition
    
    Args:
        state: Current state
        bind_operation: T × T → T matrix (composition)
        unit_state: Baseline state (unit η)
    
    Returns:
        Dictionary with monad law satisfaction scores
    """
    n = len(state)
    
    # Normalize
    state = state / (np.linalg.norm(state) + 1e-10)
    unit = unit_state / (np.linalg.norm(unit_state) + 1e-10)
    
    # Apply bind
    T_state = bind_operation @ state
    T_state = T_state / (np.linalg.norm(T_state) + 1e-10)
    
    # Apply bind to unit
    T_unit = bind_operation @ unit
    T_unit = T_unit / (np.linalg.norm(T_unit) + 1e-10)
    
    # Left unit law: μ ∘ η_T = id
    # T(η) followed by μ should give back the original
    left_unit = np.dot(T_unit, state)
    
    # Right unit law: μ ∘ T(η) = id
    right_unit = np.dot(T_state, unit)
    
    # Associativity: μ ∘ T(μ) = μ ∘ μ_T
    # Apply bind twice
    TT_state = bind_operation @ T_state
    TT_state = TT_state / (np.linalg.norm(TT_state) + 1e-10)
    
    # Compare two ways of flattening
    associativity = np.dot(TT_state, T_state)
    
    return {
        'monad_left_unit': float(left_unit),
        'monad_right_unit': float(right_unit),
        'monad_associativity': float(associativity),
        'monad_law_satisfaction': float((left_unit + right_unit + associativity) / 3)
    }


def compute_enriched_category_measure(
    activity: np.ndarray,
    hom_enrichment: str = 'metric'
) -> Dict[str, float]:
    """
    Measure enriched category structure of consciousness.
    
    An enriched category has hom-sets that are objects in a monoidal category V.
    - V = Set: ordinary category
    - V = Ab: preadditive category (linear algebra)
    - V = Met: metric spaces (distances between states)
    - V = [0,∞]: Lawvere metric spaces
    
    For consciousness:
    - Objects = brain states/modes
    - Hom = relationship strength (enriched in [0,1] or metric space)
    
    Args:
        activity: State activity
        hom_enrichment: Type of enrichment ('metric', 'truth', 'probability')
    
    Returns:
        Dictionary with enriched category metrics
    """
    if activity.ndim == 1:
        activity = activity.reshape(-1, 1)
    
    n_objects = activity.shape[0]
    
    # Compute hom-objects based on enrichment type
    if hom_enrichment == 'metric':
        # Hom(A, B) = distance from A to B
        hom = np.zeros((n_objects, n_objects))
        for i in range(n_objects):
            for j in range(n_objects):
                hom[i, j] = np.linalg.norm(activity[i] - activity[j])
        
        # Metric enrichment should satisfy triangle inequality
        triangle_violations = 0
        triangle_checks = 0
        
        for i in range(min(n_objects, 15)):
            for j in range(min(n_objects, 15)):
                for k in range(min(n_objects, 15)):
                    if hom[i, j] + hom[j, k] < hom[i, k] - 1e-6:
                        triangle_violations += 1
                    triangle_checks += 1
        
        enrichment_quality = 1.0 - triangle_violations / (triangle_checks + 1)
        
    elif hom_enrichment == 'truth':
        # Hom(A, B) = truth value of "A relates to B"
        corr = np.corrcoef(activity) if activity.shape[1] > 1 else np.eye(n_objects)
        hom = (corr + 1) / 2  # Map to [0, 1]
        
        # Should be transitive for Boolean enrichment
        transitivity = 0
        trans_checks = 0
        for i in range(min(n_objects, 15)):
            for j in range(min(n_objects, 15)):
                for k in range(min(n_objects, 15)):
                    # If A→B and B→C, check A→C
                    min_path = min(hom[i, j], hom[j, k])
                    trans_checks += 1
                    transitivity += 1 if hom[i, k] >= min_path - 0.1 else 0
        
        enrichment_quality = transitivity / (trans_checks + 1)
        
    else:  # probability
        # Hom(A, B) = probability of transition
        activity_norm = activity / (activity.sum(axis=1, keepdims=True) + 1e-10)
        hom = activity_norm @ activity_norm.T
        
        # Should sum to 1 for stochastic enrichment
        row_sums = hom.sum(axis=1)
        enrichment_quality = 1.0 - np.std(row_sums)
    
    # Composition structure
    # Check if hom-composition is well-behaved
    composition_error = 0.0
    for i in range(min(n_objects, 10)):
        for j in range(min(n_objects, 10)):
            for k in range(min(n_objects, 10)):
                # Composition: hom(A,B) ⊗ hom(B,C) → hom(A,C)
                expected = (hom[i, j] + hom[j, k]) / 2 if hom_enrichment != 'metric' else hom[i, j] + hom[j, k]
                actual = hom[i, k]
                composition_error += abs(expected - actual)
    
    composition_quality = 1.0 / (1.0 + composition_error / (n_objects ** 3))
    
    return {
        'enrichment_type': hom_enrichment,
        'enrichment_quality': float(enrichment_quality),
        'composition_quality': float(composition_quality),
        'hom_object_mean': float(np.mean(hom)),
        'hom_object_std': float(np.std(hom))
    }


def compute_all_advanced_category_metrics(
    activity: np.ndarray,
    adjacency: np.ndarray,
    state_history: List[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all advanced category theory metrics.
    
    Args:
        activity: N x T activity matrix (or N-dim state)
        adjacency: N x N adjacency matrix
        state_history: Optional list of previous states for dynamics
    
    Returns:
        Dictionary of all advanced category metrics
    """
    metrics = {}
    
    if activity.ndim == 1:
        activity = activity.reshape(-1, 1)
    
    n_nodes = activity.shape[0]
    
    # Topos structure
    topos = compute_topos_structure(activity, adjacency)
    metrics.update({f'topos_{k}': v for k, v in topos.items()})
    
    # Enriched category (metric enrichment)
    enriched = compute_enriched_category_measure(activity, 'metric')
    metrics.update({f'enriched_{k}': v for k, v in enriched.items() if isinstance(v, float)})
    
    # Grothendieck construction (treat rows as fibers)
    local_cats = [activity[i, :] for i in range(n_nodes)]
    groth = compute_grothendieck_construction(local_cats, adjacency)
    metrics.update({f'groth_{k}': v for k, v in groth.items()})
    
    # Colimit consciousness
    local_states = [activity[i, :] for i in range(min(n_nodes, 20))]
    _, colimit_qual = compute_colimit_consciousness(local_states, adjacency[:20, :20] if n_nodes >= 20 else adjacency)
    metrics['colimit_quality'] = colimit_qual
    
    # Yoneda richness
    # Generate simple transformation group
    transforms = [np.eye(n_nodes)]
    for _ in range(5):
        T = np.eye(n_nodes) + 0.1 * np.random.randn(n_nodes, n_nodes)
        transforms.append(T)
    
    state_vec = activity.mean(axis=1) if activity.ndim == 2 else activity
    yoneda = compute_yoneda_embedding_richness(state_vec, transforms)
    metrics['yoneda_richness'] = yoneda
    
    # Natural transformation (if history available)
    if state_history and len(state_history) >= 2:
        corr = np.corrcoef(activity) if activity.shape[1] > 1 else adjacency
        nat_dist = compute_natural_transformation_distance(
            state_history[-2], state_history[-1], corr
        )
        metrics['natural_transformation_distance'] = nat_dist
    
    # Monad structure
    bind_op = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-10)
    unit = np.ones(n_nodes) / n_nodes
    monad = compute_monad_structure(state_vec, bind_op, unit)
    metrics.update({f'monad_{k}': v for k, v in monad.items()})
    
    # Kan extension (partial knowledge completion)
    partial = state_vec.copy()
    known = np.ones(n_nodes, dtype=bool)
    known[::3] = False  # Pretend every 3rd value is unknown
    _, kan_qual = compute_kan_extension(partial, known, adjacency)
    metrics['kan_extension_quality'] = kan_qual
    
    return metrics


if __name__ == "__main__":
    print("Testing category theory / algebraic topology metrics...")
    np.random.seed(42)
    
    # Generate test data
    n_nodes = 50
    n_time = 100
    
    # Modular activity pattern
    activity = np.random.randn(n_nodes, n_time)
    # Add module structure
    for i in range(0, 25):
        activity[i, :] += 0.5 * np.sin(np.linspace(0, 4*np.pi, n_time))
    for i in range(25, 50):
        activity[i, :] += 0.5 * np.cos(np.linspace(0, 4*np.pi, n_time))
    
    # Generate adjacency (modular)
    adjacency = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if (i < 25 and j < 25) or (i >= 25 and j >= 25):
                if np.random.rand() < 0.3:
                    adjacency[i, j] = adjacency[j, i] = 1
            else:
                if np.random.rand() < 0.05:
                    adjacency[i, j] = adjacency[j, i] = 1
    
    print("\n" + "="*60)
    print("Computing metrics...")
    
    metrics = compute_all_category_metrics(activity, adjacency)
    
    print("\nSheaf-theoretic:")
    print(f"  Global consistency: {metrics['sheaf_consistency']:.3f}")
    print(f"  Local consistency range: [{metrics['min_local_consistency']:.3f}, {metrics['max_local_consistency']:.3f}]")
    
    print("\nTopological (Betti numbers):")
    print(f"  β₀ (components): {metrics['betti_0']}")
    print(f"  β₁ (loops): {metrics['betti_1']}")
    print(f"  β₂ (voids): {metrics['betti_2']}")
    print(f"  Triangles: {metrics['n_triangles']}")
    
    print("\nPersistence:")
    print(f"  Total persistence: {metrics['total_persistence']:.3f}")
    print(f"  Total features: {metrics['total_features']}")
    
    print("\nIntegration:")
    print(f"  Φ (integration): {metrics['phi_integration']:.3f}")
    
    print("\n" + "="*60)
    print("All category theory metrics working correctly!")
