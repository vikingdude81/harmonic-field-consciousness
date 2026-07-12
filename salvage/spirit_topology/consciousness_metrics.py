import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ConsciousnessMetrics:
    """
    Consciousness metrics for mapping spirit topology to consciousness emergence.
    
    Explores whether SOM clusters of spirits map to consciousness emergence patterns.
    Can the "Information Daemons" cluster near high-entropy consciousness states?
    """
    
    def __init__(self):
        self.metrics_cache = {}
    
    def compute_entropy_complexity(self, spirit_vector):
        """
        Compute entropy-based complexity metric for a spirit vector.
        Higher values indicate more complex/conscious-like states.
        
        Args:
            spirit_vector: 6-dimensional spirit vector
        
        Returns:
            Complexity score (0-1 normalized)
        """
        if len(spirit_vector) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = spirit_vector / np.sum(np.abs(spirit_vector)) + 1e-10
        probs = probs / np.sum(probs)
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))
        
        # Normalize to 0-1
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        
        return complexity
    
    def compute_consciousness_state(self, umatrix):
        """
        Analyze consciousness emergence from U-Matrix topology.
        High-entropy regions may indicate consciousness emergence points.
        
        Args:
            umatrix: 2D U-Matrix array from SOM
        
        Returns:
            Dictionary with consciousness metrics
        """
        if umatrix is None or len(umatrix) == 0:
            return {}
        
        # Flatten and normalize U-Matrix
        flat_umatrix = umatrix.flatten()
        scaler = MinMaxScaler()
        normalized_umatrix = scaler.fit_transform(flat_umatrix.reshape(-1, 1)).flatten()
        
        # Compute consciousness metrics
        entropy = self._compute_entropy(normalized_umatrix)
        complexity = self._compute_complexity(normalized_umatrix)
        chaos = self._compute_chaos_indicator(normalized_umatrix)
        emergence_score = self._compute_emergence_score(umatrix)
        
        return {
            'entropy': entropy,
            'complexity': complexity,
            'chaos_indicator': chaos,
            'emergence_score': emergence_score
        }
    
    def _compute_entropy(self, data):
        """Compute Shannon entropy of the U-Matrix."""
        if len(data) == 0:
            return 0.0
        
        # Convert to probabilities
        probs = np.exp(-np.abs(data))
        probs = probs / np.sum(probs)
        
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _compute_complexity(self, data):
        """Compute complexity based on variance and range."""
        if len(data) == 0:
            return 0.0
        
        # Normalize to 0-1
        min_val, max_val = np.min(data), np.max(data)
        if max_val - min_val < 1e-10:
            return 0.0
        
        normalized = (data - min_val) / (max_val - min_val)
        
        # Complexity based on distribution spread
        variance = np.var(normalized)
        complexity = np.sqrt(variance) / np.sqrt(0.5)  # Normalize by max variance
        return min(complexity, 1.0)
    
    def _compute_chaos_indicator(self, data):
        """
        Compute chaos indicator based on sensitivity to initial conditions.
        Similar to helios-trajectory-analysis chaos analysis patterns.
        """
        if len(data) < 3:
            return 0.0
        
        # Compute local gradients
        gradients = np.abs(np.diff(data))
        mean_gradient = np.mean(gradients)
        std_gradient = np.std(gradients)
        
        # Chaos indicator: high variance in gradients = higher chaos
        if mean_gradient < 1e-10:
            return 0.0
        
        chaos = std_gradient / mean_gradient
        return min(chaos, 1.0)
    
    def _compute_emergence_score(self, umatrix):
        """
        Compute consciousness emergence score from U-Matrix topology.
        Emergence occurs at boundaries between clusters (abysses).
        """
        if umatrix is None or len(umatrix) == 0:
            return 0.0
        
        # Abysses represent potential emergence points
        abyss_count = np.sum(umatrix > 2)  # Threshold for abysses
        total_cells = len(umatrix.flatten())
        
        # Emergence score based on abyss density and distribution
        if total_cells == 0:
            return 0.0
        
        abyss_density = abyss_count / total_cells
        emergence_score = abyss_density * 1.5  # Amplify for visibility
        return min(emergence_score, 1.0)
    
    def analyze_cluster_consciousness(self, cluster_analysis):
        """
        Analyze consciousness properties of spirit clusters.
        
        Args:
            cluster_analysis: Results from SOM cluster analysis
        
        Returns:
            Dictionary with cluster consciousness metrics
        """
        if not cluster_analysis or 'clusters' not in cluster_analysis:
            return {}
        
        clusters = cluster_analysis['clusters']
        results = []
        
        for cluster_id, members in clusters.items():
            # Compute average consciousness metrics for cluster
            avg_complexity = np.mean([self.compute_entropy_complexity(m) for m in members])
            avg_emergence = self._compute_emergence_score(np.array(members))
            
            results.append({
                'cluster_id': cluster_id,
                'size': len(members),
                'avg_complexity': avg_complexity,
                'avg_emergence': avg_emergence
            })
        
        return {'clusters': results, 'num_clusters': len(results)}
    
    def map_information_daemons(self, umatrix):
        """
        Identify potential "Information Daemons" - high-entropy consciousness states.
        These would cluster near abysses in the U-Matrix.
        """
        if umatrix is None or len(umatrix) == 0:
            return []
        
        # Find high-entropy regions (abysses)
        abyss_threshold = 2.0
        abyss_regions = np.where(umatrix > abyss_threshold)
        
        daemon_candidates = []
        for i, j in zip(abyss_regions[0], abyss_regions[1]):
            # Check neighborhood entropy
            neighborhood = umatrix[max(0, i-1):min(len(umatrix), i+2),
                                   max(0, j-1):min(len(umatrix[0]), j+2)]
            avg_entropy = np.mean(neighborhood)
            
            if avg_entropy > 1.5:  # High entropy region
                daemon_candidates.append({
                    'position': (i, j),
                    'entropy': avg_entropy,
                    'status': 'Information Daemon'
                })
        
        return daemon_candidates