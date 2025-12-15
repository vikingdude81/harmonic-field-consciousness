# Category 1: Network Topology Experiments

This category explores how network structure affects consciousness metrics.

## Experiments

### exp1_topology_comparison.py
**Compare consciousness metrics across different network architectures**

Tests:
- Small-world networks (Watts-Strogatz)
- Scale-free networks (Barabási-Albert)
- Random graphs (Erdős-Rényi)
- Lattice graphs (2D grids)
- Modular networks with communities

For each topology:
- Fixed node count (100 nodes) for fair comparison
- Apply all 4 brain states (Wake, NREM, Dream, Anesthesia)
- Calculate all consciousness metrics
- Generate comparison visualizations
- Statistical analysis of differences
- Export results to CSV/JSON

**Output**: `results/exp1_topology_comparison/`

### exp2_network_scaling.py
**Test how network size affects consciousness metrics**

Tests networks of: 50, 100, 200, 500, 1000 nodes
- Use consistent topology (small-world)
- Apply wake state to all sizes
- Track how each metric scales with size
- Test computational efficiency
- Generate scaling plots (log-log if needed)
- Identify optimal network sizes

**Output**: `results/exp2_network_scaling/`

### exp3_hub_disruption.py
**Model lesions by removing hub nodes**

Procedure:
- Create scale-free network with clear hubs
- Identify top hubs by degree, betweenness, eigenvector centrality
- Progressively remove hubs (1%, 5%, 10%, 20%)
- Measure C(t) degradation after each removal
- Compare to random node removal
- Visualize network fragmentation
- Model stroke/TBI effects

**Output**: `results/exp3_hub_disruption/`

### exp4_modular_networks.py
**Test consciousness in modular architectures**

Procedure:
- Generate networks with 2, 4, 6, 8 modules
- Vary inter-module vs intra-module connectivity
- Test integration across modules
- Measure how modularity affects PR and H_mode
- Model hemisphere disconnection
- Visualize community structure and eigenmodes

**Output**: `results/exp4_modular_networks/`

## Running the Experiments

Run all experiments in this category:
```bash
python exp1_topology_comparison.py
python exp2_network_scaling.py
python exp3_hub_disruption.py
python exp4_modular_networks.py
```

Or run from the parent directory:
```bash
cd ..
python category1_network_topology/exp1_topology_comparison.py
```

## Results

All results are saved in the `results/` subdirectory with the following structure:
- CSV files for tabular data
- PNG files for visualizations
- JSON files for metadata and parameters
