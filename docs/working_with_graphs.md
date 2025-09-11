# Working with Graphs

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Graph layers represent general network structures with vertices and edges. Unlike skeletons, graphs can have cycles, multiple connected components, and any connectivity pattern. They're commonly used for level-2 graph data from connectomics pipelines.

!!! note "Shared Features"
    Graphs inherit many common features from the `PointMixin` class. For information about labels, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! info "Graphs vs Skeletons"
    **Graphs** are general network structures that can have cycles and multiple components. **Skeletons** are specialized tree structures (no cycles, single component, with a defined root). See [Working with Skeletons](working_with_skeletons.md) for tree-specific analysis.

## What is a Graph Layer?

A `GraphLayer` contains:
- **Vertices**: 3D coordinates of network nodes
- **Edges**: Connections between vertices (can form cycles)
- **Network properties**: Connectivity analysis, distance calculations between arbitrary vertices

## Creating Graph Layers

### Basic Graph Creation

```python
import numpy as np
import ossify

# Create a graph with a cycle
vertices = np.array([
    [0, 0, 0],    # Vertex 0
    [1, 0, 0],    # Vertex 1
    [2, 0, 0],    # Vertex 2
    [1, 1, 0],    # Vertex 3
])

edges = np.array([
    [0, 1],  # Edge from 0 to 1
    [1, 2],  # Edge from 1 to 2
    [2, 3],  # Edge from 2 to 3
    [3, 0],  # Edge from 3 to 0 (creates cycle)
])

# Add to cell
cell = ossify.Cell(name="graph_example")
cell.add_graph(vertices=vertices, edges=edges)

print(f"Graph has {cell.graph.n_vertices} vertices and {len(cell.graph.edges)} edges")
```

### Graph with Labels

```python
# Add vertex labels during creation
compartment_labels = np.array([0, 1, 1, 0])  # Different compartments

cell.add_graph(
    vertices=vertices,
    edges=edges,
    labels={"compartment": compartment_labels}
)
```

### Multiple Connected Components

```python
# Graph with disconnected components
vertices = np.array([
    [0, 0, 0], [1, 0, 0],    # Component 1
    [5, 0, 0], [6, 0, 0],    # Component 2 (disconnected)
])

edges = np.array([
    [0, 1],  # Component 1
    [2, 3],  # Component 2
    # No edges between components
])

cell.add_graph(vertices=vertices, edges=edges)
```

## Graph-Specific Features

### Edge Access and Connectivity

```python
graph = cell.graph

# Access edge data
edges_array = graph.edges               # Edges with vertex indices
edges_positional = graph.edges_positional  # Edges with positional indices
edge_df = graph.edge_df                # Edges as DataFrame

print(f"Number of edges: {len(graph.edges)}")
```

### Graph Representation as Sparse Matrices

```python
# Different sparse matrix representations for analysis
csgraph = graph.csgraph                    # Weighted by Euclidean distance (directed)
csgraph_binary = graph.csgraph_binary      # Unweighted/binary (directed)
csgraph_undirected = graph.csgraph_undirected  # Weighted (undirected)
csgraph_binary_undirected = graph.csgraph_binary_undirected  # Binary (undirected)

# Use with scipy.sparse.csgraph functions
from scipy.sparse.csgraph import connected_components
n_components, labels = connected_components(graph.csgraph_undirected)
print(f"Number of connected components: {n_components}")
```

### Distance Calculations Between Arbitrary Vertices

```python
# Calculate shortest path distances between any vertices
sources = graph.vertex_index[:2]  # First two vertices
targets = graph.vertex_index[2:4]  # Next two vertices

distances = graph.distance_between(
    sources=sources,
    targets=targets,
    as_positional=False,    # Using vertex indices
    limit=10.0             # Maximum distance to consider
)
print(f"Distance matrix shape: {distances.shape}")
print(f"Distances:\n{distances}")

# Find shortest path between two vertices
path = graph.path_between(
    source=graph.vertex_index[0],
    target=graph.vertex_index[2],
    as_positional=False,
    as_vertices=False      # Return vertex indices, not coordinates
)
print(f"Path from vertex {graph.vertex_index[0]} to {graph.vertex_index[2]}: {path}")
```

### Network Analysis with External Libraries

```python
# Convert to NetworkX for advanced network analysis
import networkx as nx

# Create NetworkX graph from edges
G = nx.from_edgelist(graph.edges)

# Network properties
print(f"Number of connected components: {nx.number_connected_components(G)}")
print(f"Is connected: {nx.is_connected(G)}")
print(f"Average clustering: {nx.average_clustering(G)}")

# Centrality measures
betweenness = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)

# Add centralities as labels
graph.add_label(list(betweenness.values()), name="betweenness")
graph.add_label(list(degree_centrality.values()), name="degree_centrality")
```

### Annotation Aggregation

```python
# Aggregate point annotations to graph vertices
# (requires annotations in the cell)
if "synapses" in cell.annotations.names:
    synapse_counts = graph.map_annotations_to_label(
        annotation="synapses",
        distance_threshold=1.0,     # Distance threshold for aggregation
        agg="count",                # Count synapses near each vertex
        chunk_size=1000,            # Process in chunks for efficiency
        validate=False,             # Skip mapping validation
        agg_direction="undirected"  # Consider all nearby edges
    )
    
    # Add as label
    graph.add_label(synapse_counts, name="synapse_count")

    # Custom aggregation functions
    synapse_stats = graph.map_annotations_to_label(
        annotation="synapses",
        distance_threshold=2.0,
        agg={
            "mean_confidence": ("confidence", "mean"),
            "max_size": ("size", "max"),
            "synapse_density": ("size", lambda x: len(x) / 2.0)  # Custom function
        }
    )
```

### Working with Level-2 Graph Data

```python
# Common pattern for connectomics L2 graphs
# L2 graphs often have many vertices and represent spatial connectivity

# Large graph from connectomics data
import pandas as pd

# Typical L2 vertex data
l2_data = pd.DataFrame({
    'l2_id': [100, 101, 102, 103, 104],
    'rep_coord_nm_x': [1000, 2000, 3000, 1500, 2500],
    'rep_coord_nm_y': [1000, 1000, 1000, 2000, 2000],
    'rep_coord_nm_z': [500, 500, 500, 500, 500],
    'size_nm3': [1e6, 1.5e6, 0.8e6, 1.2e6, 0.9e6]
})

# L2 edges (fewer edges than complete graph)
l2_edges = np.array([
    [100, 101],
    [101, 102], 
    [101, 103],
    [103, 104]
])

cell.add_graph(
    vertices=l2_data,
    edges=l2_edges,
    spatial_columns=['rep_coord_nm_x', 'rep_coord_nm_y', 'rep_coord_nm_z'],
    vertex_index='l2_id',
    labels={'volume': 'size_nm3'}
)

print(f"L2 graph vertex indices: {cell.graph.vertex_index}")
```

## Graph Analysis Patterns

### Connectivity Analysis

```python
# Find isolated vertices (no edges)
degrees = np.array([len(nx.neighbors(G, v)) for v in G.nodes()])
isolated_vertices = graph.vertex_index[degrees == 0]
print(f"Isolated vertices: {isolated_vertices}")

# Find high-degree vertices (hubs)
high_degree_threshold = np.percentile(degrees, 90)
hub_vertices = graph.vertex_index[degrees > high_degree_threshold]
print(f"Hub vertices: {hub_vertices}")
```

### Subgraph Extraction

```python
# Extract subgraph around specific vertices
center_vertices = graph.vertex_index[:3]
subgraph_mask = graph.vertex_index.isin(center_vertices)

# Expand to include neighbors
for center in center_vertices:
    neighbors = list(G.neighbors(center))
    neighbor_mask = graph.vertex_index.isin(neighbors)
    subgraph_mask = subgraph_mask | neighbor_mask

# Create masked subgraph
subgraph = graph.apply_mask(subgraph_mask, as_positional=False)
print(f"Subgraph has {subgraph.n_vertices} vertices")
```

### Distance-based Queries

```python
# Find all vertices within a certain graph distance
center_vertex = graph.vertex_index[0]
max_distance = 3

# Calculate distances from center
distances = graph.distance_between(
    sources=[center_vertex],
    targets=graph.vertex_index,
    as_positional=False,
    limit=max_distance
)

# Find vertices within distance
nearby_mask = distances[0] <= max_distance
nearby_vertices = graph.vertex_index[nearby_mask]
print(f"Vertices within distance {max_distance}: {nearby_vertices}")
```

## Key Graph-Specific Methods

### Graph Creation
- `cell.add_graph(vertices, edges, labels=None, spatial_columns=None, vertex_index=None)` - Add graph to cell

### Graph-Specific Properties
- `graph.edges` - Edge indices with vertex indices
- `graph.edges_positional` - Edge indices with positional indices
- `graph.edge_df` - Edges as DataFrame
- `graph.csgraph` - Weighted sparse graph (directed)
- `graph.csgraph_binary` - Binary sparse graph (directed)
- `graph.csgraph_undirected` - Weighted sparse graph (undirected)  
- `graph.csgraph_binary_undirected` - Binary sparse graph (undirected)

### Network Analysis
- `graph.distance_between(sources, targets, as_positional=False, limit=None)` - Calculate shortest path distances
- `graph.path_between(source, target, as_positional=False, as_vertices=False)` - Find shortest path
- `graph.map_annotations_to_label(annotation, distance_threshold, agg="count", agg_direction="undirected", ...)` - Aggregate annotations

### Integration with External Libraries
- Convert edges to NetworkX: `nx.from_edgelist(graph.edges)`
- Use with scipy.sparse.csgraph functions: `connected_components(graph.csgraph_undirected)`

!!! note "Additional Features"
    For comprehensive information about vertex access, labels, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! tip "When to Use Graphs"
    - Level-2 connectomics data with spatial connectivity
    - Network structures that may have cycles or multiple components  
    - General connectivity analysis between arbitrary vertex pairs
    - Integration with network analysis libraries like NetworkX