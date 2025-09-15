# Working with Skeletons

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Skeleton layers represent tree-structured morphologies with a defined root and no cycles. They're specialized for analyzing branching structures like neuronal arbors, providing tree-specific properties and measurements not available in general graphs.

!!! note "Shared Features"
    Skeletons inherit many common features from the `PointMixin` class. For information about features, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! info "Skeletons vs Graphs"
    **Skeletons** are rooted tree structures (no cycles, single component) with specialized tree analysis capabilities. **Graphs** are general networks that can have cycles and multiple components. See [Working with Graphs](working_with_graphs.md) for general network analysis.

## What is a Skeleton Layer?

A `SkeletonLayer` contains:
- **Vertices**: 3D coordinates representing points along the morphology
- **Edges**: Connections forming a tree structure (no cycles)
- **Root**: A designated root vertex that defines tree orientation
- **Tree properties**: Branch points, end points, paths, distances to root

## Inspecting Skeleton Layers

### Quick Overview with `describe()`

The `describe()` method provides a comprehensive summary of skeleton layers, showing vertex/edge counts, features, and connections to other layers:

```python
# Individual skeleton layer
cell.skeleton.describe()
```

**Output:**
```
# Cell: my_neuron
# Layer: skeleton (SkeletonLayer)
├── 150 vertices, 149 edges
├── features: [radius, branch_type, distance_to_root]
└── Links: mesh <-> skeleton, synapses → skeleton
```

The output shows:
- **Cell context**: Which cell this skeleton belongs to
- **Layer type**: Confirms this is a SkeletonLayer
- **Metrics**: Vertex and edge counts
- **features**: Available data columns beyond spatial coordinates
- **Links**: Connections to other layers (`<->` = bidirectional, `→` = unidirectional)

### Layer Manager Overview

You can also inspect all morphological layers at once:

```python
# All layers in the cell
cell.layers.describe()
```

**Output:**
```
# Layers (3)
├── skeleton (SkeletonLayer)
│   ├── 150 vertices, 149 edges
│   ├── features: [radius, branch_type]
│   └── Links: mesh <-> skeleton, synapses → skeleton
├── mesh (MeshLayer)
│   ├── 2847 vertices, 5691 faces
│   ├── features: [compartment]
│   └── Links: skeleton <-> mesh
└── graph (GraphLayer)
    ├── 45 vertices, 67 edges
    ├── features: []
    └── Links: []
```

This gives you a detailed overview of all morphological layers and their relationships.

## Creating Skeleton Layers

### Basic Skeleton Creation

```python
import numpy as np
import ossify

# Create a branching tree structure
vertices = np.array([
    [0, 0, 0],    # Root (0)
    [1, 0, 0],    # Branch point (1)
    [2, 0, 0],    # End point (2)
    [1, 1, 0],    # End point (3)
    [1, -1, 0],   # End point (4)
])

edges = np.array([
    [0, 1],  # Root to branch point
    [1, 2],  # Branch point to end 1
    [1, 3],  # Branch point to end 2
    [1, 4],  # Branch point to end 3
])

cell = ossify.Cell(name="skeleton_example")
cell.add_skeleton(
    vertices=vertices,
    edges=edges,
    root=0  # Specify root vertex (required for skeletons)
)

print(f"Skeleton has {cell.skeleton.n_vertices} vertices")
print(f"Root location: {cell.skeleton.root_location}")
```

### Skeleton with Morphological features

```python
# Add radius and compartment information
radius_values = np.array([1.0, 0.8, 0.5, 0.5, 0.5])
compartments = np.array([0, 0, 1, 1, 1])  # 0=dendrite, 1=axon

cell.add_skeleton(
    vertices=vertices,
    edges=edges,
    root=0,
    features={
        "radius": radius_values,
        "compartment": compartments
    }
)
```

## Tree Structure Properties

### Root and Topological Points

```python
skeleton = cell.skeleton

# Root information (unique to skeletons)
print(f"Root vertex: {skeleton.root}")
print(f"Root location: {skeleton.root_location}")
print(f"Root positional index: {skeleton.root_positional}")

# Topological points in the tree
print(f"Branch points: {skeleton.branch_points}")
print(f"End points: {skeleton.end_points}")
print(f"All topological points: {skeleton.topo_points}")

# Counts
print(f"Number of branch points: {skeleton.n_branch_points}")
print(f"Number of end points: {skeleton.n_end_points}")
```

### Directed vs Undirected Analysis

```python
# Directed analysis (considers parent-child relationships)
branch_points_directed = skeleton.branch_points
end_points_directed = skeleton.end_points

# Undirected analysis (ignores parent-child direction)  
branch_points_undirected = skeleton.branch_points_undirected
end_points_undirected = skeleton.end_points_undirected

print(f"Directed branch points: {branch_points_directed}")
print(f"Undirected branch points: {branch_points_undirected}")
```

## Tree Navigation (Unique to Skeletons)

### Parent-Child Relationships

```python
# Parent-child relationships (directed tree structure)
parent_array = skeleton.parent_node_array  # Parent of each vertex (positional indices)
parentless_nodes = skeleton.parentless_nodes  # Should only be the root

print(f"Parent array: {parent_array}")
print(f"Parentless nodes: {parentless_nodes}")

# Get children of specific vertices
children_dict = skeleton.child_vertices(
    vertices=[skeleton.root],
    as_positional=False
)
print(f"Children of root: {children_dict}")
```

### Downstream Navigation

```python
# Get all vertices downstream from a point (toward leaves)
branch_point = skeleton.branch_points[0]
downstream = skeleton.downstream_vertices(
    vertex=branch_point,
    inclusive=True,                   # Include the vertex itself
    as_positional=False
)
print(f"Downstream from branch point {branch_point}: {downstream}")

# Downstream from root includes entire tree
all_downstream = skeleton.downstream_vertices(
    vertex=skeleton.root,
    inclusive=True,
    as_positional=False
)
print(f"Total vertices downstream from root: {len(all_downstream)}")
```

## Distance Analysis (Root-Based)

### Distance to Root

```python
# Distance to root for all vertices (unique to rooted trees)
distances_to_root = skeleton.distance_to_root()
print(f"Distances to root: {distances_to_root}")

# Distance to root for specific vertices
specific_distances = skeleton.distance_to_root(
    vertices=skeleton.end_points,
    as_positional=False
)
print(f"End point distances to root: {specific_distances}")

# Hop count to root (number of edges, not spatial distance)
hops_to_root = skeleton.hops_to_root()
print(f"Hops to root: {hops_to_root}")
```

### Tree-Based Path Analysis

```python
# Lowest common ancestor (requires tree structure)
if len(skeleton.end_points) >= 2:
    lca = skeleton.lowest_common_ancestor(
        skeleton.end_points[0],
        skeleton.end_points[1],
        as_positional=False
    )
    print(f"LCA of first two end points: {lca}")

# Path between vertices goes through tree (not direct)
path = skeleton.path_between(
    source=skeleton.end_points[0],
    target=skeleton.end_points[1], 
    as_positional=False,
    as_vertices=False
)
print(f"Tree path between end points: {path}")
```

## Path-Based Analysis (Unique to Skeletons)

### Cover Paths

```python
# Cover paths from end points toward root
cover_paths = skeleton.cover_paths  # With vertex indices
cover_paths_pos = skeleton.cover_paths_positional  # With positional indices

print(f"Number of cover paths: {len(cover_paths)}")
for i, path in enumerate(cover_paths):
    print(f"Path {i}: {path}")

# Get cover paths from specific sources
custom_paths = skeleton.cover_paths_specific(
    sources=skeleton.branch_points,
    as_positional=False
)
print(f"Paths from branch points: {len(custom_paths)}")
```

### Segment Analysis

```python
# Segments are unbranched spans between topological points
segments = skeleton.segments  # With vertex indices
segments_pos = skeleton.segments_positional  # With positional indices

print(f"Number of segments: {len(segments)}")
for i, segment in enumerate(segments):
    print(f"Segment {i}: {segment}")

# Segments including their parent connection
segments_plus = skeleton.segments_plus
print(f"Segments with parent connections: {len(segments_plus)}")

# Map from vertices to their segment
segment_map = skeleton.segment_map
print(f"Segment map shape: {segment_map.shape}")

# Expand vertices to their full segments
selected_vertices = skeleton.end_points[:2]
expanded_segments = skeleton.expand_to_segment(
    vertices=selected_vertices,
    as_positional=False
)
print(f"Expanded segments: {expanded_segments}")
```

## Morphological Measurements

### Cable Length Analysis

```python
# Total cable length (sum of edge lengths)
total_length = skeleton.cable_length()
print(f"Total cable length: {total_length}")

# Cable length of specific subtrees
downstream_vertices = skeleton.downstream_vertices(
    vertex=skeleton.branch_points[0],
    inclusive=True,
    as_positional=False
)
subtree_length = skeleton.cable_length(
    vertices=downstream_vertices,
    as_positional=False
)
print(f"Subtree cable length: {subtree_length}")

# Half-edge lengths (sum of half-edges touching each vertex)
half_edge_lengths = skeleton.half_edge_length
print(f"Half-edge lengths: {half_edge_lengths}")
```

### Surface Area and Volume (with radius)

```python
# If skeleton has radius information
if "radius" in skeleton.feature_names:
    # Calculate surface area treating skeleton as cylinders
    surface_area = skeleton.surface_area()
    print(f"Surface area: {surface_area}")
    
    # Volume calculations
    volume = skeleton.volume()
    print(f"Volume: {volume}")
```

## Root Management (Unique to Skeletons)

### Changing the Root

```python
# Change the root to reorient the tree
new_root = skeleton.end_points[0]
print(f"Original root: {skeleton.root}")

skeleton.reroot(new_root, as_positional=False)
print(f"New root: {skeleton.root}")
print(f"New root location: {skeleton.root_location}")

# Tree properties change with new root
print(f"New branch points: {skeleton.branch_points}")
print(f"New end points: {skeleton.end_points}")
```

### Base Properties (Pre-masking)

```python
# Original properties before any masking operations
print(f"Original root: {skeleton.base_root}")
print(f"Original root location: {skeleton.base_root_location}")

# Original graph matrices
base_csgraph = skeleton.base_csgraph
base_csgraph_binary = skeleton.base_csgraph_binary
```

## Directional Annotation Analysis

### Skeleton-Specific Aggregation

```python
# Aggregate annotations with tree direction awareness
if "synapses" in cell.annotations.names:
    # Count synapses considering tree direction
    synapse_density = skeleton.map_annotations_to_feature(
        annotation="synapses",
        distance_threshold=2.0,
        agg="density",              # Density per unit cable length
        chunk_size=1000,
        validate=False,
        agg_direction="directed"    # Consider tree direction
    )
    
    # Custom aggregation with tree context
    synapse_stats = skeleton.map_annotations_to_feature(
        annotation="synapses",
        distance_threshold=2.0,
        agg={
            "pre_count": ("type", lambda x: sum(x == "pre")),
            "post_count": ("type", lambda x: sum(x == "post")),
            "total_strength": ("strength", "sum")
        }
    )
    
    skeleton.add_feature(synapse_stats)
```

## Advanced Skeleton Analysis

### Compartment Analysis

```python
# Analyze different compartments
if "compartment" in skeleton.feature_names:
    compartments = skeleton.get_feature("compartment")
    
    # Axon vs dendrite analysis
    axon_mask = compartments == 1
    dendrite_mask = compartments == 0
    
    # Cable length by compartment
    axon_length = skeleton.cable_length(
        vertices=skeleton.vertex_index[axon_mask],
        as_positional=False
    )
    dendrite_length = skeleton.cable_length(
        vertices=skeleton.vertex_index[dendrite_mask], 
        as_positional=False
    )
    
    print(f"Axon cable length: {axon_length}")
    print(f"Dendrite cable length: {dendrite_length}")
```

### Branch Analysis

```python
# Analyze branching patterns
branch_points = skeleton.branch_points

# Get branching degrees
branch_degrees = []
for bp in branch_points:
    children = skeleton.child_vertices([bp], as_positional=False)[bp]
    branch_degrees.append(len(children))

print(f"Branch degrees: {branch_degrees}")

# Distance between branch points
if len(branch_points) >= 2:
    bp_distances = skeleton.distance_between(
        sources=branch_points[:2],
        targets=branch_points[:2],
        as_positional=False
    )
    print(f"Distances between branch points: {bp_distances}")
```

## Key Skeleton-Specific Methods

### Skeleton Creation
- `cell.add_skeleton(vertices, edges, root=None, features=None, spatial_columns=None, vertex_index=None)` - Add skeleton to cell

### Tree Structure Properties
- `skeleton.root` / `skeleton.root_positional` - Root vertex
- `skeleton.root_location` - Root coordinates
- `skeleton.branch_points` / `skeleton.branch_points_positional` - Branch vertices
- `skeleton.end_points` / `skeleton.end_points_positional` - Leaf vertices
- `skeleton.topo_points` / `skeleton.topo_points_positional` - All topological vertices
- `skeleton.parent_node_array` - Parent of each vertex (positional indices)
- `skeleton.parentless_nodes` / `skeleton.parentless_nodes_positional` - Vertices with no parent

### Tree Navigation
- `skeleton.child_vertices(vertices, as_positional=False)` - Get children of vertices
- `skeleton.downstream_vertices(vertex, inclusive=False, as_positional=False)` - Get subtree
- `skeleton.lowest_common_ancestor(u, v, as_positional=False)` - Find LCA
- `skeleton.path_between(source, target, as_positional=False, as_vertices=False)` - Tree path

### Root-Based Analysis
- `skeleton.distance_to_root(vertices=None, as_positional=False)` - Distance to root
- `skeleton.hops_to_root(vertices=None, as_positional=False)` - Hop count to root

### Path and Segment Analysis
- `skeleton.cover_paths` / `skeleton.cover_paths_positional` - Paths from tips to root
- `skeleton.cover_paths_specific(sources, as_positional=False)` - Custom cover paths
- `skeleton.segments` / `skeleton.segments_positional` - Unbranched spans
- `skeleton.segments_plus` / `skeleton.segments_plus_positional` - Segments with parent
- `skeleton.segment_map` - Vertex to segment mapping
- `skeleton.expand_to_segment(vertices, as_positional=False)` - Get full segments

### Morphological Measurements
- `skeleton.cable_length(vertices=None, as_positional=False)` - Cable length
- `skeleton.half_edge_length` - Half-edge lengths per vertex
- `skeleton.surface_area(vertices=None, as_positional=False)` - Surface area (with radius)
- `skeleton.volume(vertices=None, as_positional=False)` - Volume (with radius)

### Root Management
- `skeleton.reroot(new_root, as_positional=False)` - Change root vertex
- `skeleton.base_root` / `skeleton.base_root_location` - Original root
- `skeleton.base_csgraph` / `skeleton.base_csgraph_binary` - Original matrices

### Tree-Aware Analysis
- `skeleton.map_annotations_to_feature(annotation, distance_threshold, agg="count"/"density", agg_direction="directed", ...)` - Aggregate with tree direction

!!! note "Additional Features"
    For comprehensive information about vertex access, features, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! tip "When to Use Skeletons"
    - Neuronal morphology analysis requiring tree structure
    - Root-based distance and path analysis
    - Compartment analysis (axon vs dendrite)
    - Branch pattern analysis
    - Cable length and morphological measurements