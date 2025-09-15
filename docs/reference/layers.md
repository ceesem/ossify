# Data Layer Classes

Data layer classes represent different geometric and topological views of neuromorphological data. Each layer type is optimized for specific types of analysis while maintaining a consistent interface.

## Overview

| Layer Type | Purpose | Key Features |
|------------|---------|--------------|
| **[`SkeletonLayer`](#skeletonlayer)** | Tree-structured representations | Hierarchical analysis, pathfinding, root-based operations |
| **[`GraphLayer`](#graphlayer)** | General graph connectivity | Shortest paths, spatial queries, flexible topology |  
| **[`MeshLayer`](#meshlayer)** | 3D surface geometry | Surface area, face connectivity, mesh operations |
| **[`PointCloudLayer`](#pointcloudlayer)** | Sparse annotations | Lightweight markers, flexible metadata |

---

## SkeletonLayer {: .doc-heading}

**Specialized for tree-structured neuronal representations with hierarchical analysis capabilities.**

::: ossify.SkeletonLayer
    options:
        heading_level: 3
        show_root_heading: true 
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"
          - "!^base_"

### Key Properties for Tree Analysis

- **`root`**, **`root_location`**: Root node identification and coordinates
- **`parent_node_array`**: Parent relationships defining tree structure
- **`branch_points`**, **`end_points`**: Topological feature identification
- **`segments`**, **`cover_paths`**: Path-based decomposition of tree structure
- **`distance_to_root()`**, **`cable_length()`**: Distance measurements along tree paths

### Usage Example

```python
# Access tree structure
skeleton = cell.skeleton
root_pos = skeleton.root_location
branch_nodes = skeleton.branch_points

# Analyze tree topology  
distances = skeleton.distance_to_root([10, 20, 30])
subtree = skeleton.downstream_vertices(branch_node_id)

# Path analysis
path = skeleton.path_between(source_id, target_id)
total_length = skeleton.cable_length(path)

# Segment-based analysis
segment_id = skeleton.segment_map[vertex_id]
full_segment = skeleton.segments[segment_id]
```

---

## GraphLayer {: .doc-heading}

**General-purpose graph representation for spatial connectivity analysis.**

::: ossify.GraphLayer
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false  
        show_signature_annotations: true
        separate_signature: true
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"

### Key Properties for Graph Analysis

- **`csgraph`**, **`csgraph_binary`**: Compressed sparse graph representations
- **`edges`**, **`edges_positional`**: Edge connectivity in different index formats
- **`kdtree`**: Spatial indexing for efficient nearest neighbor queries
- **`distance_between()`**, **`path_between()`**: Graph-based distance and pathfinding

### Usage Example

```python
# Graph connectivity
graph = cell.graph
adjacency = graph.csgraph_binary
edge_list = graph.edges

# Spatial queries using KDTree
tree = graph.kdtree
nearest_ids = tree.query(query_points, k=5)

# Distance calculations
distances = graph.distance_between(
    sources=[1, 2, 3], 
    targets=[10, 20, 30],
    limit=1000  # Maximum search distance
)

# Pathfinding
path = graph.path_between(source_id, target_id)
```

---

## MeshLayer {: .doc-heading}

**3D triangulated mesh representation for surface-based analysis.**

::: ossify.MeshLayer
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true  
        separate_signature: true
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"

### Key Properties for Mesh Analysis

- **`faces`**, **`faces_positional`**: Triangle face connectivity
- **`as_trimesh`**: Integration with trimesh library
- **`surface_area()`**: Surface area calculations for vertex regions
- **`edges`**: Edge connectivity derived from face topology

### Usage Example

```python
# Mesh geometry
mesh = cell.mesh
triangles = mesh.faces_positional
vertices = mesh.vertices

# Surface analysis
total_area = mesh.surface_area()
vertex_areas = mesh.surface_area([1, 2, 3])

# Mesh connectivity
edge_graph = mesh.csgraph  # Graph from mesh edges
mesh_obj = mesh.as_trimesh  # trimesh.Trimesh object

# Export formats
vertices_array, faces_array = mesh.as_tuple
```

---

## PointCloudLayer {: .doc-heading}

**Lightweight point-based annotations and sparse markers.**

::: ossify.PointCloudLayer
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true  
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"

### Usage Example

```python
# Point annotations
synapses = cell.annotations["pre_syn"]
locations = synapses.vertices
synapse_count = len(synapses)

# Link to other layers via mapping
mapped_skeleton_ids = synapses.map_index_to_layer("skeleton")
distances_to_root = synapses.distance_to_root()

# Filter annotations
active_synapses = synapses.filter(activity_mask, layer="skeleton")
```

---

## Common Layer Operations

All layer types share a consistent interface for common operations:

### **Spatial Operations** {: .text-primary }

```python
# All layers support transformations
layer.transform(transform_matrix, inplace=False)
layer.transform(custom_function, inplace=True)

# Bounding box calculation
bbox_min, bbox_max = layer.bbox

# Spatial indexing (available for all layers with vertices)
tree = layer.kdtree
```

### **Data Management** {: .text-primary }

```python
# feature management
layer.add_feature(values, name="new_feature")
feature_data = layer.get_feature("existing_feature") 
all_features = layer.features  # DataFrame with all features

# Masking and filtering
masked_layer = layer.apply_mask(boolean_mask)
subset = layer.apply_mask([1, 5, 10, 15])  # Vertex IDs

# Copy operations
layer_copy = layer.copy()
```

### **Index Management** {: .text-primary }

```python
# Flexible indexing support
vertex_ids = layer.vertex_index        # DataFrame indices  
n_vertices = layer.n_vertices           # Total count
vertices_array = layer.vertices         # (N, 3) coordinate array

# Convert between index types
positional_idx = layer._convert_to_positional([100, 200, 300])
vertex_ids = layer.vertex_index[positional_idx]
```

### **Cross-Layer Mapping** {: .text-secondary }

```python  
# Map data between layers
target_values = layer.map_index_to_layer("target_layer_name")
region_mapping = layer.map_region_to_layer("target_layer_name")

# feature propagation with aggregation
aggregated = layer.map_features_to_layer(
    features=["synapse_count", "activity"],
    layer="skeleton", 
    agg={"synapse_count": "sum", "activity": "mean"}
)

# Boolean mask mapping  
skeleton_mask = graph_layer.map_mask_to_layer("skeleton", boolean_mask)
```

---

## Advanced Usage Patterns

### **Multi-Layer Analysis Pipeline**

```python
# Chain operations across layers
cell = ossify.load_cell("neuron.osy")

# 1. Filter skeleton by compartment  
axon_mask = cell.skeleton.get_feature("is_axon")
axon_cell = cell.apply_mask("skeleton", axon_mask)

# 2. Map synapses to axon skeleton
with axon_cell.mask_context("skeleton", axon_mask) as masked:
    synapse_mapping = masked.annotations["pre_syn"].map_index_to_layer("skeleton")
    
# 3. Analyze synapse density along axon
density = masked.skeleton.map_annotations_to_feature(
    "pre_syn", 
    distance_threshold=500,
    agg="count"
)
```

### **Custom Layer Properties**

```python
class CustomAnalysis:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        
    @property  
    def branch_angles(self):
        \"\"\"Calculate angles at branch points\"\"\"
        angles = []
        for bp in self.skeleton.branch_points:
            children = self.skeleton.child_vertices([bp])
            # Calculate angles between child branches
            # ... angle calculation logic
        return np.array(angles)
        
    @property
    def tortuosity(self):
        \"\"\"Path tortuosity for each segment\"\"\"
        tortuosities = []
        for segment in self.skeleton.segments:
            path_length = self.skeleton.cable_length(segment) 
            euclidean = np.linalg.norm(
                self.skeleton.vertices[segment[-1]] - 
                self.skeleton.vertices[segment[0]]
            )
            tortuosities.append(path_length / euclidean)
        return np.array(tortuosities)

# Usage
analysis = CustomAnalysis(cell.skeleton)
angles = analysis.branch_angles
tortuous_segments = analysis.tortuosity > 1.5
```

!!! info "Layer Design Principles"
    
    **Consistent Interface**: All layers implement the same core methods for transformations, masking, and feature management.
    
    **Efficient Storage**: Layers use appropriate data structures (sparse matrices, KDTrees) optimized for their specific use cases.
    
    **Cross-Layer Integration**: The `Link` system enables seamless data flow between different geometric representations.

!!! tip "Performance Optimization"
    
    - **Spatial Queries**: Use `kdtree` property for fast nearest neighbor searches
    - **Graph Operations**: Leverage `csgraph` for efficient pathfinding and connectivity analysis  
    - **Memory Management**: Use `mask_context()` for temporary operations without copying data
    - **Batch Operations**: Process multiple vertices/faces together rather than iterating