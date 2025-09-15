# The Cell Object

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

The `Cell` is the central container in ossify that holds all morphological data for a single cellular structure. It manages different types of data layers and their relationships through a unified interface.

## Cell Architecture

A `Cell` object contains:

- **Layers**: Primary morphological data (mesh, skeleton, graph, and additional point clouds)
- **Annotations**: Sparse point data representing specific features
- **Metadata**: Additional information about the cell
- **Links**: Mappings that connect data between different layers

## Creating a Cell

### Basic Creation

```python
import ossify

# Create an empty cell
cell = ossify.Cell(name="my_cell")

# Create with metadata
cell = ossify.Cell(
    name="neuron_12345",
    meta={"source": "experiment_A", "date": "2024-01-15"}
)

print(cell.name)  # "neuron_12345"
print(cell.meta)  # {"source": "experiment_A", "date": "2024-01-15"}
```

### Cell Properties

```python
# Basic information
print(f"Cell name: {cell.name}")
print(f"Metadata: {cell.meta}")

# Layer access
print(f"Available layers: {cell.layers.names}")
print(f"Available annotations: {cell.annotations.names}")

# Quick layer access
skeleton = cell.skeleton  # or cell.s
mesh = cell.mesh         # or cell.m  
graph = cell.graph       # or cell.g
annotations = cell.annotations  # or cell.a
layers = cell.layers     # or cell.l
```

## Core Layer Types

The `Cell` supports three main types of morphological layers:

### Skeleton Layer
Tree-structured data representing the branching morphology:

```python
import numpy as np

vertices = np.array([[0,0,0], [1,0,0], [2,0,0]])
edges = np.array([[0,1], [1,2]])

cell.add_skeleton(
    vertices=vertices,
    edges=edges,
    root=0,  # Index of root vertex
    features={"radius": [0.5, 0.3, 0.2]}  # Optional features
)

# Access skeleton properties
print(f"Root location: {cell.skeleton.root_location}")
print(f"End points: {cell.skeleton.end_points}")
print(f"Branch points: {cell.skeleton.branch_points}")
```

### Mesh Layer
Surface mesh data with faces:

```python
mesh_vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
faces = np.array([[0,1,2]])  # Triangle

cell.add_mesh(vertices=mesh_vertices, faces=faces)

# Access mesh properties
print(f"Surface area: {cell.mesh.surface_area()}")
print(f"Number of faces: {len(cell.mesh.faces)}")
```

### Graph Layer
General graph structure without tree constraints:

```python
graph_vertices = np.random.randn(5, 3)
graph_edges = np.array([[0,1], [1,2], [2,3], [3,4], [4,0]])  # Cycle

cell.add_graph(vertices=graph_vertices, edges=graph_edges)

print(f"Graph vertices: {cell.graph.n_vertices}")
```

## Annotations System

Annotations are sparse point data representing specific features:

```python
# Add synaptic sites
synapse_locations = np.array([[0.5, 0, 0], [1.5, 0, 0]])

cell.add_point_annotations(
    name="pre_synapses",
    vertices=synapse_locations,
    spatial_columns=["x", "y", "z"]
)

# Access annotations
print(f"Number of synapses: {len(cell.annotations.pre_synapses.vertices)}")
print(f"Annotation names: {cell.annotations.names}")
```

## Layer Management

### Adding Additional Point Layers

Beyond the three core layer types, you can add additional point cloud layers:

```python
# Add a general point cloud layer (not an annotation)
point_data = np.random.randn(100, 3)

cell.add_point_layer(
    name="sampling_points",
    vertices=point_data,
    spatial_columns=["x", "y", "z"]
)
```

### Removing Layers

```python
# Remove a layer
cell.remove_layer("sampling_points")

# Remove an annotation
cell.remove_annotation("pre_synapses")
```

## Working with features

Both layers and annotations can have associated feature data:

```python
# Add features to skeleton
import pandas as pd

# Using arrays
radius_values = np.array([0.5, 0.3, 0.2])
cell.skeleton.add_feature(radius_values, name="radius")

# Using dictionaries
features_dict = {"compartment": [0, 1, 1]}  # 0=dendrite, 1=axon
cell.skeleton.add_feature(features_dict)

# Access features
print(f"Available features: {cell.skeleton.feature_names}")
radius = cell.skeleton.get_feature("radius")
```

## Cell Information and Inspection

### Summary View with `describe()`

The `describe()` method provides a comprehensive overview of your cell's structure and contents:

```python
# Get a hierarchical summary of the cell
cell.describe()

# Example output:
# Cell: my_cell
# ├── Layers (3)
# │   ├── skeleton: 150 vertices, 149 edges
# │   ├── mesh: 2847 vertices, 5691 faces  
# │   └── graph: 45 vertices, 67 edges
# ├── Annotations (2)
# │   ├── synapses: 23 points
# │   └── spines: 47 points
# └── Linkage (3 connections)
#     ├── skeleton → mesh: 150 mappings
#     ├── synapses → skeleton: 23 mappings
#     └── spines → skeleton: 47 mappings

# HTML view in Jupyter notebooks (interactive tree view)
cell.describe(html=True)

# Simple string representation
print(cell)  # Shows basic layers and annotations summary
```

The `describe()` method is particularly useful for:
- **Quick overview** of cell contents and structure
- **Debugging** linkage issues between layers
- **Validation** that data loaded correctly
- **Documentation** of cell composition in notebooks

### Multi-Level Inspection with `describe()`

Ossify provides detailed inspection at multiple levels of granularity:

```python
# 1. Cell overview (high-level summary)
cell.describe()

# 2. All morphological layers (detailed)
cell.layers.describe()
# Output:
# Layers (3)
# ├── skeleton (SkeletonLayer)
# │   ├── 150 vertices, 149 edges
# │   ├── features: [radius, branch_type]
# │   └── Links: mesh <-> skeleton, synapses → skeleton
# ├── mesh (MeshLayer)
# │   ├── 2847 vertices, 5691 faces
# │   ├── features: [compartment]
# │   └── Links: skeleton <-> mesh
# └── graph (GraphLayer)
#     ├── 45 vertices, 67 edges
#     ├── features: []
#     └── Links: []

# 3. All annotations (detailed)
cell.annotations.describe()
# Output:
# Annotations (2)
# ├── synapses (PointCloudLayer)
# │   ├── 23 vertices
# │   ├── features: [synapse_type, confidence]
# │   └── Links: skeleton → synapses
# └── spines (PointCloudLayer)
#     ├── 47 vertices
#     ├── features: [spine_type]
#     └── Links: skeleton → spines

# 4. Individual layers (with cell context)
cell.skeleton.describe()
# Output:
# Cell: my_neuron
# Layer: skeleton (SkeletonLayer)
# ├── 150 vertices, 149 edges
# ├── features: [radius, branch_type]
# └── Links: mesh <-> skeleton, synapses → skeleton

cell.annotations.synapses.describe()
# Output:
# Cell: my_neuron
# Layer: synapses (PointCloudLayer)
# ├── 23 vertices
# ├── features: [synapse_type, confidence]
# └── Links: skeleton → synapses
```

This hierarchical inspection system lets you:
- **Start broad** with `cell.describe()` for overall structure
- **Focus on groups** with `cell.layers.describe()` or `cell.annotations.describe()`
- **Drill down** to individual layers for detailed analysis
- **Maintain context** with cell name shown in individual layer descriptions

### Accessing All features

```python
# Get DataFrame of all features across layers
all_features = cell.features
print(all_features)

# Map features from one layer to another
mapped_features = cell.get_features(
    features="radius",
    target_layer="mesh",
    source_layers="skeleton",
    agg="mean"  # How to aggregate when mapping
)
```

## Copying and Transforming Cells

### Creating Copies

```python
# Deep copy of the cell
cell_copy = cell.copy()

# Copy preserves all data and structure
print(cell_copy.name)  # Same name
print(len(cell_copy.skeleton.vertices))  # Same data
```

### Spatial Transformations

```python
# Apply a transformation to all spatial layers
# Using a transformation matrix
transform_matrix = np.eye(4)  # Identity matrix
transform_matrix[:3, 3] = [10, 0, 0]  # Translation

cell.transform(transform_matrix, inplace=True)

# Using a function
def scale_by_two(vertices):
    return vertices * 2

cell.transform(scale_by_two, inplace=False)  # Returns new cell
```

## Advanced Features

### Layer Aliases

```python
# Short aliases for common access patterns
skel = cell.s      # skeleton
mesh = cell.m      # mesh  
graph = cell.g     # graph
annos = cell.a     # annotations
layers = cell.l    # layers
```

### Context Managers

```python
# Temporarily apply a mask
mask = cell.skeleton.vertex_index < 10

with cell.mask_context("skeleton", mask) as masked_cell:
    # Work with filtered data
    result = some_analysis_function(masked_cell)
# Original cell unchanged
```

## Key Methods Reference

### Cell Creation and Management
- `ossify.Cell(name, meta=None)` - Create new cell
- `cell.copy()` - Deep copy of cell
- `cell.describe()` - Summary view
- `cell.transform(transform, inplace=False)` - Apply spatial transformation

### Adding Layers
- `cell.add_skeleton(vertices, edges, root=None, features=None)` - Add skeleton
- `cell.add_mesh(vertices, faces, features=None)` - Add mesh  
- `cell.add_graph(vertices, edges, features=None)` - Add graph
- `cell.add_point_layer(name, vertices, features=None)` - Add point cloud
- `cell.add_point_annotations(name, vertices, features=None)` - Add annotations

### Layer Management  
- `cell.remove_layer(name)` - Remove morphological layer
- `cell.remove_annotation(name)` - Remove annotation layer

### Data Access
- `cell.layers` - Access to layer manager
- `cell.annotations` - Access to annotation manager  
- `cell.features` - DataFrame of all features
- `cell.get_features(features, target_layer, source_layers=None, agg="median")` - Map features between layers

### Masking and Filtering
- `cell.apply_mask(layer, mask, as_positional=False)` - Apply mask to create new cell
- `cell.mask_context(layer, mask)` - Temporary masking context manager