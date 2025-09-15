# Shared Layer Features

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

All ossify layers (meshes, graphs, skeletons, and point clouds) inherit common functionality through the `PointMixin` class. This document covers the features that work the same way across all layer types.

## Common Properties

Every layer has these basic properties:

```python
# Assuming you have a layer (mesh, graph, skeleton, or point cloud)
layer = cell.skeleton  # or cell.mesh, cell.graph, etc.

# Basic properties
print(f"Layer name: {layer.name}")
print(f"Number of vertices: {layer.n_vertices}")
print(f"Bounding box: {layer.bbox}")
print(f"Spatial columns: {layer.spatial_columns}")
print(f"Available features: {layer.feature_names}")
```

## Vertex Data Access

All layers provide consistent access to vertex data:

```python
# Vertex coordinates
vertices_array = layer.vertices          # Nx3 numpy array
vertices_df = layer.vertex_df           # Pandas DataFrame with index
vertex_indices = layer.vertex_index     # Array of vertex indices
vertex_index_map = layer.vertex_index_map  # Dict mapping indices to positions

# Complete node data (coordinates + features)
full_data = layer.nodes                 # DataFrame with all data
features_only = layer.features              # DataFrame with just features
```

## Working with features

All layers support adding and accessing vertex features:

```python
# Add features using arrays
quality_values = np.array([0.9, 0.8, 0.7, 0.6])
layer.add_feature(quality_values, name="quality")

# Add multiple features at once
new_features = {
    "region": [0, 0, 1, 1],
    "processed": [True, True, False, False]
}
layer.add_feature(new_features)

# Access feature data
quality = layer.get_feature("quality")
all_features = layer.features
print(f"Available features: {layer.feature_names}")
```

## Spatial Queries

All layers provide KDTree functionality for spatial searches:

```python
# Get KDTree for spatial queries
kdtree = layer.kdtree

# Find nearest vertices to a point
query_point = [0.5, 0.5, 0.5]
distances, indices = kdtree.query(query_point, k=2)  # 2 nearest

# Get vertex indices (not positional indices)
nearest_vertex_ids = layer.vertex_index[indices]
print(f"Nearest vertices: {nearest_vertex_ids}")
print(f"Distances: {distances}")
```

## Copying and Transformations

All layers support copying and spatial transformations:

```python
# Create a deep copy
layer_copy = layer.copy()

# Apply spatial transformation using a function
def scale_and_translate(vertices):
    scaled = vertices * 2.0  # Scale by 2
    translated = scaled + [10, 0, 0]  # Translate
    return translated

# Transform (returns new layer)
transformed_layer = layer.transform(scale_and_translate, inplace=False)

# Transform in place
layer.transform(scale_and_translate, inplace=True)

# Transform using a matrix
transform_matrix = np.eye(4)
transform_matrix[:3, 3] = [5, 0, 0]  # Translation
layer.transform(transform_matrix, inplace=True)
```

## Masking and Filtering

All layers support masking to create filtered subsets:

```python
# Create boolean mask
mask = layer.get_feature("quality") > 0.7

# Apply mask (creates new layer)
filtered_layer = layer.apply_mask(mask, as_positional=False)

# Use as context manager for temporary filtering
with layer.mask_context(mask) as filtered_layer:
    # Work with filtered data
    result = some_analysis_function(filtered_layer)
# Original layer unchanged

# Mask using vertex indices instead of boolean
vertex_indices = layer.vertex_index[:10]  # First 10 vertices
subset_layer = layer.apply_mask(vertex_indices, as_positional=False)

# Mask using positional indices
positional_mask = np.array([0, 1, 2, 5, 8])  # Specific positions
subset_layer = layer.apply_mask(positional_mask, as_positional=True)
```

## Cross-Layer Mapping

When layers are part of a cell, you can map data between them:

### Index Mapping

```python
# Map indices one-to-one between layers
target_indices = layer.map_index_to_layer(
    layer="skeleton",           # Target layer
    source_index=None,          # None = all vertices
    as_positional=False,        # Use vertex indices
    validate=False              # Check for ambiguous mappings
)

# Map a region to all corresponding vertices
all_target_indices = layer.map_region_to_layer(
    layer="mesh",
    source_index=layer.vertex_index[:5],  # First 5 vertices
    as_positional=False
)

# Map each vertex to lists of all corresponding vertices
mapping_dict = layer.map_index_to_layer_region(
    layer="graph",
    source_index=layer.vertex_index[:3],
    as_positional=False
)
print(f"Vertex mappings: {mapping_dict}")
```

### feature Mapping

```python
# Map features from one layer to another
mapped_features = layer.map_features_to_layer(
    features="quality",           # feature to map
    layer="skeleton",           # Target layer
    agg="mean"                  # Aggregation method
)

# Map multiple features with different aggregations
mapped_features = layer.map_features_to_layer(
    features=["quality", "region"],
    layer="mesh",
    agg={
        "quality": "mean",
        "region": "majority"    # Custom aggregation function
    }
)

# Use the mapping
target_layer = cell.layers["skeleton"]
target_layer.add_feature(mapped_features)
```

### Mask Mapping

```python
# Map a boolean mask to another layer
source_mask = layer.get_feature("processed") == True
target_mask = layer.map_mask_to_layer("mesh", source_mask)

# Apply the mapped mask to target layer
target_layer = cell.layers["mesh"]
filtered_target = target_layer.apply_mask(target_mask, as_positional=True)
```

## Vertex Index vs Positional Index

Understanding the difference between vertex indices and positional indices is crucial:

```python
# Vertex indices: The actual IDs/features of vertices
vertex_ids = layer.vertex_index  # e.g., [100, 205, 350, 401]

# Positional indices: Array positions (0, 1, 2, 3...)  
# Used for indexing into arrays like layer.vertices

# Convert between them
vertex_index_map = layer.vertex_index_map  # Dict: {100: 0, 205: 1, 350: 2, 401: 3}

# Most functions have as_positional parameter
distances = layer.distance_to_root(
    vertices=[100, 205],        # Vertex indices
    as_positional=False         # Specify we're using vertex indices
)

distances = layer.distance_to_root(
    vertices=[0, 1],            # Positional indices  
    as_positional=True          # Specify we're using positions
)
```

## Finding Unmapped Vertices

Find vertices that don't have mappings to other layers:

```python
# Find vertices with no mapping to any other layer
unmapped = layer.get_unmapped_vertices()

# Find vertices with no mapping to specific layers
unmapped_to_mesh = layer.get_unmapped_vertices(target_layers="mesh")
unmapped_to_multiple = layer.get_unmapped_vertices(
    target_layers=["mesh", "skeleton"]
)

# Remove unmapped vertices
clean_layer = layer.mask_out_unmapped(target_layers="mesh")

# Remove vertices unmapped to any other layer
fully_clean = layer.mask_out_unmapped()  # Checks all other layers
```

## Key Shared Methods Reference

### Basic Properties
- `layer.name` - Layer name
- `layer.n_vertices` - Number of vertices
- `layer.bbox` - Bounding box
- `layer.spatial_columns` - List of coordinate column names
- `layer.feature_names` - List of available features

### Data Access
- `layer.vertices` - Vertex coordinates as numpy array
- `layer.vertex_df` - Vertices as indexed DataFrame
- `layer.vertex_index` - Array of vertex indices  
- `layer.vertex_index_map` - Dict mapping indices to positions
- `layer.nodes` - Complete DataFrame with coordinates and features
- `layer.features` - DataFrame of all features

### features
- `layer.add_feature(feature, name=None)` - Add vertex features
- `layer.get_feature(key)` - Get feature array

### Spatial Operations
- `layer.kdtree` - KDTree for spatial queries
- `layer.transform(transform, inplace=False)` - Apply transformation
- `layer.copy()` - Create deep copy

### Masking
- `layer.apply_mask(mask, as_positional=False, self_only=False)` - Create masked subset
- `layer.mask_context(mask)` - Temporary masking context manager
- `layer.get_unmapped_vertices(target_layers=None)` - Find unmapped vertices
- `layer.mask_out_unmapped(target_layers=None, self_only=False)` - Remove unmapped vertices

### Cross-layer Mapping
- `layer.map_index_to_layer(layer, source_index=None, as_positional=False, validate=False)` - Map indices 1:1
- `layer.map_region_to_layer(layer, source_index=None, as_positional=False)` - Map region to region  
- `layer.map_index_to_layer_region(layer, source_index=None, as_positional=False)` - Map to lists
- `layer.map_features_to_layer(features, layer, agg="mean")` - Map features between layers
- `layer.map_mask_to_layer(layer, mask)` - Map boolean mask