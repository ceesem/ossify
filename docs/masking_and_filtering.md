# Masking and Filtering

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Masking is a powerful feature in ossify that allows you to create filtered views of your data without permanently modifying the original. You can apply masks to individual layers or entire cells, enabling focused analysis on subsets of your morphological data.

!!! note "Shared Features"
    Masking functionality is available across all layer types through the `PointMixin` class. For information about other shared features, see [Shared Layer Features](shared_layer_features.md).

!!! info "Non-Destructive Filtering"
    Masking in ossify creates new objects with filtered data while preserving the original. This allows you to try different analyses without losing data.

## What is Masking?

Masking creates filtered views of your data by:
- **Selecting subsets** of vertices based on conditions
- **Preserving relationships** (edges, faces, linkages) for selected vertices
- **Maintaining data integrity** across linked layers
- **Creating new objects** without modifying originals

## Basic Masking Concepts

### Boolean Masks

```python
import numpy as np
import ossify

# Create a sample cell with skeleton
vertices = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0]])
edges = np.array([[0,1], [1,2], [2,3], [3,4]])
features = {"quality": [0.9, 0.7, 0.8, 0.6, 0.9]}

cell = ossify.Cell(name="mask_example")
cell.add_skeleton(vertices=vertices, edges=edges, root=0, features=features)

# Create boolean mask - high quality vertices
skeleton = cell.skeleton
quality_mask = skeleton.get_feature("quality") > 0.8

print(f"Original vertices: {skeleton.n_vertices}")
print(f"High quality vertices: {sum(quality_mask)}")
print(f"Boolean mask: {quality_mask}")
```

### Index Masks

```python
# Select specific vertices by index
vertex_indices = skeleton.vertex_index[[0, 2, 4]]  # First, third, fifth vertices
positional_indices = [0, 2, 4]  # Positional indices

print(f"Selected vertex indices: {vertex_indices}")
print(f"Selected positional indices: {positional_indices}")
```

## Applying Masks to Layers

### Layer-Level Masking

```python
# Apply boolean mask to create filtered skeleton
filtered_skeleton = skeleton.apply_mask(
    mask=quality_mask,
    as_positional=True    # Mask is boolean array (positional)
)

print(f"Filtered skeleton vertices: {filtered_skeleton.n_vertices}")
print(f"Filtered skeleton edges: {len(filtered_skeleton.edges)}")

# Apply index mask
subset_skeleton = skeleton.apply_mask(
    mask=vertex_indices,
    as_positional=False   # Mask contains vertex indices
)

print(f"Subset skeleton vertices: {subset_skeleton.n_vertices}")
```

### Preserving Connectivity

```python
# Edges are automatically filtered to maintain valid connections
original_edges = skeleton.edges
filtered_edges = filtered_skeleton.edges

print(f"Original edges: {original_edges}")
print(f"Filtered edges: {filtered_edges}")

# Only edges between retained vertices are kept
# Edge indices are remapped to new vertex positions
```

### Mask Effects on Different Layer Types

```python
# Add a mesh to demonstrate masking across layer types
mesh_vertices = np.random.randn(10, 3)
mesh_faces = np.array([[0,1,2], [1,2,3], [4,5,6], [7,8,9]])

cell.add_mesh(vertices=mesh_vertices, faces=mesh_faces)

# Mask mesh vertices
mesh_mask = np.array([True, True, False, True, False, True, True, False, True, False])
filtered_mesh = cell.mesh.apply_mask(mesh_mask, as_positional=True)

print(f"Original mesh: {cell.mesh.n_vertices} vertices, {len(cell.mesh.faces)} faces")
print(f"Filtered mesh: {filtered_mesh.n_vertices} vertices, {len(filtered_mesh.faces)} faces")

# Faces with any removed vertices are filtered out
```

## Cell-Level Masking

### Masking Entire Cells

```python
# Apply mask to entire cell (affects all layers)
filtered_cell = cell.apply_mask(
    layer="skeleton",        # Layer to base mask on
    mask=quality_mask,       # Boolean mask
    as_positional=True
)

print(f"Original cell layers: {cell.layers.names}")
print(f"Filtered cell layers: {filtered_cell.layers.names}")

# Check how masking affected each layer
for layer_name in cell.layers.names:
    original_n = getattr(cell, layer_name).n_vertices
    filtered_n = getattr(filtered_cell, layer_name).n_vertices
    print(f"{layer_name}: {original_n} → {filtered_n} vertices")
```

### Cross-Layer Mask Propagation

```python
# Add annotations to demonstrate cross-layer effects
annotation_points = np.array([[0.5, 0.1, 0.0], [2.5, 0.1, 0.0], [4.1, 0.1, 0.0]])
cell.add_point_annotations(
    name="markers",
    vertices=annotation_points,
    linkage=ossify.Link(target="skeleton", distance_threshold=0.5)
)

# Mask propagates through linkages
filtered_cell_with_annos = cell.apply_mask("skeleton", quality_mask, as_positional=True)

print(f"Original annotations: {len(cell.annotations.markers.vertices)}")
print(f"Filtered annotations: {len(filtered_cell_with_annos.annotations.markers.vertices)}")

# Annotations linked to removed skeleton vertices are filtered out
```

## Temporary Masking with Context Managers

### Layer Context Managers

```python
# Temporarily mask a layer for analysis
with skeleton.mask_context(quality_mask) as temp_skeleton:
    # Work with high-quality vertices only
    cable_length = temp_skeleton.cable_length()
    branch_points = temp_skeleton.branch_points
    
    print(f"High-quality cable length: {cable_length}")
    print(f"High-quality branch points: {branch_points}")

# Original skeleton unchanged after context
print(f"Original skeleton vertices: {skeleton.n_vertices}")
```

### Cell Context Managers

```python
# Temporarily mask entire cell
with cell.mask_context(layer="skeleton", mask=quality_mask) as temp_cell:
    # Analyze filtered cell
    temp_cell.describe()
    
    # All layers are consistently filtered
    if temp_cell.mesh is not None:
        mesh_area = temp_cell.mesh.surface_area()
        print(f"Filtered mesh surface area: {mesh_area}")

# Original cell unchanged
cell.describe()
```

### Advanced Masking with Visualization

Here's a sophisticated example that combines masking with algorithmic analysis and visualization:

```python
# Load example cell and add Strahler analysis
cell = ossify.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')

from ossify.algorithms import strahler_number
strahler_values = strahler_number(cell.skeleton)
cell.skeleton.add_feature(strahler_values, 'strahler_number')

# Mask to axon compartment (compartment == 2) and visualize
with cell.skeleton.mask_context(cell.skeleton.features['compartment'] == 2) as masked_cell:
    fig = ossify.plot_cell_2d(
        masked_cell,
        color='strahler_number',    # Color by branching complexity
        palette='coolwarm',         # Blue to red colormap
        linewidth='radius',         # Width varies with radius
        widths=(1, 5.0),           # Min/max line widths
        units_per_inch=100_000,    # Precise scaling
        root_marker=True,          # Show root location
        root_color='k',            # Black root marker
        root_size=50,              # Root marker size
        dpi=100,                   # Figure resolution
        projection='xy'            # XY projection
    )
    
    # Print analysis of masked data
    print(f"Axon cable length: {masked_cell.skeleton.cable_length():.0f} nm")
    print(f"Axon branch points: {len(masked_cell.skeleton.branch_points)}")
    print(f"Axon Strahler range: {masked_cell.skeleton.get_feature('strahler_number').min()}-{masked_cell.skeleton.get_feature('strahler_number').max()}")

# Original cell unchanged - can analyze other compartments
with cell.skeleton.mask_context(cell.skeleton.features['compartment'] == 3) as dendrite_cell:
    print(f"Dendrite cable length: {dendrite_cell.skeleton.cable_length():.0f} nm")
```

This example demonstrates:
- **Compartment-specific analysis**: Focus on axon (compartment 2)
- **Multi-dimensional visualization**: Color by Strahler order, width by radius
- **Quantitative analysis**: Cable length and topological measurements
- **Publication-ready styling**: Precise scaling, clean markers, appropriate resolution

## Advanced Masking Techniques

### Combining Masks

```python
# Combine multiple conditions
quality = skeleton.get_feature("quality")
distances = skeleton.distance_to_root()

# Complex boolean logic
high_quality_and_distant = (quality > 0.7) & (distances > 1.5)
quality_or_close = (quality > 0.8) | (distances < 1.0)

print(f"High quality AND distant: {sum(high_quality_and_distant)}")
print(f"High quality OR close: {sum(quality_or_close)}")

# Apply combined mask
complex_filtered = skeleton.apply_mask(high_quality_and_distant, as_positional=True)
```

### Region-Based Masking

```python
# Mask based on spatial regions
vertices = skeleton.vertices
x_coords = vertices[:, 0]
y_coords = vertices[:, 1]

# Spatial region mask
region_mask = (x_coords > 1.0) & (x_coords < 3.0) & (y_coords > -0.5) & (y_coords < 0.5)
region_filtered = skeleton.apply_mask(region_mask, as_positional=True)

print(f"Vertices in region: {region_filtered.n_vertices}")
```

### Tree-Specific Masking (Skeletons)

```python
# Mask based on tree properties (skeleton-specific)
if hasattr(skeleton, 'branch_points'):  # Skeleton-specific feature
    # Get subtree from a branch point
    branch_point = skeleton.branch_points[0] if len(skeleton.branch_points) > 0 else skeleton.root
    
    # Get all downstream vertices
    downstream_mask = skeleton.vertex_index.isin(
        skeleton.downstream_vertices(branch_point, inclusive=True, as_positional=False)
    )
    
    subtree = skeleton.apply_mask(downstream_mask, as_positional=False)
    print(f"Subtree vertices: {subtree.n_vertices}")
    
    # Mask to specific paths
    if len(skeleton.cover_paths) > 0:
        first_path = skeleton.cover_paths[0]
        path_mask = skeleton.vertex_index.isin(first_path)
        path_skeleton = skeleton.apply_mask(path_mask, as_positional=False)
        print(f"Single path vertices: {path_skeleton.n_vertices}")
```

## Handling Unmapped Vertices

### Finding Unmapped Data

```python
# Find vertices that don't map to other layers
unmapped_skeleton = skeleton.get_unmapped_vertices(target_layers="mesh")
unmapped_mesh = cell.mesh.get_unmapped_vertices(target_layers="skeleton") if cell.mesh else []

print(f"Skeleton vertices unmapped to mesh: {len(unmapped_skeleton)}")
print(f"Mesh vertices unmapped to skeleton: {len(unmapped_mesh)}")

# Find vertices unmapped to any other layer
completely_unmapped = skeleton.get_unmapped_vertices()  # Checks all other layers
```

### Cleaning Unmapped Data

```python
# Remove unmapped vertices
clean_skeleton = skeleton.mask_out_unmapped(target_layers="mesh")
print(f"Cleaned skeleton: {clean_skeleton.n_vertices} vertices")

# Clean entire cell (removes unmapped from all layers)
clean_cell = cell.apply_mask(
    layer="skeleton", 
    mask=skeleton.get_unmapped_vertices(target_layers="mesh"),
    as_positional=False
)

# Alternative: use mask_out_unmapped at cell level
fully_clean_skeleton = skeleton.mask_out_unmapped()  # Removes vertices unmapped to any layer
```

## Masking Validation

### Checking Mask Results

```python
# Validate mask results
original_vertex_count = skeleton.n_vertices
mask_true_count = sum(quality_mask)
filtered_vertex_count = filtered_skeleton.n_vertices

print(f"Mask validation:")
print(f"  Original vertices: {original_vertex_count}")
print(f"  Mask true count: {mask_true_count}")  
print(f"  Filtered vertices: {filtered_vertex_count}")
print(f"  Match: {mask_true_count == filtered_vertex_count}")

# Check edge preservation
original_edges = len(skeleton.edges)
filtered_edges = len(filtered_skeleton.edges)
print(f"Edge preservation: {original_edges} → {filtered_edges}")
```

### Inspecting Filtered Data

```python
# Compare properties before and after masking
print("Before masking:")
print(f"  Quality range: {np.min(skeleton.get_feature('quality')):.2f} - {np.max(skeleton.get_feature('quality')):.2f}")
print(f"  Cable length: {skeleton.cable_length():.2f}")

print("After masking:")
filtered_quality = filtered_skeleton.get_feature('quality')
print(f"  Quality range: {np.min(filtered_quality):.2f} - {np.max(filtered_quality):.2f}")
print(f"  Cable length: {filtered_skeleton.cable_length():.2f}")
```

## Common Masking Patterns

### Quality-Based Filtering

```python
# Common pattern: filter by confidence/quality
def filter_by_quality(layer, quality_feature="quality", threshold=0.8):
    """Filter layer to high-quality vertices."""
    quality_values = layer.get_feature(quality_feature)
    quality_mask = quality_values >= threshold
    return layer.apply_mask(quality_mask, as_positional=True)

high_quality_skeleton = filter_by_quality(skeleton, threshold=0.8)
```

### Anatomical Region Filtering

```python
# Filter to anatomical regions
def filter_to_region(layer, x_range=None, y_range=None, z_range=None):
    """Filter layer to spatial region."""
    vertices = layer.vertices
    mask = np.ones(len(vertices), dtype=bool)
    
    if x_range:
        mask &= (vertices[:, 0] >= x_range[0]) & (vertices[:, 0] <= x_range[1])
    if y_range:
        mask &= (vertices[:, 1] >= y_range[0]) & (vertices[:, 1] <= y_range[1])
    if z_range:
        mask &= (vertices[:, 2] >= z_range[0]) & (vertices[:, 2] <= z_range[1])
    
    return layer.apply_mask(mask, as_positional=True)

# Filter to specific anatomical region
region_skeleton = filter_to_region(skeleton, x_range=[1.0, 3.0], y_range=[-1.0, 1.0])
```

### Size-Based Filtering

```python
# Filter components by size
def filter_large_components(cell, min_vertices=10):
    """Keep only large connected components."""
    # This would require component analysis
    # Implementation depends on specific layer type and connectivity
    pass
```

## Key Masking Methods

### Layer Masking
- `layer.apply_mask(mask, as_positional=False, self_only=False)` - Apply mask to layer
- `layer.mask_context(mask)` - Temporary masking context manager
- `layer.get_unmapped_vertices(target_layers=None)` - Find unmapped vertices
- `layer.mask_out_unmapped(target_layers=None, self_only=False)` - Remove unmapped vertices

### Cell Masking
- `cell.apply_mask(layer, mask, as_positional=False)` - Apply mask to entire cell
- `cell.mask_context(layer, mask)` - Temporary cell masking context manager

### Mask Creation
- Boolean arrays: `layer.get_feature("quality") > threshold`
- Index arrays: `layer.vertex_index[selection]` 
- Spatial conditions: `(vertices[:, 0] > x_min) & (vertices[:, 0] < x_max)`
- Tree-based: `skeleton.downstream_vertices(vertex, as_positional=False)`

!!! note "Additional Features"
    For information about other layer operations and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! tip "Best Practices"
    - Use context managers for temporary analysis
    - Always check mask results with validation
    - Consider cross-layer effects when masking cells
    - Combine simple masks with boolean logic for complex conditions
    - Use `describe()` to inspect filtered results