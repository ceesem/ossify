# Core Classes

The core classes provide the foundational architecture for neuromorphological data management in ossify.

## Cell {: .doc-heading}

::: ossify.Cell
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

---

## Link {: .doc-heading}

::: ossify.Link
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

---

## Manager Classes {: .doc-heading}

These classes are used internally by `Cell` to organize and access different types of data layers.

### LayerManager

::: ossify.base.LayerManager
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"

### AnnotationManager

::: ossify.base.AnnotationManager  
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false
        group_by_category: true
        members_order: source
        filters:
          - "!^_"

---

## Usage Examples

### Creating and Working with Cells

```python
import ossify
import numpy as np

# Create an empty cell
cell = ossify.Cell(name="example_neuron")

# Add a skeleton layer
vertices = np.random.rand(100, 3) * 1000  # Random 3D points
edges = np.column_stack([np.arange(99), np.arange(1, 100)])  # Linear chain
cell.add_skeleton(vertices=vertices, edges=edges, root=0)

# Add a mesh layer  
mesh_vertices = np.random.rand(200, 3) * 1000
faces = np.random.randint(0, 200, (150, 3))  # Random triangular faces
cell.add_mesh(vertices=mesh_vertices, faces=faces)

# Add point annotations
annotation_points = np.random.rand(50, 3) * 1000
cell.add_point_annotations(
    name="synapses", 
    vertices=annotation_points
)
```

### Layer Access and Management

```python
# Access layers by name
skeleton = cell.layers["skeleton"]  
mesh = cell.layers["mesh"]

# Access via properties
skeleton = cell.skeleton
mesh = cell.mesh

# Access annotations
synapses = cell.annotations["synapses"]
synapse_locations = cell.annotations.synapses  # Alternative syntax

# List all available layers
print("Morphological layers:", cell.layers.names)
print("Annotation layers:", cell.annotations.names)
print("All labels across layers:")
print(cell.labels)
```

### Data Linking and Mapping

```python
# Create a link between skeleton and annotations
from ossify import Link

# Map synapses to nearest skeleton vertices
link = Link(
    mapping=skeleton_vertex_ids,  # Array mapping synapses to skeleton vertices
    source="synapses",
    target="skeleton"
)

cell.add_point_annotations(
    name="linked_synapses",
    vertices=annotation_points,
    linkage=link
)

# Map labels between layers
compartment_labels = cell.get_labels(
    labels=["compartment"], 
    target_layer="skeleton",
    source_layers=["mesh"],
    agg="majority"  # Use majority vote for mapping
)
```

### Masking and Filtering

```python
# Apply a mask to create a subset
axon_mask = skeleton.get_label("is_axon") == True
axon_cell = cell.apply_mask("skeleton", axon_mask)

# Temporary masking with context manager
with cell.mask_context("skeleton", axon_mask) as masked_cell:
    # Work with axon-only data
    axon_length = masked_cell.skeleton.cable_length()
    print(f"Axon length: {axon_length:.2f}")

# The original cell is unchanged after exiting context
```

### Spatial Transformations

```python
# Apply a transformation matrix
transform_matrix = np.array([
    [1, 0, 0],    # Scale x by 1
    [0, 1, 0],    # Scale y by 1  
    [0, 0, 2]     # Scale z by 2
])

# Transform all spatial layers
transformed_cell = cell.transform(transform_matrix)

# Or apply in-place
cell.transform(transform_matrix, inplace=True)

# Apply custom transformation function
def rotate_90_degrees(vertices):
    \"\"\"Rotate 90 degrees around z-axis\"\"\"
    rotation = np.array([
        [0, -1, 0],
        [1,  0, 0], 
        [0,  0, 1]
    ])
    return vertices @ rotation.T

rotated_cell = cell.transform(rotate_90_degrees)
```

!!! info "Key Design Principles"
    
    **Consistent Interface**: All layer types share common methods for transformations, masking, and label management.
    
    **Method Chaining**: Most methods return `self` to enable fluent interfaces: `cell.add_skeleton(...).add_mesh(...)`
    
    **Non-destructive Operations**: Operations like `apply_mask()` return new objects, preserving the original data.
    
    **Flexible Indexing**: Support both DataFrame indices (arbitrary integers) and positional indices (0-based arrays).

!!! tip "Performance Tips"
    
    - Use `mask_context()` for temporary operations to avoid copying data
    - Leverage the `LinkageSystem` for efficient cross-layer operations
    - Cache expensive computations using layer properties like `csgraph`