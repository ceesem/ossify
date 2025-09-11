# Working with Meshes

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Mesh layers in ossify represent surface data as collections of vertices and triangular faces. They're ideal for representing the 3D surface of cellular structures like neuron soma, dendrites, or entire cell bodies.

!!! note "Shared Features"
    Meshes inherit many common features from the `PointMixin` class. For information about labels, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

## What is a Mesh Layer?

A `MeshLayer` contains:
- **Vertices**: 3D coordinates of points on the surface
- **Faces**: Triangular faces connecting vertices (as indices)  
- **Surface properties**: Face connectivity, area calculations, trimesh integration

## Creating Mesh Layers

### Basic Mesh Creation

```python
import numpy as np
import ossify

# Create a simple tetrahedron
vertices = np.array([
    [0, 0, 0],    # Vertex 0
    [1, 0, 0],    # Vertex 1  
    [0, 1, 0],    # Vertex 2
    [0, 0, 1],    # Vertex 3
])

# Faces as triangles (indices into vertices)
faces = np.array([
    [0, 1, 2],    # Bottom face
    [0, 1, 3],    # Side face 1
    [0, 2, 3],    # Side face 2  
    [1, 2, 3],    # Top face
])

# Add to cell
cell = ossify.Cell(name="mesh_example")
cell.add_mesh(vertices=vertices, faces=faces)

print(f"Mesh has {cell.mesh.n_vertices} vertices and {len(cell.mesh.faces)} faces")
```

### Mesh with Labels

```python
# Add vertex labels during creation
region_labels = np.array([0, 0, 1, 1])  # Two regions

cell.add_mesh(
    vertices=vertices,
    faces=faces,
    labels={"region": region_labels}
)
```

## Mesh-Specific Properties

### Face Data Access

```python
mesh = cell.mesh

# Access face data (unique to meshes)
faces_array = mesh.faces               # Faces with vertex indices
faces_positional = mesh.faces_positional  # Faces with positional indices

print(f"Number of faces: {len(mesh.faces)}")
```

### Surface Area Calculations

```python
# Total surface area
total_area = mesh.surface_area()

# Surface area for specific vertices
vertex_subset = mesh.vertex_index[:2]  # First two vertices
partial_area = mesh.surface_area(
    vertices=vertex_subset,
    as_positional=False,     # Using vertex indices
    inclusive=True           # Include faces touching any vertex
)

# Surface area for faces fully covered by vertices
exclusive_area = mesh.surface_area(
    vertices=vertex_subset,
    as_positional=False,
    inclusive=False         # Only faces with all vertices in subset
)
```

### Trimesh Integration

Ossify integrates with the [trimesh](https://trimsh.org/) library for advanced mesh operations:

```python
# Get as trimesh object
tmesh = mesh.as_trimesh

# Trimesh provides many useful properties
print(f"Is watertight: {tmesh.is_watertight}")
print(f"Volume: {tmesh.volume}")
print(f"Center of mass: {tmesh.center_mass}")

# Get edges from trimesh
edge_indices = mesh.edges               # With vertex indices
edge_positions = mesh.edges_positional  # With positional indices
```

### Graph Representation

Meshes can be treated as graphs for connectivity analysis:

```python
# Get sparse graph representation for mesh connectivity
csgraph = mesh.csgraph  # Weighted by edge lengths

# Get edge connectivity (derived from faces)
edge_indices = mesh.edges               # With vertex indices
edge_positions = mesh.edges_positional  # With positional indices
```

## Key Mesh-Specific Methods

### Mesh Creation
- `cell.add_mesh(vertices, faces, labels=None, spatial_columns=None, vertex_index=None)` - Add mesh to cell

### Mesh-Specific Properties
- `mesh.faces` - Face indices using vertex indices
- `mesh.faces_positional` - Face indices using positional indices
- `mesh.surface_area(vertices=None, as_positional=True, inclusive=False)` - Calculate surface area
- `mesh.as_trimesh` - Get as trimesh.Trimesh object
- `mesh.as_tuple` - Get (vertices, faces) tuple for external libraries
- `mesh.edges` - Edge connectivity derived from faces (with vertex indices)
- `mesh.edges_positional` - Edge connectivity derived from faces (with positional indices)
- `mesh.csgraph` - Sparse graph representation for connectivity analysis

!!! note "Additional Features"
    For comprehensive information about vertex access, labels, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).