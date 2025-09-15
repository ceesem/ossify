# Getting Started with Ossify

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

This guide will walk you through the basics of creating and working with cellular morphology data in ossify.

## What is Ossify?

Ossify is a Python package for analyzing cellular morphology data, particularly neuronal structures. It provides tools for working with meshes, skeletons, graphs, and point annotations in a unified framework.

## Basic Concepts

The core of ossify is the `Cell` object, which acts as a container for different types of morphological data:

- **Layers**: The main morphological data (meshes, skeletons, graphs)
- **Annotations**: Sparse point data attached to specific locations
- **Links**: Connections that map data between different layers

## Your First Cell

Let's start by creating a simple cell with synthetic data:

```python
import numpy as np
import ossify

# Create some simple synthetic data
vertices = np.array([
    [0, 0, 0],    # Root
    [1, 0, 0],    # Branch point
    [2, 0, 0],    # End point 1
    [1, 1, 0],    # End point 2
])

edges = np.array([
    [0, 1],  # Root to branch point
    [1, 2],  # Branch point to end 1
    [1, 3],  # Branch point to end 2
])

# Create a cell with a skeleton
cell = ossify.Cell(name="my_first_cell")
cell.add_skeleton(
    vertices=vertices,
    edges=edges,
    root=0  # Root is at index 0
)

print(cell)
```

## Exploring Your Cell

Once you have a cell, you can explore its properties:

```python
# Access the skeleton layer
skeleton = cell.skeleton  # or cell.s for short

# Basic properties
print(f"Number of vertices: {skeleton.n_vertices}")
print(f"Root location: {skeleton.root_location}")
print(f"End points: {skeleton.end_points}")
print(f"Branch points: {skeleton.branch_points}")

# Get a detailed view
cell.describe()
```

## Adding Different Layer Types

Ossify supports multiple types of morphological data:

### Adding a Mesh

```python
# Simple triangular mesh
mesh_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0], 
    [0, 1, 0],
    [0, 0, 1]
])

faces = np.array([
    [0, 1, 2],  # Triangle 1
    [0, 1, 3],  # Triangle 2
    [0, 2, 3],  # Triangle 3
    [1, 2, 3],  # Triangle 4
])

cell.add_mesh(vertices=mesh_vertices, faces=faces)
print(f"Mesh has {cell.mesh.n_vertices} vertices")
```

### Adding a Graph

```python
# Graph is like a skeleton but without tree constraints
graph_vertices = np.random.randn(10, 3) * 5
graph_edges = np.array([[i, i+1] for i in range(9)])  # Simple chain

cell.add_graph(vertices=graph_vertices, edges=graph_edges)
print(f"Graph has {cell.graph.n_vertices} vertices")
```

## Adding Annotations

Annotations are sparse point data that represent specific features:

```python
# Add some synaptic sites
synapse_locations = np.array([
    [0.5, 0, 0],  # Near the root
    [1.5, 0, 0],  # Near branch point
])

cell.add_point_annotations(
    name="synapses",
    vertices=synapse_locations,
    spatial_columns=["x", "y", "z"]  # Column names for coordinates
)

print(f"Added {len(cell.annotations.synapses.vertices)} synapses")
```

## Basic Visualization

```python
import matplotlib.pyplot as plt

# Simple 2D plot of the skeleton
fig, ax = plt.subplots(figsize=(8, 6))
ossify.plot_morphology_2d(
    cell, 
    projection="xy",
    color="compartment",  # Color by compartment (if available)
    palette={1: 'navy', 2: 'tomato', 3: 'black'},
    ax=ax
)
plt.title("Neuron Morphology")
plt.show()
```

## Loading Real Data

For real data, you can load from files or external sources:

```python
# From a saved ossify file (try this example!)
cell = ossify.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')

print("Cable length:", cell.skeleton.cable_length(), "nm")
print("Number of presynaptic sites:", len(cell.annotations.pre_syn))
print("Available skeleton features:", cell.skeleton.feature_names)

# From CAVEclient (requires caveclient)
# cell = ossify.load_cell_from_client(root_id=12345, client=cave_client)

# From legacy MeshWork files (requires h5py)
# cell, mask = ossify.import_legacy_meshwork("path/to/meshwork.h5")
```

## Next Steps

Now that you have a basic cell, you can:

1. **Explore layer properties** - Learn about meshes, graphs, and skeletons
2. **Work with annotations** - Add features and sparse features  
3. **Apply masks** - Filter your data for analysis
4. **Use algorithms** - Compute Strahler numbers, classify compartments
5. **Create visualizations** - Make publication-ready plots

Each of these topics is covered in detail in the following guides.

## Key Functions Reference

- `ossify.Cell(name)` - Create a new cell
- `cell.add_skeleton(vertices, edges, root)` - Add skeleton data
- `cell.add_mesh(vertices, faces)` - Add mesh data  
- `cell.add_graph(vertices, edges)` - Add graph data
- `cell.add_point_annotations(name, vertices)` - Add sparse annotations
- `cell.describe()` - Get a summary of the cell
- `ossify.plot_morphology_2d(cell)` - Basic 2D visualization
- `ossify.load_cell(path)` - Load from file
- `ossify.save_cell(cell, path)` - Save to file