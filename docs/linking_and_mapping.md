# Linking and Mapping

Linking is ossify's core differentiator. It solves the problem of working with multiple representations of the same neuron by maintaining explicit mappings between vertex sets.

## Why Linking Matters

A neuron's skeleton, graph, mesh, and annotations all have different vertices. When you want to answer questions like "how many synapses are on each skeleton branch?" or "what's the total mesh volume per skeleton vertex?", you need to know which vertices in one layer correspond to which vertices in another.

Nearest-neighbor matching is the obvious approach, but it fails often enough to matter. Neurons are tortuous — a synapse on one branch can be spatially closer to a *different* branch than to its own. Ossify's links preserve the correct correspondence, so your analyses are reliable.

## What is a Link?

A `Link` is a mapping from each vertex in a source layer to a vertex in a target layer. It's stored as an array of target vertex indices, one per source vertex.

Links are established when you:

- Load a cell from CAVE (the database provides the correspondence)
- Load an `.osy` file (links are saved with the cell)
- Manually specify a link when adding a layer or annotation

Once a link exists, you can use it to move features, indices, and masks between layers.

## Mapping Features Between Layers

`map_features_to_layer` is the workhorse method. It takes a feature from one layer and maps it onto another, applying an aggregation function when multiple source vertices map to the same target vertex.

```python
import ossify as osy

cell = osy.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')

# Map graph volume onto skeleton, summing contributions
volume = cell.graph.map_features_to_layer("size_nm3", layer='skeleton', agg='sum')
cell.skeleton.add_feature(volume)
```

### Aggregation Options

When multiple source vertices map to a single target vertex (which is common — many graph or mesh vertices map to each skeleton vertex), you choose how to combine them:

- `'sum'` — total value (use for counts, volumes)
- `'mean'` — average value (use for intensities, ratios)
- `'majority'` — most common value (use for categorical labels like compartment)
- `'min'`, `'max'` — extremes

```python
# Map multiple features with different aggregation
mapped = cell.graph.map_features_to_layer(
    features=["size_nm3", "max_dt_nm"],
    layer='skeleton',
    agg={'size_nm3': 'sum', 'max_dt_nm': 'mean'}
)
cell.skeleton.add_feature(mapped)
```

## Mapping Indices Between Layers

Sometimes you need to find which vertex in layer B corresponds to a vertex in layer A, without moving feature data.

```python
# Find the skeleton vertex corresponding to each graph vertex
skeleton_indices = cell.graph.map_index_to_layer(
    layer='skeleton',
    as_positional=False
)

# Find the skeleton vertex for specific graph vertices
specific = cell.graph.map_index_to_layer(
    layer='skeleton',
    source_index=cell.graph.vertex_index[:5],
    as_positional=False
)
```

### One-to-Many Mapping

Since many source vertices can map to a single target, you can also go the other direction — find *all* source vertices that map to a given target:

```python
# For each skeleton vertex, get all graph vertices that map to it
region_map = cell.skeleton.map_index_to_layer_region(
    layer='graph',
    as_positional=False
)
# Returns a dict: {skeleton_vertex_id: [list of graph vertex ids]}
```

Or map a set of vertices to all their corresponding vertices in another layer:

```python
# Get all graph vertices corresponding to the first 5 skeleton vertices
all_graph_verts = cell.skeleton.map_region_to_layer(
    layer='graph',
    source_index=cell.skeleton.vertex_index[:5],
    as_positional=False
)
```

## Mapping Masks Between Layers

When you create a boolean mask on one layer, you can map it to a linked layer:

```python
# Create a dendrite mask on the skeleton
dendrite_mask = cell.skeleton.features['compartment'] == 3

# Map it to the graph layer
graph_dendrite_mask = cell.skeleton.map_mask_to_layer('graph', dendrite_mask)
```

This is what happens automatically when you use `mask_context` on a cell — the mask propagates through links to all connected layers and annotations.

## Creating Links

### When Loading from CAVE

Links are created automatically when you use `load_cell_from_client`. The database knows which graph vertices correspond to which skeleton vertices, and synapse annotations are linked to the graph.

### When Adding Layers Manually

You can specify a link when adding a layer or annotation:

```python
from ossify import Link

# Add annotations with an explicit link to the skeleton
cell.add_point_annotations(
    name="my_synapses",
    vertices=synapse_data,
    spatial_columns=['x', 'y', 'z'],
    linkage=Link(
        mapping=skeleton_vertex_ids,  # One per annotation
        target='skeleton',
        map_value_is_index=True
    )
)
```

### Annotations from Linkage

If your annotations are defined by their link to another layer (e.g., you know which skeleton vertex each synapse belongs to, but not its exact coordinates), you can derive the coordinates from the link:

```python
cell.add_point_annotations(
    name="linked_points",
    vertices=annotation_data,
    linkage=Link(mapping=skeleton_ids, target='skeleton', map_value_is_index=True),
    vertices_from_linkage=True  # Coordinates come from the target layer
)
```

## Finding Unmapped Vertices

Not every vertex in one layer necessarily maps to a vertex in another. You can find and handle these:

```python
# Find skeleton vertices with no mapping to the graph
unmapped = cell.skeleton.get_unmapped_vertices(target_layers='graph')

# Remove them
clean_skeleton = cell.skeleton.mask_out_unmapped(target_layers='graph')
```

## Worked Example: Synapse Count Per Skeleton Vertex

Here's a complete example that counts synapses per skeleton vertex by mapping through links:

```python
import ossify as osy

cell = osy.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')

# Map pre-synaptic annotation indices to the skeleton
pre_syn_skel_ids = cell.annotations.pre_syn.map_index_to_layer(
    layer='skeleton',
    as_positional=False
)

# Count occurrences per skeleton vertex
import numpy as np
skel_ids, counts = np.unique(pre_syn_skel_ids, return_counts=True)
print(f"Skeleton vertices with pre-synapses: {len(skel_ids)}")
print(f"Max pre-synapses on one vertex: {counts.max()}")

# Or use the built-in aggregation for skeleton vertices that
# accounts for cable length around each vertex
pre_density = cell.skeleton.map_annotations_to_feature(
    annotation='pre_syn',
    agg='density'
)
cell.skeleton.add_feature(pre_density, 'pre_syn_density')
```

## Next Steps

- [The Cell Object](cell_object.md) — creating cells and managing layers
- [Features and Data](shared_layer_features.md) — working with vertex-level features
- [Masking and Filtering](masking_and_filtering.md) — how masks propagate through links
