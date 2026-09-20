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
# `size_nm3` was added by the example above, and each feature name may only be
# used once, so rename the mapped columns to say how they were aggregated.
cell.skeleton.add_feature(
    mapped.rename(columns={"size_nm3": "size_nm3_sum", "max_dt_nm": "max_dt_nm_mean"})
)
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
import numpy as np
from ossify import Link

# Two synapse locations, and the skeleton vertex each one belongs to.
synapse_data = cell.skeleton.vertices[[10, 20]]
skeleton_vertex_ids = cell.skeleton.vertex_index[[10, 20]]

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
# These points have no coordinates of their own, only a link and some
# per-point features. `vertices` must be a DataFrame here; its spatial columns
# are filled in from the target layer.
import pandas as pd

skeleton_ids = cell.skeleton.vertex_index[[30, 40, 50]]
annotation_data = pd.DataFrame({"score": [0.7, 0.8, 0.9]})

cell.add_point_annotations(
    name="linked_points",
    vertices=annotation_data,
    linkage=Link(mapping=skeleton_ids, target='skeleton', map_value_is_index=True),
    vertices_from_linkage=True  # Coordinates come from the target layer
)
```

## Mapping Completeness

A link existing does not mean every vertex on both sides is covered by it. An
incompletely mapped cell is perfectly well defined — much like a mesh that
isn't watertight — but not every operation is meaningful on one.

**Completeness is directional.** Every mesh vertex can reach the graph while
some graph vertices are reached by no mesh vertex, so ask in the direction you
intend to use:

```python
# The four-layer example cell, so the mesh is present too.
cell = osy.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529_full.osy')

print(cell.mesh.is_fully_mapped_to('graph'))     # True  — every mesh vertex maps
print(cell.graph.is_fully_mapped_to('mesh'))     # False — the mesh misses part of the arbor
print(f"{cell.graph.mapping_coverage('mesh'):.0%}")
```

`describe()` marks the directions that are incomplete, so you can see it at a
glance:

```python
cell.describe()
```

```
└── Linkage (4 connections)
    ├── mesh <-> graph (graph 81% mapped)
    ├── post_syn <-> graph (graph 18% mapped)
    ├── pre_syn <-> graph (graph 7% mapped)
    └── graph <-> skeleton
```

`graph <-> skeleton` carries no note, meaning it is complete in both
directions. On a cell with millions of mesh vertices this walk costs a second
or so; pass `cell.describe(coverage=False)` to skip it.

!!! important "Partial coverage is not automatically a fault"

    A sparse annotation is *expected* to reach only a few vertices — `pre_syn`
    covering 7% of the graph is what "synapses are sparse" looks like, not a
    broken link. It is the layers meant to cover each other that are worth a
    second look: `graph` only 81% mapped to `mesh` says the mesh does not span
    the whole arbor.

### Which operations need a complete mapping

| Operation | With unmapped vertices |
|---|---|
| `map_index_to_layer` | **Raises `KeyError`** naming them, unless you pass `missing=` |
| `map_index_to_layer_region` | Returns a dict, omitting the unmapped keys |
| `map_region_to_layer` | Returns only the mapped targets |
| `map_features_to_layer` | Returns a row per target vertex; unmapped ones are null |
| `map_mask_to_layer` | Returns only the mapped targets |
| `mask_out_unmapped` | Exists precisely to remove them |

`map_index_to_layer` is the strict one because it promises one target per
source: there is no honest answer for a vertex with no counterpart. Either fix
the input first, or say what should stand in:

```python
# Check, then act
if not cell.skeleton.is_fully_mapped_to('mesh'):
    clean = cell.skeleton.mask_out_unmapped(target_layers='mesh', return_cell=False)

# Or map in one call with an explicit fill
mapped = cell.skeleton.map_index_to_layer('mesh', missing=-1)
```

Pick the fill yourself rather than letting a null be invented for you: vertex
ids here routinely exceed 2**53, and a `NaN` would force the array to float64
and silently change every id in it. An integer fill keeps it exact.

## Finding Unmapped Vertices

Not every vertex in one layer necessarily maps to a vertex in another. You can find and handle these:

```python
# Find skeleton vertices with no mapping to the graph
unmapped = cell.skeleton.get_unmapped_vertices(target_layers='graph')

# Remove them
clean_skeleton = cell.skeleton.mask_out_unmapped(
    target_layers='graph', return_cell=False
)
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
# `distance_threshold` is required: it sets the cable distance, in the
# skeleton's units, over which each vertex collects nearby annotations.
pre_density = cell.skeleton.map_annotations_to_feature(
    annotation='pre_syn',
    distance_threshold=3000,
    agg='density'
)
cell.skeleton.add_feature(pre_density, 'pre_syn_density')
```

## Next Steps

- [The Cell Object](cell_object.md) — creating cells and managing layers
- [Features and Data](shared_layer_features.md) — working with vertex-level features
- [Masking and Filtering](masking_and_filtering.md) — how masks propagate through links
