# Working with Annotations

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Annotations are sparse point cloud layers that represent specific features or events in cellular morphology. Unlike the main morphological layers (mesh, graph, skeleton), annotations are designed for discrete features like synapses, spines, or markers that occur at specific locations.

!!! note "Shared Features"
    Annotations inherit many common features from the `PointMixin` class. For information about labels, masking, transformations, spatial queries, and cross-layer mapping, see [Shared Layer Features](shared_layer_features.md).

!!! info "Annotations vs Layer Labels"
    **Annotations** are sparse point clouds representing discrete features. **Layer labels** are arrays of values attached to every vertex in a layer. Use annotations for sparse features (synapses, spines) and labels for properties of existing vertices (radius, compartment).

## What are Annotations?

Annotations (`PointCloudLayer` in annotation context) contain:
- **Vertices**: 3D coordinates of feature locations
- **Labels**: Metadata about each annotation (confidence, type, size, etc.)
- **Linkage**: Optional connections to morphological layers
- **Sparse nature**: Represent discrete events, not continuous morphology

## Creating Annotations

### Basic Point Annotations

```python
import numpy as np
import ossify

# Create a cell with a skeleton
cell = ossify.Cell(name="annotated_cell")
skeleton_vertices = np.array([[0,0,0], [1,0,0], [2,0,0]])
skeleton_edges = np.array([[0,1], [1,2]])
cell.add_skeleton(vertices=skeleton_vertices, edges=skeleton_edges, root=0)

# Add synaptic sites as annotations
synapse_locations = np.array([
    [0.5, 0.1, 0.0],   # Near first edge
    [1.5, -0.1, 0.0],  # Near second edge
    [0.8, 0.2, 0.0],   # Between vertices
])

cell.add_point_annotations(
    name="synapses",
    vertices=synapse_locations,
    spatial_columns=["x", "y", "z"]
)

print(f"Added {len(cell.annotations.synapses.vertices)} synapses")
print(f"Available annotations: {cell.annotations.names}")
```

### Annotations with Metadata

```python
import pandas as pd

# Create annotation data with metadata
synapse_data = pd.DataFrame({
    'x': [0.5, 1.5, 0.8],
    'y': [0.1, -0.1, 0.2], 
    'z': [0.0, 0.0, 0.0],
    'confidence': [0.95, 0.87, 0.92],
    'synapse_type': ['excitatory', 'inhibitory', 'excitatory'],
    'size': [1.2, 0.8, 1.0],
    'partner_id': [12345, 67890, 11111]
})

cell.add_point_annotations(
    name="synapses_detailed",
    vertices=synapse_data,
    spatial_columns=['x', 'y', 'z']
)

# Access annotation metadata
synapses = cell.annotations.synapses_detailed
print(f"Confidence values: {synapses.get_label('confidence')}")
print(f"Synapse types: {synapses.get_label('synapse_type')}")
```

### Multiple Annotation Types

```python
# Add different types of annotations
# Presynaptic sites
pre_syn_locations = np.array([[0.2, 0.1, 0.0], [1.8, -0.1, 0.0]])
cell.add_point_annotations(
    name="pre_syn",
    vertices=pre_syn_locations,
    labels={"strength": [0.8, 0.6]}
)

# Postsynaptic sites  
post_syn_locations = np.array([[0.7, 0.2, 0.0], [1.3, 0.1, 0.0]])
cell.add_point_annotations(
    name="post_syn", 
    vertices=post_syn_locations,
    labels={"receptor_type": ["AMPA", "GABA"]}
)

# Dendritic spines
spine_locations = np.array([[0.3, 0.3, 0.0], [0.9, -0.2, 0.0], [1.7, 0.3, 0.0]])
cell.add_point_annotations(
    name="spines",
    vertices=spine_locations,
    labels={
        "spine_type": ["mushroom", "thin", "stubby"],
        "volume": [0.05, 0.02, 0.03]
    }
)

print(f"All annotations: {cell.annotations.names}")
```

## Linking Annotations to Morphological Layers

### Manual Linkage Specification

```python
from ossify import Link

# Create annotations with explicit linkage to skeleton
# Link by providing skeleton vertex IDs that annotations map to
mapping_to_skeleton = np.array([0, 1, 0])  # Synapses map to skeleton vertices 0, 1, 0

cell.add_point_annotations(
    name="linked_synapses",
    vertices=synapse_locations,
    linkage=Link(
        mapping=mapping_to_skeleton,
        target="skeleton",           # Target layer
        map_value_is_index=True     # Values are vertex indices
    ),
    labels={"confidence": [0.9, 0.8, 0.7]}
)
```

### Automatic Spatial Linkage

```python
# Let ossify automatically find nearest skeleton vertices
cell.add_point_annotations(
    name="auto_linked_synapses",
    vertices=synapse_locations,
    linkage=Link(
        target="skeleton",
        distance_threshold=0.5      # Max distance to link
    ),
    vertices_from_linkage=True     # Use target layer coordinates
)

# The annotation coordinates will be set to the linked skeleton vertices
linked = cell.annotations.auto_linked_synapses
print(f"Linked annotation coordinates: {linked.vertices}")
```

## Working with Annotation Data

### Accessing Annotation Properties

```python
synapses = cell.annotations.synapses_detailed

# Basic properties
print(f"Number of annotations: {synapses.n_vertices}")
print(f"Spatial columns: {synapses.spatial_columns}")
print(f"Available labels: {synapses.label_names}")

# Coordinates and data
coordinates = synapses.vertices           # Nx3 coordinate array
full_data = synapses.nodes               # DataFrame with coordinates + labels
labels_only = synapses.labels            # DataFrame with just labels

# Individual label access
confidence = synapses.get_label('confidence')
types = synapses.get_label('synapse_type')
```

### Filtering and Queries

```python
# Filter by label values
high_confidence = synapses.get_label('confidence') > 0.9
high_conf_annotations = synapses.apply_mask(high_confidence, as_positional=True)

print(f"High confidence annotations: {high_conf_annotations.n_vertices}")

# Filter by type
excitatory_mask = synapses.get_label('synapse_type') == 'excitatory'
excitatory_synapses = synapses.apply_mask(excitatory_mask, as_positional=True)

# Spatial filtering using KDTree
query_point = [1.0, 0.0, 0.0]
distances, indices = synapses.kdtree.query(query_point, k=2, distance_upper_bound=0.5)

# Get nearby annotations (excluding infinite distances)
nearby_mask = distances < np.inf
nearby_indices = indices[nearby_mask]
nearby_annotations = synapses.apply_mask(nearby_indices, as_positional=True)
```

### Annotation Statistics

```python
# Summary statistics by type
types = synapses.get_label('synapse_type')
confidence = synapses.get_label('confidence')

for syn_type in np.unique(types):
    type_mask = types == syn_type
    type_confidence = confidence[type_mask]
    print(f"{syn_type}: count={sum(type_mask)}, mean_confidence={np.mean(type_confidence):.3f}")

# Size distribution
sizes = synapses.get_label('size')
print(f"Size range: {np.min(sizes):.2f} - {np.max(sizes):.2f}")
print(f"Mean size: {np.mean(sizes):.2f}")
```

## Mapping Annotations to Morphological Layers

### Aggregating to Layer Vertices

```python
# Count annotations near each skeleton vertex
if cell.skeleton is not None:
    synapse_counts = cell.skeleton.map_annotations_to_label(
        annotation="synapses_detailed",
        distance_threshold=0.3,     # Search radius
        agg="count",                # Count annotations
        chunk_size=1000,
        validate=False
    )
    
    # Add as skeleton label
    cell.skeleton.add_label(synapse_counts, name="synapse_count")
    
    # Aggregate by type
    type_counts = cell.skeleton.map_annotations_to_label(
        annotation="synapses_detailed",
        distance_threshold=0.3,
        agg={
            "excitatory_count": ("synapse_type", lambda x: sum(x == 'excitatory')),
            "inhibitory_count": ("synapse_type", lambda x: sum(x == 'inhibitory')),
            "mean_confidence": ("confidence", "mean"),
            "total_strength": ("size", "sum")
        }
    )
    
    cell.skeleton.add_label(type_counts)
    print(f"Skeleton now has labels: {cell.skeleton.label_names}")
```

### Finding Annotations Linked to Specific Vertices

```python
# Find annotations that map to specific skeleton vertices
target_vertices = cell.skeleton.vertex_index[:2]  # First two skeleton vertices

# Get annotations linked to these vertices
linked_annotation_indices = cell.annotations.synapses_detailed.map_index_to_layer(
    layer="skeleton",
    source_index=target_vertices,
    as_positional=False
)

print(f"Annotations linked to vertices {target_vertices}: {linked_annotation_indices}")

# Reverse mapping: find which skeleton vertex each annotation links to
skeleton_links = cell.annotations.synapses_detailed.map_index_to_layer(
    layer="skeleton",
    as_positional=False
)
print(f"Annotation-to-skeleton mapping: {skeleton_links}")
```

## Annotation Management

### Adding and Removing Annotations

```python
# Add more annotations
additional_spines = np.array([[0.4, 0.4, 0.0], [1.6, -0.3, 0.0]])
cell.add_point_annotations(
    name="additional_spines",
    vertices=additional_spines,
    labels={"spine_type": ["thin", "mushroom"]}
)

# Remove annotation layer
cell.remove_annotation("additional_spines")
print(f"Remaining annotations: {cell.annotations.names}")

# Clear all annotations
original_names = cell.annotations.names.copy()
for name in original_names:
    cell.remove_annotation(name)
print(f"Annotations after clearing: {cell.annotations.names}")
```

### Copying and Transforming Annotations

```python
# Recreate some annotations for demonstration
cell.add_point_annotations("test_synapses", vertices=synapse_locations)

# Copy annotation layer
synapses_copy = cell.annotations.test_synapses.copy()

# Transform annotation coordinates
def shift_up(vertices):
    return vertices + [0, 0, 1.0]  # Move up by 1 unit

transformed_synapses = cell.annotations.test_synapses.transform(
    shift_up, 
    inplace=False
)
print(f"Original Z coords: {cell.annotations.test_synapses.vertices[:, 2]}")
print(f"Transformed Z coords: {transformed_synapses.vertices[:, 2]}")
```

## Advanced Annotation Analysis

### Density Analysis

```python
# Recreate detailed synapses for analysis
cell.add_point_annotations("synapses", vertices=synapse_data)

# Calculate annotation density along skeleton
if cell.skeleton is not None:
    # Density per unit length
    synapse_density = cell.skeleton.map_annotations_to_label(
        annotation="synapses",
        distance_threshold=0.5,
        agg="density",              # Annotations per unit cable length
        chunk_size=1000
    )
    
    cell.skeleton.add_label(synapse_density, name="synapse_density")
    
    # Total density
    total_cable = cell.skeleton.cable_length()
    total_synapses = len(cell.annotations.synapses.vertices)
    overall_density = total_synapses / total_cable
    print(f"Overall synapse density: {overall_density:.3f} synapses/unit")
```

### Spatial Clustering

```python
# Find clusters of annotations
from sklearn.cluster import DBSCAN

coordinates = cell.annotations.synapses.vertices
clustering = DBSCAN(eps=0.3, min_samples=2).fit(coordinates)
cluster_labels = clustering.labels_

# Add cluster information
cell.annotations.synapses.add_label(cluster_labels, name="cluster")

# Analyze clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

### Co-localization Analysis

```python
# Analyze spatial relationship between different annotation types
if "pre_syn" in cell.annotations.names and "post_syn" in cell.annotations.names:
    pre_coords = cell.annotations.pre_syn.vertices
    post_coords = cell.annotations.post_syn.vertices
    
    # Find nearest post-synaptic site for each pre-synaptic site
    from scipy.spatial.distance import cdist
    distances = cdist(pre_coords, post_coords)
    nearest_post = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    # Add co-localization info
    cell.annotations.pre_syn.add_label(nearest_post, name="nearest_post_idx")
    cell.annotations.pre_syn.add_label(min_distances, name="distance_to_post")
    
    print(f"Mean pre-post distance: {np.mean(min_distances):.3f}")
```

## Key Annotation Methods

### Annotation Creation
- `cell.add_point_annotations(name, vertices, spatial_columns=None, labels=None, linkage=None, vertices_from_linkage=False)` - Add annotation layer

### Annotation Management
- `cell.remove_annotation(name)` - Remove annotation layer
- `cell.annotations.names` - List of annotation names
- `cell.annotations[name]` - Access specific annotation

### Annotation Properties
- `annotation.n_vertices` - Number of annotations
- `annotation.vertices` - Coordinate array
- `annotation.nodes` - DataFrame with coordinates and labels
- `annotation.labels` - DataFrame with just labels
- `annotation.label_names` - Available label names

### Data Access and Analysis
- `annotation.get_label(key)` - Get label values
- `annotation.add_label(label, name=None)` - Add new labels
- `annotation.apply_mask(mask, as_positional=False)` - Filter annotations
- `annotation.kdtree` - Spatial queries

### Cross-layer Integration
- `layer.map_annotations_to_label(annotation, distance_threshold, agg="count"/"density", ...)` - Aggregate to layer vertices
- `annotation.map_index_to_layer(layer, source_index=None, as_positional=False)` - Find layer mappings

### Linkage
- `Link(mapping, target, map_value_is_index=True)` - Explicit linkage
- `Link(target, distance_threshold)` - Automatic spatial linkage

!!! note "Additional Features"
    For comprehensive information about vertex access, transformations, spatial queries, and other shared functionality, see [Shared Layer Features](shared_layer_features.md).

!!! tip "When to Use Annotations"
    - Sparse features like synapses, spines, or markers
    - Event locations that don't correspond to morphological vertices
    - Features with rich metadata (confidence, type, size, etc.)
    - Analysis requiring spatial aggregation to morphological layers
    - Co-localization analysis between different feature types