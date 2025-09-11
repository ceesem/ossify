# Algorithms and Analysis

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Ossify provides a suite of algorithms for analyzing neuronal morphology, particularly focused on skeleton trees. These algorithms help classify compartments, analyze branching patterns, and compute morphological properties essential for neuroscience research.

!!! note "Skeleton-Focused Algorithms"
    Most algorithms in ossify are designed for skeleton layers since they represent tree structures essential for morphological analysis. Some algorithms can work with graphs, but they're optimized for tree topologies.

## Tree Analysis Algorithms

### Strahler Number

The Strahler number is a numerical measure of branching complexity, useful for analyzing dendritic trees and river networks.

```python
import ossify
import numpy as np

# Load a real neuron for analysis
cell = ossify.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')

# Compute Strahler numbers
from ossify.algorithms import strahler_number
strahler_values = strahler_number(cell.skeleton)

# Add as skeleton label
cell.skeleton.add_label(strahler_values, name="strahler")

print(f"Strahler order range: {np.min(strahler_values)} - {np.max(strahler_values)}")
print(f"Max Strahler order: {np.max(strahler_values)}")
print(f"Unique Strahler orders: {len(np.unique(strahler_values))}")

# Strahler number interpretation:
# - End points (tips): Strahler = 1  
# - Branch points: max(children) if all children different, max+1 if some equal
# - Higher orders represent main stems, lower orders represent fine branches
```

### Strahler Analysis

```python
# Analyze branching complexity
strahler = cell.skeleton.get_label("strahler")

# Find vertices by Strahler order
for order in range(1, np.max(strahler) + 1):
    vertices_at_order = cell.skeleton.vertex_index[strahler == order]
    print(f"Strahler order {order}: {len(vertices_at_order)} vertices")

# Higher-order branches (main stems)
main_stem_mask = strahler == np.max(strahler)
main_stem_vertices = cell.skeleton.vertex_index[main_stem_mask]

# Lower-order branches (fine processes)
fine_processes_mask = strahler == 1
fine_process_vertices = cell.skeleton.vertex_index[fine_processes_mask]

print(f"Main stem vertices: {len(main_stem_vertices)}")
print(f"Fine process vertices: {len(fine_process_vertices)}")
```

## Compartment Classification

### Axon/Dendrite Classification by Synapse Flow

Classify neuron compartments as axon or dendrite based on synapse distributions:

```python
# Add synapse annotations for classification
pre_syn_locations = np.array([
    [1.8, 0.1, 0.0],   # Near end of one branch
    [1.9, 1.0, 0.0],   # Near another end
])

post_syn_locations = np.array([
    [0.2, 0.1, 0.0],   # Near root
    [0.8, 0.1, 0.0],   # Middle region
    [1.1, -0.8, 0.0], # Another branch
])

cell.add_point_annotations("pre_syn", vertices=pre_syn_locations)
cell.add_point_annotations("post_syn", vertices=post_syn_locations)

# Classify compartments using synapse flow
is_axon = ossify.label_axon_from_synapse_flow(
    cell=cell,
    pre_syn="pre_syn",           # Presynaptic annotation name
    post_syn="post_syn",         # Postsynaptic annotation name
    extend_label_to_segment=True,  # Extend labels to full segments
    ntimes=1,                    # Number of split iterations
    return_segregation_index=False,
    segregation_index_threshold=0.0,  # Minimum segregation to accept split
    as_postitional=False         # Use vertex indices
)

# Add classification as label
cell.skeleton.add_label(is_axon, name="is_axon")

# Analyze results
axon_vertices = cell.skeleton.vertex_index[is_axon]
dendrite_vertices = cell.skeleton.vertex_index[~is_axon]

print(f"Axon vertices: {len(axon_vertices)}")
print(f"Dendrite vertices: {len(dendrite_vertices)}")

# Convert to compartment labels (0=dendrite, 1=axon)
compartment_labels = is_axon.astype(int)
cell.skeleton.add_label(compartment_labels, name="compartment")
```

### Advanced Synapse Flow Classification

```python
# Multiple iterations for better classification
is_axon_multi, segregation_idx = ossify.label_axon_from_synapse_flow(
    cell=cell,
    pre_syn="pre_syn",
    post_syn="post_syn",
    extend_label_to_segment=True,
    ntimes=3,                    # Multiple iterations
    return_segregation_index=True,  # Return quality metric
    segregation_index_threshold=0.1,  # Require minimum segregation
)

print(f"Segregation index: {segregation_idx:.3f}")
print(f"Classification quality: {'Good' if segregation_idx > 0.3 else 'Poor'}")

# Segregation index interpretation:
# - Close to 1: Strong segregation (pre/post spatially separated)
# - Close to 0: Poor segregation (pre/post mixed)
```

### Spectral Compartment Classification

Alternative classification method using spectral analysis:

```python
# Spectral split method (smoother boundaries)
is_axon_spectral = ossify.label_axon_from_spectral_split(
    cell=cell,
    pre_syn="pre_syn",
    post_syn="post_syn",
    aggregation_distance=1.0,    # Distance to aggregate synapses
    smoothing_alpha=0.99,        # Smoothing strength (0-1)
    axon_bias=0,                 # Bias toward axon classification
    raw_split=False,             # Apply refinement
    extend_label_to_segment=True,
    max_times=None,              # Maximum iterations
    segregation_index_threshold=0.5,
    return_segregation_index=False
)

cell.skeleton.add_label(is_axon_spectral, name="is_axon_spectral")

# Compare methods
flow_axon = cell.skeleton.get_label("is_axon")
spectral_axon = cell.skeleton.get_label("is_axon_spectral")

agreement = np.sum(flow_axon == spectral_axon) / len(flow_axon)
print(f"Method agreement: {agreement:.2%}")
```

## Synapse Analysis

### Synapse Betweenness

Measure how many synapse-to-synapse paths pass through each vertex:

```python
# Convert annotations to positional indices for algorithm
pre_syn_indices = cell.annotations.pre_syn.map_index_to_layer(
    "skeleton", as_positional=True
)
post_syn_indices = cell.annotations.post_syn.map_index_to_layer(
    "skeleton", as_positional=True
)

# Compute synapse betweenness
syn_betweenness = ossify.synapse_betweenness(
    skel=cell.skeleton,
    pre_inds=pre_syn_indices,
    post_inds=post_syn_indices
)

cell.skeleton.add_label(syn_betweenness, name="synapse_betweenness")

# Find vertices with high synapse traffic
high_traffic_threshold = np.percentile(syn_betweenness, 90)
high_traffic_vertices = cell.skeleton.vertex_index[syn_betweenness > high_traffic_threshold]

print(f"High synapse traffic vertices: {len(high_traffic_vertices)}")
print(f"Max betweenness: {np.max(syn_betweenness)}")
```

### Segregation Index

Quantify how well pre- and post-synaptic sites are segregated:

```python
# Count synapses by compartment
axon_mask = cell.skeleton.get_label("is_axon")

# Count pre/post synapses in each compartment
axon_pre = len(cell.annotations.pre_syn.map_index_to_layer("skeleton")[
    cell.skeleton.vertex_index[axon_mask]
])
axon_post = len(cell.annotations.post_syn.map_index_to_layer("skeleton")[
    cell.skeleton.vertex_index[axon_mask]
])

dendrite_pre = len(cell.annotations.pre_syn.map_index_to_layer("skeleton")[
    cell.skeleton.vertex_index[~axon_mask]
])
dendrite_post = len(cell.annotations.post_syn.map_index_to_layer("skeleton")[
    cell.skeleton.vertex_index[~axon_mask]
])

# Calculate segregation index
segregation = ossify.segregation_index(
    axon_pre=axon_pre,
    axon_post=axon_post,
    dendrite_pre=dendrite_pre,
    dendrite_post=dendrite_post
)

print(f"Segregation index: {segregation:.3f}")
print(f"Axon: {axon_pre} pre, {axon_post} post")
print(f"Dendrite: {dendrite_pre} pre, {dendrite_post} post")
```

## Label Smoothing

Smooth discrete labels along the skeleton topology:

```python
# Create noisy compartment labels
np.random.seed(42)
noisy_labels = cell.skeleton.get_label("compartment").copy()
# Add some noise
noise_indices = np.random.choice(len(noisy_labels), size=2, replace=False)
noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]

cell.skeleton.add_label(noisy_labels, name="noisy_compartment")

# Smooth the labels
smoothed_labels = ossify.smooth_labels(
    cell=cell.skeleton,
    label=noisy_labels,
    alpha=0.90  # Smoothing strength (0-1, higher = more smoothing)
)

cell.skeleton.add_label(smoothed_labels, name="smoothed_compartment")

# Compare original, noisy, and smoothed
original = cell.skeleton.get_label("compartment")
print(f"Original vs noisy differences: {np.sum(original != noisy_labels)}")
print(f"Original vs smoothed differences: {np.sum(original != (smoothed_labels > 0.5))}")
```

## Morphological Measurements

### Cable Length Analysis

```python
# Total cable length
total_length = cell.skeleton.cable_length()
print(f"Total cable length: {total_length:.2f}")

# Cable length by compartment
axon_length = cell.skeleton.cable_length(
    vertices=cell.skeleton.vertex_index[axon_mask],
    as_positional=False
)
dendrite_length = cell.skeleton.cable_length(
    vertices=cell.skeleton.vertex_index[~axon_mask],
    as_positional=False
)

print(f"Axon cable length: {axon_length:.2f}")
print(f"Dendrite cable length: {dendrite_length:.2f}")
print(f"Axon/Dendrite ratio: {axon_length/dendrite_length:.2f}")

# Cable length by Strahler order
strahler = cell.skeleton.get_label("strahler")
for order in range(1, np.max(strahler) + 1):
    order_vertices = cell.skeleton.vertex_index[strahler == order]
    order_length = cell.skeleton.cable_length(order_vertices, as_positional=False)
    print(f"Strahler {order} cable length: {order_length:.2f}")
```

### Branch Analysis

```python
# Branching statistics
branch_points = cell.skeleton.branch_points
end_points = cell.skeleton.end_points

print(f"Branch points: {len(branch_points)}")
print(f"End points: {len(end_points)}")
print(f"Branching ratio: {len(end_points) / max(len(branch_points), 1):.2f}")

# Branch angles (requires additional calculation)
def calculate_branch_angles(skeleton):
    """Calculate angles at branch points."""
    angles = []
    for bp in skeleton.branch_points:
        children = skeleton.child_vertices([bp], as_positional=False)[bp]
        if len(children) >= 2:
            # Calculate angles between child branches
            bp_pos = skeleton.vertices[skeleton.vertex_index_map[bp]]
            child_vectors = []
            for child in children:
                child_pos = skeleton.vertices[skeleton.vertex_index_map[child]]
                vector = child_pos - bp_pos
                vector = vector / np.linalg.norm(vector)
                child_vectors.append(vector)
            
            # Calculate angle between first two children
            if len(child_vectors) >= 2:
                dot_product = np.dot(child_vectors[0], child_vectors[1])
                angle = np.arccos(np.clip(dot_product, -1, 1))
                angles.append(np.degrees(angle))
    
    return np.array(angles)

branch_angles = calculate_branch_angles(cell.skeleton)
if len(branch_angles) > 0:
    print(f"Mean branch angle: {np.mean(branch_angles):.1f}°")
    print(f"Branch angle range: {np.min(branch_angles):.1f}° - {np.max(branch_angles):.1f}°")
```

## Distance and Path Analysis

### Root Distance Analysis

```python
# Distance from root
distances_to_root = cell.skeleton.distance_to_root()
cell.skeleton.add_label(distances_to_root, name="distance_to_root")

print(f"Max distance from root: {np.max(distances_to_root):.2f}")
print(f"Mean distance from root: {np.mean(distances_to_root):.2f}")

# Distance distribution by compartment
axon_distances = distances_to_root[axon_mask]
dendrite_distances = distances_to_root[~axon_mask]

print(f"Axon distance range: {np.min(axon_distances):.2f} - {np.max(axon_distances):.2f}")
print(f"Dendrite distance range: {np.min(dendrite_distances):.2f} - {np.max(dendrite_distances):.2f}")
```

### Path Length Analysis

```python
# Analyze path lengths from tips to root
cover_paths = cell.skeleton.cover_paths
path_lengths = []

for path in cover_paths:
    # Calculate path length
    path_vertices = cell.skeleton.vertices[path]
    path_length = np.sum(np.linalg.norm(np.diff(path_vertices, axis=0), axis=1))
    path_lengths.append(path_length)

path_lengths = np.array(path_lengths)
print(f"Number of paths: {len(path_lengths)}")
print(f"Mean path length: {np.mean(path_lengths):.2f}")
print(f"Path length range: {np.min(path_lengths):.2f} - {np.max(path_lengths):.2f}")

# Longest and shortest paths
longest_path_idx = np.argmax(path_lengths)
shortest_path_idx = np.argmin(path_lengths)

print(f"Longest path: {len(cover_paths[longest_path_idx])} vertices, {path_lengths[longest_path_idx]:.2f} length")
print(f"Shortest path: {len(cover_paths[shortest_path_idx])} vertices, {path_lengths[shortest_path_idx]:.2f} length")
```

## Custom Analysis Workflows

### Complete Morphological Characterization

```python
def characterize_morphology(cell):
    """Complete morphological analysis of a neuron."""
    
    results = {}
    skeleton = cell.skeleton
    
    # Basic topology
    results['n_vertices'] = skeleton.n_vertices
    results['n_edges'] = len(skeleton.edges)
    results['n_branch_points'] = skeleton.n_branch_points
    results['n_end_points'] = skeleton.n_end_points
    
    # Cable metrics
    results['total_cable_length'] = skeleton.cable_length()
    
    # Strahler analysis
    strahler = ossify.strahler_number(skeleton)
    results['max_strahler_order'] = np.max(strahler)
    results['strahler_complexity'] = len(np.unique(strahler))
    
    # Distance metrics
    distances = skeleton.distance_to_root()
    results['max_distance_from_root'] = np.max(distances)
    results['mean_distance_from_root'] = np.mean(distances)
    
    # Path analysis
    cover_paths = skeleton.cover_paths
    path_lengths = []
    for path in cover_paths:
        vertices = skeleton.vertices[path]
        length = np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))
        path_lengths.append(length)
    
    results['n_primary_paths'] = len(path_lengths)
    results['mean_path_length'] = np.mean(path_lengths)
    results['max_path_length'] = np.max(path_lengths)
    
    return results

# Analyze morphology
morphology_stats = characterize_morphology(cell)
for metric, value in morphology_stats.items():
    print(f"{metric}: {value}")
```

### Synapse Distribution Analysis

```python
def analyze_synapse_distribution(cell):
    """Analyze spatial distribution of synapses."""
    
    if "pre_syn" not in cell.annotations.names or "post_syn" not in cell.annotations.names:
        print("No synapse annotations found")
        return
    
    # Map synapses to skeleton
    pre_counts = cell.skeleton.map_annotations_to_label(
        "pre_syn", distance_threshold=0.5, agg="count"
    )
    post_counts = cell.skeleton.map_annotations_to_label(
        "post_syn", distance_threshold=0.5, agg="count"
    )
    
    # Synapse density
    cable_lengths = cell.skeleton.half_edge_length
    pre_density = pre_counts / (cable_lengths + 1e-10)  # Avoid division by zero
    post_density = post_counts / (cable_lengths + 1e-10)
    
    cell.skeleton.add_label(pre_density, name="pre_synapse_density")
    cell.skeleton.add_label(post_density, name="post_synapse_density")
    
    print(f"Total pre-synapses: {np.sum(pre_counts)}")
    print(f"Total post-synapses: {np.sum(post_counts)}")
    print(f"Mean pre-synapse density: {np.mean(pre_density):.3f} synapses/unit")
    print(f"Mean post-synapse density: {np.mean(post_density):.3f} synapses/unit")
    
    # Hot spots (high synapse density)
    pre_hotspots = cell.skeleton.vertex_index[pre_density > np.percentile(pre_density, 90)]
    post_hotspots = cell.skeleton.vertex_index[post_density > np.percentile(post_density, 90)]
    
    print(f"Pre-synapse hotspots: {len(pre_hotspots)} vertices")
    print(f"Post-synapse hotspots: {len(post_hotspots)} vertices")

analyze_synapse_distribution(cell)
```

## Key Algorithm Functions

### Tree Analysis
- `ossify.strahler_number(cell)` - Compute Strahler numbers for branching complexity
- `ossify.smooth_labels(cell, label, alpha=0.90)` - Smooth labels along skeleton topology

### Compartment Classification
- `ossify.label_axon_from_synapse_flow(cell, pre_syn, post_syn, extend_label_to_segment=False, ntimes=1, ...)` - Classify using synapse flow
- `ossify.label_axon_from_spectral_split(cell, pre_syn, post_syn, aggregation_distance=1, smoothing_alpha=0.99, ...)` - Classify using spectral method

### Synapse Analysis
- `ossify.synapse_betweenness(skel, pre_inds, post_inds)` - Compute synapse traffic through vertices
- `ossify.segregation_index(axon_pre, axon_post, dendrite_pre, dendrite_post)` - Quantify compartment segregation

### Morphological Measurements
- `skeleton.cable_length(vertices=None, as_positional=False)` - Cable length measurements
- `skeleton.distance_to_root(vertices=None, as_positional=False)` - Distance from root
- `skeleton.cover_paths` / `skeleton.segments` - Path and segment analysis

!!! tip "Algorithm Best Practices"
    - Ensure synapses are properly linked to skeleton before classification
    - Use `extend_label_to_segment=True` for biologically meaningful compartments
    - Validate classification results with `segregation_index`
    - Combine multiple metrics for robust morphological characterization
    - Consider smoothing noisy labels with `smooth_labels()`