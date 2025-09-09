# Analysis & Algorithms

Ossify provides computational methods for analyzing neuromorphological structures, with a focus on tree topology, synapse analysis, and signal processing on neuronal graphs.

## Overview

| Algorithm Category | Functions | Purpose |
|--------------------|-----------|---------|
| **[Morphological Analysis](#morphological-analysis)** | `strahler_number` | Tree structure characterization |
| **[Synapse Analysis](#synapse-analysis)** | `synapse_betweenness`, `label_axon_from_*`, `segregation_index` | Connectivity and compartmentalization |
| **[Smoothing & Filtering](#smoothing-and-filtering)** | `smooth_labels` | Signal processing on graphs |

---

## Morphological Analysis {: .doc-heading}

### strahler_number

::: ossify.algorithms.strahler_number
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false
        
**The Strahler number provides a hierarchical ordering of tree branches, starting from 1 at terminal branches and incrementing when branches of equal order merge.**

#### Usage Example

```python
import ossify

# Load a cell with skeleton
cell = ossify.load_cell("neuron.osy")

# Calculate Strahler numbers
strahler = ossify.strahler_number(cell.skeleton)

# Add as label to skeleton
cell.skeleton.add_label(strahler, "strahler_order")

# Visualize with color coding
fig, ax = ossify.plot_morphology_2d(
    cell, 
    color="strahler_order",
    palette="viridis"
)

# Analyze branch hierarchy
max_order = strahler.max()
primary_branches = (strahler == max_order).sum()
print(f"Maximum Strahler order: {max_order}")
print(f"Primary branches: {primary_branches}")
```

#### Interpretation

- **Order 1**: Terminal branches (end segments)
- **Order 2**: Branches formed by merging two Order 1 branches  
- **Order n**: Branches formed by merging two Order (n-1) branches
- **Higher orders** indicate more "central" or "primary" branches

---

## Synapse Analysis {: .doc-heading}

### synapse_betweenness  

::: ossify.algorithms.synapse_betweenness
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Calculates betweenness centrality for synaptic flow, measuring how many presynaptic→postsynaptic paths pass through each vertex.**

#### Usage Example

```python
# Get synapse locations mapped to skeleton
pre_syn_ids = cell.annotations["pre_syn"].map_index_to_layer(
    "skeleton", as_positional=True
)
post_syn_ids = cell.annotations["post_syn"].map_index_to_layer(
    "skeleton", as_positional=True
)

# Calculate synapse betweenness
betweenness = ossify.synapse_betweenness(
    cell.skeleton, 
    pre_syn_ids, 
    post_syn_ids
)

# Add as skeleton label
cell.skeleton.add_label(betweenness, "syn_betweenness")

# Find vertices with highest betweenness (potential branch points)
high_betweenness = betweenness > np.percentile(betweenness, 95)
```

### label_axon_from_synapse_flow

::: ossify.algorithms.label_axon_from_synapse_flow
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Identifies axon vs dendrite compartments based on synaptic connectivity patterns using flow-based analysis.**

#### Usage Example

```python
# Basic axon identification
is_axon = ossify.label_axon_from_synapse_flow(
    cell,
    pre_syn="pre_syn",      # Annotation layer with presynaptic sites
    post_syn="post_syn",    # Annotation layer with postsynaptic sites
    extend_label_to_segment=True,  # Extend to full segments
    return_segregation_index=True  # Return quality metric
)

if isinstance(is_axon, tuple):
    axon_mask, segregation = is_axon
    print(f"Segregation quality: {segregation:.3f}")
else:
    axon_mask = is_axon

# Add compartment labels
compartment = np.where(axon_mask, "axon", "dendrite")
cell.skeleton.add_label(compartment, "compartment")

# Iterative refinement for complex morphologies
is_axon_refined = ossify.label_axon_from_synapse_flow(
    cell,
    pre_syn="pre_syn",
    post_syn="post_syn", 
    ntimes=3,  # Multiple iterations
    segregation_index_threshold=0.5  # Quality threshold
)
```

### label_axon_from_spectral_split

::: ossify.algorithms.label_axon_from_spectral_split
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Alternative axon identification using spectral analysis of smoothed synapse density.**

#### Usage Example

```python
# Spectral method with density smoothing
is_axon_spectral = ossify.label_axon_from_spectral_split(
    cell,
    pre_syn="pre_syn",
    post_syn="post_syn",
    aggregation_distance=2000,      # Synapse aggregation radius (nm)
    smoothing_alpha=0.95,           # Smoothing strength
    axon_bias=0.1,                  # Bias towards axon classification
    segregation_index_threshold=0.6
)

# Compare with flow-based method
agreement = (is_axon_spectral == axon_mask).mean()
print(f"Method agreement: {agreement:.2%}")
```

### segregation_index

::: ossify.algorithms.segregation_index
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Quantifies the segregation of presynaptic and postsynaptic sites between compartments.**

#### Usage Example

```python
# Calculate segregation metrics
axon_vertices = cell.skeleton.vertex_index[axon_mask]
dendrite_vertices = cell.skeleton.vertex_index[~axon_mask]

# Count synapses by compartment
pre_axon = cell.annotations["pre_syn"].map_mask_to_layer("skeleton", axon_mask).sum()
pre_dendrite = len(cell.annotations["pre_syn"]) - pre_axon
post_axon = cell.annotations["post_syn"].map_mask_to_layer("skeleton", axon_mask).sum()  
post_dendrite = len(cell.annotations["post_syn"]) - post_axon

# Calculate segregation index
seg_index = ossify.segregation_index(
    axon_pre=pre_axon,
    axon_post=post_axon,
    dendrite_pre=pre_dendrite, 
    dendrite_post=post_dendrite
)

print(f"Segregation index: {seg_index:.3f}")
print(f"Pre/post ratio (axon): {pre_axon/(post_axon+1):.2f}")
print(f"Pre/post ratio (dendrite): {pre_dendrite/(post_dendrite+1):.2f}")
```

#### Interpretation

- **0.0**: No segregation (random distribution)
- **1.0**: Perfect segregation (complete separation)
- **>0.7**: Strong segregation (typical for mature neurons)
- **<0.3**: Poor segregation (may indicate immature or damaged neurons)

---

## Smoothing & Filtering {: .doc-heading}

### smooth_labels

::: ossify.algorithms.smooth_labels
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Applies graph-based smoothing to labels using a heat equation approach.**

#### Usage Example

```python
# Create noisy label data
np.random.seed(42)
noisy_signal = np.random.randn(cell.skeleton.n_vertices)

# Apply different levels of smoothing
smoothed_light = ossify.smooth_labels(cell.skeleton, noisy_signal, alpha=0.5)
smoothed_heavy = ossify.smooth_labels(cell.skeleton, noisy_signal, alpha=0.95)

# Add to skeleton for comparison
cell.skeleton.add_label(noisy_signal, "original")
cell.skeleton.add_label(smoothed_light, "smoothed_light") 
cell.skeleton.add_label(smoothed_heavy, "smoothed_heavy")

# Multi-channel smoothing
synapse_density = np.column_stack([
    cell.skeleton.map_annotations_to_label("pre_syn", distance_threshold=1000, agg="count"),
    cell.skeleton.map_annotations_to_label("post_syn", distance_threshold=1000, agg="count")
])

smoothed_density = ossify.smooth_labels(
    cell.skeleton, 
    synapse_density, 
    alpha=0.9
)

# Extract smoothed channels
pre_density_smooth = smoothed_density[:, 0]
post_density_smooth = smoothed_density[:, 1]
```

#### Parameters

- **`alpha`**: Smoothing strength (0=no smoothing, 1=maximum smoothing)
- **Input shape**: Can be 1D (single label) or 2D (multiple labels)
- **Output**: Same shape as input with smoothed values

---

## Workflow Examples

### **Complete Compartment Analysis Pipeline**

```python
def analyze_compartmentalization(cell, min_segregation=0.5):
    \"\"\"Complete pipeline for axon-dendrite analysis\"\"\"
    
    # 1. Calculate synapse betweenness
    pre_ids = cell.annotations["pre_syn"].map_index_to_layer("skeleton", as_positional=True)
    post_ids = cell.annotations["post_syn"].map_index_to_layer("skeleton", as_positional=True) 
    
    betweenness = ossify.synapse_betweenness(cell.skeleton, pre_ids, post_ids)
    
    # 2. Flow-based compartmentalization
    axon_flow, seg_flow = ossify.label_axon_from_synapse_flow(
        cell, 
        return_segregation_index=True,
        segregation_index_threshold=min_segregation
    )
    
    # 3. Spectral alternative
    axon_spectral = ossify.label_axon_from_spectral_split(
        cell,
        segregation_index_threshold=min_segregation
    )
    
    # 4. Method comparison
    agreement = (axon_flow == axon_spectral).mean()
    
    # 5. Add all results as labels
    cell.skeleton.add_label(betweenness, "syn_betweenness")
    cell.skeleton.add_label(axon_flow.astype(str), "axon_flow")
    cell.skeleton.add_label(axon_spectral.astype(str), "axon_spectral")
    
    results = {
        'segregation_flow': seg_flow,
        'method_agreement': agreement,
        'axon_fraction_flow': axon_flow.mean(),
        'axon_fraction_spectral': axon_spectral.mean()
    }
    
    return results

# Run analysis
results = analyze_compartmentalization(cell)
print(f"Flow segregation: {results['segregation_flow']:.3f}")
print(f"Method agreement: {results['method_agreement']:.2%}")
```

### **Morphological Feature Extraction**

```python
def extract_morphology_features(cell):
    \"\"\"Extract comprehensive morphological features\"\"\"
    
    skeleton = cell.skeleton
    
    # Basic metrics
    total_length = skeleton.cable_length()
    n_branches = len(skeleton.branch_points)
    n_tips = len(skeleton.end_points)
    
    # Strahler analysis
    strahler = ossify.strahler_number(skeleton)
    max_order = strahler.max()
    primary_branches = (strahler == max_order).sum()
    
    # Compartment analysis (if synapses available)
    if "pre_syn" in cell.annotations.names:
        axon_mask, segregation = ossify.label_axon_from_synapse_flow(
            cell, return_segregation_index=True
        )
        axon_length = skeleton.cable_length(skeleton.vertex_index[axon_mask])
        dendrite_length = total_length - axon_length
    else:
        segregation = None
        axon_length = None
        dendrite_length = None
    
    # Distance metrics
    distances = skeleton.distance_to_root()
    max_distance = distances.max()
    mean_distance = distances.mean()
    
    features = {
        'total_length': total_length,
        'n_branches': n_branches, 
        'n_tips': n_tips,
        'max_strahler_order': max_order,
        'primary_branches': primary_branches,
        'max_distance_to_root': max_distance,
        'mean_distance_to_root': mean_distance,
        'segregation_index': segregation,
        'axon_length': axon_length,
        'dendrite_length': dendrite_length
    }
    
    return features

# Extract features
features = extract_morphology_features(cell)
for name, value in features.items():
    if value is not None:
        print(f"{name}: {value:.2f}")
```

!!! info "Algorithm Design Principles"
    
    **Graph-Based**: All algorithms leverage the graph structure of neuronal trees for efficient computation.
    
    **Biologically Informed**: Methods incorporate known principles of neuronal organization (e.g., synaptic segregation).
    
    **Flexible Input**: Functions accept both `Cell` objects and individual `SkeletonLayer` instances.
    
    **Quality Metrics**: Many functions return quality/confidence measures alongside results.

!!! tip "Performance & Usage Tips"
    
    - **Preprocessing**: Use `smooth_labels()` to reduce noise in raw synapse data
    - **Parameter Tuning**: Adjust `segregation_index_threshold` based on data quality
    - **Method Comparison**: Use multiple algorithms and compare results for robustness
    - **Batch Processing**: Process multiple cells using the same parameters for consistency