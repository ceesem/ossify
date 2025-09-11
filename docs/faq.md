# Frequently Asked Questions

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

This FAQ addresses common questions about using ossify effectively, including when to use different features and best practices for morphological analysis.

## Choosing the Right Data Structure

### When should I use graphs vs skeletons?

**Use skeletons when:**
- Working with neuronal morphologies (dendrites, axons)
- You have tree-structured data with a clear root
- You need tree-specific analysis (Strahler numbers, path analysis, compartment classification)
- Data represents a single connected component without cycles

**Use graphs when:**
- Working with level-2 graph data from connectomics pipelines
- Your network has cycles or multiple disconnected components
- You need general network analysis (shortest paths between arbitrary nodes)
- Data doesn't have a natural root or tree structure

```python
# Skeleton: Tree with root, no cycles
cell.add_skeleton(vertices, edges, root=0)  # Root required

# Graph: General network, cycles OK
cell.add_graph(vertices, edges)  # No root needed
```

### When should I use annotations vs layer labels?

**Use annotations for:**
- Sparse, discrete features (synapses, spines, markers)
- Features that don't correspond to existing vertices
- Data you want to map to multiple layers
- Features with their own spatial coordinates

**Use layer labels for:**
- Properties of existing vertices (radius, compartment, quality)
- Dense data where every vertex has a value
- Computational results (Strahler numbers, distances)

```python
# Annotations: Sparse features with own coordinates
cell.add_point_annotations("synapses", vertices=synapse_locations)

# Labels: Properties of existing skeleton vertices
cell.skeleton.add_label([1,2,2,3,3], "compartment")
```

### Should I use meshes or skeletons for surface analysis?

**Use meshes for:**
- Surface area calculations
- Volume measurements
- Detailed surface geometry
- Integration with trimesh/other mesh libraries

**Use skeletons for:**
- Cable length measurements
- Branching analysis
- Topological properties
- Fast visualization and analysis

Many cells have both - use the appropriate layer for each analysis.

### How do I add a mesh to a cell with an L2 skeleton?

The [`cortical_tools`](https://www.csdashm.com/cortical-tools/) package has a function to map mesh vertices to L2 IDs, which can then be used to link the mesh to the graph layer in ossify.
It takes one to several minutes to compute the mapping for large meshes, but can be useful for generating surface mesh visualizations masked by compartment or other labels. The function `compute_vertex_to_l2_mapping` provides an L2 ID for each mesh vertex, which can be aligned with an ossify cell via the "graph" layer.

```python
import ossify as osy
from cortical_tools.datasets.microns_public import client # note that this is NOT a CAVEclient

root_id = 864691135336055529
cell = osy.load_cell_from_client(
    root_id,
    client.cave,
    synapses=True,
    restore_graph=True,
)

mesh = client.mesh.get_mesh(root_id)
mesh_labels = client.mesh.compute_vertex_to_l2_mapping(root_id=root_id, vertices=mesh.vertices, faces=mesh.faces)

cell.add_mesh(
    vertices=mesh.vertices,
    faces=mesh.faces,
    linkage=Link(mesh_labels, target='graph')
)
```

## Data Management

### How do I handle different coordinate units?

Always use the cell-level `transform()` method to ensure all layers are converted consistently:

```python
# Convert from nanometers to micrometers
cell_um = cell.transform(lambda x: x / 1000)

# Don't do this - only transforms skeleton
vertices_um = cell.skeleton.vertices / 1000  # Wrong!
```

### How do I work with large datasets efficiently?

- **Use masking** to focus on subsets rather than loading everything
- **Load data progressively** - start with skeleton, add other layers as needed
- **Cache expensive computations** like Strahler numbers
- **Use context managers** for temporary analysis

```python
# Efficient: analyze subsets
with cell.skeleton.mask_context(quality_mask) as high_quality:
    results = expensive_analysis(high_quality)

# Inefficient: analyze everything then filter
all_results = expensive_analysis(cell.skeleton)
filtered = all_results[quality_mask]
```

## Analysis Patterns

### What's the typical analysis workflow?

1. **Load and explore** your cell
2. **Mask to region of interest** if needed
3. **Apply algorithms** (Strahler, compartment classification)
4. **Visualize results** with appropriate coloring
5. **Extract quantitative measurements**

```python
# Standard workflow
cell = ossify.load_cell(path)
cell.describe()  # Explore structure

# Focus on high-quality data
with cell.mask_context("skeleton", quality > 0.8) as clean_cell:
    # Add analysis
    strahler = strahler_number(clean_cell.skeleton)
    clean_cell.skeleton.add_label(strahler, "strahler")
    
    # Visualize
    fig = ossify.plot_cell_2d(clean_cell, color="strahler")
    
    # Measure
    cable_length = clean_cell.skeleton.cable_length()
```

### How do I compare morphologies across cells?

- **Normalize units** consistently across cells
- **Use common reference frames** (align to soma, etc.)
- **Extract comparable metrics** (cable length, branch points, Strahler max)
- **Account for data quality** differences

```python
# Normalize for comparison
cells_um = [cell.transform(lambda x: x / 1000) for cell in cells]

# Extract comparable metrics
metrics = []
for cell in cells_um:
    metrics.append({
        'cable_length': cell.skeleton.cable_length(),
        'branch_points': len(cell.skeleton.branch_points),
        'max_strahler': cell.skeleton.get_label('strahler').max()
    })
```

## Masking and Filtering

### When should I use temporary vs permanent masking?

**Use temporary masking (context managers) for:**
- Exploratory analysis
- Computing statistics on subsets
- Comparing different filter criteria
- When you need the original data later

**Use permanent masking (`apply_mask`) for:**
- Creating clean datasets for sharing
- Removing low-quality data permanently
- Creating focused datasets for specific analysis

```python
# Temporary: original unchanged
with cell.mask_context("skeleton", mask) as subset:
    analyze(subset)

# Permanent: creates new filtered cell
clean_cell = cell.apply_mask("skeleton", mask)
```

### How do I combine multiple mask criteria?

Use boolean logic to combine conditions:

```python
# Multiple criteria
quality = skeleton.get_label("quality")
compartment = skeleton.get_label("compartment") 
distance = skeleton.distance_to_root()

# Combine with boolean logic
axon_mask = (compartment == 2) & (quality > 0.8)
proximal_mask = (distance < 50_000) & (quality > 0.7)
complex_mask = (quality > 0.9) | ((compartment == 3) & (distance < 20_000))
```

## Visualization

### Why can I only plot skeletons?

Currently, ossify's plotting functions are optimized for skeleton analysis since:
- Skeletons are the most common analysis target
- Tree structures have well-defined visualization patterns
- Most morphological algorithms work on skeleton data

Plotting support for other layer types is planned for future releases.

### How do I create publication-quality figures?

- **Use consistent units** (convert to micrometers for readability)
- **Set precise scaling** with `units_per_inch`
- **Choose appropriate colormaps** (compartment-specific colors, perceptually uniform for continuous data)
- **Add scale bars** for spatial reference
- **Use high DPI** for crisp output

```python
# Publication workflow
display_cell = cell.transform(lambda x: x / 1000)  # Convert to μm

fig, ax = ossify.single_panel_figure(
    data_bounds_min=display_cell.skeleton.bbox[0],
    data_bounds_max=display_cell.skeleton.bbox[1],
    units_per_inch=50,  # 50 μm per inch
    despine=True,
    dpi=300
)

ossify.plot_morphology_2d(
    display_cell,
    color="compartment",
    palette={1: 'navy', 2: 'tomato', 3: 'forestgreen'},
    linewidth="radius",
    root_marker=True,
    ax=ax
)
```

## Integration and Import

### What's the difference between .osy and MeshWork files?

- **.osy files**: Native ossify format, preserves all features, recommended for new projects
- **MeshWork files**: Legacy format, use `import_legacy_meshwork()` for existing data
- **CAVEclient**: Live data from databases, use `load_cell_from_client()`

### How do I handle missing data when importing?

```python
# Check what's available after import
cell = ossify.load_cell(path)
cell.describe()  # Shows available layers and annotations

# Handle missing layers gracefully
if cell.skeleton is not None:
    cable_length = cell.skeleton.cable_length()
else:
    print("No skeleton data available")

# Check for specific annotations
if hasattr(cell.annotations, 'pre_syn'):
    n_synapses = len(cell.annotations.pre_syn)
```

## Performance and Memory

### How do I work with very large cells?

- **Start with quality filtering** to reduce data size
- **Use spatial masking** to focus on regions of interest
- **Process compartments separately** instead of whole cells
- **Cache intermediate results** to avoid recomputation

```python
# Efficient large cell workflow
cell = ossify.load_cell(large_cell_path)

# Filter early
quality_mask = cell.skeleton.get_label("quality") > 0.9
with cell.mask_context("skeleton", quality_mask) as clean_cell:
    
    # Process compartments separately
    for compartment_id in [2, 3]:  # axon, dendrite
        comp_mask = clean_cell.skeleton.get_label("compartment") == compartment_id
        with clean_cell.mask_context("skeleton", comp_mask) as comp_cell:
            analyze_compartment(comp_cell, compartment_id)
```

### Should I keep multiple cells in memory?

For batch processing:
- **Load one at a time** if memory is limited
- **Extract metrics immediately** and store results
- **Use generators** for large datasets
- **Save intermediate results** to disk

```python
# Memory-efficient batch processing
def process_cell_batch(cell_paths):
    results = []
    for path in cell_paths:
        cell = ossify.load_cell(path)
        
        # Extract metrics immediately
        metrics = extract_metrics(cell)
        results.append(metrics)
        
        # Cell goes out of scope and can be garbage collected
        del cell
    
    return results
```

## Common Patterns

### How do I find synapses near branch points?

```python
# Get branch points
branch_points = cell.skeleton.branch_points
branch_coords = cell.skeleton.vertices[branch_points]

# Find nearby synapses
if hasattr(cell.annotations, 'pre_syn'):
    synapse_coords = cell.annotations.pre_syn.vertices
    
    # Use spatial queries
    nearby_synapses = []
    for bp_coord in branch_coords:
        distances = np.linalg.norm(synapse_coords - bp_coord, axis=1)
        nearby = synapse_coords[distances < 1000]  # Within 1μm
        nearby_synapses.extend(nearby)
```

### How do I compute path-specific metrics?

```python
# Analyze individual paths
for path in cell.skeleton.cover_paths:
    path_mask = cell.skeleton.vertex_index.isin(path)
    
    with cell.skeleton.mask_context(path_mask) as path_skeleton:
        path_length = path_skeleton.cable_length()
        path_synapses = count_synapses_on_path(path_skeleton)
        
        print(f"Path: {path_length:.0f} nm, {path_synapses} synapses")
```

## Troubleshooting

### Why are my edge counts different after masking?

Masking removes vertices, and edges are automatically filtered to maintain valid connections. Only edges between retained vertices are kept:

```python
# Check edge preservation
print(f"Original: {skeleton.n_vertices} vertices, {len(skeleton.edges)} edges")
filtered = skeleton.apply_mask(mask, as_positional=True)
print(f"Filtered: {filtered.n_vertices} vertices, {len(filtered.edges)} edges")

# Edges are remapped to new vertex indices
```

### Why don't my annotations show up after masking?

Annotations are filtered based on their linkage to the masked layer:

```python
# Check linkage
if hasattr(cell.annotations, 'synapses'):
    linkage = cell.annotations.synapses.linkage
    print(f"Synapses linked to: {linkage.target}")
    
    # Annotations follow their linked layer's masking
```

### How do I debug coordinate system issues?

```python
# Check coordinate ranges and units
print(f"Skeleton bbox: {cell.skeleton.bbox}")
print(f"Coordinate range: {cell.skeleton.vertices.min(axis=0)} to {cell.skeleton.vertices.max(axis=0)}")

# Check if units make sense (nm vs μm vs mm)
cable_length = cell.skeleton.cable_length()
print(f"Cable length: {cable_length} (units: nm if ~millions, μm if ~thousands)")
```

---

!!! tip "Getting Help"
    If you have questions not covered here, please [open an issue on GitHub](https://github.com/ceesem/ossify/issues) with a minimal example of what you're trying to accomplish.