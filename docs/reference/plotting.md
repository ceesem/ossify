# Visualization & Plotting

Ossify provides comprehensive 2D plotting capabilities for neuromorphological data with precise unit control, flexible styling, and publication-ready output.

## Overview

| Function Category | Functions | Purpose |
|-------------------|-----------|---------|
| **[Cell Plotting](#cell-plotting)** | `plot_cell_2d`, `plot_cell_multiview` | Integrated visualization of complete cells |
| **[Layer Plotting](#layer-plotting)** | `plot_morphology_2d`, `plot_annotations_2d`, `plot_skeleton`, `plot_points` | Individual layer visualization |
| **[Figure Management](#figure-management)** | `single_panel_figure`, `multi_panel_figure`, `add_scale_bar` | Layout and annotation utilities |

---

## Cell Plotting {: .doc-heading}

### plot_cell_2d

::: ossify.plot.plot_cell_2d
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Comprehensive 2D visualization of complete cells with skeleton, annotations, and flexible styling.**

#### Usage Examples

```python
import ossify
import matplotlib.pyplot as plt

# Load cell with skeleton and synapses
cell = ossify.load_cell("neuron.osy")

# Basic plot with compartment coloring
fig, ax = ossify.plot_cell_2d(
    cell,
    color="compartment",  # Color by axon/dendrite
    projection="xy",      # XY projection
    synapses=True         # Show both pre/post synapses
)

# Advanced styling with custom parameters
fig, ax = ossify.plot_cell_2d(
    cell,
    color="strahler_order",           # Hierarchical coloring
    palette="viridis",                # Scientific colormap
    linewidth="radius",               # Width by branch radius
    widths=(1, 20),                   # Min/max linewidth range
    alpha=0.8,                        # Semi-transparent
    root_as_sphere=True,              # Mark root location
    root_size=150,                    # Root marker size
    synapses="pre",                   # Only presynaptic sites
    pre_color="red",                  # Synapse color
    syn_size=30,                      # Synapse marker size
    projection="zy",                  # Side view
    units_per_inch=50000,             # 50,000 nm per inch
    despine=True                      # Clean axes
)
plt.show()
```

#### Color and Style Options

```python
# Discrete compartment coloring
plot_cell_2d(cell, color="compartment", palette={"axon": "red", "dendrite": "blue"})

# Continuous property coloring  
plot_cell_2d(cell, color="distance_to_root", palette="plasma", color_norm=(0, 500000))

# Multi-property styling
plot_cell_2d(
    cell,
    color="strahler_order",      # Color by hierarchy
    alpha="confidence",          # Transparency by confidence
    linewidth="thickness",       # Width by morphology
    alpha_norm=(0.3, 1.0),      # Alpha range
    linewidth_norm=(0.5, 5.0),  # Width range
    widths=(2, 40)              # Final width scaling
)
```

#### Synapse Visualization

```python
# Separate pre/post synapse styling
plot_cell_2d(
    cell,
    synapses="both",
    pre_anno="pre_syn",           # Presynaptic annotation layer
    pre_color="red",              # Presynaptic color
    pre_palette="Reds",           # If coloring by property
    post_anno="post_syn",         # Postsynaptic annotation layer  
    post_color="blue",            # Postsynaptic color
    post_palette="Blues",         # If coloring by property
    syn_size="activity",          # Size by synapse activity
    syn_sizes=(10, 100),          # Size range
    syn_alpha=0.7                 # Synapse transparency
)
```

### plot_cell_multiview

::: ossify.plot.plot_cell_multiview
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Multi-panel visualization showing different anatomical projections.**

#### Usage Examples

```python
# Three-panel L-shaped layout (default)
fig, axes = ossify.plot_cell_multiview(
    cell,
    layout="three_panel",        # xy (bottom-left), xz (top-left), zy (bottom-right) 
    color="compartment",
    units_per_inch=100000,       # Scale factor
    gap_inches=0.3               # Panel spacing
)

# Side-by-side comparison (xy | zy)
fig, axes = ossify.plot_cell_multiview(
    cell,
    layout="side_by_side",
    color="strahler_order",
    palette="cmc.hawaii",
    synapses=True,
    units_per_inch=75000
)

# Stacked layout (xz over xy)
fig, axes = ossify.plot_cell_multiview(
    cell,
    layout="stacked", 
    color="distance_to_root",
    palette="magma",
    root_as_sphere=True
)

# Access individual axes
xy_ax = axes["xy"]
xz_ax = axes["xz"] 
zy_ax = axes["zy"]

# Add titles or annotations
xy_ax.set_title("Horizontal View (XY)")
zy_ax.set_title("Side View (ZY)")
```

---

## Layer Plotting {: .doc-heading}

### plot_morphology_2d  

::: ossify.plot.plot_morphology_2d
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Flexible 2D plotting of skeleton morphology with advanced styling.**

#### Usage Examples

```python
# Basic skeleton plotting
ax = ossify.plot_morphology_2d(
    cell.skeleton,           # Can use Cell or SkeletonLayer
    projection="xy"
)

# Advanced styling
ax = ossify.plot_morphology_2d(
    cell,
    color="branch_order",        # Color by topological property
    palette="Set1",              # Discrete colormap
    alpha="confidence",          # Transparency by data quality
    linewidth="diameter",        # Width by branch diameter  
    linewidth_norm=(0.1, 10.0), # Diameter range (μm)
    widths=(0.5, 25),           # Plot width range (points)
    projection="xz",             # Sagittal view
    offset_h=1000,              # Horizontal offset (nm)
    offset_v=500,               # Vertical offset (nm) 
    zorder=3,                   # Drawing order
    invert_y=True,              # Invert Y axis
    ax=existing_axis            # Plot on existing axes
)
```

### plot_annotations_2d

::: ossify.plot.plot_annotations_2d
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Scatter plots for point cloud annotations with flexible styling.**

#### Usage Examples

```python
# Basic annotation plotting
ax = ossify.plot_annotations_2d(
    cell.annotations["pre_syn"],
    projection="xy"
)

# Advanced styling
ax = ossify.plot_annotations_2d(
    cell.annotations["synapses"],
    color="activity_level",      # Color by annotation property
    palette="plasma",            # Continuous colormap
    color_norm=(0, 1),          # Value range for coloring
    size="strength",            # Size by property
    size_norm=(0.1, 2.0),       # Property range
    sizes=(5, 80),              # Marker size range (points²)
    alpha=0.8,                  # Transparency
    projection="zy",            # Side projection
    offset_h=500,               # Position offset
    offset_v=0,
    ax=ax,                      # Add to existing plot
    edgecolors='black',         # Marker edge color
    linewidths=0.5              # Edge width
)
```

### plot_skeleton & plot_points

Lower-level functions for precise control:

::: ossify.plot.plot_skeleton
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

::: ossify.plot.plot_points
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

#### Direct Array Styling

```python
# Direct control with pre-computed arrays
skeleton = cell.skeleton

# Prepare styling arrays
colors = ossify.strahler_number(skeleton)  # Color values
alphas = np.ones(skeleton.n_vertices) * 0.8  # Uniform transparency  
widths = skeleton.get_label("radius") * 2    # Width by radius

# Plot with explicit arrays
ax = ossify.plot_skeleton(
    skeleton,
    projection="xy",
    colors=colors,              # (N, 3) RGB or (N, 4) RGBA array
    alpha=alphas,               # (N,) alpha values  
    linewidths=widths,          # (N,) linewidth values
    zorder=2
)

# Plot points with arrays
points = cell.annotations["pre_syn"].vertices
sizes = np.random.uniform(10, 50, len(points))
colors = np.random.choice(['red', 'blue'], len(points))

ax = ossify.plot_points(
    points=points,
    sizes=sizes,
    colors=colors,
    projection="xy",
    ax=ax,
    zorder=3
)
```

---

## Figure Management {: .doc-heading}

### single_panel_figure

::: ossify.plot.single_panel_figure
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Create figures with precise unit-based sizing for consistent scaling.**

#### Usage Examples

```python
# Calculate data bounds
bounds_min = cell.skeleton.bbox[0]  # [x_min, y_min, z_min]
bounds_max = cell.skeleton.bbox[1]  # [x_max, y_max, z_max]

# Create figure with specific scale
fig, ax = ossify.single_panel_figure(
    data_bounds_min=bounds_min[:2],    # [x_min, y_min] 
    data_bounds_max=bounds_max[:2],    # [x_max, y_max]
    units_per_inch=50000,              # 50,000 nm per inch
    despine=True,                      # Remove axes
    dpi=300                            # High resolution
)

# Plot data on precisely-sized axes
ossify.plot_morphology_2d(cell, ax=ax)

# The figure size automatically matches data aspect ratio
print(f"Figure size: {fig.get_size_inches()}")
```

### multi_panel_figure

::: ossify.plot.multi_panel_figure
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Create aligned multi-panel layouts with consistent scaling.**

#### Usage Examples

```python
# Three-panel layout with precise alignment
fig, axes = ossify.multi_panel_figure(
    data_bounds_min=cell.skeleton.bbox[0],
    data_bounds_max=cell.skeleton.bbox[1], 
    units_per_inch=75000,
    layout="three_panel",
    gap_inches=0.4,                    # Panel spacing
    despine=True,
    dpi=300
)

# Plot different projections
for projection, ax in axes.items():
    ossify.plot_morphology_2d(
        cell,
        projection=projection,
        color="compartment", 
        ax=ax
    )
    ax.set_title(f"{projection.upper()} View")

plt.suptitle("Multi-View Neuron Analysis")
```

### add_scale_bar

::: ossify.plot.add_scale_bar
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Add calibrated scale bars to plots.**

#### Usage Examples

```python
# Add scale bars to plots
fig, ax = ossify.single_panel_figure(
    bounds_min, bounds_max, 
    units_per_inch=100000  # 100,000 nm per inch
)

ossify.plot_morphology_2d(cell, ax=ax)

# Horizontal scale bar (bottom-left)
ossify.add_scale_bar(
    ax,
    length=50000,                      # 50 μm in nm
    position=(0.05, 0.05),            # Fractional position
    label="50 μm",                    # Text label
    color="black",
    linewidth=8,
    fontsize=12
)

# Vertical scale bar (top-right)
ossify.add_scale_bar(
    ax,
    length=25000,                     # 25 μm 
    position=(0.9, 0.7),
    orientation="vertical",           # or "v"
    label="25 μm",
    label_offset=0.02,               # Label spacing
    color="white",                   # For dark backgrounds
    linewidth=6
)
```

---

## Advanced Plotting Workflows

### **Publication Figure Pipeline**

```python
def create_publication_figure(cell, output_path="figure.pdf"):
    \"\"\"Create a publication-ready multi-panel figure\"\"\"
    
    # Calculate compartmentalization
    is_axon = ossify.label_axon_from_synapse_flow(cell)
    compartment = np.where(is_axon, "Axon", "Dendrite")
    cell.skeleton.add_label(compartment, "compartment")
    
    # Create multi-panel layout
    fig, axes = ossify.plot_cell_multiview(
        cell,
        layout="three_panel",
        color="compartment",
        palette={"Axon": "#d62728", "Dendrite": "#1f77b4"},  # Red/Blue
        linewidth="radius",
        widths=(1, 15),
        synapses="both", 
        pre_color="#ff7f0e",       # Orange presynaptic
        post_color="#2ca02c",      # Green postsynaptic
        syn_size=25,
        units_per_inch=100000,     # 100k nm/inch
        gap_inches=0.3,
        despine=True,
        dpi=300
    )
    
    # Add scale bars
    for proj, ax in axes.items():
        if proj == "xy":
            ossify.add_scale_bar(ax, 100000, (0.05, 0.05), "100 μm", fontsize=14)
        elif proj == "xz":  
            ossify.add_scale_bar(ax, 50000, (0.05, 0.05), "50 μm", fontsize=12)
        elif proj == "zy":
            ossify.add_scale_bar(ax, 75000, (0.05, 0.05), "75 μm", fontsize=12)
    
    # Add panel labels
    axes["xy"].text(0.02, 0.98, "A", transform=axes["xy"].transAxes, 
                   fontsize=20, fontweight='bold', va='top')
    axes["xz"].text(0.02, 0.98, "B", transform=axes["xz"].transAxes,
                   fontsize=20, fontweight='bold', va='top')
    axes["zy"].text(0.02, 0.98, "C", transform=axes["zy"].transAxes,
                   fontsize=20, fontweight='bold', va='top')
    
    # Add figure title and save
    fig.suptitle("Neuronal Compartmentalization Analysis", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

# Generate publication figure
fig = create_publication_figure(cell, "neuron_analysis.pdf")
```

### **Interactive Analysis Workflow**

```python
def plot_analysis_comparison(cell, algorithms=['flow', 'spectral']):
    \"\"\"Compare different analysis algorithms visually\"\"\"
    
    results = {}
    
    # Run different algorithms
    if 'flow' in algorithms:
        results['flow'] = ossify.label_axon_from_synapse_flow(cell)
        
    if 'spectral' in algorithms:
        results['spectral'] = ossify.label_axon_from_spectral_split(cell)
    
    # Create comparison plot
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, is_axon) in enumerate(results.items()):
        compartment = np.where(is_axon, "Axon", "Dendrite")
        
        ossify.plot_morphology_2d(
            cell,
            color=compartment,
            palette={"Axon": "red", "Dendrite": "blue"},
            ax=axes[i]
        )
        
        axes[i].set_title(f"{method.capitalize()} Method")
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    return fig, results

# Compare methods
fig, results = plot_analysis_comparison(cell)

# Calculate agreement
if len(results) > 1:
    methods = list(results.keys())
    agreement = (results[methods[0]] == results[methods[1]]).mean()
    print(f"Agreement between methods: {agreement:.1%}")
```

### **Custom Styling Functions**

```python
def create_custom_colormap():
    \"\"\"Create custom colormap for morphological data\"\"\"
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis-like
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def style_by_distance_to_branch(cell, ax):
    \"\"\"Style skeleton by distance to nearest branch point\"\"\"
    skeleton = cell.skeleton
    branch_points = skeleton.branch_points
    
    # Calculate distance to nearest branch for each vertex
    distances = []
    for vertex in skeleton.vertex_index:
        dist_to_branches = skeleton.distance_between(
            [vertex], branch_points, limit=100000
        )
        distances.append(dist_to_branches.min() if len(dist_to_branches) > 0 else np.inf)
    
    distances = np.array(distances)
    
    # Plot with custom styling
    ossify.plot_morphology_2d(
        cell,
        color=distances,
        palette=create_custom_colormap(),
        color_norm=(0, np.percentile(distances[distances < np.inf], 95)),
        linewidth=2,
        ax=ax
    )
    
    return distances

# Apply custom styling
fig, ax = plt.subplots(figsize=(10, 8))
distances = style_by_distance_to_branch(cell, ax)
ax.set_title("Distance to Nearest Branch Point")
```

!!! info "Plotting Design Principles"
    
    **Unit Consistency**: All plotting functions support precise unit control for consistent scaling across figures.
    
    **Flexible Styling**: Multiple ways to specify colors, sizes, and transparency - from simple constants to complex property mappings.
    
    **Layer Integration**: Seamless integration between different data layers in combined plots.
    
    **Publication Ready**: Built-in support for high-DPI output, precise scaling, and clean styling.

!!! tip "Best Practices"
    
    - **Scale Bars**: Always include scale bars for spatial data
    - **Color Schemes**: Use perceptually uniform colormaps (viridis, plasma, cividis) for continuous data
    - **Resolution**: Set `dpi=300` for publication figures
    - **Aspect Ratios**: Use `units_per_inch` for consistent scaling across different data sizes
    - **Layering**: Use `zorder` to control drawing order when combining multiple plot elements