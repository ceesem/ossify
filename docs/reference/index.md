# API Reference

Ossify is a Python package for working with neuromorphological data, providing tools for analyzing and visualizing neuron structures across multiple geometric representations.

## Quick Start

```python
import ossify

example_cell_path = "https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy"
# Load a cell from file locally or in the cloud
cell = ossify.load_cell(example_cell_path)

# Create from CAVE client (requires caveclient)
cell = ossify.load_cell_from_client(root_id=12345, client=cave_client)

# Analyze morphology
strahler = ossify.strahler_number(cell)
is_axon = ossify.label_axon_from_synapse_flow(cell)

# Create visualizations
fig, ax = ossify.plot_cell_2d(cell, color="compartment")
```

## Package Structure

### **Core Classes** {: .text-primary }
The foundation classes for representing neuromorphological data:

- **[`Cell`](core.md#ossify.Cell)**: Main container for morphological data with multiple data layers
- **[`Link`](core.md#ossify.Link)**: Manages relationships between different data layers

### **Data Layer Classes** {: .text-primary }
Specialized classes for different geometric representations:

- **[`SkeletonLayer`](layers.md#ossify.SkeletonLayer)**: Rooted tree-structured neuronal skeletons
- **[`GraphLayer`](layers.md#ossify.GraphLayer)**: Graph-based representation for spatial connectivity
- **[`MeshLayer`](layers.md#ossify.MeshLayer)**: 3D mesh surfaces with face-based geometry  
- **[`PointCloudLayer`](layers.md#ossify.PointCloudLayer)**: Sparse point annotations and markers

### **Analysis & Algorithms** {: .text-secondary }
Computational methods for morphological analysis:

- **[Morphological Analysis](algorithms.md#morphological-analysis)**: Strahler number, compartment classification

### **Visualization & Plotting** {: .text-secondary }
Plotting and visualization utilities:

- **[Cell Plotting](plotting.md#cell-plotting)**: Integrated cell visualization with multiple projections
- **[Figure Management](plotting.md#figure-management)**: Multi-panel layouts and precise sizing

### **File I/O Operations** {: .text-accent }
Loading and saving morphological data:

- **[Core I/O Functions](io.md#core-functions)**: `load_cell()`, `save_cell()`
- **[File Management](io.md#file-management)**: `CellFiles` for cloud and local storage

### **External Integrations** {: .text-accent }
Interfaces with external data sources and tools:

- **[CAVE Integration](external.md#cave-integration)**: `cell_from_client()` for connectome data

## Key Features

### **Multi-Scale Data Integration**
Seamlessly work with data at different scales - from high resolution meshes to coarse skeletons, with automatic mapping between representations.

### **Flexible Analysis Pipeline**
Chain operations across different data types with consistent APIs and automatic data propagation between layers.

### **Publication-Ready Visualization**
Create high-quality figures with precise unit control, multiple projections, and customizable styling.

### **Cloud-Native I/O**
Load and save data from local files, cloud storage (S3, GCS) via `cloud-files`, or directly from CAVE.

---

## Navigation

| Module | Description | Key Classes |
|--------|-------------|-------------|
| **[Core Classes](core.md)** | Foundation classes and containers | `Cell`, `Link` |
| **[Data Layers](layers.md)** | Spatial and graph representation classes | `SkeletonLayer`, `GraphLayer`, `MeshLayer`, `PointCloudLayer` |
| **[Algorithms](algorithms.md)** | Analysis and computation functions | `strahler_number`, `label_axon_*`, `smooth_labels` |
| **[Plotting](plotting.md)** | Visualization and figure creation | `plot_cell_*`, `plot_morphology_*` |
| **[File I/O](io.md)** | Data loading and saving | `load_cell`, `save_cell`, `CellFiles` |
| **[External](external.md)** | Third-party integrations | `cell_from_client` |

!!! tip "Best Practices"
    - Use `Cell.apply_mask()` for non-destructive filtering
    - Use `mask_context()` for temporary operations
    - Use annotations for sparse point-like data and labels for dense data where every vertex has a value.
    - Leverage `Link` objects for complex data relationships  
    - Take advantage of method chaining for concise workflows
