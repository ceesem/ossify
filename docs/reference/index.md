# API Reference

Ossify is a Python package for working with neuromorphological data, providing tools for analyzing and visualizing neuron structures across multiple geometric representations.

## Quick Start

```python
import ossify

# Load a cell from file
cell = ossify.load_cell("path/to/cell.osy")

# Create from CAVE client (requires caveclient)
cell = ossify.cell_from_client(root_id=12345, client=cave_client)

# Analyze morphology
strahler = ossify.strahler_number(cell)
is_axon = ossify.label_axon_from_synapse_flow(cell)

# Create visualizations
fig, ax = ossify.plot_cell_2d(cell, color="compartment")
```

## Package Structure

### **Core Classes** {: .text-primary }
The foundation classes for representing neuromorphological data:

- **[`Cell`](core.md#ossify.Cell)**: Main container for neuromorphological data with multiple layers
- **[`Link`](core.md#ossify.Link)**: Manages relationships between different data layers

### **Data Layer Classes** {: .text-primary }
Specialized classes for different geometric representations:

- **[`SkeletonLayer`](layers.md#ossify.SkeletonLayer)**: Tree-structured neuronal skeletons with hierarchical analysis
- **[`GraphLayer`](layers.md#ossify.GraphLayer)**: Graph-based representation for spatial connectivity
- **[`MeshLayer`](layers.md#ossify.MeshLayer)**: 3D mesh surfaces with face-based geometry  
- **[`PointCloudLayer`](layers.md#ossify.PointCloudLayer)**: Sparse point annotations and markers

### **Analysis & Algorithms** {: .text-secondary }
Computational methods for morphological analysis:

- **[Morphological Analysis](algorithms.md#morphological-analysis)**: Strahler numbers, compartment classification
- **[Synapse Analysis](algorithms.md#synapse-analysis)**: Betweenness centrality, flow-based segmentation
- **[Smoothing & Filtering](algorithms.md#smoothing-and-filtering)**: Label spreading and signal processing

### **Visualization & Plotting** {: .text-secondary }
2D plotting and visualization utilities:

- **[Cell Plotting](plotting.md#cell-plotting)**: Integrated cell visualization with multiple projections
- **[Layer Plotting](plotting.md#layer-plotting)**: Individual layer plotting with flexible styling
- **[Figure Management](plotting.md#figure-management)**: Multi-panel layouts and precise sizing

### **File I/O Operations** {: .text-accent }
Loading and saving neuromorphological data:

- **[Core I/O Functions](io.md#core-functions)**: `load_cell()`, `save_cell()`
- **[File Management](io.md#file-management)**: `CellFiles` for cloud and local storage

### **External Integrations** {: .text-accent }
Interfaces with external data sources and tools:

- **[CAVE Integration](external.md#cave-integration)**: `cell_from_client()` for connectome data

## Key Features

### **Multi-Scale Data Integration**
Seamlessly work with data at different scales - from electron microscopy meshes to light microscopy skeletons, with automatic mapping between representations.

### **Flexible Analysis Pipeline**
Chain operations across different data types with consistent APIs and automatic data propagation between layers.

### **Publication-Ready Visualization**
Create publication-quality figures with precise unit control, multiple projections, and customizable styling.

### **Cloud-Native I/O**
Load and save data from local files, cloud storage (S3, GCS), or directly from connectome databases.

---

## Navigation

| Module | Description | Key Classes |
|--------|-------------|-------------|
| **[Core Classes](core.md)** | Foundation classes and containers | `Cell`, `Link` |
| **[Data Layers](layers.md)** | Geometric representation classes | `SkeletonLayer`, `GraphLayer`, `MeshLayer`, `PointCloudLayer` |
| **[Algorithms](algorithms.md)** | Analysis and computation functions | `strahler_number`, `label_axon_*`, `smooth_labels` |
| **[Plotting](plotting.md)** | Visualization and figure creation | `plot_cell_*`, `plot_morphology_*` |
| **[File I/O](io.md)** | Data loading and saving | `load_cell`, `save_cell`, `CellFiles` |
| **[External](external.md)** | Third-party integrations | `cell_from_client` |

!!! tip "Best Practices"
    - Use `Cell.apply_mask()` for non-destructive filtering
    - Leverage `Link` objects for complex data relationships  
    - Take advantage of method chaining for concise workflows
    - Use `mask_context()` for temporary operations