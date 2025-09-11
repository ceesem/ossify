# Data Import and Export

!!! warning "AI-Generated Documentation"
    This documentation was generated with assistance from AI. While we strive for accuracy, errors may be present. If you find issues, unclear explanations, or have suggestions for improvement, please [report them on GitHub](https://github.com/ceesem/ossify/issues).

Ossify provides multiple ways to load and save cellular morphology data, from native ossify formats to external sources like CAVEclient and legacy MeshWork files. This guide covers all the import and export options available.

## Native Ossify Format

### Saving Cells

The native `.osy` format preserves all ossify data structures, including layers, annotations, linkages, and metadata.

```python
import ossify

# Save to current directory with automatic naming
cell = ossify.Cell(name="my_neuron")
# ... add layers and annotations ...

ossify.save_cell(cell)  # Creates "my_neuron.osy"

# Save with explicit path
ossify.save_cell(cell, "path/to/my_cell.osy")

# Save with overwrite protection
ossify.save_cell(cell, "existing_file.osy", allow_overwrite=False)  # Raises error if exists
ossify.save_cell(cell, "existing_file.osy", allow_overwrite=True)   # Overwrites existing

# Save to file object
with open("my_cell.osy", "wb") as f:
    ossify.save_cell(cell, f)
```

### Loading Cells

```python
# Load from file path
cell = ossify.load_cell("path/to/my_cell.osy")

# Load from file object
with open("my_cell.osy", "rb") as f:
    cell = ossify.load_cell(f)

# Inspect loaded cell
cell.describe()
print(f"Loaded cell '{cell.name}' with {len(cell.layers.names)} layers")
```

### Cloud Storage Support

Ossify supports cloud storage through the `cloudfiles` library:

```python
# Save to cloud storage
ossify.save_cell(cell, "gs://my-bucket/cells/neuron_001.osy")  # Google Cloud
ossify.save_cell(cell, "s3://my-bucket/cells/neuron_001.osy")  # AWS S3

# Load from cloud storage
cell = ossify.load_cell("gs://my-bucket/cells/neuron_001.osy")
cell = ossify.load_cell("s3://my-bucket/cells/neuron_001.osy")

# Also supports local file:// URLs
cell = ossify.load_cell("file:///absolute/path/to/cell.osy")
```

### Advanced File Management

```python
# Using CellFiles for advanced operations
from ossify import CellFiles

# Initialize file manager
cf = CellFiles("path/to/directory")  # Local directory
cf = CellFiles("gs://my-bucket/cells")  # Cloud storage

# Check if writable
print(f"Can save: {cf.saveable}")
print(f"Is remote: {cf.remote}")

# Save and load with manager
cf.save(cell, "neuron_001.osy", allow_overwrite=True)
loaded_cell = cf.load("neuron_001.osy")
```

## CAVEclient Integration

Load cells directly from connectomics databases using CAVEclient:

### Basic CAVEclient Import

```python
from caveclient import CAVEclient
import ossify

# Initialize client (replace with your datastack)
client = CAVEclient("minnie65_public")

# Load basic cell (skeleton + L2 graph)
root_id = 864691135639806264  # Example root ID
cell = ossify.load_cell_from_client(
    root_id=root_id,
    client=client
)

print(f"Loaded cell {cell.name}")
print(f"Skeleton: {cell.skeleton.n_vertices} vertices")
print(f"Graph: {cell.graph.n_vertices} L2 vertices")
```

### CAVEclient Import with Synapses

```python
# Load with synapse data
cell = ossify.load_cell_from_client(
    root_id=root_id,
    client=client,
    synapses=True,                    # Include synapse annotations
    include_partner_root_id=True,     # Include partner neuron IDs
    omit_self_synapses=True,          # Remove autapses (usually artifacts)
)

print(f"Pre-synaptic sites: {len(cell.annotations.pre_syn.vertices)}")
print(f"Post-synaptic sites: {len(cell.annotations.post_syn.vertices)}")

# Access synapse metadata
pre_synapses = cell.annotations.pre_syn
partner_ids = pre_synapses.get_label("post_pt_root_id") if "post_pt_root_id" in pre_synapses.label_names else None
```

### CAVEclient Import Options

```python
import datetime

# Specific timestamp for consistency
timestamp = datetime.datetime(2024, 1, 15, 12, 0, 0)

cell = ossify.load_cell_from_client(
    root_id=root_id,
    client=client,
    synapses=True,
    restore_graph=True,               # Include all L2 graph edges (slower)
    restore_properties=True,          # Include all L2 vertex properties
    synapse_spatial_point="ctr_pt_position",  # Synapse coordinate column
    timestamp=timestamp,              # Specific time point
    skeleton_version=4,               # Skeleton service version
)

# Check what was loaded
print(f"L2 graph edges: {len(cell.graph.edges) if cell.graph else 0}")
print(f"L2 vertex properties: {cell.graph.label_names if cell.graph else []}")
print(f"Skeleton labels: {cell.skeleton.label_names if cell.skeleton else []}")
```

### Working with CAVEclient Data

```python
# CAVEclient loads create specific structure:
# - cell.graph: L2 spatial graph with coordinates in nanometers
# - cell.skeleton: Skeleton with radius and compartment labels
# - cell.annotations.pre_syn/post_syn: Synaptic sites (if synapses=True)

# L2 graph coordinates are in nanometers
if cell.graph:
    l2_coords = cell.graph.vertices
    print(f"L2 coordinate range (nm): {l2_coords.min(axis=0)} to {l2_coords.max(axis=0)}")

# Skeleton coordinates are also in nanometers
if cell.skeleton:
    skel_coords = cell.skeleton.vertices  
    print(f"Skeleton coordinate range (nm): {skel_coords.min(axis=0)} to {skel_coords.max(axis=0)}")
    
    # Check for radius and compartment info
    if "radius" in cell.skeleton.label_names:
        radius = cell.skeleton.get_label("radius")
        print(f"Radius range: {radius.min():.2f} - {radius.max():.2f}")
    
    if "compartment" in cell.skeleton.label_names:
        compartment = cell.skeleton.get_label("compartment")
        print(f"Compartments: {np.unique(compartment)}")
```

## Legacy MeshWork Import

Import from legacy MeshWork `.h5` files (requires `h5py`):

### Basic MeshWork Import

```python
# Import legacy MeshWork file
cell, node_mask = ossify.import_legacy_meshwork(
    "path/to/meshwork_file.h5",
    l2_skeleton=True,        # Import mesh as L2 graph (True) or mesh layer (False)
    as_pcg_skel=False       # Process PCG skeleton annotations (False for raw import)
)

print(f"Imported cell: {cell.name}")
print(f"Node mask shape: {node_mask.shape}")
print(f"Layers: {cell.layers.names}")
print(f"Annotations: {cell.annotations.names}")

# The node_mask indicates which mesh vertices correspond to skeleton nodes
# It's not automatically applied - you can apply it manually if needed
if cell.graph and len(node_mask) == cell.graph.n_vertices:
    # Apply mask to keep only skeleton-corresponding vertices
    masked_graph = cell.graph.apply_mask(node_mask, as_positional=True)
    print(f"Masked graph: {masked_graph.n_vertices} vertices")
```

### MeshWork Import Options

```python
# Import as mesh layer instead of L2 graph
cell, node_mask = ossify.import_legacy_meshwork(
    "meshwork_file.h5",
    l2_skeleton=False,       # Import mesh data as actual mesh layer
    as_pcg_skel=False
)

print(f"Has mesh: {cell.mesh is not None}")
if cell.mesh:
    print(f"Mesh: {cell.mesh.n_vertices} vertices, {len(cell.mesh.faces)} faces")

# Process PCG skeleton annotations automatically
cell, node_mask = ossify.import_legacy_meshwork(
    "meshwork_file.h5", 
    l2_skeleton=True,
    as_pcg_skel=True        # Automatically process segment properties, etc.
)

# PCG processing moves annotation data to layer labels
print(f"Skeleton labels after PCG processing: {cell.skeleton.label_names}")
print(f"Graph labels after PCG processing: {cell.graph.label_names if cell.graph else []}")
```

### Understanding MeshWork Structure

```python
# MeshWork files typically contain:
# - mesh: 3D mesh data (vertices, faces, link_edges)
# - skeleton: Tree skeleton mapped to mesh
# - annotations: Various data tables

# After import:
cell.describe()

# Check linkages between layers
if cell.skeleton and cell.graph:
    # Skeleton is linked to graph/mesh
    skeleton_to_graph = cell.skeleton.map_index_to_layer("graph", as_positional=False)
    print(f"Skeleton-graph mappings: {len(skeleton_to_graph)}")

# Annotations may be linked to mesh/graph or skeleton
for anno_name in cell.annotations.names:
    anno = cell.annotations[anno_name]
    print(f"Annotation {anno_name}: {anno.n_vertices} points")
```

## Working with External Formats

### Exporting to External Libraries

```python
# Export mesh to trimesh
if cell.mesh:
    tmesh = cell.mesh.as_trimesh
    # Can then save to various formats via trimesh
    tmesh.export("output.ply")
    tmesh.export("output.obj")

# Export skeleton to NetworkX
if cell.skeleton:
    import networkx as nx
    G = nx.from_edgelist(cell.skeleton.edges)
    
    # Add vertex positions as node attributes
    pos_dict = {vid: cell.skeleton.vertices[i] for i, vid in enumerate(cell.skeleton.vertex_index)}
    nx.set_node_attributes(G, pos_dict, 'pos')
    
    # Save as GraphML
    nx.write_graphml(G, "skeleton.graphml")

# Export to pandas/CSV
skeleton_data = cell.skeleton.nodes  # DataFrame with coordinates + labels
skeleton_data.to_csv("skeleton_data.csv")

annotation_data = cell.annotations.synapses.nodes if "synapses" in cell.annotations.names else None
if annotation_data is not None:
    annotation_data.to_csv("synapse_data.csv")
```

### Creating Cells from External Data

```python
import pandas as pd
import numpy as np

# Load from CSV/pandas
vertex_data = pd.read_csv("neuron_vertices.csv")
edge_data = pd.read_csv("neuron_edges.csv")

# Create cell from DataFrames
cell = ossify.Cell(name="imported_neuron")

# Add skeleton from DataFrame
cell.add_skeleton(
    vertices=vertex_data,
    edges=edge_data[["source", "target"]].values,
    root=vertex_data.iloc[0]["vertex_id"],  # First vertex as root
    spatial_columns=["x", "y", "z"],
    vertex_index="vertex_id",
    labels={
        "radius": "radius_um", 
        "compartment": "compartment_type"
    }
)

# Add annotations from CSV
synapse_data = pd.read_csv("synapses.csv")
cell.add_point_annotations(
    name="synapses",
    vertices=synapse_data,
    spatial_columns=["pos_x", "pos_y", "pos_z"],
    labels={"confidence": "conf_score", "type": "synapse_type"}
)
```

### Coordinate System Handling

```python
# Handle different coordinate systems and units
def convert_coordinates(vertices, scale_factor=1000, offset=[0, 0, 0]):
    """Convert coordinates (e.g., micrometers to nanometers)."""
    vertices_scaled = vertices * scale_factor
    vertices_offset = vertices_scaled + offset
    return vertices_offset

# Example: convert from nanometers to micrometers and back
if cell.skeleton:
    # Convert entire cell from nm to μm (all layers and annotations)
    cell_um = cell.transform(lambda x: x / 1000)
    
    # Or apply more complex transformation
    def complex_transform(coords):
        """Custom transformation: scale and translate."""
        scaled = coords * 0.001  # nm to μm
        translated = scaled + [100, 100, 0]  # Add offset
        return translated
    
    converted_cell = cell.transform(complex_transform)
    converted_cell.name = f"{cell.name}_converted"
```

## Batch Processing

### Processing Multiple Files

```python
import os
from pathlib import Path

def process_cell_directory(input_dir, output_dir, file_pattern="*.h5"):
    """Process all MeshWork files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for meshwork_file in input_path.glob(file_pattern):
        print(f"Processing {meshwork_file.name}...")
        
        try:
            # Import MeshWork
            cell, mask = ossify.import_legacy_meshwork(
                str(meshwork_file),
                l2_skeleton=True,
                as_pcg_skel=True
            )
            
            # Save as ossify format
            output_file = output_path / f"{meshwork_file.stem}.osy"
            ossify.save_cell(cell, str(output_file))
            
            print(f"  Saved: {output_file.name}")
            
        except Exception as e:
            print(f"  Error processing {meshwork_file.name}: {e}")

# Process directory
process_cell_directory("legacy_files/", "ossify_files/", "*.h5")
```

### Batch CAVEclient Downloads

```python
def download_multiple_cells(root_ids, client, output_dir):
    """Download multiple cells from CAVEclient."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for root_id in root_ids:
        print(f"Downloading {root_id}...")
        
        try:
            cell = ossify.load_cell_from_client(
                root_id=root_id,
                client=client,
                synapses=True
            )
            
            # Save with root_id as filename
            output_file = output_path / f"cell_{root_id}.osy"
            ossify.save_cell(cell, str(output_file))
            
            print(f"  Saved: {output_file.name}")
            
        except Exception as e:
            print(f"  Error downloading {root_id}: {e}")

# Download list of cells
root_ids = [864691135639806264, 864691135639806265, 864691135639806266]
download_multiple_cells(root_ids, client, "downloaded_cells/")
```

## File Format Details

### Ossify Format Structure

The `.osy` format is a compressed tar archive containing:

```
cell.osy/
├── metadata.json              # Cell metadata and structure
├── layers/
│   ├── skeleton/
│   │   ├── meta.json         # Layer metadata
│   │   ├── nodes.feather     # Vertex data
│   │   ├── edges.npz         # Edge connectivity
│   │   └── base_properties/  # Cached properties
│   ├── mesh/
│   │   ├── meta.json
│   │   ├── nodes.feather
│   │   └── faces.npz
│   └── graph/
│       ├── meta.json
│       ├── nodes.feather
│       └── edges.npz
├── annotations/
│   ├── synapses/
│   │   ├── meta.json
│   │   └── nodes.feather
│   └── spines/
│       ├── meta.json
│       └── nodes.feather
└── linkage/
    ├── skeleton/mesh/
    │   └── linkage.feather
    └── synapses/skeleton/
        └── linkage.feather
```

### Compression and Performance

```python
# Ossify files use efficient compression
# - Feather format for DataFrames (fast, compact)
# - NPZ format for arrays (compressed numpy)
# - JSON for metadata (human readable)

# Check file sizes
import os
original_size = os.path.getsize("cell.osy")
print(f"Compressed cell file: {original_size / 1024 / 1024:.1f} MB")

# Files are optimized for loading speed and storage efficiency
```

## Key Import/Export Methods

### Native Format
- `ossify.save_cell(cell, file=None, allow_overwrite=False)` - Save to ossify format
- `ossify.load_cell(source)` - Load from ossify format
- `CellFiles(path)` - Advanced file management

### CAVEclient Integration
- `ossify.load_cell_from_client(root_id, client, synapses=False, restore_graph=False, ...)` - Load from CAVE

### Legacy MeshWork
- `ossify.import_legacy_meshwork(filename, l2_skeleton=True, as_pcg_skel=False)` - Import MeshWork files

### External Format Support
- `mesh.as_trimesh` - Export mesh to trimesh library
- `skeleton.nodes.to_csv()` - Export to CSV
- NetworkX integration for skeleton graphs

!!! tip "Best Practices"
    - Use ossify native format for long-term storage and performance
    - Set explicit timestamps when using CAVEclient for reproducibility
    - Apply node masks from MeshWork import based on your analysis needs
    - Use cloud storage URLs for large-scale collaborative projects
    - Batch process files with error handling for robustness