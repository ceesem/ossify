# Data Import and Export

Ossify supports several data sources: its native `.osy` format, live CAVE databases via CAVEclient, legacy MeshWork files, and raw arrays for custom pipelines. This page covers how to load, save, and convert data between these formats.

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
partner_ids = pre_synapses.get_feature("post_pt_root_id") if "post_pt_root_id" in pre_synapses.feature_names else None
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
print(f"L2 vertex properties: {cell.graph.feature_names if cell.graph else []}")
print(f"Skeleton features: {cell.skeleton.feature_names if cell.skeleton else []}")
```

### Working with CAVEclient Data

```python
# CAVEclient loads create specific structure:
# - cell.graph: L2 spatial graph with coordinates in nanometers
# - cell.skeleton: Skeleton with radius and compartment features
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
    if "radius" in cell.skeleton.feature_names:
        radius = cell.skeleton.get_feature("radius")
        print(f"Radius range: {radius.min():.2f} - {radius.max():.2f}")
    
    if "compartment" in cell.skeleton.feature_names:
        compartment = cell.skeleton.get_feature("compartment")
        print(f"Compartments: {np.unique(compartment)}")
```

### Batch Loading Many Cells

When loading more than a handful of cells, `load_cell_batch_from_client` is much faster than
calling `load_cell_from_client` in a loop. It produces **identical** cells but pools the two
dominant per-cell fetches — the synapse queries and the L2 property lookup — across the whole
batch, and downloads skeletons in a single bulk request. For a batch of `N` cells the number of
network round trips drops from roughly `9 × N` to a small constant.

```python
# Load a batch of cells in a few pooled queries instead of one round trip per cell.
root_ids = [864691135639806264, 864691135639806265, 864691135639806266]

cells = ossify.load_cell_batch_from_client(
    root_ids,
    client,
    synapses=True,
    reference_tables=["synapse_target_predictions_ssa_v2"],
    reference_suffixes={"synapse_target_predictions_ssa_v2": "ssa"},
    timestamp=timestamp,   # one shared timestamp for the whole batch (see note below)
)

# Returns {root_id: Cell}; each Cell is identical to the single-cell load.
for root_id, cell in cells.items():
    print(root_id, cell.skeleton.n_vertices, len(cell.annotations.pre_syn.vertices))
```

The signature mirrors `load_cell_from_client` and adds a few batch-only options:

- `skip_invalid=True` — drop root ids that are not valid at the timestamp (or whose skeleton is
  unavailable) instead of raising.
- `skeleton_download_method` — `"gcs"` (default) downloads cached skeletons directly from the
  storage bucket, bypassing the skeleton service and avoiding its request rate limits;
  `"server"` routes the download through the service instead. Only already-cached skeletons are
  returned — generate them first (see the tip below).
- `row_limit` — guards against a silently-truncated synapse query (see the batch-size note).

!!! tip "Two batching scales: generate first, then load"

    Skeletons are the expensive, server-generated artifact. For a large job, **generate them up
    front** with a single async pre-pass, then let your workers load in small batches against a
    warm cache:

    ```python
    # 1. Pre-pass (once for the whole population): trigger server-side generation.
    #    Up to 10,000 root ids per call.
    client.skeleton.generate_bulk_skeletons_async(all_root_ids)

    # ... poll client.skeleton.skeletons_exist(root_ids=chunk) until ready ...

    # 2. Per worker: load small batches (skeletons are already warm).
    for chunk in chunked(all_root_ids, 10):
        cells = ossify.load_cell_batch_from_client(chunk, client, synapses=True,
                                                   timestamp=timestamp)
        # ... your post-processing / saving ...
    ```

!!! note "Choosing a batch size (~10)"

    A batch size of about **10** is a good default. Beyond that the pooled queries become
    payload-bound, so larger batches stop improving per-cell time, and a batch of ~10 keeps the
    combined synapse query safely under the server's row limit (500,000 rows) even for
    heavily-connected cells. (Skeletons are no longer the constraint — the cached bulk download
    fetches up to 500 per call — so the synapse query is what sets the batch size.) If a batch
    ever does exceed the row limit, the `row_limit` guard raises rather than returning a silently
    truncated result — reduce the batch size if you hit it.

!!! warning "Use a shared timestamp"

    All cells in a batch are queried at one shared `timestamp`. Pass an explicit `timestamp` (or
    pin `client.version`) so the pooled queries are consistent and the batch is reproducible; if
    omitted, the current materialization timestamp is used.

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

# PCG processing moves annotation data to layer features
print(f"Skeleton features after PCG processing: {cell.skeleton.feature_names}")
print(f"Graph features after PCG processing: {cell.graph.feature_names if cell.graph else []}")
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

## SWC Import and Export

Ossify reads and writes the SWC format used by the neuromorphology
community. SWC is a simple seven-column text format
([specification](https://swc-specification.readthedocs.io/en/latest/swc.html)),
which makes it convenient for sharing skeletons with tools like NEURON,
NeuTu, Vaa3D, and many morphology databases.

### Loading SWC Files

```python
import ossify

# Read SWC into a SkeletonLayer
skel = ossify.load_swc("path/to/neuron.swc")

print(f"Loaded {skel.n_vertices} vertices, root at index {skel.root_positional}")
print(f"Features: {skel.feature_names}")
# → ["compartment", "radius"]

# Unit conversion at load time. SWC files are usually in micrometers;
# convert to the nanometer convention used by CAVE data with rescale=1000.
skel_nm = ossify.load_swc("neuron.swc", rescale=1000.0)

# Attach the loaded skeleton to a Cell
cell = ossify.Cell(name="from_swc")
cell.add_skeleton(
    vertices=skel.vertices,
    edges=skel.edges,
    root=skel.root_positional,
    features={
        "compartment": skel.get_feature("compartment"),
        "radius": skel.get_feature("radius"),
    },
)
```

### Exporting to SWC

`export_swc` accepts either a `Cell` (using its skeleton layer) or a
`SkeletonLayer` directly:

```python
# Simple export — uses cell.name for the filename
ossify.export_swc(cell)                          # writes "<cell.name>.swc"
ossify.export_swc(cell, file="neuron.swc")       # explicit path

# Export a SkeletonLayer directly (no enclosing Cell required)
ossify.export_swc(cell.skeleton, file="skeleton_only.swc")
```

#### Compartment Labels and Mapping

The SWC `type` column uses small integers (1=soma, 2=axon, 3=dendrite, …).
By default the exporter writes `0` for every vertex. Point at a feature
to fill in real compartment labels:

```python
ossify.export_swc(
    cell,
    file="neuron.swc",
    compartment="compartment",         # feature name on the skeleton
)
```

If your compartment feature uses different values than the SWC convention,
remap them with `compartment_mapping`. Unmapped values fall back to
`default_compartment_label`:

```python
# Map ossify's axon/dendrite labels to SWC's 2/3 codes
ossify.export_swc(
    cell,
    file="neuron.swc",
    compartment="compartment",
    compartment_mapping={"axon": 2, "dendrite": 3, "soma": 1},
    default_compartment_label=0,
)
```

You can also pass a list of fallback feature names — useful when your
pipeline might attach the compartment under either of several names:

```python
ossify.export_swc(
    cell,
    compartment=["compartment", "compartment_label", "axon_dendrite"],
)
```

#### Radius

Same lookup rules as compartment: feature name, list of fallback names,
or omitted entirely (uses `default_radius`):

```python
ossify.export_swc(
    cell,
    file="neuron.swc",
    radius="radius",
    default_radius=1.0,   # used only when no radius feature is found
)
```

#### Unit Conversion

SWC convention is micrometers. Vertex coordinates in CAVE-derived cells
are in nanometers. Use `rescale` for the conversion — applied to
coordinates and, by default, to radius values as well:

```python
# nm → µm on export
ossify.export_swc(
    cell,
    file="neuron_um.swc",
    radius="radius",
    rescale=0.001,            # scale coordinates from nm → µm
    rescale_radius=True,      # default — also rescale the radius column
)

# Coordinates in µm but radius already in some other unit you want kept
ossify.export_swc(
    cell,
    file="neuron.swc",
    radius="radius_um",
    rescale=0.001,
    rescale_radius=False,
)
```

#### Resampling Before Export

Tools that consume SWC sometimes prefer evenly-spaced vertices. The
optional `resample_distance` parameter resamples the skeleton to
approximately that vertex spacing (in the skeleton's native units, before
`rescale` is applied):

```python
ossify.export_swc(
    cell,
    file="neuron_uniform.swc",
    resample_distance=2000.0,  # ~2 µm spacing for a cell stored in nm
    rescale=0.001,
)
```

#### Headers

`export_swc` writes a plain SWC body by default. Use `header` to prepend
comment lines (the function adds `# ` prefixes automatically when they
are missing):

```python
ossify.export_swc(
    cell,
    file="neuron.swc",
    header="Generated by analysis pipeline v3.2\nRoot ID 864691135336055529",
    compartment="compartment",
    radius="radius",
)
```

### Inspecting Before Writing

If you want to look at the SWC table before committing to a file (or
adjust it programmatically), `export_swc_dataframe` returns the
seven-column DataFrame that `export_swc` would write:

```python
df = ossify.export_swc_dataframe(
    cell.skeleton,
    compartment="compartment",
    radius="radius",
    rescale=0.001,
)
df.head()
# Columns: ["id", "type", "x", "y", "z", "radius", "parent"]
# Rows are in topological order (each row's parent precedes it).
```

### Round-Trip Example

```python
# Export → reload to confirm
ossify.export_swc(cell, file="roundtrip.swc", compartment="compartment",
                  radius="radius", rescale=0.001)

reloaded = ossify.load_swc("roundtrip.swc", rescale=1000.0)
assert reloaded.n_vertices == cell.skeleton.n_vertices
```

### SWC Compartment Constants

The SWC spec assigns integer codes to standard compartment types
(0 = undefined, 1 = soma, 2 = axon, 3 = basal dendrite, 4 = apical
dendrite, 5 = custom). Ossify exposes these as both an
:class:`~ossify.SWCCompartment` `IntEnum` and flat module-level aliases
so palettes can be written by name instead of by number:

```python
import ossify
from ossify import SWC_SOMA, SWC_AXON, SWC_DENDRITE, SWC_APICAL_DENDRITE

# Compact palette dict keyed by SWC type
compartment_palette = {
    SWC_SOMA: "black",
    SWC_AXON: "steelblue",
    SWC_DENDRITE: "tomato",
    SWC_APICAL_DENDRITE: "darkorange",
}

# Works directly with the 2D and 3D plot APIs
ossify.plot_cell_2d(cell, color="compartment", palette=compartment_palette)

# The IntEnum members are ints, so they can also drive export_swc's
# compartment_mapping argument without casting:
ossify.export_swc(
    cell,
    file="neuron.swc",
    compartment="compartment_label",
    compartment_mapping={
        "soma": SWC_SOMA,
        "axon": SWC_AXON,
        "dendrite": SWC_DENDRITE,
    },
)
```

If you prefer the namespaced form, use the enum class directly:

```python
from ossify import SWCCompartment

palette = {
    SWCCompartment.SOMA: "black",
    SWCCompartment.DENDRITE: "tomato",
}
```

Both forms are interchangeable — the flat aliases are just convenient
references to the same enum members.

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
skeleton_data = cell.skeleton.nodes  # DataFrame with coordinates + features
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
    features={
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
    features={"confidence": "conf_score", "type": "synapse_type"}
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

Use `load_cell_batch_from_client` and process the batch in chunks (see
[Batch Loading Many Cells](#batch-loading-many-cells) for the details and the recommended chunk
size of ~10):

```python
def download_multiple_cells(root_ids, client, output_dir, timestamp=None, chunk_size=10):
    """Download and save many cells, batching the CAVE queries."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for start in range(0, len(root_ids), chunk_size):
        chunk = root_ids[start:start + chunk_size]
        print(f"Downloading {len(chunk)} cells ({start + 1}-{start + len(chunk)})...")

        # One batched load for the whole chunk (skip roots that are invalid/unskeletonized).
        cells = ossify.load_cell_batch_from_client(
            chunk,
            client,
            synapses=True,
            timestamp=timestamp,
            skip_invalid=True,
        )

        for root_id, cell in cells.items():
            output_file = output_path / f"cell_{root_id}.osy"
            ossify.save_cell(cell, str(output_file))
            print(f"  Saved: {output_file.name}")

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
- `ossify.load_cell_from_client(root_id, client, synapses=False, restore_graph=False, ...)` - Load one cell from CAVE
- `ossify.load_cell_batch_from_client(root_ids, client, synapses=False, ...)` - Load many cells with pooled queries (returns `{root_id: Cell}`)

### Legacy MeshWork
- `ossify.import_legacy_meshwork(filename, l2_skeleton=True, as_pcg_skel=False)` - Import MeshWork files

### SWC Format
- `ossify.load_swc(source, name=None, rescale=None)` - Read an `.swc` file into a `SkeletonLayer`
- `ossify.export_swc(cell_or_skeleton, file=None, ...)` - Write a `Cell` or `SkeletonLayer` as `.swc`
- `ossify.export_swc_dataframe(skeleton, ...)` - Build the seven-column SWC DataFrame without writing
- `ossify.SWCCompartment` and `ossify.SWC_SOMA`/`SWC_AXON`/`SWC_DENDRITE`/`SWC_APICAL_DENDRITE`/`SWC_CUSTOM`/`SWC_UNDEFINED` - Named constants for the SWC `type` column

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