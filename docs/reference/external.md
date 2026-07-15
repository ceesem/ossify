# External Integrations

Ossify integrates with external data sources and analysis platforms to streamline neuromorphological data workflows.

## Overview

| Integration Category | Functions | Purpose |
|---------------------|-----------|---------|
| **[CAVE Integration](#cave-integration)** | `load_cell_from_client`, `load_cell_batch_from_client`, `fetch_frames_batch` | Connectome analysis via CAVE infrastructure |

---

## CAVE Integration {: .doc-heading}

[CAVE](https://www.caveconnecto.me/) (Connectome Analysis Versioning Engine) provides an interface for managing and analysis in large-scale densely segmented anatomical datasets datasets.
Ossify integrates with CAVE to import neurons with their meshes, skeletons, and synaptic connectivity.
If you have previously used `pcg_skel`, this is the equivalent of `get_meshwork_from_client` to import a neuron and synapses from the CAVE skeleton service.

### load_cell_from_client

::: ossify.load_cell_from_client
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Import neurons directly from CAVE databases with automatic skeleton generation, graph reconstruction, and synapse mapping.**

#### Prerequisites

```bash
# Install CAVE client
pip install caveclient
``

#### Basic Usage

```python
import ossify
from caveclient import CAVEclient

# Initialize CAVE client
client = CAVEclient("minnie65_public")  # MICrONS dataset

# Import neuron with basic skeleton
cell = ossify.load_cell_from_client(
    root_id=864691135336055529,
    client=client
)

print(f"Loaded cell {cell.name}")
print(f"Skeleton vertices: {cell.skeleton.n_vertices}")
print(f"Graph vertices: {cell.graph.n_vertices}")
```

#### Advanced Import Options

```python
from datetime import datetime

# Import with synapses and full graph
cell = ossify.load_cell_from_client(
    root_id=864691135336055529,
    client=client,
    synapses=True,                    # Include synapse annotations
    restore_graph=True,               # Include complete L2 graph
    restore_properties=True,          # Include all vertex properties
    include_partner_root_id=True,     # Include synaptic partner IDs
    omit_self_synapses=True,          # Remove autapses
    skeleton_version=4                # Skeleton service version
)

# Check imported data
print(f"Graph vertices: {cell.graph.n_vertices}")
print(f"Skeleton vertices: {cell.skeleton.n_vertices}")
print(f"Presynaptic sites: {len(cell.annotations['pre_syn'])}")
print(f"Postsynaptic sites: {len(cell.annotations['post_syn'])}")
print(f"Available features: {cell.skeleton.features.columns.tolist()}")
```

#### Synapse Analysis Workflow

```python
# Import cell with synapses
cell = ossify.load_cell_from_client(
    root_id=864691135336055529,
    client=client,
    synapses=True,
    timestamp=datetime(2023, 6, 1)  # Consistent analysis timestamp
)

# Analyze compartmentalization
is_axon, segregation = ossify.label_axon_from_synapse_flow(
    cell, 
    return_segregation_index=True
)

compartment = ["dendrite" if not ax else "axon" for ax in is_axon]
cell.skeleton.add_feature(compartment, "compartment")

print(f"Segregation index: {segregation:.3f}")
print(f"Axon fraction: {is_axon.mean():.2%}")

# Visualize results
fig, ax = ossify.plot_cell_2d(
    cell,
    color="compartment",
    palette={"axon": "red", "dendrite": "blue"},
    synapses=True,
    pre_color="orange",
    post_color="green",
    units_per_inch=100_000 # nm
)
```

### load_cell_batch_from_client

::: ossify.load_cell_batch_from_client
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Load many cells at once, pooling the synapse and L2 queries across the whole batch and
downloading skeletons in a single bulk request.** Each returned `Cell` is identical to what
`load_cell_from_client` produces for that root id, but a batch of `N` cells costs a handful of
round trips instead of ~`9 × N`.

```python
root_ids = [864691135639806264, 864691135639806265, 864691135639806266]

cells = ossify.load_cell_batch_from_client(
    root_ids,
    client,
    synapses=True,
    reference_tables=["synapse_target_predictions_ssa_v2"],
    reference_suffixes={"synapse_target_predictions_ssa_v2": "ssa"},
    timestamp=timestamp,     # one shared timestamp for the batch
    skip_invalid=True,       # drop invalid/unskeletonized roots instead of raising
)

for root_id, cell in cells.items():
    print(root_id, cell.skeleton.n_vertices)
```

!!! tip "Recommended batch size ~10"

    Around 10 cells per batch is a good default: past that the pooled queries are payload-bound
    (per-cell time stops improving), and ~10 stays under the server's 500,000-row query limit even
    for heavily-connected cells while matching the synchronous bulk-skeleton download cap. For
    large jobs, generate skeletons up front with `client.skeleton.generate_bulk_skeletons_async`
    (up to 10,000 ids/call), then load small batches against the warm cache. See the
    [Data Import and Export guide](../data_import_export.md#batch-loading-many-cells) for the full
    workflow.

### fetch_frames_batch

::: ossify.fetch_frames_batch
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

Lower-level helper used by `load_cell_batch_from_client`. It returns the pooled per-root synapse
and L2 frames (`{root_id: {"pre_syn_df", "post_syn_df", "l2_df"}}`) without assembling `Cell`
objects, which are byte-identical to the single-cell path. Most users should call
`load_cell_batch_from_client`; reach for this only when you want the raw frames — for example to
feed them into `load_cell_from_client` via its `pre_syn_df` / `post_syn_df` / `l2_df` injection
parameters in a custom pipeline.

### **Cross-Platform Analysis Pipeline**

```python
def cave_to_analysis_pipeline(root_ids, client, output_format="both"):
    \"\"\"Complete pipeline from CAVE import to analysis results\"\"\"
    
    results = {}
    
    for root_id in root_ids:
        print(f"Processing cell {root_id}...")
        
        # Import from CAVE
        cell = ossify.load_cell_from_client(
            root_id=root_id,
            client=client, 
            synapses=True,
            restore_properties=True
        )
        
        # Morphological analysis
        strahler = ossify.strahler_number(cell)
        cell.skeleton.add_feature(strahler, "strahler_order")
        
        # Compartment analysis
        is_axon, segregation = ossify.label_axon_from_synapse_flow(
            cell, return_segregation_index=True
        )
        compartment = ["dendrite" if not ax else "axon" for ax in is_axon]
        cell.skeleton.add_feature(compartment, "compartment")
        
        # Generate visualization
        fig, axes = ossify.plot_cell_multiview(
            cell,
            color="compartment",
            palette={"axon": "red", "dendrite": "blue"},
            synapses=True,
            units_per_inch=100000
        )
        
        # Save results
        if output_format in ["ossify", "both"]:
            ossify.save_cell(cell, f"cell_{root_id}.osy")
        
        if output_format in ["figure", "both"]:
            fig.savefig(f"cell_{root_id}_analysis.pdf", dpi=300, bbox_inches='tight')
        
        # Store metrics
        results[root_id] = {
            'total_length_um': cell.skeleton.cable_length() / 1000,
            'n_synapses': len(cell.annotations["pre_syn"]) + len(cell.annotations["post_syn"]),
            'segregation_index': segregation,
            'axon_fraction': is_axon.mean()
        }
        
        print(f"  Length: {results[root_id]['total_length_um']:.1f} μm")
        print(f"  Synapses: {results[root_id]['n_synapses']}")
        print(f"  Segregation: {segregation:.3f}")
    
    return results

# Run complete pipeline
root_ids = [864691135336055529, 864691135174324866]
analysis_results = cave_to_analysis_pipeline(root_ids, client, "both")
```

!!! info "CAVE Integration Features"
    
    **Multi-Scale Data**: Automatically combines L2 graph connectivity with skeleton representations.
    
    **Temporal Consistency**: Support for timestamp-locked analyses across datasets.
    
    **Synapse Mapping**: Automatic mapping of synaptic sites to skeleton structures.
    
    **Quality Control**: Built-in validation for data integrity and biological plausibility via compartments.

!!! tip "Best Practices for CAVE Integration"
    
    - **Use Timestamps**: Always specify timestamps for reproducible analyses
    - **Batch Processing**: Use `load_cell_batch_from_client` (batches of ~10) to load many cells with pooled queries instead of looping `load_cell_from_client`  
    - **Error Handling**: Implement robust error handling for network operations
    - **Data Validation**: Validate imported data before analysis
    - **Version Control**: Track skeleton service and dataset versions used
    - **Memory Management**: Consider memory usage when importing large datasets