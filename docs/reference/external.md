# External Integrations

Ossify integrates with external data sources and analysis platforms to streamline neuromorphological data workflows.

## Overview

| Integration Category | Functions | Purpose |
|---------------------|-----------|---------|
| **[CAVE Integration](#cave-integration)** | `cell_from_client` | Connectome analysis via CAVE infrastructure |

---

## CAVE Integration {: .doc-heading}

[CAVE](https://caveclient.readthedocs.io/) (Connectome Analysis and Visualization Engine) provides access to large-scale electron microscopy connectome datasets. Ossify integrates seamlessly with CAVE to import neurons with their meshes, skeletons, and synaptic connectivity.

### cell_from_client

::: ossify.cell_from_client
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

# Install optional dependencies for enhanced functionality  
pip install meshparty  # For advanced mesh operations
pip install cloud-volume  # For image data access
```

#### Basic Usage

```python
import ossify
from caveclient import CAVEclient

# Initialize CAVE client
client = CAVEclient("minnie65_public")  # Allen MICrONS dataset
# client = CAVEclient("flywire_fafb_public")  # FlyWire dataset

# Import neuron with basic skeleton
cell = ossify.cell_from_client(
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
cell = ossify.cell_from_client(
    root_id=864691135336055529,
    client=client,
    synapses=True,                    # Include synapse annotations
    restore_graph=True,               # Include complete L2 graph
    restore_properties=True,          # Include all vertex properties
    include_partner_root_id=True,     # Include synaptic partner IDs
    timestamp=datetime(2023, 6, 1),   # Consistent timestamp
    omit_self_synapses=True,          # Remove autapses
    skeleton_version=4                # Skeleton service version
)

# Check imported data
print(f"Graph vertices: {cell.graph.n_vertices}")
print(f"Skeleton vertices: {cell.skeleton.n_vertices}")
print(f"Presynaptic sites: {len(cell.annotations['pre_syn'])}")
print(f"Postsynaptic sites: {len(cell.annotations['post_syn'])}")
print(f"Available labels: {cell.skeleton.labels.columns.tolist()}")
```

#### Synapse Analysis Workflow

```python
# Import cell with synapses
cell = ossify.cell_from_client(
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
cell.skeleton.add_label(compartment, "compartment")

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
    units_per_inch=100000
)
```

#### Batch Import Pipeline

```python
def import_cell_batch(root_ids, client, output_dir="./cells/"):
    \"\"\"Import multiple cells from CAVE and save locally\"\"\"
    
    import os
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for root_id in root_ids:
        try:
            print(f"Importing cell {root_id}...")
            
            # Import with consistent parameters
            cell = ossify.cell_from_client(
                root_id=root_id,
                client=client,
                synapses=True,
                restore_graph=False,  # Skip for speed
                restore_properties=False,
                omit_self_synapses=True
            )
            
            # Add analysis labels
            strahler = ossify.strahler_number(cell)
            cell.skeleton.add_label(strahler, "strahler_order")
            
            if len(cell.annotations["pre_syn"]) > 0:
                is_axon = ossify.label_axon_from_synapse_flow(cell)
                compartment = ["dendrite" if not ax else "axon" for ax in is_axon]
                cell.skeleton.add_label(compartment, "compartment")
            
            # Save to file
            output_path = os.path.join(output_dir, f"cell_{root_id}.osy")
            ossify.save_cell(cell, output_path)
            
            results.append({
                'root_id': root_id,
                'status': 'success',
                'path': output_path,
                'n_vertices': cell.skeleton.n_vertices,
                'n_synapses': len(cell.annotations["pre_syn"]) + len(cell.annotations["post_syn"])
            })
            
        except Exception as e:
            print(f"Error importing {root_id}: {e}")
            results.append({
                'root_id': root_id,
                'status': 'error',
                'error': str(e)
            })
    
    return results

# Import multiple cells
root_ids = [864691135336055529, 864691135174324866, 864691136027963697]
results = import_cell_batch(root_ids, client)

# Summary
successful = [r for r in results if r['status'] == 'success']
print(f"Successfully imported {len(successful)}/{len(root_ids)} cells")
```

#### Cell Type Analysis

```python
def analyze_cell_type(root_id, client):
    \"\"\"Comprehensive cell type analysis using CAVE data\"\"\"
    
    # Import with all features
    cell = ossify.cell_from_client(
        root_id=root_id,
        client=client,
        synapses=True,
        restore_graph=True,
        restore_properties=True,
        include_partner_root_id=True
    )
    
    # Morphological features
    skeleton = cell.skeleton
    total_length = skeleton.cable_length()
    n_branch_points = len(skeleton.branch_points)
    n_end_points = len(skeleton.end_points)
    
    # Strahler analysis
    strahler = ossify.strahler_number(skeleton)
    max_strahler = strahler.max()
    
    # Compartment analysis
    if len(cell.annotations["pre_syn"]) > 5:  # Minimum synapses for analysis
        is_axon, segregation = ossify.label_axon_from_synapse_flow(
            cell, return_segregation_index=True
        )
        axon_fraction = is_axon.mean()
    else:
        segregation = None
        axon_fraction = None
    
    # Synapse analysis
    n_pre_syn = len(cell.annotations["pre_syn"])
    n_post_syn = len(cell.annotations["post_syn"])
    
    # Input/output ratio
    if n_pre_syn > 0 and n_post_syn > 0:
        io_ratio = n_pre_syn / n_post_syn
    else:
        io_ratio = None
    
    # Compile features
    features = {
        'root_id': root_id,
        'total_length_um': total_length / 1000,  # Convert to micrometers
        'n_branch_points': n_branch_points,
        'n_end_points': n_end_points,
        'max_strahler_order': max_strahler,
        'branching_complexity': n_branch_points / (total_length / 1000),
        'n_presynaptic': n_pre_syn,
        'n_postsynaptic': n_post_syn,
        'input_output_ratio': io_ratio,
        'segregation_index': segregation,
        'axon_fraction': axon_fraction
    }
    
    return features, cell

# Analyze cell type
features, cell = analyze_cell_type(864691135336055529, client)

print("Cell Type Analysis:")
for key, value in features.items():
    if value is not None:
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
```

#### Working with Different Datasets

```python
# MICrONS (mouse visual cortex)
microns_client = CAVEclient("minnie65_public")
microns_cell = ossify.cell_from_client(
    root_id=864691135336055529,
    client=microns_client,
    synapses=True
)

# FlyWire (whole fly brain)
flywire_client = CAVEclient("flywire_fafb_public") 
flywire_cell = ossify.cell_from_client(
    root_id=720575940609550077,  # Example FlyWire ID
    client=flywire_client,
    synapses=True
)

# Compare morphologies
print(f"MICrONS cell length: {microns_cell.skeleton.cable_length()/1000:.1f} μm")
print(f"FlyWire cell length: {flywire_cell.skeleton.cable_length()/1000:.1f} μm")

# Species-specific analysis parameters
if "microns" in client.server_address:
    # Mouse cortical neurons - expect stronger compartmentalization
    segregation_threshold = 0.6
    axon_algorithm = "flow"  # Flow-based works well for mammalian neurons
else:
    # Fly neurons - may have different patterns
    segregation_threshold = 0.4
    axon_algorithm = "spectral"  # Spectral may work better for fly neurons
```

#### Error Handling and Validation

```python
def robust_cave_import(root_id, client, max_retries=3):
    \"\"\"Import with error handling and validation\"\"\"
    
    for attempt in range(max_retries):
        try:
            # Check if root_id is valid
            if not client.chunkedgraph.is_latest_roots([root_id])[0]:
                raise ValueError(f"Root ID {root_id} is not current")
            
            # Import cell
            cell = ossify.cell_from_client(
                root_id=root_id,
                client=client,
                synapses=True,
                restore_graph=False
            )
            
            # Validation checks
            if cell.skeleton.n_vertices < 10:
                raise ValueError("Skeleton too small (< 10 vertices)")
            
            total_length = cell.skeleton.cable_length()
            if total_length < 1000:  # < 1 μm
                raise ValueError(f"Skeleton too short: {total_length:.0f} nm")
            
            if total_length > 50_000_000:  # > 50 mm
                raise ValueError(f"Skeleton unreasonably long: {total_length:.0f} nm")
            
            # Check for disconnected components
            if len(cell.skeleton.cover_paths) > 1:
                print(f"Warning: Skeleton has {len(cell.skeleton.cover_paths)} components")
            
            return cell
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            
            # Wait before retry
            import time
            time.sleep(2 ** attempt)  # Exponential backoff

# Robust import with validation
try:
    cell = robust_cave_import(864691135336055529, client)
    print(f"Successfully imported cell {cell.name}")
except Exception as e:
    print(f"Failed to import cell: {e}")
```

---

## Integration Examples

### **Cross-Platform Analysis Pipeline**

```python
def cave_to_analysis_pipeline(root_ids, client, output_format="both"):
    \"\"\"Complete pipeline from CAVE import to analysis results\"\"\"
    
    results = {}
    
    for root_id in root_ids:
        print(f"Processing cell {root_id}...")
        
        # Import from CAVE
        cell = ossify.cell_from_client(
            root_id=root_id,
            client=client, 
            synapses=True,
            restore_properties=True
        )
        
        # Morphological analysis
        strahler = ossify.strahler_number(cell)
        cell.skeleton.add_label(strahler, "strahler_order")
        
        # Compartment analysis
        is_axon, segregation = ossify.label_axon_from_synapse_flow(
            cell, return_segregation_index=True
        )
        compartment = ["dendrite" if not ax else "axon" for ax in is_axon]
        cell.skeleton.add_label(compartment, "compartment")
        
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
    
    **Quality Control**: Built-in validation for data integrity and biological plausibility.

!!! tip "Best Practices for CAVE Integration"
    
    - **Use Timestamps**: Always specify timestamps for reproducible analyses
    - **Batch Processing**: Process multiple cells with consistent parameters  
    - **Error Handling**: Implement robust error handling for network operations
    - **Data Validation**: Validate imported data before analysis
    - **Version Control**: Track skeleton service and dataset versions used
    - **Memory Management**: Consider memory usage when importing large datasets