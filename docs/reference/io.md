# File I/O Operations

Ossify provides robust file I/O capabilities supporting local files, cloud storage, and binary file objects with efficient compression and validation.

## Overview

| Function Category | Functions | Purpose |
|-------------------|-----------|---------|
| **[Core Functions](#core-functions)** | `load_cell`, `save_cell` | Primary interface for loading and saving cells |
| **[File Management](#file-management)** | `CellFiles` | Advanced file operations and cloud storage |

---

## Core Functions {: .doc-heading}

### load_cell

::: ossify.load_cell
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Load neuromorphological data from various sources including local files, cloud storage, and binary file objects.**

#### Usage Examples

```python
import ossify
from io import BytesIO

# Load from local file
cell = ossify.load_cell("data/neuron_12345.osy")

# Load from cloud storage (S3)
cell = ossify.load_cell("s3://my-bucket/neurons/cell_67890.osy")

# Load from Google Cloud Storage
cell = ossify.load_cell("gs://neuron-data/processed/cell_abc.osy")

# Load from HTTPS URL
cell = ossify.load_cell("https://example.com/data/neuron.osy")

# Load from binary file object
with open("neuron.osy", "rb") as f:
    cell = ossify.load_cell(f)

# Load from bytes in memory
with open("neuron.osy", "rb") as f:
    data_bytes = f.read()
    
cell = ossify.load_cell(BytesIO(data_bytes))
```

#### Supported Formats

- **Local Files**: Absolute or relative paths
- **Cloud Storage**: S3 (`s3://`), Google Cloud Storage (`gs://`)
- **HTTP/HTTPS**: Direct URLs to `.osy` files
- **File Objects**: Any binary file-like object
- **In-Memory**: BytesIO objects with ossify data

### save_cell

::: ossify.save_cell
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Save Cell objects to various destinations with automatic compression and metadata.**

#### Usage Examples

```python
# Save to local file
ossify.save_cell(cell, "output/processed_neuron.osy")

# Save with automatic naming (uses cell.name)
ossify.save_cell(cell)  # Creates "{cell.name}.osy"

# Save to cloud storage
ossify.save_cell(cell, "s3://results-bucket/analysis/neuron_final.osy")

# Save to file object
with open("neuron_backup.osy", "wb") as f:
    ossify.save_cell(cell, f)

# Overwrite protection
try:
    ossify.save_cell(cell, "existing_file.osy", allow_overwrite=False)
except FileExistsError:
    print("File already exists!")

# Allow overwriting
ossify.save_cell(cell, "existing_file.osy", allow_overwrite=True)
```

#### Compression and Storage

The `.osy` format uses efficient compression:

- **Vertices/Labels**: Feather format with Zstandard compression
- **Connectivity**: Sparse matrix compression for edges/faces  
- **Metadata**: JSON with numpy array serialization
- **Archive**: TAR format for multiple file organization

---

## File Management {: .doc-heading}

### CellFiles

::: ossify.CellFiles
    options:
        heading_level: 4
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        separate_signature: true
        show_source: false

**Advanced file management for batch operations and cloud storage integration.**

#### Usage Examples

```python
# Local directory management
files = ossify.CellFiles("./neuron_data/")

# Cloud storage management  
s3_files = ossify.CellFiles("s3://my-bucket/neurons/", use_https=True)
gcs_files = ossify.CellFiles("gs://neuron-dataset/processed/")

# Check capabilities
print(f"Can save files: {files.saveable}")
print(f"Is remote storage: {files.remote}")

# Save with custom filename
files.save(cell, "custom_name.osy", allow_overwrite=True)

# Auto-generated filename
files.save(cell)  # Uses f"{cell.name}.osy"

# Load specific file
loaded_cell = files.load("neuron_12345.osy")

# Batch operations
filenames = ["cell_001.osy", "cell_002.osy", "cell_003.osy"]
cells = [files.load(fname) for fname in filenames]
```

#### Cloud Storage Configuration

```python
# S3 with custom configuration
s3_files = ossify.CellFiles(
    "s3://bucket-name/path/",
    use_https=True  # Use HTTPS for transfers
)

# Google Cloud Storage
gcs_files = ossify.CellFiles("gs://bucket-name/neurons/")

# Memory-based storage (for testing)
mem_files = ossify.CellFiles("mem://test-storage/")

# Check write permissions
if not s3_files.saveable:
    print("Cannot write to this location")
```

---

## Advanced I/O Operations

### **Batch Processing Pipeline**

```python
def process_neuron_batch(input_path, output_path, processing_func):
    \"\"\"Process multiple neurons with consistent I/O handling\"\"\"
    
    # Setup file managers
    input_files = ossify.CellFiles(input_path)
    output_files = ossify.CellFiles(output_path)
    
    if not output_files.saveable:
        raise ValueError(f"Cannot write to {output_path}")
    
    # Process all .osy files in input directory
    import os
    results = []
    
    for filename in os.listdir(input_path):
        if not filename.endswith('.osy'):
            continue
            
        print(f"Processing {filename}...")
        
        try:
            # Load cell
            cell = input_files.load(filename)
            
            # Apply processing function
            processed_cell = processing_func(cell)
            
            # Save result
            output_filename = f"processed_{filename}"
            output_files.save(
                processed_cell, 
                output_filename, 
                allow_overwrite=True
            )
            
            results.append({
                'input': filename,
                'output': output_filename, 
                'status': 'success',
                'cell_name': cell.name
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                'input': filename,
                'status': 'error',
                'error': str(e)
            })
    
    return results

# Example processing function
def add_analysis_labels(cell):
    \"\"\"Add standard analysis labels to cell\"\"\"
    
    # Add Strahler numbers
    strahler = ossify.strahler_number(cell)
    cell.skeleton.add_label(strahler, "strahler_order")
    
    # Add compartment labels if synapses available
    if "pre_syn" in cell.annotations.names:
        is_axon = ossify.label_axon_from_synapse_flow(cell)
        compartment = np.where(is_axon, "axon", "dendrite")
        cell.skeleton.add_label(compartment, "compartment")
    
    return cell

# Run batch processing
results = process_neuron_batch(
    input_path="s3://raw-data/neurons/",
    output_path="s3://processed-data/analyzed/", 
    processing_func=add_analysis_labels
)

# Summary
successful = sum(1 for r in results if r['status'] == 'success')
print(f"Successfully processed {successful}/{len(results)} files")
```

### **Data Validation and Integrity**

```python
def validate_cell_file(filepath):
    \"\"\"Validate cell file integrity and structure\"\"\"
    
    try:
        # Load cell
        cell = ossify.load_cell(filepath)
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic structure validation
        if cell.skeleton is None:
            validation_results['errors'].append("No skeleton layer found")
            
        if len(cell.layers.names) == 0:
            validation_results['errors'].append("No morphological layers found")
        
        # Skeleton validation
        if cell.skeleton is not None:
            skeleton = cell.skeleton
            
            # Check for disconnected components
            n_components = len(skeleton.cover_paths)
            if n_components > 1:
                validation_results['warnings'].append(
                    f"Skeleton has {n_components} disconnected components"
                )
            
            # Check for reasonable size
            total_length = skeleton.cable_length()
            if total_length > 10_000_000:  # > 10mm
                validation_results['warnings'].append(
                    f"Unusually long skeleton: {total_length:.0f} nm"
                )
            elif total_length < 1000:  # < 1μm
                validation_results['warnings'].append(
                    f"Unusually short skeleton: {total_length:.0f} nm"
                )
        
        # Annotation validation
        for anno_name in cell.annotations.names:
            anno = cell.annotations[anno_name]
            if len(anno) == 0:
                validation_results['warnings'].append(
                    f"Empty annotation layer: {anno_name}"
                )
        
        # Collect info
        validation_results['info'] = {
            'cell_name': cell.name,
            'n_layers': len(cell.layers.names),
            'layer_names': cell.layers.names,
            'n_annotations': len(cell.annotations.names),
            'annotation_names': cell.annotations.names
        }
        
        if cell.skeleton:
            validation_results['info'].update({
                'n_skeleton_vertices': cell.skeleton.n_vertices,
                'skeleton_length': cell.skeleton.cable_length(),
                'n_branch_points': len(cell.skeleton.branch_points),
                'n_end_points': len(cell.skeleton.end_points)
            })
        
        # Set overall validity
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        return validation_results
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Failed to load file: {str(e)}"],
            'warnings': [],
            'info': {}
        }

# Validate files
validation = validate_cell_file("neuron.osy")

if validation['valid']:
    print("✓ File is valid")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
else:
    print("✗ File validation failed")
    for error in validation['errors']:
        print(f"  ✗ {error}")

print(f"\nFile info: {validation['info']}")
```

### **Memory-Efficient Streaming**

```python
def stream_large_dataset(file_list, processing_func):
    \"\"\"Process large datasets without loading everything into memory\"\"\"
    
    for filepath in file_list:
        print(f"Processing {filepath}...")
        
        # Load one cell at a time
        cell = ossify.load_cell(filepath)
        
        try:
            # Process cell
            result = processing_func(cell)
            
            # Yield result without storing
            yield {
                'filepath': filepath,
                'result': result,
                'cell_name': cell.name
            }
            
        finally:
            # Explicit cleanup
            del cell

# Example: Extract features from many cells
def extract_features(cell):
    \"\"\"Extract numerical features from cell\"\"\"
    
    features = {}
    
    if cell.skeleton:
        skeleton = cell.skeleton
        features.update({
            'total_length': skeleton.cable_length(),
            'n_branch_points': len(skeleton.branch_points),
            'n_end_points': len(skeleton.end_points),
            'max_distance_to_root': skeleton.distance_to_root().max()
        })
        
        # Strahler analysis
        strahler = ossify.strahler_number(skeleton)
        features['max_strahler_order'] = strahler.max()
    
    return features

# Stream processing
file_paths = [f"neuron_{i:03d}.osy" for i in range(1000)]

all_features = []
for result in stream_large_dataset(file_paths, extract_features):
    all_features.append({
        'cell_name': result['cell_name'],
        **result['result']
    })
    
    # Optional: Save intermediate results
    if len(all_features) % 100 == 0:
        print(f"Processed {len(all_features)} files...")

# Convert to DataFrame for analysis
import pandas as pd
features_df = pd.DataFrame(all_features)
```

### **Cross-Platform Path Handling**

```python
from pathlib import Path
import os

def robust_file_operations(base_path):
    \"\"\"Handle file paths across different systems and storage types\"\"\"
    
    # Convert to Path object for cross-platform compatibility
    if isinstance(base_path, str) and not base_path.startswith(('s3://', 'gs://', 'http')):
        base_path = Path(base_path).expanduser().absolute()
    
    files = ossify.CellFiles(str(base_path))
    
    # Platform-aware path joining
    if files.remote:
        # Cloud storage - use forward slashes
        subdir_path = str(base_path) + "/subdirectory/"
    else:
        # Local storage - use os.path.join or Path
        subdir_path = base_path / "subdirectory"
    
    return files, subdir_path

# Usage examples
local_files, local_subdir = robust_file_operations("~/neuron_data")
cloud_files, cloud_subdir = robust_file_operations("s3://bucket/data")

# Check what type of storage we're using
if local_files.remote:
    print("Using remote storage")
else:
    print("Using local storage")
```

!!! info "File Format Details"
    
    **Archive Structure**: The `.osy` format is a TAR archive containing:
    
    - `metadata.json`: Cell metadata and structure information
    - `layers/{name}/`: Morphological layer data (nodes, edges, faces)
    - `annotations/{name}/`: Annotation layer data  
    - `linkage/{source}/{target}/`: Inter-layer mapping data
    
    **Compression**: Uses Feather format with Zstandard compression for optimal balance of speed and size.
    
    **Validation**: Built-in structure validation ensures data integrity during load operations.

!!! tip "Performance Tips"
    
    - **Batch Operations**: Use `CellFiles` for multiple operations in the same location
    - **Cloud Storage**: Enable `use_https=True` for secure cloud transfers
    - **Memory Management**: Use streaming approaches for large datasets
    - **Error Handling**: Always wrap I/O operations in try-except blocks
    - **Path Handling**: Use `pathlib.Path` for cross-platform compatibility