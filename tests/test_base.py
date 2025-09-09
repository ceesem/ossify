import fastremap
import numpy as np
import pandas as pd
import pytest

from ossify import Cell, Link


def test_create_meshwork(
    root_id,
    skel_dict,
    l2_graph,
    l2_df,
    pre_syn_df,
    post_syn_df,
    synapse_spatial_columns,
    l2_spatial_columns,
):
    l2_map = {v: k for k, v in l2_df["l2_id"].to_dict().items()}
    edges = fastremap.remap(
        l2_graph,
        l2_map,
    )

    nrn = (
        Cell(
            name=root_id,
        )
        .add_graph(
            vertices=l2_df,
            spatial_columns=l2_spatial_columns,
            edges=edges,
            vertex_index="l2_id",
        )
        .add_skeleton(
            vertices=np.array(skel_dict["vertices"]),
            edges=np.array(skel_dict["edges"]),
            labels={
                "radius": skel_dict["radius"],
                "compartment": skel_dict["compartment"],
            },
            linkage=Link(
                mapping=skel_dict["mesh_to_skel_map"],
                source="graph",
                map_value_is_index=False,
            ),
        )
        .add_point_annotations(
            "pre_syn",
            vertices=pre_syn_df,
            spatial_columns=synapse_spatial_columns,
            vertex_index="id",
            linkage=Link(mapping="pre_pt_l2_id", target="graph"),
        )
        .add_point_annotations(
            "post_syn",
            vertices=post_syn_df,
            spatial_columns=synapse_spatial_columns,
            vertex_index="id",
            linkage=Link(mapping="post_pt_l2_id", target="graph"),
        )
    )
    assert len(nrn.layers) == 2


def test_morphology_loading(nrn):
    assert nrn is not None
    assert isinstance(nrn, Cell)
    # Updated expected value based on actual data
    assert np.isclose(nrn.skeleton.distance_to_root()[0], 400552.1214790344, rtol=1e-3)


# ============================================================================
# Unit Tests with Mock Data
# ============================================================================


def test_cell_creation_empty():
    """Test creating an empty cell."""
    cell = Cell()
    assert cell.name is None
    assert len(cell.layers.names) == 0
    assert cell._morphsync is not None


def test_cell_creation_with_name():
    """Test creating a cell with a name."""
    cell_name = "test_neuron_123"
    cell = Cell(name=cell_name)
    assert cell.name == cell_name
    assert len(cell.layers.names) == 0


def test_cell_add_skeleton_basic(simple_skeleton_data, spatial_columns, mock_labels):
    """Test adding a basic skeleton to a cell."""
    vertices, edges, vertex_indices = simple_skeleton_data

    # Create DataFrame from vertices
    vertex_df = pd.DataFrame(vertices, columns=spatial_columns)
    vertex_df.index = vertex_indices

    # Fix edges to reference actual vertex indices instead of positional indices
    edges_with_indices = np.array(
        [
            [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
            [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
            [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
            [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
        ]
    )

    cell = Cell(name="test_cell")
    cell.add_skeleton(
        vertices=vertex_df,
        edges=edges_with_indices,
        spatial_columns=spatial_columns,
        root=vertex_indices[0],  # First vertex as root
        labels=mock_labels,
    )

    assert "skeleton" in cell.layers.names
    assert cell.skeleton.n_vertices == 5
    assert cell.skeleton.root == vertex_indices[0]
    assert len(cell.skeleton.edges) == 4


def test_cell_add_mesh_basic(simple_mesh_data, spatial_columns):
    """Test adding a basic mesh to a cell."""
    vertices, faces, vertex_indices = simple_mesh_data

    vertex_df = pd.DataFrame(vertices, columns=spatial_columns)
    vertex_df.index = vertex_indices

    # Fix faces to reference actual vertex indices instead of positional indices
    faces_with_indices = np.array(
        [
            [vertex_indices[0], vertex_indices[1], vertex_indices[2]],
            [vertex_indices[0], vertex_indices[1], vertex_indices[3]],
            [vertex_indices[0], vertex_indices[2], vertex_indices[3]],
            [vertex_indices[1], vertex_indices[2], vertex_indices[3]],
        ]
    )

    cell = Cell(name="test_cell")
    cell.add_mesh(
        vertices=vertex_df, faces=faces_with_indices, spatial_columns=spatial_columns
    )

    assert "mesh" in cell.layers.names
    assert cell.mesh.n_vertices == 4
    assert len(cell.mesh.faces) == 4


def test_cell_add_graph_basic(simple_graph_data, spatial_columns):
    """Test adding a basic graph to a cell."""
    vertices, edges, vertex_indices = simple_graph_data

    vertex_df = pd.DataFrame(vertices, columns=spatial_columns)
    vertex_df.index = vertex_indices

    # Fix edges to reference actual vertex indices instead of positional indices
    edges_with_indices = np.array(
        [
            [vertex_indices[0], vertex_indices[1]],  # 300 -> 301
            [vertex_indices[1], vertex_indices[2]],  # 301 -> 302
            [vertex_indices[1], vertex_indices[3]],  # 301 -> 303
            [vertex_indices[3], vertex_indices[4]],  # 303 -> 304
        ]
    )

    cell = Cell(name="test_cell")
    cell.add_graph(
        vertices=vertex_df, edges=edges_with_indices, spatial_columns=spatial_columns
    )

    assert "graph" in cell.layers.names
    assert cell.graph.n_vertices == 5
    assert len(cell.graph.edges) == 4


def test_cell_add_point_annotations_basic(mock_point_annotations, spatial_columns):
    """Test adding point annotations to a cell."""
    cell = Cell(name="test_cell")
    cell.add_point_annotations(
        name="synapses",
        vertices=mock_point_annotations,
        spatial_columns=spatial_columns,
    )

    # Point annotations create annotation layers, not main layers
    assert "synapses" in [layer.name for layer in cell.annotations]
    assert cell.annotations["synapses"].n_vertices == 4

    # Check that annotation labels are preserved in the point cloud
    assert "annotation_type" in cell.annotations["synapses"].label_names
    assert "confidence" in cell.annotations["synapses"].label_names


def test_cell_layer_management(simple_skeleton_data, simple_mesh_data, spatial_columns):
    """Test layer access and management methods."""
    vertices_skel, edges_skel, indices_skel = simple_skeleton_data
    vertices_mesh, faces_mesh, indices_mesh = simple_mesh_data

    # Create DataFrames
    skel_df = pd.DataFrame(vertices_skel, columns=spatial_columns, index=indices_skel)
    mesh_df = pd.DataFrame(vertices_mesh, columns=spatial_columns, index=indices_mesh)

    # Fix edges and faces to use vertex indices
    edges_with_indices = np.array(
        [
            [indices_skel[1], indices_skel[0]],
            [indices_skel[2], indices_skel[1]],
            [indices_skel[3], indices_skel[2]],
            [indices_skel[4], indices_skel[3]],
        ]
    )

    faces_with_indices = np.array(
        [
            [indices_mesh[0], indices_mesh[1], indices_mesh[2]],
            [indices_mesh[0], indices_mesh[1], indices_mesh[3]],
            [indices_mesh[0], indices_mesh[2], indices_mesh[3]],
            [indices_mesh[1], indices_mesh[2], indices_mesh[3]],
        ]
    )

    cell = Cell(name="test_cell")
    cell.add_skeleton(
        vertices=skel_df,
        edges=edges_with_indices,
        spatial_columns=spatial_columns,
        root=indices_skel[0],
    )
    cell.add_mesh(
        vertices=mesh_df, faces=faces_with_indices, spatial_columns=spatial_columns
    )

    # Test layer access
    assert len(cell.layers.names) == 2
    assert "skeleton" in cell.layers.names
    assert "mesh" in cell.layers.names

    # Test direct access
    assert cell.skeleton.n_vertices == 5
    assert cell.mesh.n_vertices == 4


def test_cell_with_linkage(simple_skeleton_data, simple_graph_data, spatial_columns):
    """Test creating linked layers."""
    vertices_skel, edges_skel, indices_skel = simple_skeleton_data
    vertices_graph, edges_graph, indices_graph = simple_graph_data

    skel_df = pd.DataFrame(vertices_skel, columns=spatial_columns, index=indices_skel)
    graph_df = pd.DataFrame(
        vertices_graph, columns=spatial_columns, index=indices_graph
    )

    # Fix edges to use vertex indices
    edges_skel_fixed = np.array(
        [
            [indices_skel[1], indices_skel[0]],
            [indices_skel[2], indices_skel[1]],
            [indices_skel[3], indices_skel[2]],
            [indices_skel[4], indices_skel[3]],
        ]
    )

    edges_graph_fixed = np.array(
        [
            [indices_graph[0], indices_graph[1]],
            [indices_graph[1], indices_graph[2]],
            [indices_graph[1], indices_graph[3]],
            [indices_graph[3], indices_graph[4]],
        ]
    )

    # Create complete mapping (all skeleton vertices map to graph vertices)
    skeleton_to_graph_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}

    cell = Cell(name="test_cell")
    cell.add_graph(
        vertices=graph_df, edges=edges_graph_fixed, spatial_columns=spatial_columns
    )
    cell.add_skeleton(
        vertices=skel_df,
        edges=edges_skel_fixed,
        spatial_columns=spatial_columns,
        root=indices_skel[0],
        linkage=Link(mapping=skeleton_to_graph_mapping, target="graph"),
    )

    # Test that linkage was created
    links = cell._morphsync.links
    assert len(links) > 0

    # Test mapping functionality
    mapped_indices = cell.skeleton.map_index_to_layer(
        "graph", source_index=np.array([100, 101])
    )
    assert 300 in mapped_indices
    assert 301 in mapped_indices


def test_cell_copy_and_properties():
    """Test cell copying and basic property access."""
    cell = Cell(name="original_cell")

    # Test basic properties
    assert cell.name == "original_cell"
    assert hasattr(cell, "_morphsync")

    # Add a simple layer for copy testing
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )

    # Fix edges to use vertex indices instead of positional indices
    edges = np.array([[11, 10], [12, 11]])  # Use actual vertex indices

    cell.add_skeleton(
        vertices=vertices, edges=edges, spatial_columns=["x", "y", "z"], root=10
    )

    # Test that layers were added correctly
    assert "skeleton" in cell.layers.names
    assert cell.skeleton.n_vertices == 3


# ============================================================================
# Layer Deletion Tests
# ============================================================================


def test_layer_manager_internal_remove():
    """Test LayerManager _remove method with error handling."""
    from ossify.base import LayerManager
    from ossify.data_layers import SkeletonLayer

    manager = LayerManager()

    # Create and add a mock layer
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges = np.array([[11, 10], [12, 11]])

    layer = SkeletonLayer(
        name="test_skeleton",
        vertices=vertices,
        edges=edges,
        spatial_columns=["x", "y", "z"],
        root=10,
    )
    manager._add(layer)

    # Test successful removal
    assert "test_skeleton" in manager
    manager._remove("test_skeleton")
    assert "test_skeleton" not in manager

    # Test error on removing non-existent layer
    with pytest.raises(ValueError, match='Layer "nonexistent" does not exist'):
        manager._remove("nonexistent")


def test_cell_remove_layer_basic():
    """Test basic layer removal from Cell."""
    cell = Cell(name="test_cell")

    # Add a skeleton layer
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges = np.array([[11, 10], [12, 11]])

    cell.add_skeleton(
        vertices=vertices, edges=edges, spatial_columns=["x", "y", "z"], root=10
    )

    # Verify layer was added
    assert "skeleton" in cell.layers.names
    assert "skeleton" in cell._managed_layers
    assert "skeleton" in cell._morphsync.layers

    # Remove the layer
    result = cell.remove_layer("skeleton")
    assert result == cell  # Test method chaining

    # Verify layer was removed from all locations
    assert "skeleton" not in cell.layers.names
    assert "skeleton" not in cell._managed_layers
    assert "skeleton" not in cell._morphsync.layers


def test_cell_remove_annotation_basic():
    """Test basic annotation removal from Cell."""
    cell = Cell(name="test_cell")

    # Add point annotations
    annotations = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 0, 1, 1],
            "z": [0, 1, 0, 1],
            "annotation_type": ["synapse", "synapse", "bouton", "bouton"],
            "confidence": [0.95, 0.87, 0.92, 0.78],
        }
    )

    cell.add_point_annotations(
        name="synapses", vertices=annotations, spatial_columns=["x", "y", "z"]
    )

    # Verify annotation was added
    assert "synapses" in cell.annotations.names
    assert "synapses" in cell._annotations._layers
    assert "synapses" in cell._morphsync.layers

    # Remove the annotation
    result = cell.remove_annotation("synapses")
    assert result == cell  # Test method chaining

    # Verify annotation was removed from all locations
    assert "synapses" not in cell.annotations.names
    assert "synapses" not in cell._annotations._layers
    assert "synapses" not in cell._morphsync.layers


def test_cell_remove_layer_with_links():
    """Test layer removal with linkage cleanup."""
    cell = Cell(name="test_cell")

    # Add graph layer
    graph_vertices = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4], "y": [0, 0, 1, 1, 2], "z": [0, 1, 0, 1, 0]},
        index=[300, 301, 302, 303, 304],
    )
    graph_edges = np.array([[300, 301], [301, 302], [301, 303], [303, 304]])

    cell.add_graph(
        vertices=graph_vertices, edges=graph_edges, spatial_columns=["x", "y", "z"]
    )

    # Add skeleton layer with linkage to graph
    skel_vertices = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4], "y": [0, 0, 1, 1, 2], "z": [0, 1, 0, 1, 0]},
        index=[100, 101, 102, 103, 104],
    )
    skel_edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
    skeleton_to_graph_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}

    cell.add_skeleton(
        vertices=skel_vertices,
        edges=skel_edges,
        spatial_columns=["x", "y", "z"],
        root=100,
        linkage=Link(mapping=skeleton_to_graph_mapping, target="graph"),
    )

    # Verify linkage was created
    initial_links = len(cell._morphsync.links)
    assert initial_links > 0

    # Remove skeleton layer
    cell.remove_layer("skeleton")

    # Verify links involving skeleton were removed
    remaining_links = len(cell._morphsync.links)
    assert remaining_links < initial_links

    # Verify any remaining links don't reference skeleton
    for link_key in cell._morphsync.links.keys():
        assert "skeleton" not in link_key


def test_cell_remove_nonexistent_layer():
    """Test error handling when removing non-existent layers."""
    cell = Cell(name="test_cell")

    # Try to remove non-existent layer
    with pytest.raises(ValueError, match='Layer "nonexistent" does not exist'):
        cell.remove_layer("nonexistent")

    # Try to remove non-existent annotation
    with pytest.raises(ValueError, match='Annotation "nonexistent" does not exist'):
        cell.remove_annotation("nonexistent")


def test_cell_remove_layer_edge_cases():
    """Test edge cases for layer removal."""
    cell = Cell(name="test_cell")

    # Add multiple layers
    vertices1 = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges1 = np.array([[11, 10], [12, 11]])

    vertices2 = pd.DataFrame(
        {"x": [0, 1, 2, 3], "y": [0, 0, 1, 1], "z": [0, 1, 0, 1]},
        index=[20, 21, 22, 23],
    )
    faces2 = np.array([[20, 21, 22], [20, 21, 23], [20, 22, 23], [21, 22, 23]])

    cell.add_skeleton(
        vertices=vertices1, edges=edges1, spatial_columns=["x", "y", "z"], root=10
    )

    cell.add_mesh(vertices=vertices2, faces=faces2, spatial_columns=["x", "y", "z"])

    # Verify both layers exist
    assert len(cell.layers.names) == 2
    assert "skeleton" in cell.layers.names
    assert "mesh" in cell.layers.names

    # Remove one layer
    cell.remove_layer("skeleton")

    # Verify only mesh remains
    assert len(cell.layers.names) == 1
    assert "mesh" in cell.layers.names
    assert "skeleton" not in cell.layers.names

    # Remove remaining layer
    cell.remove_layer("mesh")

    # Verify cell is now empty
    assert len(cell.layers.names) == 0


# ============================================================================
# LayerManager Validation Tests
# ============================================================================


def test_layer_manager_custom_validation():
    """Test LayerManager with custom validation functions."""
    from ossify.base import LayerManager
    from ossify.data_layers import GraphLayer, SkeletonLayer

    # Custom validation that only accepts skeleton layers
    def skeleton_only(layer):
        return isinstance(layer, SkeletonLayer)

    manager = LayerManager(validation=skeleton_only, context="custom")

    # Create skeleton layer (should pass)
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges = np.array([[11, 10], [12, 11]])

    skeleton_layer = SkeletonLayer(
        name="test_skeleton",
        vertices=vertices,
        edges=edges,
        spatial_columns=["x", "y", "z"],
        root=10,
    )

    # Should succeed
    manager._add(skeleton_layer)
    assert "test_skeleton" in manager

    # Create graph layer (should fail)
    graph_layer = GraphLayer(
        name="test_graph",
        vertices=vertices,
        edges=edges,
        spatial_columns=["x", "y", "z"],
    )

    # Should fail validation
    with pytest.raises(ValueError, match="Layer validation failed for custom"):
        manager._add(graph_layer)


def test_layer_manager_initial_layers():
    """Test LayerManager with initial_layers parameter."""
    from ossify.base import LayerManager
    from ossify.data_layers import PointCloudLayer

    # Create some initial layers
    annotations1 = pd.DataFrame(
        {"x": [0, 1], "y": [0, 1], "z": [0, 1], "type": ["synapse", "bouton"]}
    )

    annotations2 = pd.DataFrame(
        {"x": [2, 3], "y": [2, 3], "z": [2, 3], "type": ["dendrite", "spine"]}
    )

    layer1 = PointCloudLayer(
        name="synapses", vertices=annotations1, spatial_columns=["x", "y", "z"]
    )

    layer2 = PointCloudLayer(
        name="dendrites", vertices=annotations2, spatial_columns=["x", "y", "z"]
    )

    initial_layers = [layer1, layer2]

    # Create manager with initial layers
    manager = LayerManager(
        validation="point_cloud_only",
        context="annotation",
        initial_layers=initial_layers,
    )

    # Verify initial layers were added
    assert len(manager) == 2
    assert "synapses" in manager
    assert "dendrites" in manager


def test_layer_manager_point_cloud_validation():
    """Test strict point cloud validation."""
    from ossify.base import LayerManager
    from ossify.data_layers import PointCloudLayer, SkeletonLayer

    manager = LayerManager(validation="point_cloud_only", context="annotation")

    # Create point cloud layer (should pass)
    annotations = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "z": [0, 1, 2],
            "type": ["synapse", "bouton", "spine"],
        }
    )

    point_layer = PointCloudLayer(
        name="test_points", vertices=annotations, spatial_columns=["x", "y", "z"]
    )

    # Should succeed
    manager._add(point_layer)
    assert "test_points" in manager

    # Create skeleton layer (should fail)
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges = np.array([[11, 10], [12, 11]])

    skeleton_layer = SkeletonLayer(
        name="test_skeleton",
        vertices=vertices,
        edges=edges,
        spatial_columns=["x", "y", "z"],
        root=10,
    )

    # Should fail validation
    with pytest.raises(
        ValueError, match="Annotation layers must be PointCloudLayer instances"
    ):
        manager._add(skeleton_layer)


def test_layer_manager_invalid_validation_mode():
    """Test LayerManager with invalid validation mode."""
    from ossify.base import LayerManager

    # Should raise error for unknown validation mode
    with pytest.raises(ValueError, match="Unknown validation mode: invalid_mode"):
        manager = LayerManager(validation="invalid_mode")
        # Need to trigger validation
        from ossify.data_layers import PointCloudLayer

        annotations = pd.DataFrame({"x": [0], "y": [0], "z": [0], "type": ["test"]})
        layer = PointCloudLayer(
            name="test", vertices=annotations, spatial_columns=["x", "y", "z"]
        )
        manager._add(layer)


# ============================================================================
# Cell Property and Meta Tests
# ============================================================================


def test_cell_meta_property():
    """Test Cell meta property access and modification."""
    initial_meta = {"source": "test", "timestamp": "2024-01-01", "version": 1.0}
    cell = Cell(name="test_cell", meta=initial_meta)

    # Test meta access
    meta = cell.meta
    assert meta["source"] == "test"
    assert meta["timestamp"] == "2024-01-01"
    assert meta["version"] == 1.0

    # Test meta modification
    cell.meta["new_field"] = "added_value"
    assert cell.meta["new_field"] == "added_value"

    # Test that meta is a copy (original dict shouldn't change)
    initial_meta["should_not_appear"] = "invisible"
    assert "should_not_appear" not in cell.meta


def test_cell_name_edge_cases():
    """Test Cell name property with various types."""
    # Test with string name
    cell_str = Cell(name="string_name")
    assert cell_str.name == "string_name"

    # Test with integer name
    cell_int = Cell(name=12345)
    assert cell_int.name == 12345

    # Test with None name
    cell_none = Cell(name=None)
    assert cell_none.name is None

    # Test default (no name provided)
    cell_default = Cell()
    assert cell_default.name is None


def test_cell_layer_property_access():
    """Test Cell layer properties when layers don't exist."""
    cell = Cell(name="test_cell")

    # Test properties when layers don't exist
    assert cell.skeleton is None
    assert cell.graph is None
    assert cell.mesh is None

    # Test _all_objects when empty
    all_objects = cell._all_objects
    assert len(all_objects) == 0
    assert isinstance(all_objects, dict)


def test_cell_all_objects_property():
    """Test Cell _all_objects property with mixed layers and annotations."""
    cell = Cell(name="test_cell")

    # Add a skeleton layer
    vertices = pd.DataFrame(
        {"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0]}, index=[10, 11, 12]
    )
    edges = np.array([[11, 10], [12, 11]])

    cell.add_skeleton(
        vertices=vertices, edges=edges, spatial_columns=["x", "y", "z"], root=10
    )

    # Add annotations
    annotations = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 0, 1, 1],
            "z": [0, 1, 0, 1],
            "type": ["synapse", "synapse", "bouton", "bouton"],
        }
    )

    cell.add_point_annotations(
        name="synapses", vertices=annotations, spatial_columns=["x", "y", "z"]
    )

    # Test _all_objects includes both layers and annotations
    all_objects = cell._all_objects
    assert len(all_objects) == 2
    assert "skeleton" in all_objects
    assert "synapses" in all_objects

    # Verify objects are actual layer instances
    assert hasattr(all_objects["skeleton"], "n_vertices")
    assert hasattr(all_objects["synapses"], "n_vertices")
