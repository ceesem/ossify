import numpy as np
import pandas as pd
import pytest

from ossify.sync_classes import Link, MorphSync


class TestMorphSync:
    """Tests for MorphSync functionality."""

    def test_morphsync_creation(self):
        """Test creating a MorphSync object."""
        morphsync = MorphSync(name="test_sync")
        assert morphsync.name == "test_sync"
        assert len(morphsync.layer_names) == 0
        assert len(morphsync.links) == 0

    def test_layer_addition_and_removal(self, simple_skeleton_data, spatial_columns):
        """Test adding and removing layers."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        morphsync = MorphSync()

        # Add a graph layer
        morphsync.add_graph(
            graph=(vertex_df, edges), name="test_graph", spatial_columns=spatial_columns
        )

        assert "test_graph" in morphsync.layer_names
        assert morphsync.has_layer("test_graph")

        # Test layer access
        layer = morphsync.get_layer("test_graph")
        assert layer.n_vertices == 5

        # Test layer removal
        morphsync.drop_layer("test_graph")
        assert "test_graph" not in morphsync.layer_names
        assert not morphsync.has_layer("test_graph")

    def test_layer_types_property(
        self, simple_skeleton_data, simple_mesh_data, spatial_columns
    ):
        """Test the layer_types property."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_mesh, faces_mesh, indices_mesh = simple_mesh_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        mesh_df = pd.DataFrame(
            vertices_mesh, columns=spatial_columns, index=indices_mesh
        )

        morphsync = MorphSync()

        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )

        morphsync.add_mesh(
            mesh=(mesh_df, faces_mesh), name="mesh", spatial_columns=spatial_columns
        )

        layer_types = morphsync.layer_types
        assert len(layer_types) == 2
        assert "skeleton" in layer_types
        assert "mesh" in layer_types

    def test_link_creation_strategies(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test different link creation strategies."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Test index mapping
        morphsync.add_link("skeleton", "graph", mapping="index")
        links = morphsync.links
        assert len(links) > 0

        # Test specified mapping with dictionary
        specified_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        morphsync.add_link("skeleton", "graph", mapping=specified_mapping)

        # Test mapping retrieval
        link_df = morphsync.get_link("skeleton", "graph")
        assert len(link_df) > 0
        assert "skeleton" in link_df.columns
        assert "graph" in link_df.columns

    def test_link_creation_with_arrays(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test link creation with numpy array mapping."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Create numpy array mapping
        array_mapping = np.array(
            [300, 301, 302, 303, 304]
        )  # Maps skeleton indices to graph indices

        morphsync.add_link("skeleton", "graph", mapping=array_mapping)

        # Test that link was created
        link_df = morphsync.get_link("skeleton", "graph")
        assert len(link_df) == 5

    def test_link_creation_with_series(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test link creation with pandas Series mapping."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Create pandas Series mapping
        series_mapping = pd.Series(data=[300, 301, 302, 303, 304], index=indices_skel)

        morphsync.add_link("skeleton", "graph", mapping=series_mapping)

        # Test that link was created
        link_df = morphsync.get_link("skeleton", "graph")
        assert len(link_df) == 5

    def test_mapping_path_traversal(
        self, simple_skeleton_data, simple_graph_data, simple_mesh_data, spatial_columns
    ):
        """Test mapping across multiple layers."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data
        vertices_mesh, faces_mesh, indices_mesh = simple_mesh_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )
        mesh_df = pd.DataFrame(
            vertices_mesh, columns=spatial_columns, index=indices_mesh
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )
        morphsync.add_mesh(
            mesh=(mesh_df, faces_mesh), name="mesh", spatial_columns=spatial_columns
        )

        # Create chain of mappings: skeleton -> graph -> mesh
        skel_to_graph = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        graph_to_mesh = {
            300: 200,
            301: 201,
            302: 202,
            303: 203,
        }  # Note: 304 -> no mapping

        morphsync.add_link("skeleton", "graph", mapping=skel_to_graph)
        morphsync.add_link("graph", "mesh", mapping=graph_to_mesh)

        # Test link graph
        link_graph = morphsync.link_graph
        assert "skeleton" in link_graph.nodes
        assert "graph" in link_graph.nodes
        assert "mesh" in link_graph.nodes

        # Test path finding
        path = morphsync.get_link_path("skeleton", "mesh")
        assert path == ["skeleton", "graph", "mesh"]

        # Test mapping traversal
        mapping = morphsync.get_mapping("skeleton", "mesh", null_strategy="drop")
        assert len(mapping) == 4  # Should lose vertex 104 due to incomplete mapping

    def test_mapping_validation_modes(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test mapping validation modes."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Create one-to-one mapping
        one_to_one_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        morphsync.add_link("skeleton", "graph", mapping=one_to_one_mapping)

        # Test validation - this should pass
        mapping = morphsync.get_mapping("skeleton", "graph", validate="one_to_one")
        assert len(mapping) == 5

        # Test with many-to-one scenario
        many_to_one_mapping = {
            100: 300,
            101: 300,
            102: 301,
            103: 301,
            104: 302,
        }  # Multiple skeletons -> same graph
        morphsync.add_link("skeleton", "graph", mapping=many_to_one_mapping)

        # This should work with many-to-one validation
        mapping = morphsync.get_mapping("skeleton", "graph", validate="many_to_one")
        assert len(mapping) == 5

    def test_null_strategy_behaviors(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test different null handling strategies."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Create partial mapping (missing some vertices)
        partial_mapping = {100: 300, 101: 301, 102: 302}  # Missing 103, 104
        morphsync.add_link("skeleton", "graph", mapping=partial_mapping)

        # Test drop strategy
        mapping_drop = morphsync.get_mapping("skeleton", "graph", null_strategy="drop")
        assert len(mapping_drop) == 3
        assert not mapping_drop.isna().any()

        # Test keep strategy
        mapping_keep = morphsync.get_mapping("skeleton", "graph", null_strategy="keep")
        assert len(mapping_keep) == 5
        assert mapping_keep.isna().any()
        assert mapping_keep.isna().sum() == 2

        # Test sentinel strategy
        mapping_sentinel = morphsync.get_mapping(
            "skeleton", "graph", null_strategy="sentinel"
        )
        assert len(mapping_sentinel) == 5
        assert (mapping_sentinel == -1).any()
        assert (mapping_sentinel == -1).sum() == 2

    def test_get_mapping_paths(
        self, simple_skeleton_data, simple_graph_data, simple_mesh_data, spatial_columns
    ):
        """Test get_mapping_paths method."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data
        vertices_mesh, faces_mesh, indices_mesh = simple_mesh_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )
        mesh_df = pd.DataFrame(
            vertices_mesh, columns=spatial_columns, index=indices_mesh
        )

        morphsync = MorphSync()

        # Add layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )
        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )
        morphsync.add_mesh(
            mesh=(mesh_df, faces_mesh), name="mesh", spatial_columns=spatial_columns
        )

        # Create chain of mappings
        skel_to_graph = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        graph_to_mesh = {300: 200, 301: 201, 302: 202, 303: 203}

        morphsync.add_link("skeleton", "graph", mapping=skel_to_graph)
        morphsync.add_link("graph", "mesh", mapping=graph_to_mesh)

        # Test get_mapping_paths
        mapping_paths = morphsync.get_mapping_paths(
            "skeleton", "mesh", null_strategy="drop"
        )

        # Should have columns for each step in the path
        assert "skeleton" in mapping_paths.columns
        assert "graph" in mapping_paths.columns
        assert "mesh" in mapping_paths.columns
        assert len(mapping_paths) == 4  # Lost vertex 104 due to incomplete mapping

    def test_delayed_link_addition(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test adding links before layers exist (delayed addition)."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        morphsync = MorphSync()

        # Try to add link before layers exist
        mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        morphsync.add_link("skeleton", "graph", mapping=mapping)

        # Should be stored in delayed links
        assert len(morphsync._delayed_add_links) > 0

        # Add the layers
        morphsync.add_graph(
            graph=(skel_df, edges_skel),
            name="skeleton",
            spatial_columns=spatial_columns,
        )

        # Still delayed until both layers exist
        assert len(morphsync._delayed_add_links) > 0

        morphsync.add_graph(
            graph=(graph_df, edges_graph), name="graph", spatial_columns=spatial_columns
        )

        # Now should be processed
        assert len(morphsync.links) > 0

        # Test that mapping works
        final_mapping = morphsync.get_mapping("skeleton", "graph")
        assert len(final_mapping) == 5


class TestLink:
    """Tests for Link class functionality."""

    def test_link_creation_with_mapping_dict(self):
        """Test creating Link with dictionary mapping."""
        mapping = {100: 200, 101: 201, 102: 202}
        link = Link(mapping=mapping, source="layer1", target="layer2")

        assert link.source == "layer1"
        assert link.target == "layer2"
        assert link.mapping == mapping

    def test_link_creation_with_mapping_string(self):
        """Test creating Link with string mapping reference."""
        link = Link(mapping="column_name", source="layer1", target="layer2")

        assert link.source == "layer1"
        assert link.target == "layer2"
        assert link.mapping == "column_name"

    def test_link_mapping_to_index_conversion(self):
        """Test converting mapping to index format."""
        # Create test data
        test_nodes = pd.DataFrame(
            {
                "x": [0, 1, 2],
                "y": [0, 0, 0],
                "z": [0, 0, 0],
                "target_col": [200, 201, 202],
            },
            index=[100, 101, 102],
        )

        # Test with column reference
        link = Link(mapping="target_col", source="layer1", target="layer2")

        # Convert to index mapping
        index_mapping = link.mapping_to_index(test_nodes)

        # Should return the column values
        expected = np.array([200, 201, 202])
        np.testing.assert_array_equal(index_mapping, expected)

    def test_link_properties(self):
        """Test Link property access."""
        mapping = {100: 200, 101: 201}
        link = Link(
            mapping=mapping,
            source="source_layer",
            target="target_layer",
            map_value_is_index=False,
        )

        assert link.source == "source_layer"
        assert link.target == "target_layer"
        assert link.mapping == mapping
        assert link.map_value_is_index == False
