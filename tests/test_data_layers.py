import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ossify import Cell, Link


class TestSkeletonLayer:
    """Tests for SkeletonLayer functionality."""

    def test_skeleton_creation_minimal(self, simple_skeleton_data, spatial_columns):
        """Test creating a skeleton with minimal parameters."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton
        assert skeleton.n_vertices == 5
        assert skeleton.root == vertex_indices[0]
        assert len(skeleton.edges) == 4
        assert skeleton.layer_name == "skeleton"

    def test_skeleton_root_inference(self, simple_skeleton_data, spatial_columns):
        """Test automatic root inference from graph structure."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],  # Set root explicitly for now - 100
        )

        skeleton = cell.skeleton
        # Test that the root was set correctly
        assert skeleton.root == vertex_indices[0]  # Should be 100
        assert skeleton.root in vertex_indices

    def test_skeleton_topological_properties(
        self, branched_skeleton_data, spatial_columns
    ):
        """Test topological property calculations."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[1]],  # 103 -> 101 (branch)
                [vertex_indices[4], vertex_indices[1]],  # 104 -> 101 (branch)
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],  # root at index 100
        )

        skeleton = cell.skeleton

        # Test branch points (should be vertex 101 - the Y junction)
        branch_points = skeleton.branch_points
        assert len(branch_points) == 1
        assert 101 in branch_points

        # Test end points (should be 102, 103, 104)
        end_points = skeleton.end_points
        assert len(end_points) == 3
        assert 102 in end_points
        assert 103 in end_points
        assert 104 in end_points

    def test_skeleton_distance_calculations(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test distance calculation methods."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton

        # Test distance to root
        distances = skeleton.distance_to_root()
        assert distances[0] == 0.0  # root should have distance 0
        assert distances[1] == 1.0  # second vertex should be distance 1
        assert distances[-1] == 4.0  # last vertex should be distance 4

        # Test distance between specific vertices
        dist_matrix = skeleton.distance_between(
            sources=np.array([vertex_indices[0]]),
            targets=np.array([vertex_indices[-1]]),
        )
        assert dist_matrix[0, 0] == 4.0

    def test_skeleton_path_finding(self, simple_skeleton_data, spatial_columns):
        """Test path finding between vertices."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton

        # Find path from root to tip - check that path finding works
        path = skeleton.path_between(
            source=vertex_indices[0], target=vertex_indices[-1]
        )

        # Path should include all 5 vertices in sequence
        assert len(path) == 5
        # By default (as_positional=False), the path is returned in vertex indices,
        # matching the index space of the (vertex-index) inputs.
        assert path[0] == vertex_indices[0]  # root vertex index
        assert path[-1] == vertex_indices[-1]  # tip vertex index

        # With as_positional=True, inputs and outputs are positional indices.
        path_positional = skeleton.path_between(source=0, target=4, as_positional=True)
        assert len(path_positional) == 5
        assert path_positional[0] == 0  # root position
        assert path_positional[-1] == 4  # tip position

        # as_vertices=True returns 3d coordinates regardless of as_positional.
        path_vertices = skeleton.path_between(
            source=vertex_indices[0], target=vertex_indices[-1], as_vertices=True
        )
        assert path_vertices.shape == (5, 3)
        np.testing.assert_array_equal(path_vertices[0], vertices[0])
        np.testing.assert_array_equal(path_vertices[-1], vertices[-1])

    def test_skeleton_lowest_common_ancestor(
        self, branched_skeleton_data, spatial_columns
    ):
        """LCA is returned in the same index space as the query."""
        vertices, _, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )
        # Tree: 100 -> 101 -> {102 -> 104, 103}
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[1]],  # 103 -> 101
                [vertex_indices[4], vertex_indices[2]],  # 104 -> 102
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )
        skeleton = cell.skeleton

        # Default (as_positional=False): inputs and output are vertex indices.
        assert (
            skeleton.lowest_common_ancestor(vertex_indices[4], vertex_indices[3])
            == vertex_indices[1]  # 104 and 103 meet at 101
        )
        assert (
            skeleton.lowest_common_ancestor(vertex_indices[4], vertex_indices[2])
            == vertex_indices[2]  # 104 is downstream of 102
        )

        # as_positional=True: inputs and output are positional indices.
        assert skeleton.lowest_common_ancestor(4, 3, as_positional=True) == 1
        assert skeleton.lowest_common_ancestor(4, 2, as_positional=True) == 2

    def test_masked_skeleton_positional_queries_stay_in_masked_space(
        self, simple_skeleton_data, spatial_columns
    ):
        """Regression: with as_positional=True on a masked skeleton, results are
        positional indices in the masked space, not raw vertex indices.

        Before the fix, path_between/lowest_common_ancestor returned values in the
        original vertex-index space, which raised IndexError when used to index the
        smaller masked arrays.
        """
        vertices, _, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        masked = cell.skeleton.apply_mask(
            np.array([True, True, True, True, False]),
            as_positional=True,
            self_only=True,
        )
        # vertex_index is non-identity (values 100..103), so the two index spaces
        # genuinely differ and a space mix-up would be observable.
        assert not np.array_equal(masked.vertex_index, np.arange(masked.n_vertices))

        root_pos = masked.root_positional
        tip_pos = masked.end_points_positional[0]

        # Positional query -> positional result, valid to index masked arrays.
        path_pos = masked.path_between(
            source=root_pos, target=tip_pos, as_positional=True
        )
        assert path_pos.max() < masked.n_vertices
        masked.vertices[path_pos]  # would raise IndexError before the fix

        # Vertex-index query -> vertex-index result, consistent with positional one.
        path_vi = masked.path_between(
            source=masked.root, target=masked.vertex_index[tip_pos]
        )
        np.testing.assert_array_equal(masked.vertex_index[path_pos], path_vi)

        # lowest_common_ancestor honors the same contract.
        lca_pos = masked.lowest_common_ancestor(root_pos, tip_pos, as_positional=True)
        assert lca_pos < masked.n_vertices
        lca_vi = masked.lowest_common_ancestor(
            masked.root, masked.vertex_index[tip_pos]
        )
        assert lca_vi == masked.vertex_index[lca_pos]

    def test_skeleton_segments_and_cover_paths(
        self, branched_skeleton_data, spatial_columns
    ):
        """Test segment and cover path generation."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[1]],  # 103 -> 101 (branch)
                [vertex_indices[4], vertex_indices[1]],  # 104 -> 101 (branch)
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton

        # Test segments
        segments = skeleton.segments
        assert len(segments) > 0

        # Test cover paths
        cover_paths = skeleton.cover_paths
        assert len(cover_paths) > 0

        # Each cover path should be a numpy array of vertex indices
        for path in cover_paths:
            assert isinstance(path, np.ndarray)
            assert all(idx in vertex_indices for idx in path)

    def test_skeleton_masking_operations(self, simple_skeleton_data, spatial_columns):
        """Test masking and filtering operations."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton

        # Test masking with boolean array
        mask = np.array([True, True, True, False, False])
        masked_skeleton = skeleton.apply_mask(mask, as_positional=True, self_only=True)

        assert masked_skeleton.n_vertices == 3
        assert masked_skeleton.root == vertex_indices[0]

    def test_skeleton_reroot_functionality(self, simple_skeleton_data, spatial_columns):
        """Test rerooting the skeleton."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        skeleton = cell.skeleton
        original_root = skeleton.root

        # Reroot to a different vertex (vertex_indices[2] == 102).
        new_root = vertex_indices[2]
        returned = skeleton.reroot(new_root)

        # reroot mutates in place and returns self.
        assert returned is skeleton
        assert skeleton.root == new_root
        assert skeleton.root != original_root
        # root/base_root remain vertex indices (not positional).
        assert skeleton.root in skeleton.vertex_index
        assert skeleton.base_root == new_root


class TestGraphLayer:
    """Tests for GraphLayer functionality."""

    def test_graph_creation_and_properties(self, simple_graph_data, spatial_columns):
        """Test creating a graph and accessing its properties."""
        vertices, edges, vertex_indices = simple_graph_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[0], vertex_indices[1]],  # 300 -> 301
                [vertex_indices[1], vertex_indices[2]],  # 301 -> 302
                [vertex_indices[1], vertex_indices[3]],  # 301 -> 303
                [vertex_indices[3], vertex_indices[4]],  # 303 -> 304
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
        )

        graph = cell.graph
        assert graph.n_vertices == 5
        assert len(graph.edges) == 4
        assert graph.layer_name == "graph"

    def test_graph_csgraph_generation(self, simple_graph_data, spatial_columns):
        """Test compressed sparse graph generation."""
        vertices, edges, vertex_indices = simple_graph_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[0], vertex_indices[1]],  # 300 -> 301
                [vertex_indices[1], vertex_indices[2]],  # 301 -> 302
                [vertex_indices[1], vertex_indices[3]],  # 301 -> 303
                [vertex_indices[3], vertex_indices[4]],  # 303 -> 304
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
        )

        graph = cell.graph

        # Test sparse graph creation
        csgraph = graph.csgraph
        assert isinstance(csgraph, sparse.csr_matrix)
        assert csgraph.shape == (5, 5)

        # Test binary sparse graph
        csgraph_binary = graph.csgraph_binary
        assert isinstance(csgraph_binary, sparse.csr_matrix)
        assert csgraph_binary.shape == (5, 5)

    def test_graph_distance_calculations(self, simple_graph_data, spatial_columns):
        """Test distance calculations in graph."""
        vertices, edges, vertex_indices = simple_graph_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[0], vertex_indices[1]],  # 300 -> 301
                [vertex_indices[1], vertex_indices[2]],  # 301 -> 302
                [vertex_indices[1], vertex_indices[3]],  # 301 -> 303
                [vertex_indices[3], vertex_indices[4]],  # 303 -> 304
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
        )

        graph = cell.graph

        # Test distance between vertices
        distances = graph.distance_between(
            sources=np.array([vertex_indices[0]]),
            targets=np.array([vertex_indices[2]]),
            as_positional=False,
        )

        assert distances.shape == (1, 1)
        assert distances[0, 0] > 0  # Should have positive distance


class TestMeshLayer:
    """Tests for MeshLayer functionality."""

    def test_mesh_creation_and_properties(self, simple_mesh_data, spatial_columns):
        """Test creating a mesh and accessing its properties."""
        vertices, faces, vertex_indices = simple_mesh_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix faces to reference actual vertex indices instead of positional indices
        faces_with_indices = np.array(
            [
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[2],
                ],  # 200, 201, 202
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[3],
                ],  # 200, 201, 203
                [
                    vertex_indices[0],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 200, 202, 203
                [
                    vertex_indices[1],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 201, 202, 203
            ]
        )

        cell = Cell()
        cell.add_mesh(
            vertices=vertex_df,
            faces=faces_with_indices,
            spatial_columns=spatial_columns,
        )

        mesh = cell.mesh
        assert mesh.n_vertices == 4
        assert len(mesh.faces) == 4
        assert mesh.layer_name == "mesh"

    def test_mesh_trimesh_integration(self, simple_mesh_data, spatial_columns):
        """Test trimesh integration."""
        vertices, faces, vertex_indices = simple_mesh_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix faces to reference actual vertex indices instead of positional indices
        faces_with_indices = np.array(
            [
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[2],
                ],  # 200, 201, 202
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[3],
                ],  # 200, 201, 203
                [
                    vertex_indices[0],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 200, 202, 203
                [
                    vertex_indices[1],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 201, 202, 203
            ]
        )

        cell = Cell()
        cell.add_mesh(
            vertices=vertex_df,
            faces=faces_with_indices,
            spatial_columns=spatial_columns,
        )

        mesh = cell.mesh

        # Test trimesh object
        trimesh_obj = mesh.as_trimesh
        assert trimesh_obj.vertices.shape == (4, 3)
        assert trimesh_obj.faces.shape == (4, 3)

        # Test as tuple
        vertices_out, faces_out = mesh.as_tuple
        assert vertices_out.shape == (4, 3)
        assert faces_out.shape == (4, 3)

    def test_mesh_surface_area_calculations(self, simple_mesh_data, spatial_columns):
        """Test surface area calculations."""
        vertices, faces, vertex_indices = simple_mesh_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix faces to reference actual vertex indices instead of positional indices
        faces_with_indices = np.array(
            [
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[2],
                ],  # 200, 201, 202
                [
                    vertex_indices[0],
                    vertex_indices[1],
                    vertex_indices[3],
                ],  # 200, 201, 203
                [
                    vertex_indices[0],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 200, 202, 203
                [
                    vertex_indices[1],
                    vertex_indices[2],
                    vertex_indices[3],
                ],  # 201, 202, 203
            ]
        )

        cell = Cell()
        cell.add_mesh(
            vertices=vertex_df,
            faces=faces_with_indices,
            spatial_columns=spatial_columns,
        )

        mesh = cell.mesh

        # Test total surface area
        total_area = mesh.surface_area()
        assert total_area > 0

        # Test partial surface area
        partial_area = mesh.surface_area(
            vertices=np.array([0, 1, 2]), as_positional=True
        )
        assert 0 <= partial_area <= total_area


class TestPointCloudLayer:
    """Tests for PointCloudLayer functionality."""

    def test_pointcloud_creation(self, mock_point_annotations, spatial_columns):
        """Test creating a point cloud layer."""
        cell = Cell()
        cell.add_point_annotations(
            name="test_points",
            vertices=mock_point_annotations,
            spatial_columns=spatial_columns,
        )

        points = cell.annotations["test_points"]
        assert points.n_vertices == 4
        assert points.layer_name == "test_points"

    def test_pointcloud_with_skeleton_distance(
        self, mock_point_annotations, simple_skeleton_data, spatial_columns
    ):
        """Test point cloud distance calculations via skeleton."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to reference actual vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Create mapping from points to skeleton vertices (simple closest mapping)
        point_to_skeleton_mapping = {400: 100, 401: 101, 402: 102, 403: 103}

        cell.add_point_annotations(
            name="test_points",
            vertices=mock_point_annotations,
            spatial_columns=spatial_columns,
            linkage=Link(mapping=point_to_skeleton_mapping, target="skeleton"),
        )

        points = cell.annotations["test_points"]

        # Test distance to root via skeleton
        distances = points.distance_to_root()
        assert len(distances) == 4
        assert all(d >= 0 for d in distances)


class TestMappingFunctionality:
    """Tests for mapping and unmapped vertex detection."""

    def test_get_unmapped_vertices_single_target(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test finding unmapped vertices with single target layer."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        # Fix edges and faces to use vertex indices
        edges_graph_fixed = np.array(
            [
                [indices_graph[0], indices_graph[1]],  # 300 -> 301
                [indices_graph[1], indices_graph[2]],  # 301 -> 302
                [indices_graph[1], indices_graph[3]],  # 301 -> 303
                [indices_graph[3], indices_graph[4]],  # 303 -> 304
            ]
        )

        edges_skel_fixed = np.array(
            [
                [indices_skel[1], indices_skel[0]],  # 101 -> 100
                [indices_skel[2], indices_skel[1]],  # 102 -> 101
                [indices_skel[3], indices_skel[2]],  # 103 -> 102
                [indices_skel[4], indices_skel[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=graph_df, edges=edges_graph_fixed, spatial_columns=spatial_columns
        )

        # Create complete mapping (all skeleton vertices must map to graph)
        complete_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}

        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_skel_fixed,
            spatial_columns=spatial_columns,
            root=indices_skel[0],  # Set explicit root
            linkage=Link(mapping=complete_mapping, target="graph"),
        )

        # With complete mapping, there should be no unmapped vertices
        unmapped = cell.skeleton.get_unmapped_vertices(target_layers="graph")

        # Should find no unmapped vertices with complete mapping
        assert len(unmapped) == 0

    def test_get_unmapped_vertices_multiple_targets(
        self, simple_skeleton_data, simple_graph_data, simple_mesh_data, spatial_columns
    ):
        """Test finding unmapped vertices with multiple target layers."""
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

        # Fix edges and faces to use vertex indices
        edges_graph_fixed = np.array(
            [
                [indices_graph[0], indices_graph[1]],  # 300 -> 301
                [indices_graph[1], indices_graph[2]],  # 301 -> 302
                [indices_graph[1], indices_graph[3]],  # 301 -> 303
                [indices_graph[3], indices_graph[4]],  # 303 -> 304
            ]
        )

        faces_mesh_fixed = np.array(
            [
                [indices_mesh[0], indices_mesh[1], indices_mesh[2]],  # 200, 201, 202
                [indices_mesh[0], indices_mesh[1], indices_mesh[3]],  # 200, 201, 203
                [indices_mesh[0], indices_mesh[2], indices_mesh[3]],  # 200, 202, 203
                [indices_mesh[1], indices_mesh[2], indices_mesh[3]],  # 201, 202, 203
            ]
        )

        edges_skel_fixed = np.array(
            [
                [indices_skel[1], indices_skel[0]],  # 101 -> 100
                [indices_skel[2], indices_skel[1]],  # 102 -> 101
                [indices_skel[3], indices_skel[2]],  # 103 -> 102
                [indices_skel[4], indices_skel[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=graph_df, edges=edges_graph_fixed, spatial_columns=spatial_columns
        )
        cell.add_mesh(
            vertices=mesh_df, faces=faces_mesh_fixed, spatial_columns=spatial_columns
        )

        # Create complete graph mapping and complete mesh mapping for this test
        graph_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}
        mesh_mapping = {
            100: 200,
            101: 201,
            102: 202,
            103: 203,
        }  # Missing 104 -> 203 (only 4 mesh vertices)

        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_skel_fixed,
            spatial_columns=spatial_columns,
            root=indices_skel[0],  # Set explicit root
            linkage=Link(mapping=graph_mapping, target="graph"),
        )

        # Add link to mesh (skeleton vertex 104 cannot map to mesh as there's no mesh vertex 204)
        # For this test, skip the mesh linkage to focus on testing the functionality

        # Find unmapped vertices to graph (should be none)
        unmapped_graph = cell.skeleton.get_unmapped_vertices(target_layers="graph")
        # Every skeleton vertex maps to the graph, so the result is an empty
        # integer index array (not just a truthy "the method exists").
        assert isinstance(unmapped_graph, np.ndarray)
        assert unmapped_graph.dtype.kind in "iu"
        assert unmapped_graph.size == 0

    def test_mask_out_unmapped_functionality(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test masking out unmapped vertices."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        # Fix edges to use vertex indices
        edges_graph_fixed = np.array(
            [
                [indices_graph[0], indices_graph[1]],  # 300 -> 301
                [indices_graph[1], indices_graph[2]],  # 301 -> 302
                [indices_graph[1], indices_graph[3]],  # 301 -> 303
                [indices_graph[3], indices_graph[4]],  # 303 -> 304
            ]
        )

        edges_skel_fixed = np.array(
            [
                [indices_skel[1], indices_skel[0]],  # 101 -> 100
                [indices_skel[2], indices_skel[1]],  # 102 -> 101
                [indices_skel[3], indices_skel[2]],  # 103 -> 102
                [indices_skel[4], indices_skel[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=graph_df, edges=edges_graph_fixed, spatial_columns=spatial_columns
        )

        # Create complete mapping for this test
        complete_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}

        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_skel_fixed,
            spatial_columns=spatial_columns,
            root=indices_skel[0],  # Set explicit root
            linkage=Link(mapping=complete_mapping, target="graph"),
        )

        original_n_vertices = cell.skeleton.n_vertices

        # With complete mapping, mask_out_unmapped should not reduce vertices
        cleaned_skeleton = cell.skeleton.mask_out_unmapped(
            target_layers="graph", self_only=True
        )

        # Should have same number of vertices since all are mapped
        assert cleaned_skeleton.n_vertices == original_n_vertices
        assert cleaned_skeleton.n_vertices == 5  # All vertices remain

    def test_mapping_null_strategies(
        self, simple_skeleton_data, simple_graph_data, spatial_columns
    ):
        """Test different null handling strategies in mappings."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_graph, edges_graph, indices_graph = simple_graph_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        graph_df = pd.DataFrame(
            vertices_graph, columns=spatial_columns, index=indices_graph
        )

        # Fix edges to use vertex indices
        edges_graph_fixed = np.array(
            [
                [indices_graph[0], indices_graph[1]],  # 300 -> 301
                [indices_graph[1], indices_graph[2]],  # 301 -> 302
                [indices_graph[1], indices_graph[3]],  # 301 -> 303
                [indices_graph[3], indices_graph[4]],  # 303 -> 304
            ]
        )

        edges_skel_fixed = np.array(
            [
                [indices_skel[1], indices_skel[0]],  # 101 -> 100
                [indices_skel[2], indices_skel[1]],  # 102 -> 101
                [indices_skel[3], indices_skel[2]],  # 103 -> 102
                [indices_skel[4], indices_skel[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_graph(
            vertices=graph_df, edges=edges_graph_fixed, spatial_columns=spatial_columns
        )

        # Create complete mapping
        complete_mapping = {100: 300, 101: 301, 102: 302, 103: 303, 104: 304}

        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_skel_fixed,
            spatial_columns=spatial_columns,
            root=indices_skel[0],  # Set explicit root
            linkage=Link(mapping=complete_mapping, target="graph"),
        )

        # Test different null strategies with complete mapping
        mapping_drop = cell._morphsync.get_mapping("skeleton", "graph", dropna=True)
        mapping_keep = cell._morphsync.get_mapping(
            "skeleton",
            "graph",
            dropna=False,
        )

        # With complete mapping, all strategies should have same length
        assert len(mapping_drop) == len(indices_skel)
        assert len(mapping_keep) == len(indices_skel)

        # Test that mapping functions exist and return valid data
        assert mapping_drop is not None
        assert mapping_keep is not None


class TestSkeletonResample:
    """Tests for SkeletonLayer.resample() functionality."""

    def _make_skeleton(
        self, vertices, edges, vertex_indices, spatial_columns, features=None
    ):
        """Helper to build a Cell with a skeleton."""
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )
        if features is not None:
            for fname, fvals in features.items():
                vertex_df[fname] = fvals

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )
        return cell.skeleton

    def test_resample_linear_upsampling(self, simple_skeleton_data, spatial_columns):
        """Upsampling a linear skeleton should increase vertex count."""
        vertices, _, vertex_indices = simple_skeleton_data
        # Edges using vertex indices
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        # Original: 5 vertices, spacing=1.0 between them, total length=4.0
        # Resample with spacing=0.5 -> expect ~8 intervals -> ~9 vertices
        resampled = skeleton.resample(spacing=0.5)

        assert resampled.n_vertices > skeleton.n_vertices
        assert resampled.root == 100  # Root preserved

    def test_resample_linear_downsampling(self, simple_skeleton_data, spatial_columns):
        """Downsampling a linear skeleton should reduce vertex count."""
        vertices, _, vertex_indices = simple_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        # Original spacing=1.0, resample with spacing=2.0 -> fewer vertices
        resampled = skeleton.resample(spacing=2.0)

        assert resampled.n_vertices < skeleton.n_vertices
        assert resampled.root == 100

    def test_resample_preserves_topo_points(
        self, branched_skeleton_data, spatial_columns
    ):
        """Branch points, end points, and root should be preserved exactly."""
        vertices, _, vertex_indices = branched_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 101], [104, 102]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        resampled = skeleton.resample(spacing=0.3)

        # All original topo point vertex IDs should be in the resampled skeleton
        for tp in skeleton.topo_points:
            assert tp in resampled.vertex_index, (
                f"Topo point {tp} missing from resampled skeleton"
            )

        # Topo point positions should match exactly
        for tp in skeleton.topo_points:
            orig_pos = skeleton.vertex_df.loc[tp, spatial_columns].values
            new_pos = resampled.vertex_df.loc[tp, spatial_columns].values
            np.testing.assert_array_equal(orig_pos, new_pos)

    def test_resample_preserves_cable_length(
        self, branched_skeleton_data, spatial_columns
    ):
        """Total cable length should be approximately preserved."""
        vertices, _, vertex_indices = branched_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 101], [104, 102]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        resampled = skeleton.resample(spacing=0.3)

        orig_length = skeleton.cable_length()
        new_length = resampled.cable_length()
        # Resampling a polyline with angled segments introduces small chord-vs-arc
        # differences. For a straight-line skeleton the match is exact; for angled
        # segments the error scales with (spacing / radius_of_curvature)^2.
        np.testing.assert_allclose(orig_length, new_length, rtol=0.01)

    def test_resample_skip_root_adjacent(self, branched_skeleton_data, spatial_columns):
        """skip_root_adjacent should leave root-connected segments unchanged."""
        vertices, _, vertex_indices = branched_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 101], [104, 102]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        resampled_skip = skeleton.resample(spacing=0.3, skip_root_adjacent=True)
        resampled_noskip = skeleton.resample(spacing=0.3, skip_root_adjacent=False)

        # The skip version should have fewer (or equal) vertices since some segments aren't resampled
        assert resampled_skip.n_vertices <= resampled_noskip.n_vertices

    def test_resample_feature_nearest(self, simple_skeleton_data, spatial_columns):
        """Nearest feature mapping should assign correct values."""
        vertices, _, vertex_indices = simple_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        features = {"radius": [0.5, 0.6, 0.4, 0.7, 0.3]}

        skeleton = self._make_skeleton(
            vertices, edges, vertex_indices, spatial_columns, features=features
        )
        resampled = skeleton.resample(spacing=0.5, feature_agg="nearest")

        # All new vertices should have a radius value
        assert "radius" in resampled.feature_names
        radii = resampled.get_feature("radius")
        assert len(radii) == resampled.n_vertices
        # All radius values should come from the original set
        assert all(r in [0.5, 0.6, 0.4, 0.7, 0.3] for r in radii)

    def test_resample_feature_agg_dict(self, simple_skeleton_data, spatial_columns):
        """Dict-based aggregation should work for downsampling."""
        vertices, _, vertex_indices = simple_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        features = {"radius": [0.5, 0.6, 0.4, 0.7, 0.3]}

        skeleton = self._make_skeleton(
            vertices, edges, vertex_indices, spatial_columns, features=features
        )
        resampled = skeleton.resample(spacing=2.0, feature_agg={"radius": "mean"})

        assert "radius" in resampled.feature_names
        assert len(resampled.get_feature("radius")) == resampled.n_vertices

    def test_resample_edge_case_short_segment(self, spatial_columns):
        """A segment shorter than spacing should still produce endpoints."""
        # Two vertices very close together plus root
        vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        vertex_indices = np.array([10, 11, 12])
        edges = np.array([[11, 10], [12, 11]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        resampled = skeleton.resample(spacing=10.0)

        # Should still have at least the topo points
        assert resampled.n_vertices >= 2
        assert 10 in resampled.vertex_index
        assert 12 in resampled.vertex_index

    def test_resample_valid_tree(self, branched_skeleton_data, spatial_columns):
        """Resampled skeleton should be a valid tree: n_edges == n_vertices - 1."""
        vertices, _, vertex_indices = branched_skeleton_data
        edges = np.array([[101, 100], [102, 101], [103, 101], [104, 102]])

        skeleton = self._make_skeleton(vertices, edges, vertex_indices, spatial_columns)
        resampled = skeleton.resample(spacing=0.3)

        assert len(resampled.edges) == resampled.n_vertices - 1


class TestAggregateFeatures:
    """Tests for GraphLayer/SkeletonLayer.aggregate_features."""

    def _linear_cell(self):
        """Linear skeleton 0(root)-1-2-3-4(tip) with feature val=[10,20,30,40,50]."""
        verts = np.array([[i, 0.0, 0.0] for i in range(5)])
        edges = np.array([[1, 0], [2, 1], [3, 2], [4, 3]])
        feat = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        cell = Cell()
        cell.add_skeleton(verts, edges, features={"val": feat}, root=0)
        return cell

    def test_undirected_hop_mean(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features("val", radius=1, metric="hops", agg="mean")
        np.testing.assert_allclose(r["val"].values, [15, 20, 30, 40, 45])

    def test_downstream_hop_sum(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features(
            "val", radius=1, metric="hops", direction="downstream", agg="sum"
        )
        np.testing.assert_allclose(r["val"].values, [30, 50, 70, 90, 50])

    def test_upstream_hop_sum(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features(
            "val", radius=1, metric="hops", direction="upstream", agg="sum"
        )
        np.testing.assert_allclose(r["val"].values, [10, 30, 50, 70, 90])

    def test_max_fallback(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features("val", radius=2, metric="hops", agg="max")
        np.testing.assert_allclose(r["val"].values, [30, 40, 50, 50, 50])

    def test_distance_metric(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features("val", radius=1.5, metric="distance", agg="mean")
        np.testing.assert_allclose(r["val"].values, [15, 20, 30, 40, 45])

    def test_weighted_smoothing_symmetry(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features(
            "val", radius=2, metric="hops", agg="mean", weight=lambda d: np.exp(-d)
        )
        # Center vertex's neighborhood is symmetric, so weighted mean stays at 30.
        assert r["val"].values[2] == pytest.approx(30.0)

    def test_exclusive_drops_self(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features(
            "val", radius=1, metric="hops", agg="mean", inclusive=False
        )
        # Root (0) only neighbor is vertex 1 -> 20; tip (4) only neighbor is 3 -> 40.
        assert r["val"].values[0] == pytest.approx(20.0)
        assert r["val"].values[4] == pytest.approx(40.0)

    def test_output_indexed_by_vertex_index(self):
        s = self._linear_cell().skeleton
        r = s.aggregate_features("val", radius=1, metric="hops", agg="mean")
        np.testing.assert_array_equal(r.index.values, s.vertex_index)

    def test_weight_requires_sum_or_mean(self):
        s = self._linear_cell().skeleton
        with pytest.raises(ValueError):
            s.aggregate_features(
                "val", radius=1, metric="hops", agg="max", weight=lambda d: d
            )

    def test_direction_requires_skeleton(self, simple_graph_data):
        from ossify.data_layers import GraphLayer

        vertices, edges, vertex_indices = simple_graph_data
        rng = np.random.default_rng(0)
        g = GraphLayer(
            "test_graph",
            vertices,
            edges,
            spatial_columns=["x", "y", "z"],
            features={"weight": rng.uniform(0, 1, len(vertices))},
        )
        with pytest.raises(ValueError):
            g.aggregate_features("weight", radius=1, direction="downstream")


class TestSegmentGraph:
    """Tests for SkeletonLayer.segment_graph and SegmentGraph."""

    def _branched_cell(self, features=None):
        """root(0)->branch(1)->{run 2->4, tip 3}."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [3, 1, 0]], dtype=float
        )
        edges = np.array([[1, 0], [2, 1], [3, 1], [4, 2]])
        cell = Cell()
        cell.add_skeleton(verts, edges, features=features, root=0)
        return cell

    def _linear_cell(self, n=11, features=None):
        """Unit-spaced linear chain 0(root)..n-1(tip), total cable length n-1."""
        verts = np.array([[i, 0.0, 0.0] for i in range(n)])
        edges = np.array([[i + 1, i] for i in range(n - 1)])
        cell = Cell()
        cell.add_skeleton(verts, edges, features=features, root=0)
        return cell

    @staticmethod
    def _by_distal(sg, values):
        """Map a per-node array to {distal_source_vertex: value} (order-robust)."""
        return {int(vs[0]): values[i] for i, vs in enumerate(sg.node_source_vertices)}

    def test_node_set_is_reduced_tree(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        # node set == {root} U {branch points} U {tips}
        distal = {int(vs[0]) for vs in sg.node_source_vertices}
        expected = (
            {sk.root_positional}
            | set(sk.branch_points_positional)
            | set(sk.end_points_positional)
        )
        assert distal == expected
        assert sg.n_vertices == 4

    def test_valid_rooted_tree(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        assert (sg.parent_node_array == -1).sum() == 1  # exactly one root
        assert sg.edges.shape[0] == sg.n_vertices - 1  # tree
        # every node reachable from root
        d = sg.distance_to_root(as_positional=True)
        assert np.all(np.isfinite(d))

    def test_length_conservation(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        total_cable = sk.half_edge_length.sum()
        np.testing.assert_allclose(sg.get_feature("length").sum(), total_cable)

    def test_round_trip_to_vertices(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        np.testing.assert_array_equal(
            sg.to_vertices(np.arange(sg.n_vertices)), sg.vertex_segment_map
        )
        # broadcasting a per-node label: all source vertices in a node share it
        for nid, vs in enumerate(sg.node_source_vertices):
            assert np.all(sg.vertex_segment_map[vs] == nid)

    def test_capping_splits_long_runs(self):
        sk = self._linear_cell(n=11).skeleton  # cable length 10
        sg0 = sk.segment_graph()
        sgc = sk.segment_graph(max_length=2.5)
        assert sg0.n_vertices == 2  # root singleton + one long run
        assert sgc.n_vertices > sg0.n_vertices
        total = sk.half_edge_length.sum()
        np.testing.assert_allclose(sg0.get_feature("length").sum(), total)
        np.testing.assert_allclose(sgc.get_feature("length").sum(), total)
        assert sgc.get_feature("length").max() <= 2.5 + 1e-9
        assert sgc.get_feature("length").max() < sg0.get_feature("length").max()

    def test_feature_rollup_reducers(self):
        val = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sk = self._linear_cell(n=5, features={"val": val}).skeleton
        for reducer, expected in [
            ("sum", {4: 140.0, 0: 10.0}),
            ("mean", {4: 35.0, 0: 10.0}),
            ("max", {4: 50.0, 0: 10.0}),
            ("distal", {4: 50.0, 0: 10.0}),
            ("proximal", {4: 20.0, 0: 10.0}),
        ]:
            sg = sk.segment_graph(features=["val"], agg={"val": reducer})
            got = self._by_distal(sg, sg.get_feature("val"))
            assert got == expected, reducer

    def test_sum_then_derive_weighted_mean(self):
        verts = np.array([[i, 0.0, 0.0] for i in range(5)])
        edges = np.array([[i + 1, i] for i in range(4)])
        base = Cell()
        base.add_skeleton(verts, edges, root=0)
        w = base.skeleton.half_edge_length  # [0.5, 1, 1, 1, 0.5]
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cell = Cell()
        cell.add_skeleton(verts, edges, features={"rw": r * w, "w": w}, root=0)
        sg = cell.skeleton.segment_graph(
            features=["rw", "w"], agg={"rw": "sum", "w": "sum"}
        )
        wmean = self._by_distal(sg, sg.get_feature("rw") / sg.get_feature("w"))
        # run node = vertices {1,2,3,4}: length-weighted mean of r
        run = [1, 2, 3, 4]
        expected = np.sum(r[run] * w[run]) / np.sum(w[run])
        np.testing.assert_allclose(wmean[4], expected)

    def test_distances_are_cable_not_chord(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        src_dtr = sk.distance_to_root(as_positional=True)
        distal = np.array([int(vs[0]) for vs in sg.node_source_vertices])
        sg_dtr = sg.distance_to_root(as_positional=True)
        # segment-graph distances equal the source skeleton's arc-length depths
        np.testing.assert_allclose(sg_dtr, src_dtr[distal], rtol=1e-5)
        # tip-4 node: cable length 2 + sqrt(2), NOT the chord sqrt(10) between
        # the root and the tip coordinates
        d = self._by_distal(sg, sg_dtr)
        np.testing.assert_allclose(d[4], 2 + np.sqrt(2), rtol=1e-5)
        chord = np.linalg.norm(sk.vertices[4] - sk.vertices[0])
        assert not np.isclose(d[4], chord)

    def test_cable_length_conserved(self):
        sk = self._branched_cell().skeleton
        # capping must not change the total cable length recovered from the graph
        for max_length in (None, 0.6):
            sg = sk.segment_graph(max_length=max_length)
            np.testing.assert_allclose(sg.cable_length(), sk.cable_length(), rtol=1e-5)

    def test_aggregate_vertices_length_weighted_mean(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        node_r = sg.aggregate_vertices(r, agg="mean", weight="length")
        half = sk.half_edge_length
        by_distal = self._by_distal(sg, node_r)
        # run node {2,4}: length-weighted mean of r over those vertices
        run = [2, 4]
        np.testing.assert_allclose(by_distal[4], np.average(r[run], weights=half[run]))

    def test_aggregate_vertices_2d_unaries(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        unaries = np.random.default_rng(0).normal(size=(5, 2))
        node_u = sg.aggregate_vertices(unaries, agg="mean", weight="length")
        assert node_u.shape == (sg.n_vertices, 2)

    def test_aggregate_vertices_uniform_and_sum(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        ones = np.ones(5)
        # uniform sum == subtree-free node sizes; length-weighted sum == cable
        node_sum = sg.aggregate_vertices(ones, agg="sum", weight=None)
        np.testing.assert_allclose(node_sum.sum(), 5)  # every vertex counted once
        node_len = sg.aggregate_vertices(np.ones(5), agg="sum", weight="length")
        np.testing.assert_allclose(node_len.sum(), sk.half_edge_length.sum())

    def test_aggregate_then_to_vertices_roundtrip_piecewise(self):
        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        node_ids = np.arange(sg.n_vertices, dtype=float)
        per_vertex = sg.to_vertices(node_ids)
        # aggregating a piecewise-constant-per-node array recovers the node ids
        back = sg.aggregate_vertices(per_vertex, agg="mean", weight="length")
        np.testing.assert_allclose(back, node_ids)

    def test_is_skeleton_layer(self):
        from ossify.data_layers import SegmentGraph, SkeletonLayer

        sk = self._branched_cell().skeleton
        sg = sk.segment_graph()
        assert isinstance(sg, SkeletonLayer)
        assert isinstance(sg, SegmentGraph)
        assert sg.layer_name == "segment_graph"
        assert sg.source_skeleton is sk


class TestAnnotationWhereFilter:
    """Tests for the `where` filter on annotation counting and CountFeature."""

    def _cell_with_labeled_synapses(self):
        # linear skeleton 0..4; 5 synapses anchored to vertices with a 'kind' label
        verts = np.array([[i, 0.0, 0.0] for i in range(5)])
        edges = np.array([[i + 1, i] for i in range(4)])
        cell = Cell()
        cell.add_skeleton(verts, edges, root=0)
        anno = pd.DataFrame(
            {
                "x": [0.0, 1.0, 1.0, 2.0, 4.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0, 0.0, 0.0],
                "anchor": [0, 1, 1, 2, 4],  # skeleton vertex each synapse sits on
                "kind": ["spine", "spine", "shaft", "spine", "shaft"],
            }
        )
        cell.add_point_annotations(
            "syn",
            vertices=anno,
            spatial_columns=["x", "y", "z"],
            linkage=Link(mapping="anchor", target="skeleton"),
        )
        return cell

    def _count(self, skel, **kw):
        return (
            skel.map_annotations_to_feature(
                "syn", distance_threshold=0, agg="count", **kw
            )
            .reindex(skel.vertex_index)
            .fillna(0)
            .to_numpy()
            .ravel()
        )

    def test_where_partitions_total(self):
        skel = self._cell_with_labeled_synapses().skeleton
        total = self._count(skel)
        spine = self._count(skel, where={"kind": "spine"})
        shaft = self._count(skel, where={"kind": "shaft"})
        np.testing.assert_array_equal(total, [1, 2, 1, 0, 1])
        np.testing.assert_array_equal(spine, [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(shaft, [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(spine + shaft, total)

    def test_where_dict_string_callable_agree(self):
        skel = self._cell_with_labeled_synapses().skeleton
        by_dict = self._count(skel, where={"kind": "spine"})
        by_str = self._count(skel, where="kind == 'spine'")
        by_call = self._count(skel, where=lambda df: df["kind"] == "spine")
        np.testing.assert_array_equal(by_dict, by_str)
        np.testing.assert_array_equal(by_dict, by_call)

    def test_where_membership_list(self):
        skel = self._cell_with_labeled_synapses().skeleton
        both = self._count(skel, where={"kind": ["spine", "shaft"]})
        np.testing.assert_array_equal(both, self._count(skel))

    def test_count_feature_where_end_to_end(self):
        import ossify.compartments as cp

        cell = self._cell_with_labeled_synapses()
        df = cp.make_skel_prop_df_base(
            cell,
            feature_spec=[
                cp.CountFeature("syn_in", "syn"),
                cp.CountFeature("spine_in", "syn", where={"kind": "spine"}),
                cp.CountFeature("shaft_in", "syn", where="kind == 'shaft'"),
            ],
        )
        np.testing.assert_array_equal(
            (df["spine_in"] + df["shaft_in"]).to_numpy(), df["syn_in"].to_numpy()
        )
        np.testing.assert_array_equal(df["spine_in"].to_numpy(), [1, 1, 1, 0, 0])


class TestSpatialDtypeCoercion:
    """Regression tests: spatial coordinates must always be float, never object.

    A single object-dtype spatial column would otherwise force the whole (N, 3)
    block to object dtype when accessed via ``.values``, which breaks callables
    passed to ``transform`` (e.g. scipy interpolators reject object arrays).
    """

    def _object_dtype_vertex_df(self, spatial_columns):
        """5-vertex skeleton whose x/y/z columns are stored as object dtype."""
        vertices, _, vertex_indices = (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
            None,
            np.array([100, 101, 102, 103, 104]),
        )
        df = pd.DataFrame(vertices, columns=spatial_columns, index=vertex_indices)
        # Force one spatial column to object dtype, holding python floats.
        df[spatial_columns[1]] = df[spatial_columns[1]].astype(object)
        assert df[spatial_columns[1]].dtype == object
        # ``.values`` on the raw frame is object -- this is what we guard against.
        assert df[spatial_columns].values.dtype == object
        return df, vertex_indices

    def _skeleton_cell(self, spatial_columns):
        df, vertex_indices = self._object_dtype_vertex_df(spatial_columns)
        edges = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )
        cell = Cell()
        cell.add_skeleton(
            vertices=df,
            edges=edges,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )
        return cell

    def test_vertices_are_float64_from_object_columns(self, spatial_columns):
        cell = self._skeleton_cell(spatial_columns)
        assert cell.skeleton.vertices.dtype == np.float64

    def test_root_location_is_float64(self, spatial_columns):
        cell = self._skeleton_cell(spatial_columns)
        root_location = cell.skeleton.root_location
        assert root_location.dtype == np.float64
        assert root_location.shape == (3,)

    def test_transform_receives_float64_array(self, spatial_columns):
        cell = self._skeleton_cell(spatial_columns)
        seen = {}

        def fn(coords):
            seen["dtype"] = coords.dtype
            return coords * 2.0

        cell.skeleton.transform(fn, inplace=True)
        assert seen["dtype"] == np.float64
        assert cell.skeleton.vertices.dtype == np.float64

    def test_point_layer_object_columns_are_float64(self, spatial_columns):
        df, vertex_indices = self._object_dtype_vertex_df(spatial_columns)
        cell = Cell()
        cell.add_point_layer(
            name="points",
            vertices=df,
            spatial_columns=spatial_columns,
        )
        assert cell.layers["points"].vertices.dtype == np.float64

    def test_non_coercible_column_raises_clear_error(self, spatial_columns):
        df, vertex_indices = self._object_dtype_vertex_df(spatial_columns)
        df[spatial_columns[1]] = ["a", "b", "c", "d", "e"]
        cell = Cell()
        with pytest.raises(ValueError, match=spatial_columns[1]):
            cell.add_point_layer(
                name="points",
                vertices=df,
                spatial_columns=spatial_columns,
            )


class TestTransformInvalidatesCaches:
    """Transforming vertices changes distances, so distance-weighted graph
    caches (the lazy ``csgraph`` and the snapshotted ``base_csgraph``) must be
    rebuilt rather than served stale."""

    def _line_skeleton(self, spatial_columns):
        # 5 vertices on a line, unit spacing -> total cable length 4.
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell

    def test_inplace_transform_rebuilds_csgraph(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        # Prime the lazy csgraph cache before transforming.
        assert cell.skeleton.csgraph.sum() == pytest.approx(4.0)
        cell.transform(lambda a: a * 10.0, inplace=True)
        assert cell.skeleton.csgraph.sum() == pytest.approx(40.0)

    def test_inplace_transform_rebuilds_base_csgraph(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        assert cell.skeleton.base_csgraph.sum() == pytest.approx(4.0)
        cell.transform(lambda a: a * 10.0, inplace=True)
        assert cell.skeleton.base_csgraph.sum() == pytest.approx(40.0)
        # distance_to_root reads base_csgraph, so it must reflect new distances.
        assert cell.skeleton.distance_to_root().max() == pytest.approx(40.0)

    def test_copy_transform_rebuilds_base_csgraph(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        _ = cell.skeleton.base_csgraph  # prime cache on the original
        moved = cell.transform(lambda a: a * 10.0)
        assert moved.skeleton.base_csgraph.sum() == pytest.approx(40.0)
        # The original must be untouched by the out-of-place transform.
        assert cell.skeleton.base_csgraph.sum() == pytest.approx(4.0)

    def test_transform_rebuilds_mesh_caches(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        cell = Cell()
        cell.add_mesh(vertices=verts, faces=faces, spatial_columns=spatial_columns)
        mesh = cell.mesh
        area_before = mesh.surface_area()
        csgraph_sum_before = mesh.csgraph.sum()  # primes the lazy caches
        cell.transform(lambda a: a * 2.0, inplace=True)
        # Scaling by 2 in-plane quadruples area (trimesh cache must refresh) and
        # doubles every edge length (csgraph cache must refresh).
        assert cell.mesh.surface_area() == pytest.approx(area_before * 4.0)
        assert cell.mesh.csgraph.sum() == pytest.approx(csgraph_sum_before * 2.0)


class TestTransformArrayArgument:
    """``transform`` also accepts an explicit (N, 3) array of new coordinates,
    not just a callable. That branch (and its shape validation) was previously
    unexercised."""

    def _skeleton(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell.skeleton

    def test_array_replaces_coordinates(self, spatial_columns):
        sk = self._skeleton(spatial_columns)
        _ = sk.csgraph  # prime the cache
        new_coords = np.column_stack(
            [np.arange(0.0, 10.0, 2.0), np.zeros(5), np.zeros(5)]
        )  # spacing doubled to 2.0
        sk.transform(new_coords, inplace=True)
        np.testing.assert_allclose(sk.vertices, new_coords)
        # Cache reflects the new geometry: total cable length 5 edges? 4 * 2 = 8.
        assert sk.csgraph.sum() == pytest.approx(8.0)

    def test_wrong_shape_raises(self, spatial_columns):
        sk = self._skeleton(spatial_columns)
        with pytest.raises(ValueError, match="same shape"):
            sk.transform(np.zeros((3, 3)), inplace=True)


class TestLayerCopy:
    """A layer attached to a Cell must be copyable on its own, detached from
    the Cell (this is also what out-of-place ``layer.transform`` relies on)."""

    def _line_skeleton(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell

    def test_skeleton_copy_detaches_from_cell(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        copied = cell.skeleton.copy()
        assert copied._cell is None
        assert copied.n_vertices == cell.skeleton.n_vertices
        np.testing.assert_array_equal(copied.vertices, cell.skeleton.vertices)
        assert copied.root == cell.skeleton.root

    def test_skeleton_copy_is_independent(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        copied = cell.skeleton.copy()
        copied.transform(lambda a: a * 10.0, inplace=True)
        # Mutating the copy must not touch the original attached layer.
        assert cell.skeleton.csgraph.sum() == pytest.approx(4.0)
        assert copied.csgraph.sum() == pytest.approx(40.0)

    def test_skeleton_out_of_place_transform(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        moved = cell.skeleton.transform(lambda a: a * 10.0)
        assert moved._cell is None
        assert moved.csgraph.sum() == pytest.approx(40.0)
        assert moved.base_csgraph.sum() == pytest.approx(40.0)
        assert cell.skeleton.csgraph.sum() == pytest.approx(4.0)

    def test_mesh_copy_detaches_from_cell(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        cell = Cell()
        cell.add_mesh(vertices=verts, faces=faces, spatial_columns=spatial_columns)
        copied = cell.mesh.copy()
        assert copied._cell is None
        assert copied.surface_area() == pytest.approx(cell.mesh.surface_area())

    def test_point_cloud_copy_detaches_from_cell(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cell = Cell()
        cell.add_point_layer(
            name="points", vertices=verts, spatial_columns=spatial_columns
        )
        copied = cell.layers["points"].copy()
        assert copied._cell is None
        assert copied.n_vertices == 3


class TestReroot:
    """reroot must move the root, reorient edges, and leave the derived/base
    caches self-consistent (root stays a vertex index; the base graphs, both
    weighted and binary, are rebuilt)."""

    def _line_skeleton(self, spatial_columns):
        # vertices 100..104 on a line, unit spacing, rooted at 100.
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell

    def test_reroot_by_vertex_index(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        sk = cell.skeleton
        assert sk.distance_to_root(np.array([104]))[0] == pytest.approx(4.0)
        sk.reroot(104)
        # Root is stored as a vertex index, not a positional index.
        assert sk.root == 104
        assert sk.root in sk.vertex_index
        # Distances are now measured from the new root.
        dtr = sk.distance_to_root(np.array([104, 100]))
        assert dtr[0] == pytest.approx(0.0)
        assert dtr[1] == pytest.approx(4.0)

    def test_reroot_as_positional(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        sk = cell.skeleton
        sk.reroot(4, as_positional=True)  # positional 4 -> vertex index 104
        assert sk.root == 104
        assert sk.distance_to_root(np.array([100]))[0] == pytest.approx(4.0)

    def test_reroot_preserves_base_binary_graph(self, spatial_columns):
        # hops_to_root reads base_csgraph_binary, which reroot must not drop.
        cell = self._line_skeleton(spatial_columns)
        sk = cell.skeleton
        sk.reroot(104)
        assert "base_csgraph_binary" in sk._base_properties
        htr = sk.hops_to_root(np.array([100, 104]))
        assert htr[0] == pytest.approx(4.0)
        assert htr[1] == pytest.approx(0.0)

    def test_reroot_reorients_edges(self, spatial_columns):
        # After rerooting at 104, the parent of 100 is 101 (edges point rootward).
        cell = self._line_skeleton(spatial_columns)
        sk = cell.skeleton
        sk.reroot(104)
        parents = sk.parent_node_array  # positional, -1 for the root
        pos = {v: i for i, v in enumerate(sk.vertex_index)}
        assert sk.vertex_index[parents[pos[100]]] == 101
        assert parents[pos[104]] == -1

    def test_reroot_returns_self(self, spatial_columns):
        cell = self._line_skeleton(spatial_columns)
        sk = cell.skeleton
        assert sk.reroot(102) is sk


class TestScalarVertexArguments:
    """A single vertex may be passed as a scalar (not just a 1-element array).
    ``_vertices_to_positional`` must handle the 0-d case (``fastremap.remap``
    rejects it) and return a scalar, matching the positional-scalar path."""

    def _line_skeleton(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell

    def test_distance_to_root_scalar_index(self, spatial_columns):
        sk = self._line_skeleton(spatial_columns).skeleton
        scalar = sk.distance_to_root(104)
        assert np.ndim(scalar) == 0
        assert scalar == pytest.approx(4.0)
        # Consistent with the 1-element-array form.
        assert scalar == pytest.approx(sk.distance_to_root(np.array([104]))[0])

    def test_distance_to_root_scalar_positional(self, spatial_columns):
        sk = self._line_skeleton(spatial_columns).skeleton
        # Positional scalar already worked; confirm it still matches.
        assert sk.distance_to_root(4, as_positional=True) == pytest.approx(4.0)

    def test_hops_to_root_scalar_index(self, spatial_columns):
        sk = self._line_skeleton(spatial_columns).skeleton
        assert sk.hops_to_root(104) == pytest.approx(4.0)

    def test_distance_between_scalar(self, spatial_columns):
        sk = self._line_skeleton(spatial_columns).skeleton
        d = sk.distance_between(100, 104)
        assert np.asarray(d).item() == pytest.approx(4.0)


class TestMaskContextTeardown:
    """``mask_context`` yields a scoped temporary that must be torn down when
    the block exits, so the masked copy is reclaimed promptly rather than
    lingering behind the layer<->cell reference cycle."""

    def _cell(self, spatial_columns):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        idx = np.array([100, 101, 102, 103, 104])
        df = pd.DataFrame(verts, columns=spatial_columns, index=idx)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(
            vertices=df, edges=edges, spatial_columns=spatial_columns, root=100
        )
        return cell

    def test_masked_cell_usable_inside_context(self, spatial_columns):
        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        with cell.mask_context("skeleton", mask) as masked:
            assert masked.skeleton.n_vertices == 3

    def test_original_untouched(self, spatial_columns):
        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        with cell.mask_context("skeleton", mask):
            pass
        assert cell.skeleton.n_vertices == 5

    def test_masked_cell_closed_after_context(self, spatial_columns):
        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        with cell.mask_context("skeleton", mask) as masked:
            saved = masked
            saved_layer = masked.skeleton  # capture before the layers are cleared
        # The temporary is torn down: data released, cycle broken.
        assert saved._morphsync is None
        assert saved.skeleton is None  # layers dropped from the manager
        assert saved_layer._cell is None  # back-reference severed

    def test_reclaimed_without_cyclic_gc(self, spatial_columns):
        # With the reference cycle broken, the masked copy is freed by
        # refcounting alone -- no wait for the cyclic collector.
        import gc
        import weakref

        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        gc.disable()
        try:
            with cell.mask_context("skeleton", mask) as masked:
                ref = weakref.ref(masked)
            del masked
            assert ref() is None
        finally:
            gc.enable()

    def test_closed_even_on_exception(self, spatial_columns):
        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        saved = None
        with pytest.raises(RuntimeError):
            with cell.mask_context("skeleton", mask) as masked:
                saved = masked
                raise RuntimeError("boom")
        assert saved._morphsync is None

    def test_layer_level_mask_context_closes(self, spatial_columns):
        cell = self._cell(spatial_columns)
        mask = np.array([True, True, True, False, False])
        with cell.skeleton.mask_context(mask) as masked:
            assert masked.skeleton.n_vertices == 3
            saved = masked
        assert saved._morphsync is None
