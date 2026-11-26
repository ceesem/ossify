"""Tests for graph skeletonization (``ossify.skeletonization``)."""

import io
from typing import Tuple

import numpy as np
import pytest
from scipy import sparse

import ossify
from ossify import file_io
from ossify.skeletonization import (
    SkeletonResult,
    _soma_core,
    add_skeleton_radius,
    contract_vertices,
    detect_soma,
    find_soma_point,
    skeletonize,
    skeletonize_graph,
)

# ============================================================================
# Helpers and synthetic graphs
# ============================================================================


def line_graph(n: int, spacing: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """A straight line of ``n`` vertices along x."""
    vertices = np.zeros((n, 3))
    vertices[:, 0] = np.arange(n) * spacing
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return vertices, edges


def y_graph(spur: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """A 21-vertex trunk along x with a ``spur``-vertex side branch at vertex 10.

    Spacing is 10, so the spur tip sits ``10 * spur`` from the trunk.
    """
    vertices, edges = line_graph(21)
    spur_vertices = np.zeros((spur, 3))
    spur_vertices[:, 0] = vertices[10, 0]
    spur_vertices[:, 1] = (np.arange(spur) + 1) * 10.0
    spur_ids = np.arange(spur) + 21
    spur_edges = np.stack([np.concatenate([[10], spur_ids[:-1]]), spur_ids], axis=1)
    return np.vstack([vertices, spur_vertices]), np.vstack([edges, spur_edges])


def ring_graph(n: int = 40, radius: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)], axis=1
    )
    edges = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    return vertices, edges


def assert_is_rooted_tree(result: SkeletonResult) -> None:
    """The skeleton is one connected tree whose edges point child -> parent."""
    n = len(result.vertices)
    assert result.edges.shape == (n - 1, 2)
    graph = sparse.csr_matrix(
        (np.ones(n - 1), (result.edges[:, 0], result.edges[:, 1])), shape=(n, n)
    )
    n_comp, _ = sparse.csgraph.connected_components(graph, directed=False)
    assert n_comp == 1
    # Every vertex but the root is the child of exactly one edge.
    children = np.sort(result.edges[:, 0])
    assert np.array_equal(children, np.setdiff1d(np.arange(n), [result.root]))


def assert_valid_mapping(result: SkeletonResult, n_graph: int) -> None:
    assert result.graph_to_skeleton.shape == (n_graph,)
    assert result.graph_to_skeleton.min() >= 0
    assert result.graph_to_skeleton.max() < len(result.vertices)
    # A skeleton vertex drawn from a graph vertex is that vertex's own image.
    real = np.flatnonzero(result.source_index >= 0)
    assert np.array_equal(result.graph_to_skeleton[result.source_index[real]], real)


# ============================================================================
# contract_vertices
# ============================================================================


class TestContractVertices:
    def test_contracts_group_to_one_appended_vertex(self):
        vertices, edges = line_graph(5)
        new_vertices, new_edges, vertex_map = contract_vertices(vertices, edges, [1, 2])

        assert len(new_vertices) == 4
        # Kept vertices hold their relative order; the new vertex comes last.
        assert np.array_equal(vertex_map, [0, 3, 3, 1, 2])
        assert np.allclose(new_vertices[:3], vertices[[0, 3, 4]])
        # Default position is the centroid of the contracted group.
        assert np.allclose(new_vertices[3], [15.0, 0.0, 0.0])
        # The internal edge 1-2 is gone, the path stays connected.
        assert {tuple(e) for e in new_edges.tolist()} == {(0, 3), (1, 3), (1, 2)}

    def test_position_can_be_off_graph(self):
        vertices, edges = line_graph(5)
        new_vertices, _, _ = contract_vertices(
            vertices, edges, [1, 2], position=[1.0, 2.0, 3.0]
        )
        assert np.allclose(new_vertices[-1], [1.0, 2.0, 3.0])

    def test_boolean_mask_matches_indices(self):
        vertices, edges = y_graph()
        mask = np.zeros(len(vertices), dtype=bool)
        mask[[9, 10, 11, 21]] = True
        from_mask = contract_vertices(vertices, edges, mask)
        from_index = contract_vertices(vertices, edges, [9, 10, 11, 21])
        for a, b in zip(from_mask, from_index):
            assert np.array_equal(a, b)

    def test_parallel_edges_are_merged(self):
        # Vertices 1 and 2 both touch 0; contracting them must leave one edge.
        vertices = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [20, 0, 0]])
        edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3]])
        _, new_edges, _ = contract_vertices(vertices, edges, [1, 2])
        assert {tuple(e) for e in new_edges.tolist()} == {(0, 2), (1, 2)}

    def test_empty_group_raises(self):
        vertices, edges = line_graph(5)
        with pytest.raises(ValueError, match="empty"):
            contract_vertices(vertices, edges, [])

    def test_wrong_length_mask_raises(self):
        vertices, edges = line_graph(5)
        with pytest.raises(ValueError, match="mask"):
            contract_vertices(vertices, edges, np.ones(3, dtype=bool))


# ============================================================================
# skeletonize_graph: core
# ============================================================================


class TestSkeletonizeGraphCore:
    def test_line_keeps_every_vertex(self):
        vertices, edges = line_graph(10)
        result = skeletonize_graph(vertices, edges, invalidation=5.0, root=0)

        assert len(result.vertices) == 10
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, 10)
        assert result.source_index[result.root] == 0
        assert np.allclose(result.vertices[result.root], vertices[0])

    def test_short_spur_is_absorbed(self):
        vertices, edges = y_graph(spur=2)  # tip 20 from the trunk
        result = skeletonize_graph(vertices, edges, invalidation=25.0, root=0)

        assert len(result.vertices) == 21
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))
        # Spur vertices map onto the trunk vertex they hang off.
        branch_vertex = result.graph_to_skeleton[10]
        assert np.all(result.graph_to_skeleton[[21, 22]] == branch_vertex)

    def test_long_spur_becomes_a_branch(self):
        vertices, edges = y_graph(spur=5)  # tip 50 from the trunk
        result = skeletonize_graph(vertices, edges, invalidation=25.0, root=0)

        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))
        assert np.isin([21, 22, 23, 24, 25], result.source_index).all()
        degree = np.bincount(result.edges.ravel(), minlength=len(result.vertices))
        assert degree[result.graph_to_skeleton[10]] == 3

    def test_mapping_is_geodesic_nearest(self):
        # With the spur absorbed, each spur vertex is nearest the branch
        # vertex along the graph, even though vertex 9/11 are close in space.
        vertices, edges = y_graph(spur=1)
        result = skeletonize_graph(vertices, edges, invalidation=15.0, root=0)
        assert result.graph_to_skeleton[21] == result.graph_to_skeleton[10]

    def test_zero_invalidation_keeps_whole_tree(self):
        vertices, edges = y_graph(spur=3)
        result = skeletonize_graph(vertices, edges, invalidation=0.0, root=0)
        assert len(result.vertices) == len(vertices)
        assert_is_rooted_tree(result)

    def test_cycle_yields_a_tree(self):
        vertices, edges = ring_graph()
        result = skeletonize_graph(vertices, edges, invalidation=5.0, root=0)
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))

    def test_thick_process_gets_one_path(self):
        # A thick process split lengthwise shows up as a ladder: two parallel
        # rails joined by short rungs. The root is on rail A, and the rung at
        # the root end is the shortest, so a branch hanging off the far end of
        # rail B has its shortest route home down the whole of rail B. It must
        # still join the path already running down rail A, not double it.
        n = 30
        rail_a = np.stack([np.arange(n) * 10.0, np.zeros(n), np.zeros(n)], axis=1)
        rail_b = np.stack([np.arange(n) * 10.0, np.full(n, 2.0), np.zeros(n)], axis=1)
        rail_b[0, 1] = 1.0
        side_a = np.stack(
            [np.full(10, 290.0), -(np.arange(10) + 1) * 10.0, np.zeros(10)], axis=1
        )
        side_b = np.stack(
            [np.full(10, 290.0), 2 + (np.arange(10) + 1) * 9.0, np.zeros(10)], axis=1
        )
        vertices = np.vstack([rail_a, rail_b, side_a, side_b])

        def chain(start: int, count: int) -> np.ndarray:
            index = np.arange(start, start + count)
            return np.stack([index[:-1], index[1:]], axis=1)

        edges = np.vstack(
            [
                chain(0, n),
                chain(n, n),
                np.stack([np.arange(n), np.arange(n) + n], axis=1),  # rungs
                [[n - 1, 2 * n]],
                chain(2 * n, 10),
                [[2 * n - 1, 2 * n + 10]],
                chain(2 * n + 10, 10),
            ]
        )
        result = skeletonize_graph(vertices, edges, invalidation=15.0, root=0)

        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))
        on_rail_b = (result.source_index >= n) & (result.source_index < 2 * n)
        assert on_rail_b.sum() <= 2
        # Both side branches are still there.
        assert np.isin([2 * n + 9, 2 * n + 19], result.source_index).all()

    def test_default_root_is_an_extremity(self):
        vertices, edges = line_graph(10)
        result = skeletonize_graph(vertices, edges, invalidation=5.0)
        assert result.source_index[result.root] in (0, 9)

    def test_duplicate_and_reversed_edges_are_harmless(self):
        vertices, edges = y_graph(spur=5)
        noisy = np.vstack([edges, edges[:7, ::-1], edges[3:9]])
        clean = skeletonize_graph(vertices, edges, invalidation=25.0, root=0)
        dirty = skeletonize_graph(vertices, noisy, invalidation=25.0, root=0)
        assert np.array_equal(clean.edges, dirty.edges)
        assert np.array_equal(clean.graph_to_skeleton, dirty.graph_to_skeleton)

    def test_coincident_vertices(self):
        vertices, edges = line_graph(6)
        vertices[3] = vertices[2]  # zero-length edge
        result = skeletonize_graph(vertices, edges, invalidation=5.0, root=0)
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, 6)

    def test_disconnected_graph_raises(self):
        vertices, edges = line_graph(6)
        with pytest.raises(ValueError, match="connected"):
            skeletonize_graph(vertices, edges[[0, 1, 3, 4]], invalidation=5.0)

    def test_negative_invalidation_raises(self):
        vertices, edges = line_graph(6)
        with pytest.raises(ValueError, match="invalidation"):
            skeletonize_graph(vertices, edges, invalidation=-1.0)

    def test_root_out_of_range_raises(self):
        vertices, edges = line_graph(6)
        with pytest.raises(ValueError, match="root"):
            skeletonize_graph(vertices, edges, invalidation=5.0, root=6)


# ============================================================================
# skeletonize_graph: soma handling
# ============================================================================


class TestSkeletonizeGraphSoma:
    def test_soma_point_alone_roots_at_nearest_vertex(self):
        vertices, edges = line_graph(10)
        result = skeletonize_graph(
            vertices, edges, invalidation=5.0, soma_point=[41.0, 3.0, 0.0]
        )
        assert result.source_index[result.root] == 4
        assert len(result.vertices) == 10

    def test_ball_collapse_adds_synthetic_root(self):
        vertices, edges = line_graph(21)
        soma_point = np.array([100.0, 7.0, 0.0])  # off the graph, beside vertex 10
        result = skeletonize_graph(
            vertices, edges, invalidation=5.0, soma_point=soma_point, soma_radius=25.0
        )

        assert_is_rooted_tree(result)
        assert_valid_mapping(result, 21)
        # The root is a new vertex at the soma point, not a graph vertex.
        assert result.source_index[result.root] == -1
        assert np.sum(result.source_index == -1) == 1
        assert np.allclose(result.vertices[result.root], soma_point)
        # Vertices 8..12 lie inside the ball and collapse onto the root.
        assert np.all(result.graph_to_skeleton[8:13] == result.root)
        assert not np.isin([8, 9, 10, 11, 12], result.source_index).any()
        assert len(result.vertices) == 21 - 5 + 1
        # Two branches leave the soma.
        assert np.sum(result.edges[:, 1] == result.root) == 2

    def test_vertices_just_outside_the_soma_are_not_stubs(self):
        # A soma is a blob: plenty of its vertices sit just outside any ball.
        # The bump here is 26 from the soma point (outside the ball) but only
        # ~11 along the graph from the collapsed region, so it is covered by
        # the paths through the soma rather than becoming a branch of its own.
        vertices, edges = line_graph(21)
        vertices = np.vstack([vertices, [[124.0, 10.0, 0.0]]])
        edges = np.vstack([edges, [[12, 21]]])
        soma_point = np.array([100.0, 0.0, 0.0])
        assert np.linalg.norm(vertices[21] - soma_point) > 25.0

        result = skeletonize_graph(
            vertices, edges, invalidation=15.0, soma_point=soma_point, soma_radius=25.0
        )
        assert_is_rooted_tree(result)
        assert result.graph_to_skeleton[21] == result.root
        assert np.sum(result.edges[:, 1] == result.root) == 2
        # The skeleton itself still runs to the soma point.
        assert np.allclose(result.vertices[result.root], soma_point)

    def test_ball_with_no_vertices_inside_uses_nearest(self):
        vertices, edges = line_graph(10)
        soma_point = np.array([40.0, 50.0, 0.0])
        result = skeletonize_graph(
            vertices, edges, invalidation=5.0, soma_point=soma_point, soma_radius=1.0
        )
        assert result.source_index[result.root] == -1
        assert np.allclose(result.vertices[result.root], soma_point)
        assert result.graph_to_skeleton[4] == result.root
        assert np.sum(result.graph_to_skeleton == result.root) == 1

    def test_ball_only_collapses_connected_region(self):
        # A hairpin: the far arm passes back through the ball, but is only
        # joined to the soma region by a long detour outside it.
        n = 30
        out = np.stack([np.arange(n) * 10.0, np.zeros(n), np.zeros(n)], axis=1)
        back = np.stack([np.arange(n)[::-1] * 10.0, np.full(n, 15.0), np.zeros(n)], 1)
        vertices = np.vstack([out, back])
        edges = np.stack([np.arange(2 * n - 1), np.arange(1, 2 * n)], axis=1)

        result = skeletonize_graph(
            vertices, edges, invalidation=5.0, soma_point=out[0], soma_radius=22.0
        )
        assert_is_rooted_tree(result)
        collapsed = np.flatnonzero(result.graph_to_skeleton == result.root)
        assert np.array_equal(collapsed, [0, 1, 2])
        # The returning arm stays a single unbranched path: no shortcut.
        assert np.sum(result.edges[:, 1] == result.root) == 1

    def test_explicit_soma_mask(self):
        vertices, edges = y_graph(spur=5)
        mask = np.zeros(len(vertices), dtype=bool)
        mask[[9, 10, 11, 21]] = True
        result = skeletonize_graph(vertices, edges, invalidation=5.0, soma_mask=mask)

        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))
        assert np.all(result.graph_to_skeleton[mask] == result.root)
        # No soma_point: the root sits at the centroid of the masked vertices.
        assert np.allclose(result.vertices[result.root], vertices[mask].mean(axis=0))
        assert np.sum(result.edges[:, 1] == result.root) == 3

    def test_soma_mask_with_soma_point_sets_position(self):
        vertices, edges = line_graph(10)
        mask = np.zeros(10, dtype=bool)
        mask[4:6] = True
        result = skeletonize_graph(
            vertices, edges, invalidation=5.0, soma_mask=mask, soma_point=[1.0, 2, 3]
        )
        assert np.allclose(result.vertices[result.root], [1.0, 2.0, 3.0])

    def test_root_conflicts_with_collapse(self):
        vertices, edges = line_graph(10)
        with pytest.raises(ValueError, match="root"):
            skeletonize_graph(
                vertices,
                edges,
                invalidation=5.0,
                root=0,
                soma_point=[40.0, 0, 0],
                soma_radius=15.0,
            )

    def test_soma_radius_without_point_raises(self):
        vertices, edges = line_graph(10)
        with pytest.raises(ValueError, match="soma_point"):
            skeletonize_graph(vertices, edges, invalidation=5.0, soma_radius=15.0)

    def test_empty_soma_mask_raises(self):
        vertices, edges = line_graph(10)
        with pytest.raises(ValueError, match="empty"):
            skeletonize_graph(
                vertices, edges, invalidation=5.0, soma_mask=np.zeros(10, dtype=bool)
            )


# ============================================================================
# Cell-level integration on a real L2 graph
# ============================================================================


@pytest.fixture
def graph_cell(root_id, l2_df, l2_graph, l2_spatial_columns):
    return ossify.Cell(name=root_id).add_graph(
        vertices=l2_df,
        spatial_columns=l2_spatial_columns,
        edges=l2_graph,
        vertex_index="l2_id",
        edges_as_positional=False,
    )


@pytest.fixture
def soma_point(skel_dict):
    meta = skel_dict["meta"]
    return np.array([meta["soma_pt_x"], meta["soma_pt_y"], meta["soma_pt_z"]])


def reference_cable_length(skel_dict) -> float:
    vertices = np.array(skel_dict["vertices"])
    edges = np.array(skel_dict["edges"])
    return float(
        np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).sum()
    )


class TestSkeletonizeCell:
    def test_l2_graph_with_soma(self, graph_cell, soma_point, skel_dict):
        cell = skeletonize(
            graph_cell, invalidation=7500, soma_point=soma_point, soma_radius=7500
        )
        assert cell is graph_cell
        skel = cell.skeleton
        assert skel is not None

        # A single rooted tree, with the root at the soma point.
        assert len(skel.edges) == skel.n_vertices - 1
        assert np.allclose(skel.root_location, soma_point)
        assert np.isfinite(skel.distance_to_root()).all()

        # Every graph vertex maps to a skeleton vertex through the link.
        assert cell.graph.mapping_coverage("skeleton") == 1.0
        mapped = cell.graph.map_index_to_layer("skeleton", as_positional=True)
        assert len(mapped) == cell.graph.n_vertices
        assert mapped.min() >= 0 and mapped.max() < skel.n_vertices

        # Vertices near the soma collapse onto the root.
        near = np.linalg.norm(cell.graph.vertices - soma_point, axis=1) < 2000
        assert near.any()
        assert np.all(mapped[near] == skel.root_positional)

        # Same data and parameters as the reference pcg_skel skeleton, so the
        # total cable should land close to it.
        ratio = skel.cable_length() / reference_cable_length(skel_dict)
        assert 0.9 < ratio < 1.1

    def test_l2_graph_without_soma(self, graph_cell):
        cell = skeletonize(graph_cell, invalidation=7500)
        skel = cell.skeleton
        assert len(skel.edges) == skel.n_vertices - 1
        assert cell.graph.mapping_coverage("skeleton") == 1.0
        # Without a soma every skeleton vertex is a graph vertex.
        assert skel.n_vertices < cell.graph.n_vertices

    def test_root_is_a_graph_vertex_index(self, graph_cell):
        root_l2 = int(graph_cell.graph.vertex_index[123])
        cell = skeletonize(graph_cell, invalidation=7500, root=root_l2)
        assert np.allclose(cell.skeleton.root_location, cell.graph.vertices[123])

        cell = skeletonize(
            graph_cell,
            invalidation=7500,
            root=123,
            root_as_positional=True,
            overwrite=True,
        )
        assert np.allclose(cell.skeleton.root_location, cell.graph.vertices[123])

    def test_soma_mask_from_feature(self, graph_cell, soma_point):
        near = np.linalg.norm(graph_cell.graph.vertices - soma_point, axis=1) < 5000
        graph_cell.graph.add_feature(near, "is_soma")
        cell = skeletonize(
            graph_cell, invalidation=7500, soma_mask="is_soma", soma_point=soma_point
        )
        mapped = cell.graph.map_index_to_layer("skeleton", as_positional=True)
        assert np.all(mapped[near] == cell.skeleton.root_positional)
        assert np.allclose(cell.skeleton.root_location, soma_point)

    def test_existing_skeleton_requires_overwrite(self, graph_cell):
        skeletonize(graph_cell, invalidation=7500)
        n_first = graph_cell.skeleton.n_vertices
        with pytest.raises(ValueError, match="overwrite"):
            skeletonize(graph_cell, invalidation=7500)

        skeletonize(graph_cell, invalidation=20000, overwrite=True)
        assert graph_cell.skeleton.n_vertices < n_first
        assert graph_cell.graph.mapping_coverage("skeleton") == 1.0

    def test_no_graph_raises(self):
        with pytest.raises(ValueError, match="graph"):
            skeletonize(ossify.Cell(name="empty"))

    def test_save_load_roundtrip(self, graph_cell, soma_point):
        skeletonize(
            graph_cell, invalidation=7500, soma_point=soma_point, soma_radius=7500
        )
        buffer = io.BytesIO()
        file_io.save_cell(graph_cell, buffer)
        buffer.seek(0)
        loaded = file_io.load_cell(buffer)

        assert loaded.skeleton.n_vertices == graph_cell.skeleton.n_vertices
        assert np.allclose(loaded.skeleton.root_location, soma_point)
        assert np.array_equal(
            loaded.graph.map_index_to_layer("skeleton", as_positional=True),
            graph_cell.graph.map_index_to_layer("skeleton", as_positional=True),
        )


# ============================================================================
# SkeletonService-compatible method
# ============================================================================


class TestPcgSkelMethod:
    """``method="pcg_skel"`` reproduces SkeletonService skeletons.

    ``skel.json`` is a version 4 skeleton from the service for the same level 2
    graph held in ``l2graph.json``, so it can be matched exactly.
    """

    def test_matches_service_skeleton_exactly(self, graph_cell, soma_point, skel_dict):
        meta = skel_dict["meta"]
        graph = graph_cell.graph
        assert np.array_equal(graph.vertex_index, skel_dict["lvl2_ids"])

        result = skeletonize_graph(
            graph.vertices,
            graph.edges_positional,
            invalidation=meta["invalidation_d"],
            soma_point=soma_point,
            soma_radius=meta["soma_radius"],
            method="pcg_skel",
        )

        assert np.array_equal(result.vertices, np.array(skel_dict["vertices"]))
        assert np.array_equal(result.edges, np.array(skel_dict["edges"]))
        assert result.root == skel_dict["root"]
        assert np.array_equal(result.graph_to_skeleton, skel_dict["mesh_to_skel_map"])
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, graph.n_vertices)
        assert result.source_index[result.root] == -1

    def test_cell_level_matches_service(self, graph_cell, soma_point, skel_dict):
        cell = skeletonize(
            graph_cell,
            invalidation=7500,
            soma_point=soma_point,
            soma_radius=7500,
            method="pcg_skel",
        )
        assert np.array_equal(cell.skeleton.vertices, np.array(skel_dict["vertices"]))
        assert cell.skeleton.root_positional == skel_dict["root"]
        assert np.array_equal(
            cell.graph.map_index_to_layer("skeleton", as_positional=True),
            skel_dict["mesh_to_skel_map"],
        )

    def test_without_soma_roots_at_extremity(self):
        vertices, edges = y_graph(spur=5)
        result = skeletonize_graph(
            vertices, edges, invalidation=25.0, method="pcg_skel"
        )
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))
        assert result.source_index[result.root] in (0, 20)
        assert np.all(result.source_index >= 0)

    def test_soma_point_alone_roots_at_nearest_skeleton_vertex(self):
        vertices, edges = line_graph(10)
        result = skeletonize_graph(
            vertices,
            edges,
            invalidation=5.0,
            soma_point=[41.0, 3.0, 0.0],
            method="pcg_skel",
        )
        assert result.source_index[result.root] == 4
        assert len(result.vertices) == 10

    def test_collapse_takes_one_vertex_beyond_the_sphere(self):
        # Vertices 8..12 are strictly inside the sphere. The service method
        # also collapses 7 and 13, one edge outside it; the default does not.
        vertices, edges = line_graph(21)
        kwargs = dict(invalidation=5.0, soma_point=[100.0, 0, 0], soma_radius=25.0)

        compat = skeletonize_graph(vertices, edges, method="pcg_skel", **kwargs)
        assert_is_rooted_tree(compat)
        assert np.array_equal(
            np.flatnonzero(compat.graph_to_skeleton == compat.root), np.arange(7, 14)
        )
        assert compat.root == len(compat.vertices) - 1
        assert np.allclose(compat.vertices[compat.root], [100.0, 0, 0])

        default = skeletonize_graph(vertices, edges, **kwargs)
        assert np.array_equal(
            np.flatnonzero(default.graph_to_skeleton == default.root), np.arange(8, 13)
        )

    def test_coincident_vertices_are_merged(self):
        vertices, edges = line_graph(8)
        vertices[3] = vertices[2]
        vertices[4] = vertices[2]  # a run of three coincident vertices
        result = skeletonize_graph(vertices, edges, invalidation=5.0, method="pcg_skel")
        assert_is_rooted_tree(result)
        assert len(result.vertices) == 6
        assert len(np.unique(result.graph_to_skeleton[[2, 3, 4]])) == 1
        lengths = np.linalg.norm(
            result.vertices[result.edges[:, 0]] - result.vertices[result.edges[:, 1]],
            axis=1,
        )
        assert np.all(lengths > 0)

    def test_repeated_edges_change_the_graph(self):
        # Faithful to the service: a doubled edge is twice as long, which can
        # move the root. Here it only has to stay a valid skeleton.
        vertices, edges = y_graph(spur=5)
        doubled = np.vstack([edges, edges[:5]])
        result = skeletonize_graph(
            vertices, doubled, invalidation=25.0, method="pcg_skel"
        )
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(vertices))

    @pytest.mark.parametrize(
        "kwargs", [{"root": 0}, {"soma_mask": np.ones(10, dtype=bool)}]
    )
    def test_unsupported_arguments_raise(self, kwargs):
        vertices, edges = line_graph(10)
        with pytest.raises(ValueError, match="pcg_skel"):
            skeletonize_graph(
                vertices, edges, invalidation=5.0, method="pcg_skel", **kwargs
            )

    def test_unknown_method_raises(self):
        vertices, edges = line_graph(10)
        with pytest.raises(ValueError, match="method"):
            skeletonize_graph(vertices, edges, invalidation=5.0, method="teasar")

    def test_disconnected_graph_raises(self):
        vertices, edges = line_graph(6)
        with pytest.raises(ValueError, match="connected"):
            skeletonize_graph(
                vertices, edges[[0, 1, 3, 4]], invalidation=5.0, method="pcg_skel"
            )


# ============================================================================
# Soma detection from the graph
# ============================================================================


def root_branch_cable(result: SkeletonResult) -> np.ndarray:
    """Cable length of each branch leaving the root, including its first edge."""
    n = len(result.vertices)
    parent = np.full(n, -1)
    parent[result.edges[:, 0]] = result.edges[:, 1]
    length = np.zeros(n)
    length[result.edges[:, 0]] = np.linalg.norm(
        result.vertices[result.edges[:, 0]] - result.vertices[result.edges[:, 1]],
        axis=1,
    )
    cable = {}
    for vertex in np.flatnonzero(parent >= 0):
        node = vertex
        while parent[node] != result.root:
            node = parent[node]
        cable[node] = cable.get(node, 0.0) + length[vertex]
    return np.array(sorted(cable.values()))


class SomaGraph:
    """A toy neuron in nanometres, built to exercise each soma rule.

    - a 5 um sphere at the origin with six 4 um spheres 3 um away: the soma;
    - a thin dendrite along +x and another along -x;
    - a short thin twig off the +y side, which is not a real process;
    - off the -y side, a thick stalk that forks 9 um out into two dendrites.
    """

    def __init__(self) -> None:
        self.vertices = [[0.0, 0.0, 0.0]]
        self.radius = [5000.0]
        self.edges = []
        self.satellite = {}
        for name, direction in {
            "+x": (1, 0, 0),
            "-x": (-1, 0, 0),
            "+y": (0, 1, 0),
            "-y": (0, -1, 0),
            "+z": (0, 0, 1),
            "-z": (0, 0, -1),
        }.items():
            self.satellite[name] = self.add(np.array(direction) * 3000.0, 4000.0, 0)

        self.dendrite_a = self.chain(self.satellite["+x"], (1, 0, 0), 40, 300.0)
        self.dendrite_b = self.chain(self.satellite["-x"], (-1, 0, 0), 40, 300.0)
        self.twig = self.chain(self.satellite["+y"], (0, 1, 0), 3, 300.0)
        self.stalk = self.chain(self.satellite["-y"], (0, -1, 0), 3, 1500.0)
        diagonal = np.sqrt(0.5)
        self.fork_a = self.chain(self.stalk[-1], (diagonal, -diagonal, 0), 40, 300.0)
        self.fork_b = self.chain(self.stalk[-1], (-diagonal, -diagonal, 0), 40, 300.0)

    def add(self, position, radius: float, neighbour: int) -> int:
        self.vertices.append(list(position))
        self.radius.append(radius)
        self.edges.append([neighbour, len(self.vertices) - 1])
        return len(self.vertices) - 1

    def chain(self, start: int, direction, count: int, radius: float) -> list:
        """``count`` vertices 2 um apart, leading away from ``start``."""
        added = []
        previous = start
        for _ in range(count):
            position = np.array(self.vertices[previous]) + np.array(direction) * 2000.0
            previous = self.add(position, radius, previous)
            added.append(previous)
        return added

    @property
    def arrays(self) -> tuple:
        radius = np.array(self.radius)
        volume = 4 / 3 * np.pi * radius**3
        return np.array(self.vertices), np.array(self.edges), radius, volume


@pytest.fixture
def soma_graph() -> SomaGraph:
    return SomaGraph()


class TestSomaCore:
    def test_engulfed_neighbours_join_and_neurites_do_not(self, soma_graph):
        vertices, edges, radius, _ = soma_graph.arrays
        core = _soma_core(vertices, edges, radius, seed=0, large_sphere_fraction=0.5)

        assert core[0]
        assert core[list(soma_graph.satellite.values())].all()
        # The first vertex of each process sits inside a 4 um sphere.
        assert core[soma_graph.dendrite_a[0]]
        # Beyond that, vertices are 2 um apart and far thinner than that.
        assert not core[soma_graph.dendrite_a[1:]].any()
        assert not core[soma_graph.stalk[1:]].any()

    def test_large_overlapping_spheres_join(self, soma_graph):
        # A second big piece of soma, 5.5 um beyond the +z sphere: too far for
        # either sphere to hold the other's centre, but they overlap deeply.
        piece = soma_graph.add([0.0, 0.0, 8500.0], 3500.0, soma_graph.satellite["+z"])
        vertices, edges, radius, _ = soma_graph.arrays

        assert _soma_core(vertices, edges, radius, 0, large_sphere_fraction=0.5)[piece]
        # Not when it is too small to count as a large sphere.
        assert not _soma_core(vertices, edges, radius, 0, large_sphere_fraction=0.9)[
            piece
        ]

    def test_overlap_needs_no_graph_edge(self, soma_graph):
        # The same piece, attached to the soma only through a thin sliver.
        sliver = soma_graph.add(
            [4000.0, 0.0, 5000.0], 200.0, soma_graph.satellite["+z"]
        )
        piece = soma_graph.add([0.0, 0.0, 8500.0], 3500.0, sliver)
        vertices, edges, radius, _ = soma_graph.arrays
        assert _soma_core(vertices, edges, radius, 0, large_sphere_fraction=0.5)[piece]


class TestDetectSoma:
    def detect(self, soma_graph: SomaGraph, **kwargs) -> np.ndarray:
        vertices, edges, radius, volume = soma_graph.arrays
        return detect_soma(
            vertices,
            edges,
            radius,
            volume,
            [0.0, 0.0, 0.0],
            invalidation=1000.0,
            **kwargs,
        )

    def test_real_processes_are_left_alone(self, soma_graph):
        mask = self.detect(soma_graph)
        for process in (soma_graph.dendrite_a, soma_graph.dendrite_b):
            assert not mask[process[1:]].any()
        assert not mask[soma_graph.fork_a].any()
        assert not mask[soma_graph.fork_b].any()

    def test_short_twig_is_absorbed(self, soma_graph):
        mask = self.detect(soma_graph)
        assert mask[soma_graph.twig].all()
        # It only counts as soma because it is short.
        assert not self.detect(soma_graph, min_branch_length=5000.0)[
            soma_graph.twig[1:]
        ].any()

    def test_thick_stalk_before_a_near_fork_is_absorbed(self, soma_graph):
        assert self.detect(soma_graph)[soma_graph.stalk].all()

    def test_stalk_is_kept_when_the_fork_is_far(self, soma_graph):
        mask = self.detect(soma_graph, base_distance_factor=0.5)
        assert not mask[soma_graph.stalk[1:]].any()

    def test_stalk_is_kept_when_it_is_thin(self, soma_graph):
        mask = self.detect(soma_graph, base_thickness_fraction=0.5)
        assert not mask[soma_graph.stalk[1:]].any()

    def test_seed_is_the_largest_sphere_near_the_soma_point(self, soma_graph):
        # A soma point several microns off still finds the same soma.
        vertices, edges, radius, volume = soma_graph.arrays
        centred = self.detect(soma_graph)
        shifted = detect_soma(
            vertices,
            edges,
            radius,
            volume,
            [4000.0, -3000.0, 2000.0],
            invalidation=1000.0,
        )
        assert np.array_equal(centred, shifted)

    def test_missing_values_count_as_zero(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        radius[soma_graph.dendrite_a[10]] = np.nan
        volume[soma_graph.dendrite_a[10]] = np.nan
        mask = detect_soma(
            vertices, edges, radius, volume, [0.0, 0.0, 0.0], invalidation=1000.0
        )
        assert np.array_equal(mask, self.detect(soma_graph))

    def test_wrong_length_arrays_raise(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        with pytest.raises(ValueError, match="one value per vertex"):
            detect_soma(vertices, edges, radius[:-1], volume, [0.0, 0.0, 0.0])


class TestSkeletonizeWithDetectedSoma:
    def skeleton(self, soma_graph: SomaGraph, **kwargs) -> SkeletonResult:
        vertices, edges, radius, volume = soma_graph.arrays
        arguments = dict(
            invalidation=1000.0,
            soma_point=[0.0, 0.0, 0.0],
            vertex_radius=radius,
            vertex_volume=volume,
        )
        arguments.update(kwargs)
        return skeletonize_graph(vertices, edges, **arguments)

    def test_each_process_leaves_the_root_separately(self, soma_graph):
        result = self.skeleton(soma_graph)
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, len(soma_graph.vertices))
        assert result.source_index[result.root] == -1
        assert np.allclose(result.vertices[result.root], 0.0)

        # Two dendrites and the two sides of the fork; no twig, no stalk.
        cable = root_branch_cable(result)
        assert len(cable) == 4
        assert cable.min() > 20000.0
        assert np.all(result.graph_to_skeleton[soma_graph.twig] == result.root)
        assert np.all(result.graph_to_skeleton[soma_graph.stalk] == result.root)

    def test_a_ball_does_worse_on_the_same_graph(self, soma_graph):
        vertices, edges, _, _ = soma_graph.arrays
        ball = skeletonize_graph(
            vertices,
            edges,
            invalidation=1000.0,
            soma_point=[0, 0, 0],
            soma_radius=7500.0,
        )
        # The ball stops short of the twig's tip and the fork, so it keeps the
        # twig as a stub and the two forked dendrites as one branch.
        cable = root_branch_cable(ball)
        assert (cable < 20000.0).sum() == 1
        assert (cable >= 20000.0).sum() == 3

    def test_everything_is_soma_when_nothing_persists(self):
        vertices, edges = line_graph(5, spacing=2000.0)
        result = skeletonize_graph(
            vertices,
            edges,
            soma_point=vertices[2],
            vertex_radius=np.full(5, 300.0),
            vertex_volume=np.full(5, 1e8),
        )
        assert len(result.vertices) == 1
        assert np.all(result.graph_to_skeleton == 0)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"vertex_volume": None}, "together"),
            ({"soma_radius": 7500.0}, "cannot be combined"),
            ({"soma_mask": np.ones(213, dtype=bool)}, "cannot be combined"),
            ({"method": "pcg_skel"}, "pcg_skel"),
        ],
    )
    def test_inconsistent_arguments_raise(self, soma_graph, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self.skeleton(soma_graph, **kwargs)


class TestDetectSomaOnCell:
    def test_l2_graph(self, graph_cell, soma_point):
        graph = graph_cell.graph
        result = skeletonize_graph(
            graph.vertices,
            graph.edges_positional,
            soma_point=soma_point,
            vertex_radius=graph.get_feature("max_dt_nm"),
            vertex_volume=graph.get_feature("size_nm3"),
        )
        assert_is_rooted_tree(result)
        assert_valid_mapping(result, graph.n_vertices)
        assert np.allclose(result.vertices[result.root], soma_point)

        # No stubs: every branch leaving the soma is a real process.
        cable = root_branch_cable(result)
        assert len(cable) >= 2
        assert cable.min() >= 20000.0

        # And it merges far less of the graph than the 7.5 um ball does.
        mask = detect_soma(
            graph.vertices,
            graph.edges_positional,
            graph.get_feature("max_dt_nm"),
            graph.get_feature("size_nm3"),
            soma_point,
        )
        near = np.linalg.norm(graph.vertices - soma_point, axis=1) < 7500
        assert 0 < mask.sum() < near.sum()

    def test_cell_level(self, graph_cell, soma_point):
        cell = skeletonize(graph_cell, soma_point=soma_point, detect_soma=True)
        skel = cell.skeleton
        assert len(skel.edges) == skel.n_vertices - 1
        assert np.allclose(skel.root_location, soma_point)
        assert cell.graph.mapping_coverage("skeleton") == 1.0

        graph = cell.graph
        expected = skeletonize_graph(
            graph.vertices,
            graph.edges_positional,
            soma_point=soma_point,
            vertex_radius=graph.get_feature("max_dt_nm"),
            vertex_volume=graph.get_feature("size_nm3"),
        )
        assert np.array_equal(skel.vertices, expected.vertices)

    def test_cell_level_custom_feature_names(self, graph_cell, soma_point):
        graph = graph_cell.graph
        graph.add_feature(graph.get_feature("max_dt_nm"), "thickness")
        graph.add_feature(graph.get_feature("size_nm3"), "volume")
        default = skeletonize(graph_cell, soma_point=soma_point, detect_soma=True)
        n_default = default.skeleton.n_vertices
        skeletonize(
            graph_cell,
            soma_point=soma_point,
            detect_soma=True,
            radius_feature="thickness",
            volume_feature="volume",
            overwrite=True,
        )
        assert graph_cell.skeleton.n_vertices == n_default

    def test_cell_level_missing_feature_raises(self, graph_cell, soma_point):
        with pytest.raises(ValueError, match="not_a_feature"):
            skeletonize(
                graph_cell,
                soma_point=soma_point,
                detect_soma=True,
                radius_feature="not_a_feature",
            )


class TestFindSomaPoint:
    def test_finds_the_soma_centre(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        point = find_soma_point(vertices, edges, radius, volume)
        assert point is not None
        # The toy soma is centred on the origin, 5 um across at its core.
        assert np.linalg.norm(point) < 1000.0

    def test_no_soma_when_nothing_is_thick(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        thin = np.minimum(radius, 1500.0)
        assert find_soma_point(vertices, edges, thin, volume) is None

    def test_no_soma_when_the_thick_region_is_small(self, soma_graph):
        # One fat vertex on a dendrite is a swelling, not a cell body.
        vertices, edges, radius, volume = soma_graph.arrays
        radius = np.full(len(radius), 300.0)
        radius[soma_graph.dendrite_a[20]] = 2500.0
        volume = 4 / 3 * np.pi * radius**3
        assert find_soma_point(vertices, edges, radius, volume) is None

    def test_thresholds_are_adjustable(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        assert (
            find_soma_point(vertices, edges, radius, volume, min_peak_radius=6000.0)
            is None
        )
        assert (
            find_soma_point(vertices, edges, radius, volume, min_soma_radius=20000.0)
            is None
        )

    def test_skeletonize_locates_the_soma_itself(self, soma_graph):
        vertices, edges, radius, volume = soma_graph.arrays
        guessed = skeletonize_graph(
            vertices,
            edges,
            invalidation=1000.0,
            vertex_radius=radius,
            vertex_volume=volume,
        )
        assert guessed.source_index[guessed.root] == -1
        assert np.linalg.norm(guessed.vertices[guessed.root]) < 1000.0
        assert len(root_branch_cable(guessed)) == 4

    def test_skeletonize_without_a_soma_roots_at_an_extremity(self):
        vertices, edges = line_graph(30, spacing=2000.0)
        result = skeletonize_graph(
            vertices,
            edges,
            invalidation=1000.0,
            vertex_radius=np.full(30, 300.0),
            vertex_volume=np.full(30, 1e8),
        )
        assert result.source_index[result.root] in (0, 29)
        assert len(result.vertices) == 30

    def test_cell_level_without_a_soma_point(self, graph_cell, soma_point):
        cell = skeletonize(graph_cell, detect_soma=True)
        assert np.linalg.norm(cell.skeleton.root_location - soma_point) < 5000


class TestSingleVertex:
    def test_single_vertex_graph(self):
        result = skeletonize_graph(
            np.array([[1.0, 2.0, 3.0]]), np.empty((0, 2), dtype=int)
        )
        assert result.vertices.shape == (1, 3)
        assert result.edges.shape == (0, 2)
        assert result.root == 0
        assert np.array_equal(result.graph_to_skeleton, [0])
        assert np.array_equal(result.source_index, [0])

    def test_single_vertex_with_soma_arguments(self):
        result = skeletonize_graph(
            np.array([[1.0, 2.0, 3.0]]),
            np.empty((0, 2), dtype=int),
            vertex_radius=np.array([5000.0]),
            vertex_volume=np.array([5e11]),
        )
        assert len(result.vertices) == 1

    def test_single_vertex_cell(self):
        cell = ossify.Cell(name="dot").add_graph(
            vertices=np.array([[1.0, 2.0, 3.0]]), edges=np.empty((0, 2), dtype=int)
        )
        skeletonize(cell)
        assert cell.skeleton.n_vertices == 1
        assert len(cell.skeleton.edges) == 0
        assert cell.graph.mapping_coverage("skeleton") == 1.0


class TestSkeletonRadius:
    def test_matches_the_skeleton_service(self, graph_cell, soma_point, skel_dict):
        # On the service's own skeleton, the equivalent radius reproduces the
        # values the service reported for it.
        skeletonize(
            graph_cell,
            invalidation=7500,
            soma_point=soma_point,
            soma_radius=7500,
            method="pcg_skel",
        )
        add_skeleton_radius(graph_cell)
        radius = graph_cell.skeleton.get_feature("radius")
        assert np.allclose(radius, skel_dict["radius"], rtol=1e-6)

    def test_equivalent_radius_is_constant_on_a_segment(self, graph_cell, soma_point):
        skeletonize(graph_cell, soma_point=soma_point, detect_soma=True)
        add_skeleton_radius(graph_cell)
        skel = graph_cell.skeleton
        radius = skel.get_feature("radius")
        for segment in skel.segments_positional[:50]:
            assert np.ptp(radius[segment]) == 0

    def test_cylinder_and_sphere(self):
        # A soma at the middle of a line with a uniform process either side:
        # vertices 1000 apart, each carrying the volume of a 1000-long
        # cylinder of radius 200.
        vertices, edges = line_graph(21, spacing=1000.0)
        volume = np.full(21, np.pi * 200.0**2 * 1000.0)
        volume[10] = 4 / 3 * np.pi * 3000.0**3
        inscribed = np.full(21, 150.0)
        inscribed[10] = 2500.0
        cell = ossify.Cell(name="rod").add_graph(
            vertices=vertices,
            edges=edges,
            features={"max_dt_nm": inscribed, "size_nm3": volume},
        )
        skeletonize(cell, invalidation=100.0, root=10, root_as_positional=True)
        add_skeleton_radius(cell)

        skel = cell.skeleton
        radius = skel.get_feature("radius")
        root = skel.root_positional
        others = np.setdiff1d(np.arange(21), [root])
        assert np.allclose(radius[others], 200.0)
        assert np.isclose(radius[root], 3000.0)
        assert np.allclose(skel.get_feature("radius_local")[others], 150.0)
        assert np.isclose(skel.get_feature("radius_local")[root], 2500.0)

    def test_root_segment_is_a_sphere_even_when_it_is_a_process(self):
        # Inherited from pcg_skel: the whole segment holding the root is given
        # a sphere's radius. With the root at the end of an unbranched process
        # that is the entire process, which is the price of matching it.
        vertices, edges = line_graph(11, spacing=1000.0)
        volume = np.full(11, np.pi * 200.0**2 * 1000.0)
        cell = ossify.Cell(name="rod").add_graph(
            vertices=vertices,
            edges=edges,
            features={"max_dt_nm": np.full(11, 150.0), "size_nm3": volume},
        )
        skeletonize(cell, invalidation=100.0, root=0, root_as_positional=True)
        add_skeleton_radius(cell)
        radius = cell.skeleton.get_feature("radius")
        assert np.allclose(radius, (3 * volume.sum() / (4 * np.pi)) ** (1 / 3))

    def test_missing_features_raise(self, graph_cell):
        skeletonize(graph_cell)
        with pytest.raises(ValueError, match="nope"):
            add_skeleton_radius(graph_cell, radius_feature="nope")

    def test_needs_a_skeleton(self, graph_cell):
        with pytest.raises(ValueError, match="skeleton"):
            add_skeleton_radius(graph_cell)
