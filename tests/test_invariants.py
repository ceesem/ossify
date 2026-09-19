"""Invariant, parametrized, and roundtrip tests.

These target the *classes* of bug that line coverage misses:

* ``_vertices_to_positional`` is the chokepoint every distance/path method
  funnels through, and it accepts many input shapes (scalar / list / array /
  bool mask, positional vs index). Parametrizing over those shapes guards a
  whole family of "one input form was never exercised" bugs.
* Cached, geometry-derived state (``csgraph``, ``base_csgraph``) must stay
  consistent with the vertices after a mutation. A single reusable helper --
  "cached graph equals a graph rebuilt from scratch" -- checks that after every
  mutating operation, so cache-invalidation regressions surface immediately.
* The "plumbing" methods (``copy``, ``apply_mask``, save/load) are easy to
  leave unexercised; roundtrip tests per layer type keep them honest.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from ossify import Cell, Link, file_io, utils

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------
SPATIAL = ["x", "y", "z"]
SKEL_IDX = np.array([100, 101, 102, 103, 104])


def line_skeleton_cell():
    """5-vertex line skeleton, unit spacing, non-positional vertex index."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    df = pd.DataFrame(verts, columns=SPATIAL, index=SKEL_IDX)
    edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
    cell = Cell()
    cell.add_skeleton(vertices=df, edges=edges, spatial_columns=SPATIAL, root=100)
    return cell


def full_cell():
    """A cell exercising every layer family: skeleton, mesh, and points."""
    cell = line_skeleton_cell()
    cell.skeleton.add_feature(np.arange(5, dtype=float), "radius")

    mesh_v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    mesh_faces = np.array([[0, 1, 2], [1, 3, 2]])
    cell.add_mesh(vertices=mesh_v, faces=mesh_faces, spatial_columns=SPATIAL)

    pts = pd.DataFrame(
        {"x": [0.1, 1.1], "y": [0.0, 0.0], "z": [0.0, 0.0], "kind": ["a", "b"]},
        index=[500, 501],
    )
    cell.add_point_annotations(name="syn", vertices=pts, spatial_columns=SPATIAL)
    return cell


def assert_weighted_csgraph_consistent(layer):
    """The cached distance-weighted ``csgraph`` must equal a graph rebuilt from
    the layer's *current* vertices. Catches stale caches after a mutation."""
    fresh = utils.build_csgraph(
        layer.vertices,
        layer.edges_positional,
        euclidean_weight=True,
        directed=True,
    )
    cached = layer.csgraph
    assert cached.shape == fresh.shape
    # Elementwise equality for sparse matrices: no differing stored entries.
    assert (cached != fresh).nnz == 0


# ===========================================================================
# Rec 3: _vertices_to_positional across every input shape
# ===========================================================================
class TestVerticesToPositional:
    def _layer(self):
        return line_skeleton_cell().skeleton

    @pytest.mark.parametrize(
        "vertices,expected",
        [
            (100, 0),  # scalar index -> scalar positional
            (104, 4),
            ([100, 102, 104], [0, 2, 4]),  # list of indices
            (np.array([104, 100]), [4, 0]),  # ndarray, order preserved
        ],
    )
    def test_index_input_maps_to_positional(self, vertices, expected):
        layer = self._layer()
        out, is_pos = layer._vertices_to_positional(vertices, as_positional=False)
        np.testing.assert_array_equal(out, expected)

    def test_scalar_index_stays_scalar(self):
        layer = self._layer()
        out, _ = layer._vertices_to_positional(104, as_positional=False)
        assert np.ndim(out) == 0
        assert out == 4

    def test_scalar_and_singleton_agree(self):
        layer = self._layer()
        scalar, _ = layer._vertices_to_positional(102, as_positional=False)
        arr, _ = layer._vertices_to_positional(np.array([102]), as_positional=False)
        assert scalar == arr[0]

    def test_positional_input_passthrough(self):
        layer = self._layer()
        out, is_pos = layer._vertices_to_positional(
            np.array([0, 2, 4]), as_positional=True
        )
        np.testing.assert_array_equal(out, [0, 2, 4])
        assert is_pos is True

    def test_none_selects_all(self):
        layer = self._layer()
        out, is_pos = layer._vertices_to_positional(None, as_positional=False)
        np.testing.assert_array_equal(out, np.arange(5))
        assert is_pos is True

    def test_bool_mask_becomes_positional(self):
        layer = self._layer()
        mask = np.array([True, False, True, False, True])
        out, is_pos = layer._vertices_to_positional(mask, as_positional=False)
        np.testing.assert_array_equal(out, [0, 2, 4])
        assert is_pos is True

    def test_bool_mask_wrong_length_raises(self):
        layer = self._layer()
        with pytest.raises(ValueError):
            layer._vertices_to_positional(np.array([True, False]), as_positional=False)


# ===========================================================================
# Rec 3 (cont.): the public distance methods over equivalent input shapes
# ===========================================================================
class TestDistanceMethodsInputShapes:
    def _layer(self):
        return line_skeleton_cell().skeleton

    @pytest.mark.parametrize("method", ["distance_to_root", "hops_to_root"])
    def test_scalar_matches_singleton_array(self, method):
        layer = self._layer()
        fn = getattr(layer, method)
        assert fn(104) == pytest.approx(fn(np.array([104]))[0])

    def test_distance_to_root_index_vs_positional_agree(self):
        layer = self._layer()
        by_index = layer.distance_to_root(np.array([104]))[0]
        by_pos = layer.distance_to_root(np.array([4]), as_positional=True)[0]
        assert by_index == pytest.approx(by_pos)

    def test_distance_between_scalar_and_array_agree(self):
        layer = self._layer()
        scalar = np.asarray(layer.distance_between(100, 104)).item()
        arr = layer.distance_between(np.array([100]), np.array([104]))
        assert scalar == pytest.approx(np.asarray(arr).ravel()[0])


# ===========================================================================
# Rec 4: cache stays consistent with geometry after every mutation
# ===========================================================================
class TestCacheConsistencyAfterMutation:
    def test_consistent_on_fresh_skeleton(self):
        layer = line_skeleton_cell().skeleton
        assert_weighted_csgraph_consistent(layer)

    def test_consistent_after_inplace_transform(self):
        cell = line_skeleton_cell()
        _ = cell.skeleton.csgraph  # prime the cache
        cell.transform(lambda a: a * 7.0, inplace=True)
        assert_weighted_csgraph_consistent(cell.skeleton)
        # base graph tracks the transform too (unmasked skeleton).
        assert cell.skeleton.base_csgraph.sum() == pytest.approx(
            cell.skeleton.csgraph.sum()
        )

    def test_consistent_after_copy_transform(self):
        cell = line_skeleton_cell()
        _ = cell.skeleton.csgraph
        moved = cell.transform(lambda a: a * 3.0)
        assert_weighted_csgraph_consistent(moved.skeleton)

    def test_consistent_after_reroot(self):
        cell = line_skeleton_cell()
        _ = cell.skeleton.csgraph
        cell.skeleton.reroot(104)
        # reroot doesn't move vertices, so the graph is unchanged but must
        # still be internally consistent, and the binary base graph survives.
        assert_weighted_csgraph_consistent(cell.skeleton)
        assert "base_csgraph_binary" in cell.skeleton._base_properties

    def test_mesh_cache_consistent_after_transform(self):
        cell = full_cell()
        _ = cell.mesh.csgraph
        area_before = cell.mesh.surface_area()
        cell.transform(lambda a: a * 2.0, inplace=True)
        # trimesh + csgraph caches both refresh.
        assert cell.mesh.surface_area() == pytest.approx(area_before * 4.0)


# ===========================================================================
# Rec 4 (cont.): roundtrip / plumbing per layer family
# ===========================================================================
class TestRoundtrips:
    def test_copy_is_independent(self):
        cell = full_cell()
        copied = cell.copy()
        assert copied is not cell
        assert copied._morphsync is not cell._morphsync
        assert copied.skeleton.n_vertices == cell.skeleton.n_vertices
        assert copied.mesh.n_vertices == cell.mesh.n_vertices
        # Mutating the copy leaves the original untouched.
        copied.transform(lambda a: a + 50.0, inplace=True)
        np.testing.assert_allclose(cell.skeleton.vertices[:, 0], [0, 1, 2, 3, 4])

    def test_apply_mask_preserves_original(self):
        # Skeleton-only cell: masking a multi-layer cell would require links to
        # propagate through, which is a separate concern from this plumbing.
        cell = line_skeleton_cell()
        mask = np.array([True, True, True, False, False])
        masked = cell.apply_mask("skeleton", mask)
        assert masked.skeleton.n_vertices == 3
        # Original is untouched by the (non-destructive) mask.
        assert cell.skeleton.n_vertices == 5

    def test_save_load_roundtrip(self):
        cell = full_cell()
        with tempfile.NamedTemporaryFile(suffix=".osy", delete=True) as tmp:
            file_io.save_cell(cell, tmp.name, allow_overwrite=True)
            loaded = file_io.load_cell(tmp.name)
        # Every layer family survives the roundtrip with matching geometry.
        np.testing.assert_allclose(loaded.skeleton.vertices, cell.skeleton.vertices)
        np.testing.assert_array_equal(
            loaded.skeleton.edges_positional, cell.skeleton.edges_positional
        )
        np.testing.assert_allclose(loaded.mesh.vertices, cell.mesh.vertices)
        np.testing.assert_array_equal(
            loaded.mesh.faces_positional, cell.mesh.faces_positional
        )
        np.testing.assert_allclose(
            np.sort(loaded.skeleton.get_feature("radius")),
            np.sort(cell.skeleton.get_feature("radius")),
        )
        assert loaded.annotations["syn"].n_vertices == 2


# ===========================================================================
# Rec 5: per-vertex accessors stay aligned with the layer's own vertices,
# including under a mask.
#
# ``distance_to_root``/``hops_to_root`` compute against the *base* (unmasked)
# graph so a masked axon still measures back to the original soma. Their
# dijkstra results are therefore in base positional space, and every vertex
# selection must be converted into that space. Mixing the two spaces produced
# values silently attached to the wrong vertices.
#
# The mask below is deliberately NOT a prefix: every pre-existing masking test
# kept vertices 0..k, where slicing a base-space array to the masked length
# happens to be correct. That is exactly why the bug survived.
# ===========================================================================
BRANCH_IDX = np.array([100, 101, 102, 103, 104])


def branched_skeleton_layer():
    """Y-shaped skeleton, unit spacing, non-positional vertex index.

        100 --- 101 --- 102 --- 104
                 |
                103

    Distances to root (100): [0, 1, 1+sqrt(2), 1+sqrt(2), 1+sqrt(2)+1]
    Hops to root:            [0, 1, 2, 2, 3]
    """
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # 100 root
            [1.0, 0.0, 0.0],  # 101 branch point
            [2.0, 1.0, 0.0],  # 102
            [2.0, -1.0, 0.0],  # 103 tip
            [3.0, 1.0, 0.0],  # 104 tip
        ]
    )
    df = pd.DataFrame(verts, columns=SPATIAL, index=BRANCH_IDX)
    edges = np.array([[101, 100], [102, 101], [103, 101], [104, 102]])
    cell = Cell()
    cell.add_skeleton(vertices=df, edges=edges, spatial_columns=SPATIAL, root=100)
    return cell.skeleton


def masked_branched_layer():
    """Keep {101, 102, 104}: drops the root (100) and an interior tip (103).

    Non-prefix, so masked positional order differs from base positional order,
    and ``base_root`` is masked out entirely.
    """
    layer = branched_skeleton_layer()
    return layer.apply_mask(np.array([101, 102, 104]), self_only=True)


@pytest.fixture(params=[False, True], ids=["unmasked", "masked"])
def branched_layer(request):
    return masked_branched_layer() if request.param else branched_skeleton_layer()


ROOT_METHODS = ["distance_to_root", "hops_to_root"]


class TestPerVertexAccessorAlignment:
    """Every per-vertex accessor must return one value per vertex of the
    *current* layer, in the current positional order, regardless of which
    equivalent call form is used."""

    def test_mask_is_not_a_prefix(self):
        # Guards the guard: if this ever became a prefix mask, the alignment
        # tests below would stop discriminating.
        masked = masked_branched_layer()
        assert not np.array_equal(masked.vertex_index, np.arange(masked.n_vertices))
        np.testing.assert_array_equal(masked.vertex_index, [101, 102, 104])
        assert masked.root is None  # base root was masked out

    @pytest.mark.parametrize("method", ROOT_METHODS)
    def test_call_forms_agree(self, branched_layer, method):
        fn = getattr(branched_layer, method)
        n = branched_layer.n_vertices
        no_arg = fn()
        by_pos = fn(np.arange(n), as_positional=True)
        by_id = fn(branched_layer.vertex_index)

        assert len(no_arg) == n
        np.testing.assert_allclose(no_arg, by_pos)
        np.testing.assert_allclose(no_arg, by_id)

    @pytest.mark.parametrize("method", ROOT_METHODS)
    def test_bool_mask_and_scalar_forms_agree(self, branched_layer, method):
        fn = getattr(branched_layer, method)
        n = branched_layer.n_vertices
        full = fn()

        # A boolean array is implicitly positional whatever the flag says.
        sel = np.zeros(n, dtype=bool)
        sel[-1] = True
        np.testing.assert_allclose(fn(sel), full[[n - 1]])

        # Scalar in -> scalar out, in both index spaces.
        last_id = branched_layer.vertex_index[-1]
        assert np.ndim(fn(last_id)) == 0
        assert fn(last_id) == pytest.approx(full[-1])
        assert fn(n - 1, as_positional=True) == pytest.approx(full[-1])

    @pytest.mark.parametrize("method", ROOT_METHODS)
    def test_subset_matches_full(self, branched_layer, method):
        fn = getattr(branched_layer, method)
        full = fn()
        order = [2, 0] if branched_layer.n_vertices > 2 else [1, 0]
        np.testing.assert_allclose(fn(np.array(order), as_positional=True), full[order])

    def test_ground_truth_unmasked(self):
        layer = branched_skeleton_layer()
        d = 1.0 + np.sqrt(2)
        np.testing.assert_allclose(layer.distance_to_root(), [0.0, 1.0, d, d, d + 1.0])
        np.testing.assert_allclose(layer.hops_to_root(), [0, 1, 2, 2, 3])

    def test_ground_truth_masked_measures_through_base_graph(self):
        # Self-consistency alone would pass if all three call forms were wrong
        # in the same way, so pin the actual values. The root (100) is masked
        # out, yet distances are still measured back to it through the base
        # graph -- that is the documented purpose of these two methods.
        masked = masked_branched_layer()
        d = 1.0 + np.sqrt(2)
        np.testing.assert_allclose(masked.distance_to_root(), [1.0, d, d + 1.0])
        np.testing.assert_allclose(masked.hops_to_root(), [1, 2, 3])
        assert np.all(masked.distance_to_root() >= 0)

    @pytest.mark.parametrize(
        "accessor",
        ["parent_node_array", "half_edge_length", "segment_map"],
    )
    def test_per_vertex_property_length(self, branched_layer, accessor):
        assert len(getattr(branched_layer, accessor)) == branched_layer.n_vertices

    @pytest.mark.parametrize("fn_name", ["strahler_number", "branch_order"])
    def test_algorithm_per_vertex_length(self, branched_layer, fn_name):
        from ossify import algorithms

        assert len(getattr(algorithms, fn_name)(branched_layer)) == (
            branched_layer.n_vertices
        )

    @pytest.mark.parametrize("method", ROOT_METHODS)
    def test_empty_selection(self, branched_layer, method):
        # An empty python list arrives as float64; passing that straight to
        # fastremap yields a float result that cannot index an array.
        fn = getattr(branched_layer, method)
        for empty in ([], np.array([], dtype=int)):
            assert len(fn(empty)) == 0
            assert len(fn(empty, as_positional=True)) == 0


def scrambled_tree_layer():
    """Tree whose stored vertex order is deliberately NOT topological.

        100 -> 101 -> 102 -> 103
                 '-> 104 -> 105
               102 -> 106

    Vertex rows are stored scrambled, so positional order carries no topological
    information. With an id order that happens to match the tree (the usual
    fixture shape), a base/masked mix-up shifts every hop count uniformly and so
    preserves ``argsort`` -- which hides the downstream corruption entirely.
    """
    parent = {1: 0, 2: 1, 3: 2, 4: 1, 5: 4, 6: 2}
    order = [3, 0, 5, 1, 6, 2, 4]
    ids = np.array([100 + i for i in order])
    verts = np.zeros((len(order), 3))
    for pos, v in enumerate(order):
        verts[pos] = [float(v), 0.0, 0.0]
    df = pd.DataFrame(verts, columns=SPATIAL, index=ids)
    edges = np.array([[100 + c, 100 + p] for c, p in parent.items()])
    cell = Cell()
    cell.add_skeleton(vertices=df, edges=edges, spatial_columns=SPATIAL, root=100)
    return cell.skeleton


def _brute_force_subtree_sizes(parent_node_array):
    n = len(parent_node_array)
    sizes = np.zeros(n)
    for i in range(n):
        for j in range(n):
            c = j
            while c >= 0:
                if c == i:
                    sizes[i] += 1
                    break
                c = parent_node_array[c]
    return sizes


class TestMaskedTraversalOrder:
    """``hops_to_root`` is used as a topological sort key by ``build_segments``
    and by the aggregate algorithms. If it is misaligned the ordering invariant
    silently breaks and the aggregates return wrong numbers."""

    @staticmethod
    def _masked():
        layer = scrambled_tree_layer()
        # Drop the root; keep everything else, in scrambled storage order.
        return layer.apply_mask(
            np.array([101, 102, 103, 104, 105, 106]), self_only=True
        )

    def test_parent_always_has_fewer_hops_than_child(self):
        masked = self._masked()
        hops = masked.hops_to_root(as_positional=True)
        pna = masked.parent_node_array
        has_parent = pna >= 0
        assert np.all(hops[pna[has_parent]] < hops[has_parent])

    def test_subtree_aggregate_matches_brute_force(self):
        from ossify import algorithms

        masked = self._masked()
        got = algorithms.subtree_aggregate(
            masked, np.ones(masked.n_vertices), agg="sum"
        )
        np.testing.assert_allclose(
            got, _brute_force_subtree_sizes(masked.parent_node_array)
        )

    def test_segments_partition_all_vertices(self):
        masked = self._masked()
        covered = np.concatenate(masked.segments_positional)
        np.testing.assert_array_equal(np.sort(covered), np.arange(masked.n_vertices))
        # Each segment runs distal -> proximal, i.e. hops strictly decrease.
        hops = masked.hops_to_root(as_positional=True)
        for seg in masked.segments_positional:
            assert np.all(np.diff(hops[seg]) < 0)


class TestPointCloudDistanceToRoot:
    """``PointCloudLayer.distance_to_root`` is the same family of accessor and
    owes the same guarantee: all equivalent call forms agree."""

    @staticmethod
    def _cell():
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        df = pd.DataFrame(verts, columns=SPATIAL, index=SKEL_IDX)
        edges = np.array([[101, 100], [102, 101], [103, 102], [104, 103]])
        cell = Cell()
        cell.add_skeleton(vertices=df, edges=edges, spatial_columns=SPATIAL, root=100)
        pts = pd.DataFrame(
            np.c_[[1.0, 3.0, 4.0], np.zeros((3, 2))],
            columns=SPATIAL,
            index=[400, 401, 402],
        )
        cell.add_point_annotations(
            name="pts",
            vertices=pts,
            spatial_columns=SPATIAL,
            linkage=Link(mapping={400: 101, 401: 103, 402: 104}, target="skeleton"),
        )
        return cell

    def test_call_forms_agree(self):
        pc = self._cell().annotations["pts"]
        n = pc.n_vertices
        no_arg = pc.distance_to_root()
        # Regression: this pre-filled vertex ids and then consumed them as
        # positional indices, raising IndexError.
        all_positional = pc.distance_to_root(as_positional=True)
        by_id = pc.distance_to_root(pc.vertex_index)
        by_pos = pc.distance_to_root(np.arange(n), as_positional=True)

        assert len(no_arg) == n
        np.testing.assert_allclose(no_arg, [1.0, 3.0, 4.0])
        np.testing.assert_allclose(no_arg, all_positional)
        np.testing.assert_allclose(no_arg, by_id)
        np.testing.assert_allclose(no_arg, by_pos)


class TestMaskedMatchesStandalone:
    """A masked skeleton and the equivalent standalone skeleton must agree on
    everything that depends only on the surviving topology."""

    @staticmethod
    def _standalone():
        """The masked subtree {101, 102, 104} rebuilt from scratch."""
        verts = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
        df = pd.DataFrame(verts, columns=SPATIAL, index=np.array([101, 102, 104]))
        edges = np.array([[102, 101], [104, 102]])
        cell = Cell()
        cell.add_skeleton(vertices=df, edges=edges, spatial_columns=SPATIAL, root=101)
        return cell.skeleton

    def test_segments_ordering(self):
        masked = masked_branched_layer()
        standalone = self._standalone()
        masked_segs = [list(s) for s in masked.segments]
        standalone_segs = [list(s) for s in standalone.segments]
        assert sorted(masked_segs) == sorted(standalone_segs)

    @pytest.mark.parametrize("fn_name", ["subtree_aggregate", "path_to_root_aggregate"])
    def test_aggregates_match_standalone(self, fn_name):
        from ossify import algorithms

        masked = masked_branched_layer()
        standalone = self._standalone()
        # Values keyed by vertex id so the two layers are compared like-for-like.
        vals = {101: 1.0, 102: 2.0, 104: 4.0}
        fn = getattr(algorithms, fn_name)
        got = fn(masked, np.array([vals[v] for v in masked.vertex_index]))
        want = fn(standalone, np.array([vals[v] for v in standalone.vertex_index]))
        got_by_id = dict(zip(masked.vertex_index, got))
        want_by_id = dict(zip(standalone.vertex_index, want))
        assert got_by_id == pytest.approx(want_by_id)
