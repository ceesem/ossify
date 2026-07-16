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
