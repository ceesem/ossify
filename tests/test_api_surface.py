"""Contract tests across the public API surface.

Most of ossify's public methods were never called by a test. That is how
``cover_paths_specific`` came to raise ``AttributeError`` on every call, and how
``segments_plus`` came to return coordinates from a property documented to
return vertex indices -- both public, both in the user guide, neither exercised.

These tests are deliberately shallow. They call each public member once and
assert the contract its name implies, which is what those two bugs violated:

* an ``X``/``X_positional`` pair must describe the same vertices,
* a positional result must be usable to index this layer's arrays,
* a vertex-index result must be drawn from ``vertex_index``,
* a per-vertex array must have one entry per vertex.

Depth belongs in the per-feature test modules; the job here is that no public
member is entirely unexercised.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from ossify import Cell, GraphLayer, MeshLayer, PointCloudLayer, SkeletonLayer

CORE_CLASSES = [Cell, SkeletonLayer, GraphLayer, MeshLayer, PointCloudLayer]


def _public_members(cls):
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or isinstance(member, property):
            yield name


def _index_pairs(cls):
    """``X`` where ``X_positional`` also exists."""
    names = {n for n in dir(cls) if not n.startswith("_")}
    return sorted(n for n in names if f"{n}_positional" in names)


# ---------------------------------------------------------------------------
# The invariant that both known bugs violated
# ---------------------------------------------------------------------------
SKELETON_PAIRS = _index_pairs(SkeletonLayer)


@pytest.mark.parametrize("name", SKELETON_PAIRS)
def test_index_and_positional_forms_describe_the_same_vertices(full_cell, name):
    """``layer.X`` must equal ``vertex_index[layer.X_positional]``.

    ``segments_plus`` failed this: it returned an (n, 3) float array of
    coordinates while ``segments_plus_positional`` returned indices.
    """
    layer = full_cell.skeleton
    by_index = getattr(layer, name)
    by_position = getattr(layer, f"{name}_positional")

    if by_index is None or by_position is None:  # e.g. an unset root
        assert by_index is None and by_position is None
        return

    # Ragged lists of paths/segments come first: np.ndim raises on them.
    if isinstance(by_index, list):
        assert len(by_index) == len(by_position)
        for ids, positions in zip(by_index, by_position):
            np.testing.assert_array_equal(ids, layer.vertex_index[positions])
        return

    if np.isscalar(by_index) or np.ndim(by_index) == 0:
        assert layer.vertex_index[by_position] == by_index
        return

    np.testing.assert_array_equal(by_index, layer.vertex_index[by_position])


@pytest.mark.parametrize("name", SKELETON_PAIRS)
def test_positional_forms_are_usable_as_indices(full_cell, name):
    """A ``_positional`` result must be able to index this layer's arrays."""
    layer = full_cell.skeleton
    value = getattr(layer, f"{name}_positional")
    if value is None:
        return
    flat = (
        np.concatenate([np.atleast_1d(v) for v in value])
        if isinstance(value, list)
        else np.atleast_1d(value)
    )
    if flat.size == 0:
        return
    assert np.issubdtype(np.asarray(flat).dtype, np.integer)
    assert flat.min() >= 0
    assert flat.max() < layer.n_vertices
    layer.vertices[flat]  # would raise if these were not valid positions


@pytest.mark.parametrize("name", SKELETON_PAIRS)
def test_index_forms_are_drawn_from_the_vertex_index(full_cell, name):
    layer = full_cell.skeleton
    value = getattr(layer, name)
    if value is None:
        return
    flat = (
        np.concatenate([np.atleast_1d(v) for v in value])
        if isinstance(value, list)
        else np.atleast_1d(value)
    )
    if flat.size == 0:
        return
    assert np.all(np.isin(flat, layer.vertex_index)), (
        f"{name} returned values that are not vertices of this layer"
    )


# ---------------------------------------------------------------------------
# Per-vertex arrays
# ---------------------------------------------------------------------------
PER_VERTEX = {
    "skeleton": ["parent_node_array", "half_edge_length", "segment_map"],
    "graph": ["half_edge_length"],
}


@pytest.mark.parametrize(
    "layer_name,attr",
    [(ln, a) for ln, attrs in PER_VERTEX.items() for a in attrs],
)
def test_per_vertex_arrays_have_one_entry_per_vertex(full_cell, layer_name, attr):
    layer = getattr(full_cell, layer_name)
    assert len(getattr(layer, attr)) == layer.n_vertices


# ---------------------------------------------------------------------------
# Nothing public is entirely unexercised
# ---------------------------------------------------------------------------
def test_every_public_member_is_reachable(full_cell):
    """Touch every public property, and call every zero-argument method.

    This is the cheap half of the sweep: it catches members that raise on any
    call at all, which is the failure mode ``cover_paths_specific`` had.
    """
    targets = {
        Cell: full_cell,
        SkeletonLayer: full_cell.skeleton,
        GraphLayer: full_cell.graph,
        MeshLayer: full_cell.mesh,
        PointCloudLayer: full_cell.annotations.syn,
    }
    failures = []
    for cls, obj in targets.items():
        for name in _public_members(cls):
            member = getattr(cls, name, None)
            try:
                if isinstance(member, property):
                    getattr(obj, name)
                else:
                    sig = inspect.signature(member)
                    required = [
                        p
                        for p in list(sig.parameters.values())[1:]
                        if p.default is inspect.Parameter.empty
                        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                    ]
                    if required:
                        continue  # needs arguments; covered by targeted tests
                    getattr(obj, name)()
            except Exception as exc:  # noqa: BLE001 - collected and reported
                failures.append(f"{cls.__name__}.{name}: {type(exc).__name__}: {exc}")
    assert not failures, "public members raised:\n  " + "\n  ".join(failures)


# ---------------------------------------------------------------------------
# Members the sweep above cannot reach, because they take required arguments
# ---------------------------------------------------------------------------
class TestMembersRequiringArguments:
    """One real call each, asserting the contract the name implies.

    These are the remainder of the never-tested surface. The sweep calls only
    zero-argument members, so without these they stay unexercised.
    """

    def test_cell_add_layer_rejects_a_foreign_layer(self, full_cell):
        from ossify import PointCloudLayer

        # A layer constructed on its own carries its own MorphSync, which the
        # cell cannot adopt; add_point_layer is the supported route.
        foreign = PointCloudLayer(
            "extra",
            np.array([[0.0, 0, 0], [1, 0, 0]]),
            spatial_columns=["x", "y", "z"],
        )
        with pytest.raises(ValueError, match="Incompatible MorphSync"):
            full_cell.add_layer(foreign)

        full_cell.add_point_layer(
            name="extra",
            vertices=np.array([[0.0, 0, 0], [1, 0, 0]]),
            spatial_columns=["x", "y", "z"],
        )
        assert "extra" in full_cell.layers.names

    def test_cell_get_features_maps_across_layers(self, full_cell):
        out = full_cell.get_features(
            "radius", target_layer="graph", source_layers="skeleton", agg="mean"
        )
        assert len(out) == full_cell.graph.n_vertices

    def test_pointcloud_filter(self, full_cell):
        syn = full_cell.annotations.syn
        skel = full_cell.skeleton
        # The mask belongs to the layer named in the second argument.
        mask = np.isin(skel.vertex_index, skel.end_points)
        out = syn.filter(mask, "skeleton")
        # Only the annotations linked to those skeleton vertices survive.
        assert len(out) <= syn.n_vertices

    def test_graph_proximity_mapping(self, full_cell):
        out = full_cell.graph.proximity_mapping(distance_threshold=1.5)
        assert out is not None

    def test_skeleton_cut_graph_splits_components(self, full_cell):
        from scipy.sparse.csgraph import connected_components

        skel = full_cell.skeleton
        whole, _ = connected_components(skel.csgraph_binary, directed=False)
        cut, _ = connected_components(
            skel.cut_graph(skel.branch_points), directed=False
        )
        assert cut > whole

    def test_skeleton_downstream_vertices(self, full_cell):
        skel = full_cell.skeleton
        branch = skel.branch_points[0]
        exclusive = skel.downstream_vertices(branch)
        inclusive = skel.downstream_vertices(branch, inclusive=True)
        assert branch in inclusive
        assert branch not in exclusive
        assert set(exclusive) < set(inclusive)
        assert np.all(np.isin(inclusive, skel.vertex_index))

    def test_skeleton_expand_to_segment(self, full_cell):
        skel = full_cell.skeleton
        tips = skel.end_points[:2]
        segments = skel.expand_to_segment(tips)
        assert len(segments) == len(tips)
        for tip, segment in zip(tips, segments):
            assert tip in segment

    def test_skeleton_segments_capped_respects_the_cap(self, full_cell):
        skel = full_cell.skeleton
        capped, seg_map = skel.segments_capped(1.0, positional=True)
        # positional=True must give positions into this layer.
        covered = np.concatenate(capped)
        assert covered.max() < skel.n_vertices
        # Every vertex still appears exactly once across the capped segments.
        assert len(covered) == skel.n_vertices
        # Capping can only split, never merge.
        assert len(capped) >= len(skel.segments_positional)
        # And the index form is the same segments, mapped through vertex_index.
        by_index, _ = skel.segments_capped(1.0, positional=False)
        for positions, ids in zip(capped, by_index):
            np.testing.assert_array_equal(skel.vertex_index[positions], ids)


class TestAsPositionalConvention:
    """``as_positional`` means the same thing everywhere it appears.

    ``MeshLayer.surface_area`` used to be the lone exception, defaulting to
    True where the other thirteen defaulted to False, so the same argument
    named the opposite index space depending on which method you called.
    """

    def test_every_as_positional_defaults_to_false(self):
        import ossify
        from ossify import data_layers

        offenders = []
        for module in (data_layers, ossify.base, ossify.algorithms):
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                members = vars(obj).values() if inspect.isclass(obj) else [obj]
                for member in members:
                    if not inspect.isfunction(member):
                        continue
                    try:
                        param = inspect.signature(member).parameters.get(
                            "as_positional"
                        )
                    except (ValueError, TypeError):
                        continue
                    if param is not None and param.default is True:
                        offenders.append(f"{obj_name}.{member.__name__}")
        assert not offenders, f"as_positional defaults to True in: {offenders}"

    def test_surface_area_default_reads_vertex_indices(self, full_cell):
        mesh = full_cell.mesh
        # This mesh is indexed 300.. so the two spaces are distinguishable.
        assert not np.array_equal(mesh.vertex_index, np.arange(mesh.n_vertices))

        positions = np.array([0, 1, 2])
        by_position = mesh.surface_area(vertices=positions, as_positional=True)
        by_index = mesh.surface_area(vertices=mesh.vertex_index[positions])

        assert by_index == pytest.approx(by_position)
        # And the default really is the vertex-index reading.
        assert mesh.surface_area(
            vertices=mesh.vertex_index[positions]
        ) == pytest.approx(by_position)


class TestApplyMaskReturnDeprecation:
    """``apply_mask`` on a linked layer returns a Cell today and a layer from
    1.0, with ``return_cell`` fixing the behaviour either way.

    Layer-in/Cell-out is the single most confusing thing in the API -- it
    prompted a user bug report and appeared wrong on six guide pages -- so the
    default is changing. Until it does, an unset call warns rather than
    silently changing under anyone.
    """

    def test_unset_warns_and_keeps_the_current_behaviour(self, full_cell):
        from ossify import Cell

        keep = full_cell.skeleton.vertex_index[:3]
        with pytest.warns(FutureWarning, match="will return a masked layer"):
            out = full_cell.skeleton.apply_mask(keep)
        assert isinstance(out, Cell)

    def test_explicit_true_returns_a_cell_without_warning(self, full_cell, recwarn):
        from ossify import Cell

        keep = full_cell.skeleton.vertex_index[:3]
        out = full_cell.skeleton.apply_mask(keep, return_cell=True)
        assert isinstance(out, Cell)
        assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]

    def test_explicit_false_returns_the_masked_layer(self, full_cell, recwarn):
        keep = full_cell.skeleton.vertex_index[:3]
        out = full_cell.skeleton.apply_mask(keep, return_cell=False)
        assert isinstance(out, SkeletonLayer)
        assert out.n_vertices == 3
        assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]

    def test_both_forms_describe_the_same_masking(self, full_cell):
        keep = full_cell.skeleton.vertex_index[:3]
        as_cell = full_cell.skeleton.apply_mask(keep, return_cell=True)
        as_layer = full_cell.skeleton.apply_mask(keep, return_cell=False)
        np.testing.assert_array_equal(
            as_cell.skeleton.vertex_index, as_layer.vertex_index
        )

    def test_the_mask_still_propagates_when_a_layer_is_returned(self, full_cell):
        """Returning a layer must not quietly turn into self_only=True."""
        keep = full_cell.skeleton.vertex_index[:3]
        as_layer = full_cell.skeleton.apply_mask(keep, return_cell=False)
        # The returned layer belongs to a masked cell, and its siblings were
        # masked too -- unlike self_only, which leaves them out entirely.
        assert as_layer._cell is not None
        assert as_layer._cell.graph.n_vertices < full_cell.graph.n_vertices

    def test_self_only_returns_a_detached_layer(self, full_cell, recwarn):
        keep = full_cell.skeleton.vertex_index[:3]
        out = full_cell.skeleton.apply_mask(keep, self_only=True)
        assert isinstance(out, SkeletonLayer)
        assert out._cell is None
        # self_only is unambiguous, so it does not warn.
        assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]

    @pytest.mark.parametrize("method", ["mask_context", "mask_out_unmapped"])
    def test_wrappers_forward_the_flag(self, full_cell, method):
        from ossify import Cell

        skel = full_cell.skeleton
        if method == "mask_context":
            with skel.mask_context(skel.vertex_index[:3], return_cell=False) as out:
                assert isinstance(out, SkeletonLayer)
            with skel.mask_context(skel.vertex_index[:3], return_cell=True) as out:
                assert isinstance(out, Cell)
        else:
            assert isinstance(
                skel.mask_out_unmapped(target_layers="graph", return_cell=False),
                SkeletonLayer,
            )
            assert isinstance(
                skel.mask_out_unmapped(target_layers="graph", return_cell=True), Cell
            )


class TestUnmappedVertexDetection:
    """``get_unmapped_vertices`` reports vertices of *this* layer that fail to
    reach the target.

    Source and target were swapped in the underlying ``get_mapping`` call, so
    it returned the target layer's vertices instead. ``mask_out_unmapped``
    then compared this layer's vertices against another layer's ids -- two
    disjoint index spaces on any real cell -- so it silently removed nothing.
    """

    SPATIAL = ["x", "y", "z"]
    VERTS = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

    def _partially_linked_cell(self):
        """A graph covering only the first two of four skeleton vertices."""
        from ossify import Link

        cell = Cell()
        cell.add_skeleton(
            vertices=pd.DataFrame(
                self.VERTS, columns=self.SPATIAL, index=[100, 101, 102, 103]
            ),
            edges=np.array([[101, 100], [102, 101], [103, 102]]),
            spatial_columns=self.SPATIAL,
            root=100,
        )
        cell.add_graph(
            vertices=pd.DataFrame(
                self.VERTS[:2], columns=self.SPATIAL, index=[200, 201]
            ),
            edges=np.array([[201, 200]]),
            spatial_columns=self.SPATIAL,
            linkage=Link(
                mapping=np.array([100, 101]),
                target="skeleton",
                map_value_is_index=True,
            ),
        )
        return cell

    def test_reports_this_layers_vertices(self):
        cell = self._partially_linked_cell()
        unmapped = cell.skeleton.get_unmapped_vertices(target_layers="graph")
        np.testing.assert_array_equal(unmapped, [102, 103])
        # And they are this layer's vertices, not the target's.
        assert np.all(np.isin(unmapped, cell.skeleton.vertex_index))

    def test_fully_mapped_layer_reports_nothing(self):
        cell = self._partially_linked_cell()
        # Every graph vertex does reach the skeleton.
        assert len(cell.graph.get_unmapped_vertices(target_layers="skeleton")) == 0

    def test_mask_out_unmapped_actually_removes_them(self):
        cell = self._partially_linked_cell()
        cleaned = cell.skeleton.mask_out_unmapped(
            target_layers="graph", return_cell=False
        )
        assert cleaned.n_vertices == 2
        np.testing.assert_array_equal(cleaned.vertex_index, [100, 101])


class TestMapIndexMissing:
    """``map_index_to_layer`` raises on unmapped vertices unless given a fill.

    The fill is the caller's to choose, not the library's, because it decides
    the dtype. Vertex ids here are routinely above 2**53, where a float64 round
    trip silently changes them -- so a library that quietly returned NaN would
    corrupt every id it handed back. An integer fill keeps the array integral.
    """

    SPATIAL = ["x", "y", "z"]
    VERTS = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    # Above 2**53, so a float64 round trip is lossy.
    BIG = np.array([173194386090230235, 173194386090230236], dtype=np.int64)

    def _cell(self):
        from ossify import Link

        cell = Cell()
        cell.add_skeleton(
            vertices=pd.DataFrame(
                self.VERTS, columns=self.SPATIAL, index=[100, 101, 102, 103]
            ),
            edges=np.array([[101, 100], [102, 101], [103, 102]]),
            spatial_columns=self.SPATIAL,
            root=100,
        )
        cell.add_graph(
            vertices=pd.DataFrame(self.VERTS[:2], columns=self.SPATIAL, index=self.BIG),
            edges=np.array([[self.BIG[1], self.BIG[0]]]),
            spatial_columns=self.SPATIAL,
            linkage=Link(
                mapping=np.array([100, 101]),
                target="skeleton",
                map_value_is_index=True,
            ),
        )
        return cell

    def test_default_raises_and_names_the_vertices(self):
        skel = self._cell().skeleton
        with pytest.raises(KeyError, match=r"102"):
            skel.map_index_to_layer("graph")

    def test_integer_fill_keeps_the_dtype_and_the_ids(self):
        skel = self._cell().skeleton
        out = skel.map_index_to_layer("graph", missing=-1)
        assert out.dtype == np.int64
        # The ids survive exactly -- this is what a float64 route would break.
        assert out[0] == self.BIG[0]
        assert out[1] == self.BIG[1]
        np.testing.assert_array_equal(out[2:], [-1, -1])
        # And the result is still a normal numpy array.
        np.testing.assert_array_equal(
            np.isin(out, self.BIG), [True, True, False, False]
        )

    def test_a_float_fill_widens_as_documented(self):
        skel = self._cell().skeleton
        out = skel.map_index_to_layer("graph", missing=np.nan)
        assert out.dtype == np.float64
        # Documented consequence of asking for a float fill, not a silent
        # default: above 2**53 the id no longer round-trips.
        assert int(out[0]) != self.BIG[0]

    def test_fully_mapped_selection_is_unaffected(self):
        skel = self._cell().skeleton
        for kwargs in ({}, {"missing": -1}):
            out = skel.map_index_to_layer(
                "graph", source_index=np.array([100, 101]), **kwargs
            )
            np.testing.assert_array_equal(out, self.BIG)
            assert out.dtype == np.int64

    def test_an_unknown_string_is_rejected(self):
        skel = self._cell().skeleton
        with pytest.raises(ValueError, match="must be 'raise' or a fill value"):
            skel.map_index_to_layer("graph", missing="drop")
