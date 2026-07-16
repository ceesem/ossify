"""Regression tests for identifier canonicalization.

Ossify identifiers are nominally uint64 but every value we support fits in the
nonnegative int64 range. Upstream sources hand us the same ID as a Python int,
a NumPy signed/unsigned integer, or inside a pandas container. Joining two
differently-typed key spaces can make pandas coerce keys through ``float64``,
which silently collapses distinct IDs above ``2**53``.

These tests pin down the canonicalization helper and the places it is applied:
layer node indexes, link endpoint columns, and -- crucially -- both sides of
every mapping join, so that legacy ``.osy`` files with mixed int64/uint64 link
dtypes still map exactly.
"""

import io

import numpy as np
import pandas as pd
import pytest

from ossify import Cell, Link, file_io
from ossify._sync.base import canonicalize_ids
from ossify._sync.morph import MorphSync

# Two distinct IDs separated by 13, both above 2**53. They become equal the
# instant either passes through a float64 representation.
ID_A = 167620961649033311
ID_B = 167620961649033324
INT64_MAX = int(np.iinfo(np.int64).max)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _two_layer_morphsync(graph_dtype="uint64", syn_dtype="int64"):
    """MorphSync with a graph and post_syn layer joined by a hand-built link.

    The link is injected directly into ``links`` so it bypasses ``add_link``'s
    canonicalization -- this reproduces a legacy on-disk object whose endpoint
    columns disagree on signedness. Synapse 501 -> graph ID_A, 502 -> ID_B.
    """
    ms = MorphSync()
    gdf = pd.DataFrame(
        {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        index=pd.Index([ID_A, ID_B], name="vid"),
    )
    ms.add_graph("graph", (gdf, np.empty((0, 2), dtype=int)))
    sdf = pd.DataFrame(
        {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        index=pd.Index([501, 502], name="id"),
    )
    ms.add_points("post_syn", sdf)
    link = pd.DataFrame(
        {
            "post_syn": pd.Series([501, 502], dtype=syn_dtype),
            "graph": pd.Series([ID_A, ID_B], dtype=graph_dtype),
        }
    )
    ms.links[("post_syn", "graph")] = link
    ms.links[("graph", "post_syn")] = link
    return ms


def _transitive_cell():
    """skeleton <-> graph <-> post_syn cell with a colliding axonal ID.

    - graph vertex ID_A is dendritic (retained), ID_B is axonal (removed);
      the two differ by 13 and are both above 2**53.
    - skeleton vertex 0 is dendrite, 1 is axon; graph maps 1:1 onto them.
    - one postsynapse (id 9001) is genuinely connected only to the axonal
      graph vertex ID_B.
    """
    gdf = pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "vid": [ID_A, ID_B]}
    )
    cell = Cell(name="transitive")
    cell.add_graph(
        vertices=gdf,
        edges=np.array([[0, 1]]),
        vertex_index="vid",
        spatial_columns=["x", "y", "z"],
    )
    cell.add_skeleton(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edges=np.array([[1, 0]]),
        root=0,
        linkage=Link(
            mapping=np.array([0, 1]), source="graph", map_value_is_index=False
        ),
    )
    sdf = pd.DataFrame(
        {"cx": [5.0], "cy": [0.0], "cz": [0.0], "id": [9001], "m": [ID_B]}
    )
    cell.add_point_annotations(
        "post_syn",
        vertices=sdf,
        spatial_columns=["cx", "cy", "cz"],
        vertex_index="id",
        linkage=Link(mapping="m", target="graph"),
    )
    return cell


def _annotation_index(cell, name):
    for a in cell.annotations:
        if a.name == name:
            return list(a.vertex_index)
    return None


def _assert_link_integrity(morphsync):
    """Every retained link endpoint must exist in its retained layer.

    Implemented as a *test* assertion, not a runtime check inside the library.
    """
    for (src, tgt), df in morphsync.links.items():
        if src not in morphsync.layers or tgt not in morphsync.layers:
            continue
        src_ids = set(np.asarray(morphsync.layers[src].nodes_index).tolist())
        tgt_ids = set(np.asarray(morphsync.layers[tgt].nodes_index).tolist())
        assert set(np.asarray(df[src]).tolist()) <= src_ids, (
            f"link {src}->{tgt} references {src} nodes absent from the layer"
        )
        assert set(np.asarray(df[tgt]).tolist()) <= tgt_ids, (
            f"link {src}->{tgt} references {tgt} nodes absent from the layer"
        )


# ---------------------------------------------------------------------------
# 1. The exact >2**53 mixed-dtype collision
# ---------------------------------------------------------------------------
class TestCollision:
    def test_raw_pandas_join_would_collide(self):
        """Document the underlying pandas hazard the fix defends against."""
        left = pd.DataFrame({"graph": pd.Series([ID_A], dtype="Int64")})
        right = pd.Series(
            [231994998],
            index=pd.Index([ID_B], dtype="uint64"),
            name="post_syn",
            dtype="Int64",
        )
        # Int64 key column joined against a uint64 index: pandas matches two IDs
        # that differ by 13. This is exactly what canonicalization prevents.
        collided = left.join(right, on="graph")["post_syn"].notna().any()
        assert collided, "expected the mixed-dtype join to mis-match (env baseline)"

    def test_mapping_never_matches_distinct_ids(self):
        ms = _two_layer_morphsync(graph_dtype="uint64", syn_dtype="int64")
        mp = ms.get_mapping_paths("post_syn", "graph")
        assert int(mp.loc[501, "graph"]) == ID_A
        assert int(mp.loc[502, "graph"]) == ID_B
        # ID_A and ID_B never cross-match.
        assert int(mp.loc[501, "graph"]) != ID_B
        assert int(mp.loc[502, "graph"]) != ID_A

    def test_unmapped_source_stays_unmatched(self):
        """A source that maps to no graph vertex must not collide onto one."""
        ms = _two_layer_morphsync(graph_dtype="uint64", syn_dtype="int64")
        # Add a synapse whose ID collides-through-float with a real graph ID but
        # has no link row at all.
        ms.add_points(
            "orphan",
            pd.DataFrame(
                {"x": [0.0], "y": [0.0], "z": [0.0]},
                index=pd.Index([9999], name="id"),
            ),
        )
        mp = ms.get_mapping_paths(
            "post_syn", "graph", source_index=np.array([501, 502])
        )
        assert set(mp["graph"].dropna().astype("int64")) == {ID_A, ID_B}


# ---------------------------------------------------------------------------
# 2. All accepted scalar / container forms
# ---------------------------------------------------------------------------
class TestAcceptedForms:
    @pytest.mark.parametrize(
        "value",
        [
            ID_A,
            np.int64(ID_A),
            np.uint64(ID_A),
            np.int32(5),
            np.uint32(5),
            0,
            INT64_MAX,
        ],
    )
    def test_scalars(self, value):
        out = canonicalize_ids(value)
        assert isinstance(out, np.int64)
        assert int(out) == int(value)

    def test_numpy_signed_array(self):
        out = canonicalize_ids(np.array([1, 2, ID_A], dtype=np.int64))
        assert out.dtype == np.int64
        assert out.tolist() == [1, 2, ID_A]

    def test_numpy_unsigned_array(self):
        out = canonicalize_ids(np.array([1, 2, ID_A], dtype=np.uint64))
        assert out.dtype == np.int64
        assert out.tolist() == [1, 2, ID_A]

    def test_python_list(self):
        out = canonicalize_ids([ID_A, ID_B])
        assert out.dtype == np.int64
        assert out.tolist() == [ID_A, ID_B]

    def test_pandas_series_preserves_index_and_name(self):
        s = pd.Series([ID_A, ID_B], index=["p", "q"], name="graph", dtype="uint64")
        out = canonicalize_ids(s)
        assert out.dtype == np.dtype("int64")
        assert out.name == "graph"
        assert list(out.index) == ["p", "q"]
        assert out.tolist() == [ID_A, ID_B]

    def test_pandas_index_preserves_name(self):
        idx = pd.Index([ID_A, ID_B], name="vid", dtype="uint64")
        out = canonicalize_ids(idx)
        assert out.dtype == np.dtype("int64")
        assert out.name == "vid"
        assert out.tolist() == [ID_A, ID_B]

    def test_nullable_integer_with_allowed_nulls(self):
        s = pd.array([ID_A, pd.NA, ID_B], dtype="Int64")
        out = canonicalize_ids(s, allow_null=True)
        assert str(out.dtype) == "Int64"
        assert out[0] == ID_A and out[2] == ID_B
        assert out[1] is pd.NA

    def test_nullable_unsigned_with_allowed_nulls(self):
        s = pd.Series([ID_A, pd.NA], dtype="UInt64", name="g")
        out = canonicalize_ids(s, allow_null=True)
        assert str(out.dtype) == "Int64"
        assert out.iloc[0] == ID_A
        assert out.isna().iloc[1]

    def test_empty_container(self):
        assert canonicalize_ids([]).dtype == np.int64
        assert canonicalize_ids(np.array([], dtype=np.uint64)).dtype == np.int64


# ---------------------------------------------------------------------------
# 3. Invalid values
# ---------------------------------------------------------------------------
class TestInvalidValues:
    @pytest.mark.parametrize("value", [-1, np.int64(-5), np.array([1, -1])])
    def test_negative(self, value):
        with pytest.raises(ValueError):
            canonicalize_ids(value)

    @pytest.mark.parametrize(
        "value",
        [
            2**63,
            np.array([2**63], dtype=np.uint64),
            INT64_MAX + 1,
        ],
    )
    def test_too_large(self, value):
        with pytest.raises(ValueError):
            canonicalize_ids(value)

    @pytest.mark.parametrize(
        "value",
        [
            1.0,
            float(ID_A),  # integral-looking float that has already lost precision
            np.float64(5.0),
            np.array([1.0, 2.0]),
            pd.array([1.5], dtype="Float64"),
        ],
    )
    def test_floats_rejected(self, value):
        with pytest.raises(TypeError):
            canonicalize_ids(value)

    def test_integral_float_actually_lost_precision(self):
        # float(ID_A) == float(ID_B): proof that rejecting integral floats is
        # not pedantry -- the value is already corrupt.
        assert float(ID_A) == float(ID_B)

    @pytest.mark.parametrize(
        "value",
        [
            True,
            np.bool_(True),
            np.array([True, False]),
            np.array(["a", "b"], dtype=object),
            np.array([object()], dtype=object),
        ],
    )
    def test_non_integral_objects_rejected(self, value):
        with pytest.raises(TypeError):
            canonicalize_ids(value)

    def test_null_without_permission_rejected(self):
        with pytest.raises(ValueError):
            canonicalize_ids(pd.array([ID_A, pd.NA], dtype="Int64"))
        with pytest.raises(ValueError):
            canonicalize_ids(np.array([ID_A, None], dtype=object))


# ---------------------------------------------------------------------------
# 4. Legacy mixed-dtype mappings
# ---------------------------------------------------------------------------
class TestLegacyMixedDtype:
    @pytest.mark.parametrize(
        "graph_dtype,syn_dtype",
        [
            ("uint64", "int64"),
            ("int64", "uint64"),
            ("uint64", "uint64"),
            ("int64", "int64"),
        ],
    )
    def test_get_mapping_paths_exact(self, graph_dtype, syn_dtype):
        ms = _two_layer_morphsync(graph_dtype=graph_dtype, syn_dtype=syn_dtype)
        mp = ms.get_mapping_paths("post_syn", "graph")
        assert int(mp.loc[501, "graph"]) == ID_A
        assert int(mp.loc[502, "graph"]) == ID_B

    def test_null_link_endpoint_is_rejected(self):
        # A stored link endpoint always connects a real source to a real target;
        # a null there is malformed and must raise rather than be silently
        # carried through the mapping.
        ms = _two_layer_morphsync()
        bad = pd.DataFrame(
            {
                "post_syn": pd.Series([501, 502], dtype="int64"),
                "graph": pd.array([ID_A, pd.NA], dtype="Int64"),
            }
        )
        ms.links[("post_syn", "graph")] = bad
        ms.links[("graph", "post_syn")] = bad
        with pytest.raises(ValueError):
            ms.get_mapping_paths("post_syn", "graph")

    def test_real_legacy_file_links_canonicalized_on_load(self, nrn):
        # The bundled test cell stores its post_syn->graph link's graph column
        # as uint64 on disk (legacy dtype optimization only touched signed ints).
        for (src, tgt), df in nrn._morphsync.links.items():
            for col in (src, tgt):
                assert df[col].dtype == np.dtype("int64"), (
                    f"link {src}->{tgt} column {col} not canonicalized on load"
                )

    def test_real_legacy_file_mapping_is_exact(self, nrn):
        mapping = nrn._morphsync.get_mapping("post_syn", "graph", dropna=True)
        graph_ids = set(np.asarray(nrn.graph.vertex_index).tolist())
        # Every mapped target really is a graph vertex (no phantom collisions).
        assert set(mapping.dropna().astype("int64")) <= graph_ids


# ---------------------------------------------------------------------------
# 5. Directional symmetry
# ---------------------------------------------------------------------------
class TestDirectionalSymmetry:
    def _pairs(self, mapping_series):
        return {
            (int(k), int(v)) for k, v in mapping_series.dropna().astype("int64").items()
        }

    def test_forward_reverse_agree(self):
        ms = _two_layer_morphsync(graph_dtype="uint64", syn_dtype="int64")
        fwd = ms.get_mapping("post_syn", "graph", dropna=True)  # syn -> graph
        rev = ms.get_mapping("graph", "post_syn", dropna=True)  # graph -> syn
        fwd_pairs = self._pairs(fwd)
        rev_pairs = {(g, s) for s, g in fwd_pairs}
        assert self._pairs(rev) == rev_pairs

    def test_result_independent_of_which_side_is_unsigned(self):
        ms_a = _two_layer_morphsync(graph_dtype="uint64", syn_dtype="int64")
        ms_b = _two_layer_morphsync(graph_dtype="int64", syn_dtype="uint64")
        pairs_a = self._pairs(ms_a.get_mapping("post_syn", "graph", dropna=True))
        pairs_b = self._pairs(ms_b.get_mapping("post_syn", "graph", dropna=True))
        assert pairs_a == pairs_b == {(501, ID_A), (502, ID_B)}


# ---------------------------------------------------------------------------
# 6. Three-layer transitive mapping + masking
# ---------------------------------------------------------------------------
class TestTransitiveMasking:
    @pytest.mark.parametrize("inject_uint64", [False, True])
    def test_axonal_postsynapse_dropped_after_dendrite_mask(self, inject_uint64):
        cell = _transitive_cell()
        if inject_uint64:
            # Simulate a legacy object: force the graph endpoint of the
            # post_syn link to uint64 so the mapping must canonicalize it.
            for key in (("graph", "post_syn"), ("post_syn", "graph")):
                cell._morphsync.links[key]["graph"] = cell._morphsync.links[key][
                    "graph"
                ].astype("uint64")

        # Keep only the dendritic skeleton vertex (index 0).
        masked = cell.skeleton.apply_mask(np.array([0]))

        # The graph keeps only the dendritic ID_A ...
        assert list(masked.graph.vertex_index) == [ID_A]
        # ... and the postsynapse wired only to the axon is NOT retained.
        assert _annotation_index(masked, "post_syn") == []

    def test_layer_mask_by_uint64_ids_is_exact(self):
        # Masking the graph by a uint64 array of big IDs must not collide ID_A
        # and ID_B through the isin membership test.
        cell = _transitive_cell()
        masked = cell.graph.apply_mask(
            np.array([ID_A], dtype=np.uint64), self_only=True
        )
        assert list(masked.vertex_index) == [ID_A]

    def test_transitive_mapping_reaches_only_connected_synapse(self):
        cell = _transitive_cell()
        # skeleton dendrite (vertex 0) -> graph -> post_syn: reaches nothing,
        # because the only synapse is on the axonal graph vertex.
        reached = cell._morphsync.get_masking(
            "skeleton", "post_syn", source_index=np.array([0])
        )
        assert list(reached) == []
        # skeleton axon (vertex 1) -> the axonal synapse.
        reached_axon = cell._morphsync.get_masking(
            "skeleton", "post_syn", source_index=np.array([1])
        )
        assert list(reached_axon) == [9001]


# ---------------------------------------------------------------------------
# 7. Masked-link integrity (by construction)
# ---------------------------------------------------------------------------
class TestMaskedLinkIntegrity:
    def test_synthetic_cell_masked_links_reference_retained_nodes(self):
        cell = _transitive_cell()
        masked = cell.skeleton.apply_mask(np.array([0]))
        _assert_link_integrity(masked._morphsync)

    def test_real_cell_masked_links_reference_retained_nodes(self, nrn):
        # Mask to an arbitrary subtree of skeleton vertices and confirm no link
        # endpoint dangles into a removed node.
        keep = nrn.skeleton.vertex_index[: max(1, nrn.skeleton.n_vertices // 3)]
        masked = nrn.skeleton.apply_mask(keep)
        _assert_link_integrity(masked._morphsync)

    def test_masking_does_not_copy_full_link_tables(self, nrn):
        keep = nrn.skeleton.vertex_index[:5]
        masked = nrn.skeleton.apply_mask(keep)
        for key, df in masked._morphsync.links.items():
            full = nrn._morphsync.links[key]
            assert len(df) <= len(full)


# ---------------------------------------------------------------------------
# 8. Serialization round-trip
# ---------------------------------------------------------------------------
class TestSerializationRoundTrip:
    def test_synthetic_cell_roundtrip_preserves_ids_and_mapping(self, tmp_path):
        cell = _transitive_cell()
        before = cell._morphsync.get_mapping("post_syn", "graph", dropna=True)

        path = str(tmp_path / "rt.osy")
        file_io.save_cell(cell, path, allow_overwrite=True)
        loaded = file_io.load_cell(path)

        # Big IDs survive exactly.
        assert list(loaded.graph.vertex_index) == [ID_A, ID_B]
        # Link columns come back canonical int64.
        for (src, tgt), df in loaded._morphsync.links.items():
            for col in (src, tgt):
                assert df[col].dtype == np.dtype("int64")
        # Mapping is identical to before the round trip.
        after = loaded._morphsync.get_mapping("post_syn", "graph", dropna=True)
        assert {(int(k), int(v)) for k, v in before.astype("int64").items()} == {
            (int(k), int(v)) for k, v in after.astype("int64").items()
        }

    def test_real_cell_roundtrip_mapping_stable(self, nrn, tmp_path):
        before = nrn._morphsync.get_mapping("post_syn", "graph", dropna=True)
        path = str(tmp_path / "real_rt.osy")
        file_io.save_cell(nrn, path, allow_overwrite=True)
        loaded = file_io.load_cell(path)
        after = loaded._morphsync.get_mapping("post_syn", "graph", dropna=True)
        assert {(int(k), int(v)) for k, v in before.astype("int64").items()} == {
            (int(k), int(v)) for k, v in after.astype("int64").items()
        }
