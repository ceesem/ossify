"""Regression tests for linkage source-direction inference in ``file_io``.

The ``.osy`` archive sorts the two layer names when it serializes a linkage
pair, so the stored column order does not preserve which endpoint was the
original *source* (the side with exactly one mapping row per vertex). The
loader must therefore infer the source by exact source-domain validation, not
by row count alone -- row count is ambiguous whenever the two layers happen to
have equal cardinality (e.g. a many-to-one graph <-> annotation link whose
counts coincide, as seen in real MICrONS cells).
"""

import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from ossify import Cell, file_io
from ossify.sync_classes import Link

SPATIAL = ["x", "y", "z"]


def _make_graph_df(ids, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.rand(len(ids), 3), columns=SPATIAL, index=list(ids))


def _build_cell(
    graph_ids,
    anno_ids,
    mapping,
    anno_name="post_syn",
    name="cell",
    graph_index_dtype=None,
    mapping_dtype=np.int64,
):
    """Build a Cell with a graph layer and a linked point-annotation layer.

    The annotation is the *source*: each annotation point maps to exactly one
    graph vertex (``mapping``), which may repeat graph ids (many-to-one).
    """
    gdf = _make_graph_df(graph_ids)
    if graph_index_dtype is not None:
        gdf.index = gdf.index.astype(graph_index_dtype)
    edges = np.array(
        [[graph_ids[i + 1], graph_ids[i]] for i in range(len(graph_ids) - 1)]
    )
    cell = Cell(name=name)
    cell.add_graph(vertices=gdf, edges=edges, spatial_columns=SPATIAL)

    adf = pd.DataFrame(
        np.random.RandomState(1).rand(len(anno_ids), 3),
        columns=SPATIAL,
        index=list(anno_ids),
    )
    cell.add_point_annotations(
        anno_name,
        vertices=adf,
        spatial_columns=SPATIAL,
        linkage=Link(
            mapping=np.asarray(mapping, dtype=mapping_dtype),
            target="graph",
            map_value_is_index=True,
        ),
    )
    return cell


def _roundtrip(cell):
    buf = io.BytesIO()
    file_io.save_cell(cell, buf)
    buf.seek(0)
    return file_io.load_cell(buf)


def _linkage_member_name(tar_bytes):
    """Return the archive path of the (single) linkage table."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
        for m in tf.getmembers():
            if m.name.startswith("linkage/") and m.name.endswith("linkage.feather"):
                return m.name
    raise AssertionError("no linkage member found")


def _rewrite_member(tar_bytes, member_name, new_bytes):
    """Return a copy of ``tar_bytes`` with one member's contents replaced."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as src:
        with tarfile.open(fileobj=out, mode="w") as dst:
            for m in src.getmembers():
                data = src.extractfile(m).read()
                if m.name == member_name:
                    data = new_bytes
                info = tarfile.TarInfo(name=m.name)
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
    return out.getvalue()


class TestEqualSizedManyToOne:
    """The core reported bug: equal row/vertex counts, annotation is source."""

    def test_equal_sized_many_to_one_loads_correctly(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        # 6 rows == 6 graph vertices, but graph column is repeated and covers
        # only 3 of the 6 graph vertices -> graph cannot be the source.
        mapping = [10, 10, 11, 11, 12, 12]
        cell = _build_cell(graph_ids, anno_ids, mapping)

        loaded = _roundtrip(cell)

        # Loading succeeds and every point maps to its correct graph vertex.
        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        assert fwd == dict(zip(anno_ids, mapping))

        # Graph -> point reverse traversal returns the correct groups.
        link = loaded._morphsync.get_link("graph", "post_syn")
        groups = (
            link.groupby("graph")["post_syn"]
            .apply(lambda s: sorted(int(v) for v in s))
            .to_dict()
        )
        assert groups == {10: [100, 101], 11: [102, 103], 12: [104, 105]}


class TestSortedNameOrderIndependence:
    """Correctness must not depend on which layer name sorts first."""

    @pytest.mark.parametrize("anno_name", ["a_pre", "z_post"])
    def test_source_inference_independent_of_sort_order(self, anno_name):
        # "a_pre" sorts before "graph"; "z_post" sorts after. Either way the
        # annotation is the true source and must be inferred as such.
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        mapping = [10, 10, 11, 11, 12, 12]
        cell = _build_cell(graph_ids, anno_ids, mapping, anno_name=anno_name)

        loaded = _roundtrip(cell)

        fwd = dict(loaded._morphsync.get_mapping(anno_name, "graph"))
        assert fwd == dict(zip(anno_ids, mapping))


class TestUnequalLayerSizes:
    """Ordinary behavior for differently-sized layers must be preserved."""

    def test_unequal_sizes_annotation_source(self):
        graph_ids = list(range(10, 22))  # 12 graph vertices
        anno_ids = [100, 101, 102, 103]  # 4 annotations
        mapping = [10, 11, 12, 13]
        cell = _build_cell(graph_ids, anno_ids, mapping)

        loaded = _roundtrip(cell)

        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        assert fwd == dict(zip(anno_ids, mapping))

    def test_graph_to_skeleton_roundtrip(self, simple_skeleton_data, spatial_columns):
        """Graph <-> skeleton (unequal sizes) still round-trips."""
        vertices, _, vertex_indices = simple_skeleton_data
        skel_df = pd.DataFrame(vertices, columns=spatial_columns, index=vertex_indices)
        skel_edges = np.array(
            [[vertex_indices[i + 1], vertex_indices[i]] for i in range(4)]
        )
        graph_ids = list(range(200, 210))  # 10 graph vertices
        graph_df = _make_graph_df(graph_ids)
        graph_edges = np.array(
            [[graph_ids[i + 1], graph_ids[i]] for i in range(len(graph_ids) - 1)]
        )
        # Each graph vertex maps to one skeleton vertex (many-to-one).
        skel_map = [
            vertex_indices[i % len(vertex_indices)] for i in range(len(graph_ids))
        ]

        cell = Cell(name="gs")
        cell.add_skeleton(
            vertices=skel_df,
            edges=skel_edges,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )
        cell.add_graph(
            vertices=graph_df,
            edges=graph_edges,
            spatial_columns=spatial_columns,
            linkage=Link(
                mapping=np.array(skel_map), target="skeleton", map_value_is_index=True
            ),
        )

        loaded = _roundtrip(cell)
        fwd = dict(loaded._morphsync.get_mapping("graph", "skeleton"))
        assert fwd == dict(zip(graph_ids, skel_map))


class TestTrueBijection:
    """Both endpoints exactly cover their layers -> deterministic + correct."""

    def test_bijection_maps_both_directions(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        mapping = [10, 11, 12, 13, 14, 15]  # one-to-one
        cell = _build_cell(graph_ids, anno_ids, mapping)

        loaded = _roundtrip(cell)

        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        rev = dict(loaded._morphsync.get_mapping("graph", "post_syn"))
        assert fwd == dict(zip(anno_ids, mapping))
        assert rev == dict(zip(mapping, anno_ids))

    def test_bijection_is_deterministic(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        mapping = [10, 11, 12, 13, 14, 15]
        cell = _build_cell(graph_ids, anno_ids, mapping)

        a = _roundtrip(cell)
        b = _roundtrip(cell)
        assert dict(a._morphsync.get_mapping("post_syn", "graph")) == dict(
            b._morphsync.get_mapping("post_syn", "graph")
        )


class TestInvalidSourceMapping:
    """Neither endpoint covers its layer -> concise ValueError, not KeyError."""

    def test_invalid_mapping_raises_bounded_valueerror(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        mapping = [10, 10, 11, 11, 12, 12]
        cell = _build_cell(graph_ids, anno_ids, mapping)

        buf = io.BytesIO()
        file_io.save_cell(cell, buf)
        tar_bytes = buf.getvalue()

        member = _linkage_member_name(tar_bytes)
        # Corrupt the table so NEITHER endpoint covers its own layer:
        # graph column repeats/omits, post column repeats/omits.
        bad = pd.DataFrame(
            {
                "graph": [10, 10, 11, 11, 12, 12],
                "post_syn": [100, 100, 101, 101, 102, 102],
            }
        )
        bad_bytes = file_io.bytesio_feather(bad)
        corrupted = _rewrite_member(tar_bytes, member, bad_bytes)

        with pytest.raises(ValueError) as exc_info:
            file_io.load_cell(io.BytesIO(corrupted))

        msg = str(exc_info.value)
        # Descriptive, bounded message -- not a giant pandas KeyError.
        assert "graph" in msg and "post_syn" in msg
        assert "vertices=6" in msg
        assert "missing_source_ids" in msg
        # Message stays small (bounded sample, not thousands of IDs).
        assert len(msg) < 2000

    def test_invalid_mapping_is_not_keyerror(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        cell = _build_cell(graph_ids, anno_ids, [10, 10, 11, 11, 12, 12])
        buf = io.BytesIO()
        file_io.save_cell(cell, buf)
        tar_bytes = buf.getvalue()
        member = _linkage_member_name(tar_bytes)
        bad = pd.DataFrame(
            {"graph": [10] * 6, "post_syn": [100, 100, 101, 101, 102, 102]}
        )
        corrupted = _rewrite_member(tar_bytes, member, file_io.bytesio_feather(bad))
        with pytest.raises(ValueError):
            file_io.load_cell(io.BytesIO(corrupted))


class TestDuplicateSourceRejected:
    """A repeated endpoint cannot qualify as source just by matching row count."""

    def test_duplicate_endpoint_not_chosen(self):
        graph_ids = [10, 11, 12, 13, 14, 15]
        anno_ids = [100, 101, 102, 103, 104, 105]
        # graph column duplicated (row count still equals graph vertex count)
        mapping = [10, 10, 11, 11, 12, 12]
        cell = _build_cell(graph_ids, anno_ids, mapping)

        loaded = _roundtrip(cell)

        # The unique, fully-covering post_syn column is the source -- NOT graph,
        # even though len(link) == n_graph_vertices.
        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        assert fwd == dict(zip(anno_ids, mapping))
        # A many-to-one reverse: some graph vertices are unmapped.
        rev_masking = loaded._morphsync.get_masking("post_syn", "graph")
        assert sorted(int(v) for v in rev_masking) == [10, 11, 12]


class TestMixedIntegerRepresentations:
    """IDs above 2**53, with uint64/int64 mixing, must match after canonicalize."""

    def test_ids_above_2_53_mixed_dtypes(self):
        base = 2**53
        graph_ids = [base + 1, base + 2, base + 3, base + 4]
        anno_ids = [base + 10, base + 11, base + 12, base + 13]
        mapping = [base + 1, base + 1, base + 2, base + 3]  # many-to-one
        # graph index stored as uint64; link mapping built as int64.
        cell = _build_cell(
            graph_ids,
            anno_ids,
            mapping,
            graph_index_dtype=np.uint64,
            mapping_dtype=np.int64,
        )

        loaded = _roundtrip(cell)

        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        # Exact identity above 2**53 -- no float collapse.
        assert fwd == dict(zip(anno_ids, mapping))
        for k, v in fwd.items():
            assert int(k) > base and int(v) > base


class TestFullArchiveRoundTripCoincidentalCounts:
    """Save/load a cell whose annotation count coincidentally equals graph count."""

    def test_coincidental_equal_counts_full_roundtrip(self):
        n = 25
        graph_ids = list(range(1000, 1000 + n))
        anno_ids = list(range(5000, 5000 + n))  # same count as graph vertices
        rng = np.random.RandomState(3)
        mapping = [graph_ids[i] for i in rng.randint(0, n, size=n)]
        cell = _build_cell(graph_ids, anno_ids, mapping, name="coincidental")

        loaded = _roundtrip(cell)

        assert loaded.graph.n_vertices == n
        assert loaded.annotations["post_syn"].n_vertices == n
        fwd = dict(loaded._morphsync.get_mapping("post_syn", "graph"))
        assert fwd == dict(zip(anno_ids, mapping))
