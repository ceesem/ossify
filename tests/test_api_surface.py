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
