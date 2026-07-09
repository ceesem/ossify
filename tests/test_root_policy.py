"""Tests for ossify.compartments.root_policy (the rooting dry-run + commit).

All cases use a tiny synthetic chain and explicit unaries -- no model or sample
data -- so they exercise the decision logic directly. ``root_policy`` is pure;
only ``RootDecision.apply`` mutates the skeleton.
"""

import numpy as np
import pytest

from ossify import Cell
from ossify.compartments import RootDecision, StructuredLabeler, root_policy
from ossify.structured_prediction import TransitionSchema


class _Encoder:
    """Encoder stub: just enough for StructuredLabeler construction.

    ``root_policy`` is always called here with explicit ``unaries``, so
    ``predict_unaries`` is never exercised; only ``classes_`` is read (by the
    StructuredLabeler class-alignment check).
    """

    classes_ = [0, 1]

    def predict_unaries(self, cell):  # pragma: no cover - not used in these tests
        raise AssertionError("unaries should be passed explicitly in tests")


def _labeler():
    schema = TransitionSchema(
        classes=["dendrite", "axon"],
        transitions={("axon", "dendrite"): 40.0, ("dendrite", "axon"): 2.0},
        root_classes=["dendrite"],
    )
    return StructuredLabeler(_Encoder(), schema)


def _chain_cell(n=8):
    verts = np.array([[i, 0, 0] for i in range(n)], dtype=float)
    edges = np.column_stack([np.arange(1, n), np.arange(0, n - 1)])
    cell = Cell()
    cell.add_skeleton(verts, edges, root=0)
    return cell


def _unaries(dendrite_margin):
    """(n, 2) log-potentials whose dendrite-vs-axon margin is ``dendrite_margin``."""
    m = np.asarray(dendrite_margin, dtype=float)
    return np.column_stack([m, np.zeros_like(m)])


def test_keeps_root_inside_a_convincing_dendrite_span():
    cell = _chain_cell(8)
    u = _unaries([5, 5, 5, 5, 5, -5, -5, -5])  # 0..4 dendrite, 5..7 axon; root 0
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert not d.rerooted
    assert d.root == 0
    assert d.root_classes == ["dendrite"]
    assert cell.skeleton.root_positional == 0  # pure: nothing mutated


def test_reroots_when_current_root_is_mislabeled():
    cell = _chain_cell(8)
    u = _unaries([-5, -5, -5, 5, 5, 5, 5, 5])  # root 0 sits in axon; dendrite is 3..7
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert d.rerooted
    assert d.root in range(3, 8)
    assert d.root_classes == ["dendrite"]
    assert d.old_root == 0
    assert cell.skeleton.root_positional == 0  # still pure until apply


def test_no_dendrite_is_a_terminal_fragment():
    cell = _chain_cell(8)
    u = _unaries([-5] * 8)  # all axon
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert not d.rerooted
    assert d.root == 0
    assert list(d.root_classes) == ["axon"]
    assert d.n_qualifying_spans == 0


def test_single_vertex_spike_does_not_qualify():
    # The guard: one isolated dendrite-like vertex (margin 5) can't clear a summed
    # window threshold of 10 -- its on-component window is just itself.
    cell = _chain_cell(8)
    u = _unaries([-5, -5, -5, -5, 5, -5, -5, -5])
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert not d.rerooted
    assert list(d.root_classes) == ["axon"]
    assert d.n_qualifying_spans == 0


def test_lower_threshold_lets_the_spike_through():
    # Same spike, but a threshold below a single vertex's margin now admits it.
    cell = _chain_cell(8)
    u = _unaries([-5, -5, -5, -5, 5, -5, -5, -5])
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=3)
    assert d.rerooted
    assert d.root == 4


def test_nucleus_short_circuits_to_soma():
    cell = _chain_cell(8)
    u = _unaries([5, 5, 5, 5, 5, -5, -5, -5])  # evidence would keep root 0...
    d = root_policy(cell, _labeler(), unaries=u, has_nucleus=True, nucleus_root=6)
    # ...but a known nucleus wins outright.
    assert d.rerooted
    assert d.root == 6
    assert d.root_classes == ["dendrite"]
    assert "nucleus" in d.reason


def test_apply_commits_the_reroot():
    cell = _chain_cell(8)
    u = _unaries([-5, -5, -5, 5, 5, 5, 5, 5])
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert d.rerooted
    returned = d.apply(cell)
    assert returned is cell
    assert cell.skeleton.root_positional == d.root


def test_apply_is_a_noop_when_not_rerooted():
    cell = _chain_cell(8)
    u = _unaries([5, 5, 5, 5, 5, -5, -5, -5])
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    assert not d.rerooted
    d.apply(cell)
    assert cell.skeleton.root_positional == 0


def test_root_class_must_be_in_schema():
    cell = _chain_cell(8)
    u = _unaries([1] * 8)
    with pytest.raises(ValueError, match="root_class"):
        root_policy(cell, _labeler(), unaries=u, root_class="apical")


def test_with_root_classes_swaps_only_the_root_set():
    base = _labeler().schema
    axon_rooted = base.with_root_classes(["axon"])
    assert axon_rooted.root_classes == ["axon"]
    assert base.root_classes == ["dendrite"]  # original untouched
    np.testing.assert_array_equal(axon_rooted.cost_matrix, base.cost_matrix)


def test_root_decision_is_dataclass_for_sweeps():
    # Sweepability: a decision flattens to a record (dataset audit -> DataFrame).
    from dataclasses import asdict

    cell = _chain_cell(8)
    u = _unaries([5, 5, 5, 5, 5, -5, -5, -5])
    d = root_policy(cell, _labeler(), unaries=u, window_hops=3, min_window_logodds=10)
    row = asdict(d)
    assert {"root", "rerooted", "window_logodds", "reason"} <= set(row)
    assert isinstance(d, RootDecision)
