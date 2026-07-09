import itertools

import numpy as np
import pytest

from ossify import Cell
from ossify.structured_prediction import (
    TransitionSchema,
    absorb_small_compartments,
    count_components,
    decode_tree,
    tree_absorb_small_compartments,
    tree_map_decode,
)


def _component_counts(parent, root, labels):
    """Number of connected same-label components per label index."""
    counts = {}
    for v in range(len(labels)):
        p = parent[v]
        if v == root or p < 0 or labels[p] != labels[v]:
            counts[int(labels[v])] = counts.get(int(labels[v]), 0) + 1
    return counts


def _objective(parent, root, unaries, C, labels, onset_penalty=None, max_components=2):
    """Score of a labeling under the MAP objective (for brute-force checks).

    With ``onset_penalty`` the cardinality term is subtracted too, with onset
    counts saturated at ``max_components`` to mirror the decoder's bounded state.
    """
    n = len(labels)
    s = unaries[np.arange(n), labels].sum()
    for v in range(n):
        p = parent[v]
        if v != root and p >= 0:
            s -= C[labels[p], labels[v]]
    if onset_penalty is not None:
        cap = max(1, int(max_components))
        for k, cnt in _component_counts(parent, root, labels).items():
            c = min(cnt, cap)
            if c > 1:
                s -= onset_penalty[k] * (c - 1)
    return s


def _brute_force(
    parent, root, unaries, C, root_mask, onset_penalty=None, max_components=2
):
    n, K = unaries.shape
    best_score, best = -np.inf, None
    for assign in itertools.product(range(K), repeat=n):
        if not root_mask[assign[root]]:
            continue
        score = _objective(
            parent, root, unaries, C, np.array(assign), onset_penalty, max_components
        )
        if score > best_score:
            best_score, best = score, np.array(assign)
    return best_score, best


class _FakeSkeleton:
    def __init__(self, parent, root):
        self.parent_node_array = np.asarray(parent)
        self.root_positional = root


class TestTransitionSchema:
    def test_cost_matrix_and_root_mask(self):
        s = TransitionSchema(
            classes=["dendrite", "axon"],
            transitions={("axon", "dendrite"): 100.0},
            root_classes=["dendrite"],
        )
        assert s.n_classes == 2
        assert s.class_index("axon") == 1
        C = s.cost_matrix
        np.testing.assert_array_equal(C, [[0.0, 0.0], [100.0, 0.0]])
        np.testing.assert_array_equal(s.root_allowed_mask, [True, False])

    def test_default_cost_is_switching_penalty(self):
        # Nonzero default_cost penalizes label changes only; the diagonal
        # (staying on a label) stays free, and explicit pairs still override.
        s = TransitionSchema(
            classes=["dendrite", "axon"],
            transitions={("axon", "dendrite"): 100.0},
            default_cost=3.0,
        )
        np.testing.assert_array_equal(s.cost_matrix, [[0.0, 3.0], [100.0, 0.0]])

    def test_explicit_diagonal_overrides(self):
        s = TransitionSchema(
            classes=["a", "b"],
            transitions={("a", "a"): 2.0},
            default_cost=5.0,
        )
        np.testing.assert_array_equal(s.cost_matrix, [[2.0, 5.0], [5.0, 0.0]])

    def test_validation(self):
        with pytest.raises(ValueError):
            TransitionSchema(classes=["a", "a"])
        with pytest.raises(ValueError):
            TransitionSchema(classes=["a", "b"], root_classes=["c"])
        with pytest.raises(ValueError):
            TransitionSchema(classes=["a", "b"], transitions={("a", "z"): 1.0})


class TestTreeMapDecode:
    def _chain(self, n=5):
        parent = np.array([-1] + list(range(n - 1)))
        return parent, 0

    def test_reproduces_root_connected_axon(self):
        # dendrite=0, axon=1; root forced dendrite; axon->dendrite discouraged.
        parent, root = self._chain(5)
        unaries = np.zeros((5, 2))
        unaries[0, 0] = 2.0  # root strongly dendrite
        unaries[1:, 1] = 1.0  # rest favor axon
        C = np.array([[0.0, 0.0], [100.0, 0.0]])
        labels, edge_cost = tree_map_decode(
            parent, root, unaries, C, np.array([True, False])
        )
        np.testing.assert_array_equal(labels, [0, 1, 1, 1, 1])
        assert edge_cost.sum() == 0.0  # only the free dendrite->axon transition

    def test_noisy_interior_node_does_not_flip(self):
        # a single mildly dendrite-leaning node inside the axon stays axon,
        # because reverting would pay the axon->dendrite cost.
        parent, root = self._chain(5)
        unaries = np.zeros((5, 2))
        unaries[0, 0] = 2.0
        unaries[1:, 1] = 1.0
        unaries[3, 0] = 0.5  # mild local dendrite preference at node 3
        unaries[3, 1] = 0.0
        C = np.array([[0.0, 0.0], [100.0, 0.0]])
        labels, _ = tree_map_decode(parent, root, unaries, C, np.array([True, False]))
        np.testing.assert_array_equal(labels, [0, 1, 1, 1, 1])

    def test_strong_evidence_overrides_and_flags(self):
        # When the surrounding cable overwhelmingly wants axon but one interior
        # node overwhelmingly wants dendrite, extending dendrite is costlier than
        # paying a single revert -- so the MAP pays it and flags the edge.
        parent, root = self._chain(5)
        unaries = np.zeros((5, 2))
        unaries[0, 0] = 1000.0  # root: dendrite
        unaries[[1, 2, 4], 1] = 1000.0  # neighbours: strongly axon
        unaries[3, 0] = 1000.0  # node 3: strongly dendrite
        C = np.array([[0.0, 0.0], [100.0, 0.0]])
        labels, edge_cost = tree_map_decode(
            parent, root, unaries, C, np.array([True, False])
        )
        np.testing.assert_array_equal(labels, [0, 1, 1, 0, 1])
        assert edge_cost[3] == 100.0  # axon(2) -> dendrite(3) violation flagged
        assert edge_cost.sum() == 100.0  # and it's the only one

    def test_root_constraint_respected(self):
        parent, root = self._chain(3)
        unaries = np.zeros((3, 2))
        unaries[:, 1] = 5.0  # everything wants axon, including the root
        C = np.zeros((2, 2))
        labels, _ = tree_map_decode(parent, root, unaries, C, np.array([True, False]))
        assert labels[0] == 0  # root forced to the allowed class despite evidence

    def test_forbids_empty_root_mask(self):
        parent, root = self._chain(3)
        with pytest.raises(ValueError):
            tree_map_decode(
                parent,
                root,
                np.zeros((3, 2)),
                np.zeros((2, 2)),
                np.array([False, False]),
            )

    @pytest.mark.parametrize("seed", range(8))
    def test_exact_vs_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        n, K = 7, 3
        # random rooted tree: parent[v] in 0..v-1, so root=0 and indices are topo
        parent = np.array([-1] + [rng.integers(0, v) for v in range(1, n)])
        root = 0
        unaries = rng.normal(size=(n, K))
        C = rng.uniform(0, 3, size=(K, K))
        np.fill_diagonal(C, 0.0)
        root_mask = np.array([True] + list(rng.random(K - 1) > 0.5))
        if not root_mask.any():
            root_mask[0] = True
        labels, _ = tree_map_decode(parent, root, unaries, C, root_mask)
        got = _objective(parent, root, unaries, C, labels)
        best, _ = _brute_force(parent, root, unaries, C, root_mask)
        np.testing.assert_allclose(got, best, rtol=1e-9)


class TestCardinalityConstraint:
    def _two_branch_tree(self):
        # root -> {1, 2}; 1 -> 3; 2 -> 4. dendrite=0, axon=1. Two dendrite
        # spacers separate two would-be axon tips into distinct components.
        parent = np.array([-1, 0, 0, 1, 2])
        unaries = np.array(
            [
                [10.0, 0.0],  # root: dendrite
                [1.0, 0.0],  # dendrite spacer
                [1.0, 0.0],  # dendrite spacer
                [0.0, 6.0],  # axon tip, stronger evidence
                [0.0, 5.0],  # axon tip, weaker evidence
            ]
        )
        C = np.zeros((2, 2))
        return parent, 0, unaries, C, np.array([True, False])

    def test_two_components_survive_when_evidence_strong(self):
        parent, root, unaries, C, mask = self._two_branch_tree()
        op = np.array([0.0, 1.0])  # cheap: both axon tips clear it
        labels, _ = tree_map_decode(parent, root, unaries, C, mask, onset_penalty=op)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1])

    def test_weak_second_component_suppressed(self):
        parent, root, unaries, C, mask = self._two_branch_tree()
        op = np.array([0.0, 10.0])  # exceeds the weaker tip's evidence (5)
        labels, _ = tree_map_decode(parent, root, unaries, C, mask, onset_penalty=op)
        # only the stronger axon (node 3) survives; the weaker reverts to dendrite
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 0])

    def test_hard_cap_forces_single_component(self):
        parent, root, unaries, C, mask = self._two_branch_tree()
        op = np.array([0.0, np.inf])
        labels, _ = tree_map_decode(parent, root, unaries, C, mask, onset_penalty=op)
        assert _component_counts(parent, root, labels).get(1, 0) == 1
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 0])

    def test_zero_penalty_matches_unconstrained(self):
        parent, root, unaries, C, mask = self._two_branch_tree()
        base, _ = tree_map_decode(parent, root, unaries, C, mask)
        zero, _ = tree_map_decode(
            parent, root, unaries, C, mask, onset_penalty=np.zeros(2)
        )
        np.testing.assert_array_equal(base, zero)

    def test_root_component_counts_as_free_first(self):
        # A chain rooted in axon is one contiguous axon component -- the free
        # first one -- so a large penalty must not perturb it.
        parent = np.array([-1, 0, 1, 2])
        unaries = np.zeros((4, 2))
        unaries[:, 1] = 5.0
        labels, _ = tree_map_decode(
            parent,
            0,
            unaries,
            np.zeros((2, 2)),
            np.array([True, True]),
            onset_penalty=np.array([0.0, 1000.0]),
        )
        np.testing.assert_array_equal(labels, [1, 1, 1, 1])

    def test_independent_caps_per_class(self):
        # root dendrite; two axon tips and two apical tips. Axon penalty is cheap
        # (both survive); apical penalty is steep (only the stronger survives).
        parent = np.array([-1, 0, 0, 1, 2, 0, 0, 5, 6])
        unaries = np.zeros((9, 3))  # dendrite=0, axon=1, apical=2
        unaries[0, 0] = 10.0
        unaries[[1, 2, 5, 6], 0] = 1.0
        unaries[[3, 4], 1] = 5.0
        unaries[7, 2] = 6.0  # stronger apical
        unaries[8, 2] = 5.0  # weaker apical
        op = np.array([0.0, 1.0, 100.0])
        labels, _ = tree_map_decode(
            parent,
            0,
            unaries,
            np.zeros((3, 3)),
            np.array([True, False, False]),
            onset_penalty=op,
        )
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 0, 0, 2, 0])
        counts = _component_counts(parent, 0, labels)
        assert counts.get(1, 0) == 2  # two axons kept
        assert counts.get(2, 0) == 1  # one apical kept

    def test_max_components_controls_saturation(self):
        # Three separated axon tips. At cap 2 the penalty saturates after the
        # first extra, so the third tip is free and all three survive; at cap 3
        # the penalty keeps accruing and only one survives.
        parent = np.array([-1, 0, 0, 0, 1, 2, 3])
        unaries = np.zeros((7, 2))
        unaries[0, 0] = 10.0
        unaries[[1, 2, 3], 0] = 1.0
        unaries[[4, 5, 6], 1] = 5.0
        C = np.zeros((2, 2))
        mask = np.array([True, False])
        op = np.array([0.0, 8.0])
        l2, _ = tree_map_decode(
            parent, 0, unaries, C, mask, onset_penalty=op, max_components=2
        )
        l3, _ = tree_map_decode(
            parent, 0, unaries, C, mask, onset_penalty=op, max_components=3
        )
        assert _component_counts(parent, 0, l2).get(1, 0) == 3
        assert _component_counts(parent, 0, l3).get(1, 0) == 1

    @pytest.mark.parametrize("max_components", [2, 3])
    @pytest.mark.parametrize("seed", range(8))
    def test_exact_vs_brute_force(self, seed, max_components):
        rng = np.random.default_rng(seed)
        n, K = 6, 3
        parent = np.array([-1] + [rng.integers(0, v) for v in range(1, n)])
        root = 0
        unaries = rng.normal(size=(n, K))
        C = rng.uniform(0, 3, size=(K, K))
        np.fill_diagonal(C, 0.0)
        root_mask = np.array([True] + list(rng.random(K - 1) > 0.5))
        # random per-class penalties, ~half the classes uncapped
        op = np.where(rng.random(K) > 0.5, rng.uniform(0, 4, size=K), 0.0)
        labels, _ = tree_map_decode(
            parent,
            root,
            unaries,
            C,
            root_mask,
            onset_penalty=op,
            max_components=max_components,
        )
        got = _objective(parent, root, unaries, C, labels, op, max_components)
        best, _ = _brute_force(parent, root, unaries, C, root_mask, op, max_components)
        np.testing.assert_allclose(got, best, rtol=1e-9)


class TestCountComponents:
    def test_counts_per_label(self):
        # dend root+{3} = one dend run; axon {1,2} and axon {4} = two axon runs.
        parent = np.array([-1, 0, 1, 0, 3])
        labels = np.array([0, 1, 1, 0, 1])
        assert count_components(parent, 0, labels) == {0: 1, 1: 2}


class TestComponentPenalties:
    def test_onset_penalty_array(self):
        s = TransitionSchema(
            classes=["dend", "axon", "apical"],
            component_penalties={"axon": 8.0, "apical": 100.0},
        )
        np.testing.assert_array_equal(s.onset_penalty, [0.0, 8.0, 100.0])

    def test_default_no_penalty(self):
        s = TransitionSchema(classes=["dend", "axon"])
        np.testing.assert_array_equal(s.onset_penalty, [0.0, 0.0])

    def test_validation_unknown_class(self):
        with pytest.raises(ValueError):
            TransitionSchema(
                classes=["dend", "axon"], component_penalties={"apical": 1.0}
            )

    def test_with_root_classes_preserves_penalties(self):
        s = TransitionSchema(
            classes=["dend", "axon"], component_penalties={"axon": 5.0}
        )
        s2 = s.with_root_classes(["axon"])
        np.testing.assert_array_equal(s2.onset_penalty, [0.0, 5.0])
        assert s2.root_classes == ["axon"]


class TestDecodeTreeCardinality:
    def test_penalty_flows_through_schema(self):
        parent = np.array([-1, 0, 0, 1, 2])
        skel = _FakeSkeleton(parent, 0)
        unaries = np.array(
            [[10.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 6.0], [0.0, 5.0]]
        )
        cheap = TransitionSchema(
            classes=["dendrite", "axon"],
            root_classes=["dendrite"],
            component_penalties={"axon": 1.0},
        )
        labels_cheap, _ = decode_tree(skel, unaries, cheap)
        np.testing.assert_array_equal(labels_cheap, [0, 0, 0, 1, 1])
        pricey = TransitionSchema(
            classes=["dendrite", "axon"],
            root_classes=["dendrite"],
            component_penalties={"axon": 10.0},
        )
        labels_pricey, _ = decode_tree(skel, unaries, pricey)
        np.testing.assert_array_equal(labels_pricey, [0, 0, 0, 1, 0])


class TestDecodeTreeWrapper:
    def test_label_output_and_topology_pull(self):
        parent = np.array([-1, 0, 1, 1])
        skel = _FakeSkeleton(parent, 0)
        schema = TransitionSchema(
            classes=["dendrite", "axon"],
            transitions={("axon", "dendrite"): 100.0},
            root_classes=["dendrite"],
        )
        unaries = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        labels, edge_cost = decode_tree(skel, unaries, schema, return_labels_as="label")
        np.testing.assert_array_equal(labels, ["dendrite", "axon", "axon", "axon"])
        assert edge_cost.shape == (4,)

    def test_runs_on_real_segment_graph(self):
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [3, 1, 0]], dtype=float
        )
        edges = np.array([[1, 0], [2, 1], [3, 1], [4, 2]])
        cell = Cell()
        cell.add_skeleton(verts, edges, root=0)
        sg = cell.skeleton.segment_graph()
        schema = TransitionSchema(
            classes=["dendrite", "axon"],
            transitions={("axon", "dendrite"): 100.0},
            root_classes=["dendrite"],
        )
        unaries = np.zeros((sg.n_vertices, 2))
        unaries[:, 1] = 1.0  # everything leans axon
        unaries[sg.root_positional, 0] = 5.0  # except the root
        labels, edge_cost = decode_tree(sg, unaries, schema)
        assert labels[sg.root_positional] == 0  # root is dendrite
        # broadcast a per-node labeling back to vertices
        per_vertex = sg.to_vertices(labels)
        assert per_vertex.shape[0] == cell.skeleton.n_vertices


class TestAbsorbSmallCompartments:
    def _chain(self, n):
        # 0 -> 1 -> 2 -> ... -> n-1, rooted at 0
        return np.array([-1] + list(range(n - 1))), 0

    def test_terminal_island_absorbed_by_count(self):
        # dendrite trunk with a 2-vertex axon tip; min_size=2 absorbs it.
        parent, root = self._chain(6)
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = tree_absorb_small_compartments(parent, root, labels, min_size=2)
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0])

    def test_island_kept_when_above_threshold(self):
        parent, root = self._chain(6)
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = tree_absorb_small_compartments(parent, root, labels, min_size=1)
        np.testing.assert_array_equal(out, labels)  # 2-vertex tip survives

    def test_interior_island_absorbed(self):
        # dendrite, one axon vertex in the middle, then dendrite again.
        parent, root = self._chain(5)
        labels = np.array([0, 0, 1, 0, 0])
        out = tree_absorb_small_compartments(parent, root, labels, min_size=1)
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0])

    def test_input_not_mutated(self):
        parent, root = self._chain(4)
        labels = np.array([0, 0, 1, 1])
        original = labels.copy()
        tree_absorb_small_compartments(parent, root, labels, min_size=2)
        np.testing.assert_array_equal(labels, original)

    def test_root_compartment_never_changed(self):
        # whole tree is a single small compartment; nothing to inherit.
        parent, root = self._chain(3)
        labels = np.array([1, 1, 1])
        out = tree_absorb_small_compartments(parent, root, labels, min_size=10)
        np.testing.assert_array_equal(out, [1, 1, 1])

    def test_cascade_nested_islands(self):
        # dend -> axon(1) -> soma(2); both small, both collapse to dendrite.
        parent, root = self._chain(4)
        labels = np.array([0, 1, 2, 2])  # comp sizes: {0:1, 1:1, 2:2}
        out = tree_absorb_small_compartments(parent, root, labels, min_size=2)
        np.testing.assert_array_equal(out, [0, 0, 0, 0])

    def test_min_weight_uses_node_weight(self):
        # 5 vertices but the axon tip spans little length -> absorbed by weight,
        # while min_size alone (tip has 2 vertices) would need min_size>=2.
        parent, root = self._chain(5)
        labels = np.array([0, 0, 0, 1, 1])
        weight = np.array([10.0, 10.0, 10.0, 0.4, 0.4])  # tip length 0.8
        out = tree_absorb_small_compartments(
            parent, root, labels, node_weight=weight, min_weight=1.0
        )
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0])

    def test_either_threshold_triggers(self):
        parent, root = self._chain(5)
        labels = np.array([0, 0, 0, 1, 1])  # tip: 2 vertices, length 20
        weight = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
        # min_size won't fire (tip has 2 > 1) but min_weight will not either
        # (20 > 5); neither active threshold triggers -> unchanged.
        out = tree_absorb_small_compartments(
            parent, root, labels, min_size=1, node_weight=weight, min_weight=5.0
        )
        np.testing.assert_array_equal(out, labels)
        # raising min_size to 2 trips the count criterion alone.
        out2 = tree_absorb_small_compartments(
            parent, root, labels, min_size=2, node_weight=weight, min_weight=5.0
        )
        np.testing.assert_array_equal(out2, [0, 0, 0, 0, 0])

    def test_requires_a_threshold(self):
        parent, root = self._chain(3)
        labels = np.array([0, 1, 1])
        with pytest.raises(ValueError):
            tree_absorb_small_compartments(parent, root, labels)
        with pytest.raises(ValueError):
            tree_absorb_small_compartments(parent, root, labels, min_weight=1.0)


class _WeightedFakeSkeleton(_FakeSkeleton):
    def __init__(self, parent, root, half_edge_length):
        super().__init__(parent, root)
        self.half_edge_length = np.asarray(half_edge_length, dtype=float)


class TestAbsorbSmallCompartmentsWrapper:
    def _chain(self, n):
        return np.array([-1] + list(range(n - 1))), 0

    def test_pulls_topology_min_size_only(self):
        # min_size alone needs no weights, so a plain skeleton is enough.
        parent, root = self._chain(6)
        skel = _FakeSkeleton(parent, root)
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = absorb_small_compartments(skel, labels, min_size=2)
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0])

    def test_min_weight_defaults_to_half_edge_length(self):
        # short tip (cable 0.8) absorbed; the wrapper supplies half_edge_length.
        parent, root = self._chain(5)
        weight = np.array([10.0, 10.0, 10.0, 0.4, 0.4])
        skel = _WeightedFakeSkeleton(parent, root, weight)
        labels = np.array([0, 0, 0, 1, 1])
        out = absorb_small_compartments(skel, labels, min_weight=1.0)
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0])
        # equivalent to passing the same weights explicitly
        explicit = tree_absorb_small_compartments(
            parent, root, labels, node_weight=weight, min_weight=1.0
        )
        np.testing.assert_array_equal(out, explicit)

    def test_explicit_node_weight_overrides_default(self):
        parent, root = self._chain(5)
        skel = _WeightedFakeSkeleton(parent, root, np.full(5, 100.0))  # would keep
        labels = np.array([0, 0, 0, 1, 1])
        out = absorb_small_compartments(
            skel, labels, node_weight=np.array([9, 9, 9, 0.4, 0.4]), min_weight=1.0
        )
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0])  # explicit weight wins
