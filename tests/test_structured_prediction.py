import itertools

import numpy as np
import pytest

from ossify import Cell
from ossify.structured_prediction import (
    TransitionSchema,
    absorb_small_compartments,
    decode_tree,
    tree_absorb_small_compartments,
    tree_map_decode,
)


def _objective(parent, root, unaries, C, labels):
    """Score of a labeling under the MAP objective (for brute-force checks)."""
    n = len(labels)
    s = unaries[np.arange(n), labels].sum()
    for v in range(n):
        p = parent[v]
        if v != root and p >= 0:
            s -= C[labels[p], labels[v]]
    return s


def _brute_force(parent, root, unaries, C, root_mask):
    n, K = unaries.shape
    best_score, best = -np.inf, None
    for assign in itertools.product(range(K), repeat=n):
        if not root_mask[assign[root]]:
            continue
        score = _objective(parent, root, unaries, C, np.array(assign))
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
