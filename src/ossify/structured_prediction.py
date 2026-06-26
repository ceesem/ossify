"""Exact structured prediction of labels on a rooted tree.

This is the decoder seat of the compartment-labeling pipeline: given per-node
*unary* scores (from any encoder -- a tabular classifier today, a GNN later) and
a small set of declared *transition* costs between labels, it finds the
maximum-a-posteriori (MAP) labeling of a rooted tree exactly via tree dynamic
programming (max-product / Viterbi on a tree).

It is deliberately classical and model-agnostic: no learning happens here. The
transition costs encode biological priors declaratively (e.g. "axon should not
revert to dendrite") as soft, large-but-finite penalties; where strong unary
evidence overrides a prior, the offending edge is reported as a transition
violation -- a morphology QC signal rather than a silent override.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "TransitionSchema",
    "tree_map_decode",
    "decode_tree",
    "tree_absorb_small_compartments",
    "absorb_small_compartments",
    "count_components",
]


def _children_and_order(
    parent_array: np.ndarray, root: int
) -> Tuple[List[List[int]], List[int]]:
    """Child adjacency and a root-first topological order for a rooted tree.

    ``children[v]`` lists v's children; ``order`` lists every node before its
    descendants (so ``reversed(order)`` is a post-order). The root's parent is
    ``< 0``. Shared by the tree decode and the compartment-absorption passes.
    """
    n = len(parent_array)
    children: List[List[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = parent_array[v]
        if v != root and p >= 0:
            children[p].append(v)
    order: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children[v])
    return children, order


class TransitionSchema:
    """Declarative label set with soft parent->child transition costs.

    Parameters
    ----------
    classes : sequence
        Ordered class labels (strings or ints). Their order defines the column
        order of unary potentials and cost matrices.
    transitions : dict, optional
        Map ``{(parent_label, child_label): cost}`` of explicitly-set transition
        costs, in the same units as the unary log-scores. Pairs not listed take
        ``default_cost``. Use a large cost for a near-impossible transition; it
        stays finite so overwhelming evidence can still override it (and the edge
        is then flagged as a violation).
    root_classes : sequence, optional
        Labels the root node is allowed to take. Defaults to all classes.
    default_cost : float
        Cost for any *label change* not named in ``transitions``. Default 0.0
        (endorsed). Self-transitions are always free (diagonal forced to 0), so
        a nonzero ``default_cost`` is a uniform switching penalty: a node needs
        its unary advantage to exceed ``default_cost`` to flip away from its
        parent's label. Acts as a total-variation regularizer on the number of
        label changes over the tree.
    component_penalties : dict, optional
        Map ``{class_label: penalty}`` of a soft cardinality prior: the penalty
        (in the same nat units as the costs) charged per *additional* connected
        component of that class beyond the first. A cell normally has one axon
        and one apical, so a second component must clear this penalty in local
        evidence to survive the decode; a large-but-finite value keeps it
        overridable, ``np.inf`` makes it a hard "at most one" constraint. Classes
        not listed are uncapped (any number of components is free). See
        :func:`tree_map_decode`.
    """

    def __init__(
        self,
        classes: Sequence,
        transitions: Optional[dict] = None,
        root_classes: Optional[Sequence] = None,
        default_cost: float = 0.0,
        component_penalties: Optional[dict] = None,
    ):
        self.classes = list(classes)
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be unique.")
        self._index = {c: i for i, c in enumerate(self.classes)}
        self.default_cost = float(default_cost)
        self.transitions = dict(transitions or {})
        self.component_penalties = dict(component_penalties or {})
        self.root_classes = (
            list(self.classes) if root_classes is None else list(root_classes)
        )
        for c in self.root_classes:
            if c not in self._index:
                raise ValueError(f"root class {c!r} is not in classes.")
        for a, b in self.transitions:
            if a not in self._index or b not in self._index:
                raise ValueError(f"transition ({a!r}, {b!r}) references unknown class.")
        for c in self.component_penalties:
            if c not in self._index:
                raise ValueError(f"component penalty class {c!r} is not in classes.")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_index(self, label) -> int:
        return self._index[label]

    def with_root_classes(self, root_classes: Sequence) -> "TransitionSchema":
        """Return a copy of this schema with ``root_classes`` replaced.

        Useful for per-cell rooting decisions: build one base schema (classes,
        transition priors, ``default_cost``) and swap only which labels the root
        may take -- e.g. ``["dendrite"]`` for a soma-rooted cell vs ``["axon"]``
        for a terminal axon fragment -- without rebuilding the cost matrix.
        """
        return TransitionSchema(
            classes=self.classes,
            transitions=self.transitions,
            root_classes=root_classes,
            default_cost=self.default_cost,
            component_penalties=self.component_penalties,
        )

    @property
    def cost_matrix(self) -> np.ndarray:
        """``(K, K)`` matrix ``C[parent, child]`` of transition costs.

        Self-transitions (the diagonal) are 0 -- staying on a label is free --
        so ``default_cost`` acts purely as a switching penalty on label
        *changes*. An explicit ``(a, a)`` entry in ``transitions`` still
        overrides its diagonal cell.
        """
        K = self.n_classes
        C = np.full((K, K), self.default_cost, dtype=float)
        np.fill_diagonal(C, 0.0)
        for (a, b), cost in self.transitions.items():
            C[self._index[a], self._index[b]] = float(cost)
        return C

    @property
    def root_allowed_mask(self) -> np.ndarray:
        """Boolean ``(K,)`` mask of classes the root may take."""
        mask = np.zeros(self.n_classes, dtype=bool)
        for c in self.root_classes:
            mask[self._index[c]] = True
        return mask

    @property
    def onset_penalty(self) -> np.ndarray:
        """``(K,)`` per-extra-component penalty from ``component_penalties``.

        Entry ``k`` is the cost charged per connected component of class ``k``
        beyond the first; ``0`` for classes with no declared cardinality prior.
        Consumed by :func:`tree_map_decode` via :func:`decode_tree`.
        """
        p = np.zeros(self.n_classes, dtype=float)
        for c, cost in self.component_penalties.items():
            p[self._index[c]] = float(cost)
        return p


def tree_map_decode(
    parent_array: np.ndarray,
    root: int,
    unaries: np.ndarray,
    cost_matrix: np.ndarray,
    root_allowed_mask: Optional[np.ndarray] = None,
    *,
    onset_penalty: Optional[np.ndarray] = None,
    max_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact MAP labeling of a rooted tree by tree dynamic programming.

    Maximizes ``sum_n unaries[n, y_n] - sum_edges cost_matrix[y_parent, y_child]``
    over all labelings ``y``, exactly, in ``O(n * K^2)``.

    With ``onset_penalty`` it also subtracts a *cardinality* term -- a soft global
    prior on how many connected components each class forms. A class's components
    are its maximal connected same-label runs; equivalently, each component is one
    "onset" edge where a non-member parent meets a member child (or the root, if it
    is a member). The penalty stays exact because that onset count is edge-additive
    and folds into the DP via a small augmented state (per capped class, an onset
    count tracked up to ``max_components``), at ``O(n * K^2 * S^2)`` with
    ``S = (max_components + 1) ** (number of capped classes)``.

    Parameters
    ----------
    parent_array : np.ndarray
        ``(n,)`` positional parent of each node; the root's parent is ``< 0``.
    root : int
        Positional index of the root node.
    unaries : np.ndarray
        ``(n, K)`` per-node log-scores; higher is more likely.
    cost_matrix : np.ndarray
        ``(K, K)`` transition costs ``C[parent_label, child_label]`` (>= 0).
    root_allowed_mask : np.ndarray, optional
        Boolean ``(K,)`` mask of classes the root may take. Default all.
    onset_penalty : np.ndarray, optional
        ``(K,)`` penalty charged per component of each class beyond the first
        (nats; ``0`` = uncapped, ``np.inf`` = hard "at most one"). When ``None``
        or all-zero the fast unconstrained path runs and the rest is unchanged.
    max_components : int
        Onset counts are tracked (and so the penalty grows) only up to this many
        components per class, then saturate. Default 2: the first component is
        free and any further ones cost ``onset_penalty`` (saturating). Raising it
        lets the penalty accumulate per extra component before saturating.

    Returns
    -------
    labels : np.ndarray
        ``(n,)`` MAP class index per node.
    edge_cost : np.ndarray
        ``(n,)`` transition cost paid on each node's edge to its parent (0 at
        the root). Nonzero entries are the transition violations for QC. This is
        the pure transition cost; the cardinality term is global, not per-edge
        (recover per-class component counts with :func:`count_components`).
    """
    parent_array = np.asarray(parent_array)
    unaries = np.asarray(unaries, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n, K = unaries.shape
    if cost_matrix.shape != (K, K):
        raise ValueError("cost_matrix must be (K, K) matching unaries' K.")
    if root_allowed_mask is None:
        root_allowed_mask = np.ones(K, dtype=bool)
    if not root_allowed_mask.any():
        raise ValueError("root_allowed_mask forbids every class.")
    if onset_penalty is not None:
        onset_penalty = np.asarray(onset_penalty, dtype=float)
        if onset_penalty.shape != (K,):
            raise ValueError("onset_penalty must be (K,) matching unaries' K.")
        if np.any(onset_penalty < 0):
            raise ValueError("onset_penalty must be non-negative.")
        if np.any(onset_penalty > 0):
            return _tree_map_decode_cardinality(
                parent_array,
                root,
                unaries,
                cost_matrix,
                root_allowed_mask,
                onset_penalty,
                int(max_components),
            )

    children, order = _children_and_order(parent_array, root)

    # Leaves -> root: msg[v, a] = best score of v's subtree given v has label a.
    msg = unaries.copy()
    back = {}  # (v, child) -> (K,) best child label for each label of v
    for v in reversed(order):
        for child in children[v]:
            # M[a, b] = score of giving child label b when v has label a
            M = msg[child][None, :] - cost_matrix
            back[(v, child)] = M.argmax(axis=1)
            msg[v] += M.max(axis=1)

    # Root choice subject to the root constraint.
    root_scores = msg[root].copy()
    root_scores[~root_allowed_mask] = -np.inf
    labels = np.full(n, -1, dtype=int)
    labels[root] = int(root_scores.argmax())

    # Root -> leaves: backtrack the stored argmax choices.
    for v in order:
        for child in children[v]:
            labels[child] = back[(v, child)][labels[v]]

    edge_cost = np.zeros(n, dtype=float)
    for v in range(n):
        p = parent_array[v]
        if v != root and p >= 0:
            edge_cost[v] = cost_matrix[labels[p], labels[v]]
    return labels, edge_cost


def _tree_map_decode_cardinality(
    parent_array: np.ndarray,
    root: int,
    unaries: np.ndarray,
    cost_matrix: np.ndarray,
    root_allowed_mask: np.ndarray,
    onset_penalty: np.ndarray,
    max_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cardinality-augmented tree MAP (see :func:`tree_map_decode`).

    State per node is ``(label, m)`` where ``m`` is an onset-count vector over the
    capped classes (those with ``onset_penalty > 0``), each count in
    ``0..max_components``. ``msg[v, a, m]`` is the best score of subtree(v) given
    ``label[v] = a`` and ``m`` onsets occurring strictly inside subtree(v); v's own
    onset (relative to its parent) is charged when v is folded into its parent, and
    the root's own onset plus the global penalty are applied once at the root.
    """
    n, K = unaries.shape
    NEG = -np.inf
    capped = np.flatnonzero(onset_penalty > 0)
    nc = int(capped.size)
    cap = max(1, int(max_components))
    radix = cap + 1
    S = radix**nc
    slot_of = np.full(K, -1, dtype=int)
    slot_of[capped] = np.arange(nc)

    # Decode each state index to its count vector, and precompute the two state
    # transitions the fold needs: saturating add of two states, and +1 onset.
    counts = np.zeros((S, nc), dtype=int)
    for s in range(S):
        x = s
        for i in range(nc):
            counts[s, i] = x % radix
            x //= radix

    def _encode(cnt: np.ndarray) -> int:
        return int(sum(int(cnt[i]) * (radix**i) for i in range(nc)))

    state_add = np.zeros((S, S), dtype=int)
    for s1 in range(S):
        for s2 in range(S):
            state_add[s1, s2] = _encode(np.minimum(counts[s1] + counts[s2], cap))
    onset_step = np.zeros((nc, S), dtype=int)
    for i in range(nc):
        for s in range(S):
            c = counts[s].copy()
            c[i] = min(c[i] + 1, cap)
            onset_step[i, s] = _encode(c)

    children, order = _children_and_order(parent_array, root)

    # Leaves -> root. msg[v] is (K, S); bp[v] holds, per folded child, the chosen
    # (child_label, child_state, prev_acc_state) for each (parent_label, new_state).
    msg = np.full((n, K, S), NEG)
    bp: List[Optional[List[np.ndarray]]] = [None] * n
    for v in reversed(order):
        acc = np.full((K, S), NEG)
        acc[:, 0] = unaries[v, :]  # state 0 = all-zero counts, no internal onsets
        child_bps: List[np.ndarray] = []
        for child in children[v]:
            new_acc = np.full((K, S), NEG)
            bp_c = np.zeros((K, S, 3), dtype=int)
            for a in range(K):
                for b in range(K):
                    base = msg[child, b, :] - cost_matrix[a, b]  # (S,) over m_c
                    if not np.any(np.isfinite(base)):
                        continue
                    # Map the child's internal state through this edge's onset
                    # (a new component of b starts iff b is capped and a != b).
                    if slot_of[b] >= 0 and a != b:
                        step = onset_step[slot_of[b]]
                        cb = np.full(S, NEG)
                        cb_mc = np.zeros(S, dtype=int)
                        for mc in range(S):
                            sp = step[mc]
                            if base[mc] > cb[sp]:
                                cb[sp] = base[mc]
                                cb_mc[sp] = mc
                    else:
                        cb = base
                        cb_mc = np.arange(S)
                    # Convolve the accumulated counts with this child's.
                    for mprev in range(S):
                        av = acc[a, mprev]
                        if not np.isfinite(av):
                            continue
                        for sp in range(S):
                            val = cb[sp]
                            if not np.isfinite(val):
                                continue
                            t = state_add[mprev, sp]
                            cand = av + val
                            if cand > new_acc[a, t]:
                                new_acc[a, t] = cand
                                bp_c[a, t, 0] = b
                                bp_c[a, t, 1] = cb_mc[sp]
                                bp_c[a, t, 2] = mprev
            acc = new_acc
            child_bps.append(bp_c)
        msg[v] = acc
        bp[v] = child_bps

    # Root choice: add the root's own onset and the global cardinality penalty.
    pen_capped = onset_penalty[capped]
    best_score, best_a, best_s = NEG, -1, 0
    for a in range(K):
        if not root_allowed_mask[a]:
            continue
        for s in range(S):
            val = msg[root, a, s]
            if not np.isfinite(val):
                continue
            tot_s = onset_step[slot_of[a], s] if slot_of[a] >= 0 else s
            cnt = counts[tot_s]
            extra = np.maximum(cnt - 1, 0)
            pen = float(np.sum(pen_capped[extra > 0] * extra[extra > 0]))
            score = val - pen
            if score > best_score:
                best_score, best_a, best_s = score, a, s
    if best_a < 0:
        raise ValueError("no labeling satisfies the root constraint.")

    # Root -> leaves: unwind each node's children in reverse fold order.
    labels = np.full(n, -1, dtype=int)
    state_at = np.zeros(n, dtype=int)
    labels[root] = best_a
    state_at[root] = best_s
    for v in order:
        child_bps = bp[v]
        if not child_bps:
            continue
        a = labels[v]
        cur = state_at[v]
        chs = children[v]
        for idx in range(len(chs) - 1, -1, -1):
            b, mc, prev = child_bps[idx][a, cur]
            c = chs[idx]
            labels[c] = int(b)
            state_at[c] = int(mc)
            cur = int(prev)

    edge_cost = np.zeros(n, dtype=float)
    for v in range(n):
        p = parent_array[v]
        if v != root and p >= 0:
            edge_cost[v] = cost_matrix[labels[p], labels[v]]
    return labels, edge_cost


def decode_tree(
    skeleton,
    unaries: np.ndarray,
    schema: TransitionSchema,
    return_labels_as: str = "index",
    max_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a rooted skeleton/segment graph with a :class:`TransitionSchema`.

    Thin wrapper over :func:`tree_map_decode` that pulls topology from a
    SkeletonLayer-like object (``parent_node_array``, ``root_positional``) and the
    cost matrix, root mask, and cardinality penalties from the schema.

    Parameters
    ----------
    skeleton : SkeletonLayer or SegmentGraph
        Provides ``parent_node_array`` and ``root_positional``.
    unaries : np.ndarray
        ``(n_nodes, n_classes)`` log-scores, columns ordered as ``schema.classes``.
    schema : TransitionSchema
    return_labels_as : {"index", "label"}
        Return class indices (default) or the schema's class labels.
    max_components : int
        Passed through to :func:`tree_map_decode` when the schema declares
        ``component_penalties``; ignored otherwise.

    Returns
    -------
    labels : np.ndarray
        ``(n_nodes,)`` per-node labels (indices or labels per ``return_labels_as``).
    edge_cost : np.ndarray
        Per-node transition cost to parent (QC).
    """
    labels, edge_cost = tree_map_decode(
        skeleton.parent_node_array,
        skeleton.root_positional,
        unaries,
        schema.cost_matrix,
        schema.root_allowed_mask,
        onset_penalty=schema.onset_penalty,
        max_components=max_components,
    )
    if return_labels_as == "label":
        lookup = np.array(schema.classes, dtype=object)
        labels = lookup[labels]
    elif return_labels_as != "index":
        raise ValueError("return_labels_as must be 'index' or 'label'.")
    return labels, edge_cost


def tree_absorb_small_compartments(
    parent_array: np.ndarray,
    root: int,
    labels: np.ndarray,
    min_size: Optional[int] = None,
    node_weight: Optional[np.ndarray] = None,
    min_weight: Optional[float] = None,
) -> np.ndarray:
    """Merge small same-label compartments into their parent's label.

    A *compartment* is a maximal connected run of equally-labeled nodes on the
    rooted tree -- its boundaries are exactly the edges where the label changes.
    Any non-root compartment that is too small is relabeled to the label of the
    node just above it (its parent compartment's label). This is a post-decode
    cleanup for the short, noise-driven label flips a tree decode produces at
    terminals and around skeletonization artifacts.

    Two independent size criteria are offered; a compartment is absorbed if it
    falls under *either* active threshold:

    * ``min_size`` -- node count. A good tell for skeletonization/topology
      artifacts, which show up as a few stray vertices regardless of scale.
    * ``min_weight`` -- summed ``node_weight`` (e.g. path length). The
      scale-aware criterion: how much evidence the compartment actually spans.

    The rule is applied root-first to a fixpoint, so absorptions cascade: a
    cleaned compartment can in turn absorb a child that has now become adjacent
    to a different label. The root compartment has no parent and is never
    changed.

    Parameters
    ----------
    parent_array : np.ndarray
        ``(n,)`` positional parent of each node; the root's parent is ``< 0``.
    root : int
        Positional index of the root node.
    labels : np.ndarray
        ``(n,)`` per-node labels (e.g. the output of :func:`tree_map_decode`).
        Not modified in place.
    min_size : int, optional
        Absorb compartments with this many nodes or fewer.
    node_weight : np.ndarray, optional
        ``(n,)`` per-node weights (e.g. path length) summed to size a
        compartment. Required when ``min_weight`` is given.
    min_weight : float, optional
        Absorb compartments whose summed ``node_weight`` is this or less.

    Returns
    -------
    labels : np.ndarray
        ``(n,)`` cleaned labels (a new array; the input is left untouched).
    """
    if min_size is None and min_weight is None:
        raise ValueError("set min_size and/or min_weight.")
    if min_weight is not None and node_weight is None:
        raise ValueError("min_weight requires node_weight.")
    parent_array = np.asarray(parent_array)
    labels = np.asarray(labels).copy()
    n = labels.shape[0]
    if node_weight is not None:
        node_weight = np.asarray(node_weight, dtype=float)

    _, order = _children_and_order(parent_array, root)

    while True:
        # Compartment id = top-most node of each maximal same-label run.
        comp = np.empty(n, dtype=int)
        for v in order:
            p = parent_array[v]
            if v == root or p < 0 or labels[p] != labels[v]:
                comp[v] = v
            else:
                comp[v] = comp[p]

        small = np.zeros(n, dtype=bool)  # indexed by compartment representative
        if min_size is not None:
            small |= np.bincount(comp, minlength=n) <= min_size
        if min_weight is not None:
            small |= np.bincount(comp, weights=node_weight, minlength=n) <= min_weight

        # Root-first relabel: a small compartment takes its parent node's
        # (already-updated) label, which cascades down the whole compartment.
        changed = False
        for v in order:
            p = parent_array[v]
            if v == root or p < 0 or not small[comp[v]]:
                continue
            if labels[v] != labels[p]:
                labels[v] = labels[p]
                changed = True
        if not changed:
            return labels


def absorb_small_compartments(
    skeleton,
    labels: np.ndarray,
    min_size: Optional[int] = None,
    node_weight: Optional[np.ndarray] = None,
    min_weight: Optional[float] = None,
) -> np.ndarray:
    """Clean small compartments on a skeleton/segment graph.

    Thin wrapper over :func:`tree_absorb_small_compartments` that pulls topology
    from a SkeletonLayer-like object (``parent_node_array``, ``root_positional``).
    When ``min_weight`` is given without an explicit ``node_weight``, the
    per-node weight defaults to ``skeleton.half_edge_length`` -- the per-vertex
    cable length that sums to true cable length over a compartment -- so
    ``min_weight`` reads directly as a cable-length threshold.

    Parameters
    ----------
    skeleton : SkeletonLayer or SegmentGraph
        Provides ``parent_node_array``, ``root_positional``, and (for the
        ``min_weight`` default) ``half_edge_length``.
    labels : np.ndarray
        ``(n_nodes,)`` per-node labels (e.g. from :func:`decode_tree`).
    min_size : int, optional
        Absorb compartments with this many nodes or fewer.
    node_weight : np.ndarray, optional
        ``(n_nodes,)`` per-node weights. Defaults to ``skeleton.half_edge_length``
        when ``min_weight`` is set and this is omitted.
    min_weight : float, optional
        Absorb compartments whose summed ``node_weight`` is this or less.

    Returns
    -------
    labels : np.ndarray
        ``(n_nodes,)`` cleaned labels (a new array).
    """
    if min_weight is not None and node_weight is None:
        node_weight = skeleton.half_edge_length
    return tree_absorb_small_compartments(
        skeleton.parent_node_array,
        skeleton.root_positional,
        labels,
        min_size=min_size,
        node_weight=node_weight,
        min_weight=min_weight,
    )


def count_components(parent_array: np.ndarray, root: int, labels: np.ndarray) -> dict:
    """Number of connected same-label components per label on a rooted tree.

    A component is a maximal connected run of equally-labeled nodes; its count
    equals the number of *onset* edges for that label (a node whose parent has a
    different label, or the root). This is the cardinality QC counterpart to the
    soft prior in :func:`tree_map_decode`: after a decode it tells you how many
    axons/apicals were kept, so a rare second component can be audited before it
    is trusted (cf. the per-edge transition violations in ``edge_cost``).

    Parameters
    ----------
    parent_array : np.ndarray
        ``(n,)`` positional parent of each node; the root's parent is ``< 0``.
    root : int
        Positional index of the root node.
    labels : np.ndarray
        ``(n,)`` per-node labels (indices or labels).

    Returns
    -------
    dict
        ``{label: n_components}`` for every label present.
    """
    parent_array = np.asarray(parent_array)
    labels = np.asarray(labels)
    counts: dict = {}
    for v in range(labels.shape[0]):
        p = parent_array[v]
        if v == root or p < 0 or labels[p] != labels[v]:
            key = labels[v].item() if hasattr(labels[v], "item") else labels[v]
            counts[key] = counts.get(key, 0) + 1
    return counts
