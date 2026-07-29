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
]


def _ossify_version() -> str:
    """The installed ossify release, for provenance in serialized dicts."""
    from . import __version__

    return __version__


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
    """

    #: Bumped when :meth:`to_dict`'s shape changes incompatibly. Independent of
    #: the ossify package version.
    _SCHEMA_VERSION = 1

    def __init__(
        self,
        classes: Sequence,
        transitions: Optional[dict] = None,
        root_classes: Optional[Sequence] = None,
        default_cost: float = 0.0,
    ):
        self.classes = list(classes)
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be unique.")
        self._index = {c: i for i, c in enumerate(self.classes)}
        self.default_cost = float(default_cost)
        self.transitions = dict(transitions or {})
        self.root_classes = (
            list(self.classes) if root_classes is None else list(root_classes)
        )
        for c in self.root_classes:
            if c not in self._index:
                raise ValueError(f"root class {c!r} is not in classes.")
        for a, b in self.transitions:
            if a not in self._index or b not in self._index:
                raise ValueError(f"transition ({a!r}, {b!r}) references unknown class.")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_index(self, label) -> int:
        return self._index[label]

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

    def to_dict(self) -> dict:
        """Plain-data representation, round-trippable via :meth:`from_dict`.

        ``transitions`` is written as a list of ``[parent, child, cost]``
        triples rather than a ``{(parent, child): cost}`` dict, since JSON/YAML
        cannot use tuples as object keys. Includes a ``version`` (this dict
        shape, bumped on breaking changes) and ``ossify_version`` (the release
        that wrote it, for provenance/debugging) alongside the parameters.
        """
        return {
            "type": "TransitionSchema",
            "version": self._SCHEMA_VERSION,
            "ossify_version": _ossify_version(),
            "classes": list(self.classes),
            "transitions": [[a, b, cost] for (a, b), cost in self.transitions.items()],
            "root_classes": list(self.root_classes),
            "default_cost": self.default_cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionSchema":
        """Reconstruct a :class:`TransitionSchema` from :meth:`to_dict` output."""
        version = d.get("version", 1)
        if version > cls._SCHEMA_VERSION:
            raise ValueError(
                f"TransitionSchema dict has schema version {version}, newer than "
                f"the {cls._SCHEMA_VERSION} this ossify release supports; "
                "upgrade ossify to load it."
            )
        return cls(
            classes=d["classes"],
            transitions={(a, b): cost for a, b, cost in d.get("transitions", [])},
            root_classes=d.get("root_classes"),
            default_cost=d.get("default_cost", 0.0),
        )


def tree_map_decode(
    parent_array: np.ndarray,
    root: int,
    unaries: np.ndarray,
    cost_matrix: np.ndarray,
    root_allowed_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact MAP labeling of a rooted tree by tree dynamic programming.

    Maximizes ``sum_n unaries[n, y_n] - sum_edges cost_matrix[y_parent, y_child]``
    over all labelings ``y``, exactly, in ``O(n * K^2)``.

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

    Returns
    -------
    labels : np.ndarray
        ``(n,)`` MAP class index per node.
    edge_cost : np.ndarray
        ``(n,)`` transition cost paid on each node's edge to its parent (0 at
        the root). Nonzero entries are the transition violations for QC.
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

    children: List[List[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = parent_array[v]
        if v != root and p >= 0:
            children[p].append(v)

    # Root-first topological order (parents before children) via DFS over the
    # child adjacency; reversed gives a post-order (children before parents).
    order: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children[v])

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


def decode_tree(
    skeleton,
    unaries: np.ndarray,
    schema: TransitionSchema,
    return_labels_as: str = "index",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a rooted skeleton/segment graph with a :class:`TransitionSchema`.

    Thin wrapper over :func:`tree_map_decode` that pulls topology from a
    SkeletonLayer-like object (``parent_node_array``, ``root_positional``).

    Parameters
    ----------
    skeleton : SkeletonLayer or SegmentGraph
        Provides ``parent_node_array`` and ``root_positional``.
    unaries : np.ndarray
        ``(n_nodes, n_classes)`` log-scores, columns ordered as ``schema.classes``.
    schema : TransitionSchema
    return_labels_as : {"index", "label"}
        Return class indices (default) or the schema's class labels.

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

    children: List[List[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = parent_array[v]
        if v != root and p >= 0:
            children[p].append(v)

    # Root-first order: every node precedes its descendants.
    order: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children[v])

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
