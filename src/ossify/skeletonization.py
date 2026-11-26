"""Skeletonization of spatially embedded graphs.

Turns a connected spatial graph (typically a level-2 chunk graph) into a rooted
tree skeleton using a TEASAR-style algorithm, keeping track of which skeleton
vertex every graph vertex belongs to.

The work is split into stages that are each usable on their own:

1. **Contract** an optional soma region of the graph into a single new vertex
   (:func:`contract_vertices`). This is how a skeleton gets a root that was
   never a vertex of the graph.
2. **Extract paths** from the farthest remaining vertex back toward the root,
   invalidating the graph around each new path.
3. **Map** every graph vertex to its nearest skeleton vertex along the graph.

:func:`skeletonize_graph` runs these on plain arrays and :func:`skeletonize`
applies the result to a :class:`~ossify.Cell`.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse, spatial

from . import utils
from ._pcg_skel_compat import pcg_skel_skeleton
from .base import Cell, Link

__all__ = [
    "SkeletonResult",
    "contract_vertices",
    "detect_soma",
    "find_soma_point",
    "add_skeleton_radius",
    "skeletonize_graph",
    "skeletonize",
]


@dataclass
class SkeletonResult:
    """Output of :func:`skeletonize_graph`. All indices are positional.

    Attributes
    ----------
    vertices : np.ndarray
        ``(S, 3)`` skeleton vertex positions.
    edges : np.ndarray
        ``(S - 1, 2)`` skeleton edges, each ``[child, parent]`` so that every
        edge points toward the root.
    root : int
        Index of the root in ``vertices``.
    graph_to_skeleton : np.ndarray
        ``(N,)`` skeleton vertex for each vertex of the input graph. Every
        graph vertex is mapped.
    source_index : np.ndarray
        ``(S,)`` graph vertex each skeleton vertex was taken from, or ``-1``
        for a vertex that was not in the graph (a contracted soma).
    """

    vertices: np.ndarray
    edges: np.ndarray
    root: int
    graph_to_skeleton: np.ndarray
    source_index: np.ndarray


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}.")
    return vertices


def _as_edges(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    """Undirected edge list with self-loops and repeated edges removed."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) > 0 and (edges.min() < 0 or edges.max() >= n_vertices):
        raise ValueError("edges refer to vertices that do not exist.")
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def _as_mask(group: Union[np.ndarray, Sequence[int]], n_vertices: int) -> np.ndarray:
    """Boolean vertex mask from either a boolean mask or a list of indices."""
    group = np.asarray(group)
    if group.dtype == bool:
        if group.shape != (n_vertices,):
            raise ValueError(
                f"A boolean mask must have one value per vertex ({n_vertices}), "
                f"got shape {group.shape}."
            )
        return group
    mask = np.zeros(n_vertices, dtype=bool)
    mask[group.astype(np.int64)] = True
    return mask


def _build_csgraph(vertices: np.ndarray, edges: np.ndarray) -> sparse.csr_matrix:
    """Symmetric graph weighted by Euclidean edge length.

    ``edges`` must already be de-duplicated (see :func:`_as_edges`): building a
    sparse matrix sums repeated entries, which would lengthen those edges.
    """
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    n = len(vertices)
    return sparse.csr_matrix(
        (
            np.concatenate([weights, weights]),
            (
                np.concatenate([edges[:, 0], edges[:, 1]]),
                np.concatenate([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(n, n),
    )


def contract_vertices(
    vertices: np.ndarray,
    edges: np.ndarray,
    group: Union[np.ndarray, Sequence[int]],
    position: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract a group of graph vertices into a single new vertex.

    The group is removed and replaced by one vertex, appended after the
    vertices that are kept. Edges into the group are rewired to the new vertex,
    edges inside it are dropped, and edges that become parallel are merged.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 3)`` vertex positions.
    edges : np.ndarray
        ``(E, 2)`` positional edges.
    group : np.ndarray or sequence of int
        Vertices to contract, as a boolean mask over vertices or as indices.
    position : array-like, optional
        Location of the new vertex. It need not coincide with any vertex of
        the graph. Defaults to the centroid of the group.

    Returns
    -------
    new_vertices : np.ndarray
        ``(N - n_group + 1, 3)`` positions. Kept vertices hold their relative
        order and the contracted vertex is last.
    new_edges : np.ndarray
        ``(E', 2)`` positional edges into ``new_vertices``, each sorted and
        unique.
    vertex_map : np.ndarray
        ``(N,)`` index into ``new_vertices`` for each original vertex.

    Notes
    -----
    Contracting a group that is not connected joins parts of the graph that
    were not adjacent, which adds shortcuts. That is occasionally the point,
    but it is never done implicitly: see ``soma_radius`` in
    :func:`skeletonize_graph`.

    Examples
    --------
    >>> vertices = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    >>> edges = np.array([[0, 1], [1, 2], [2, 3]])
    >>> new_vertices, new_edges, vertex_map = contract_vertices(
    ...     vertices, edges, [1, 2]
    ... )
    >>> vertex_map
    array([0, 2, 2, 1])
    """
    vertices = _as_vertices(vertices)
    n = len(vertices)
    mask = _as_mask(group, n)
    if not mask.any():
        raise ValueError("Cannot contract an empty group of vertices.")

    if position is None:
        new_position = vertices[mask].mean(axis=0)
    else:
        new_position = np.asarray(position, dtype=float).reshape(3)

    n_kept = n - int(mask.sum())
    vertex_map = np.full(n, n_kept, dtype=np.int64)
    vertex_map[~mask] = np.arange(n_kept)

    new_vertices = np.vstack([vertices[~mask], new_position[np.newaxis, :]])
    new_edges = _as_edges(vertex_map[np.asarray(edges, dtype=np.int64)], n_kept + 1)
    return new_vertices, new_edges, vertex_map


def _contracted_csgraph(
    vertices: np.ndarray, edges: np.ndarray, vertex_map: np.ndarray
) -> sparse.csr_matrix:
    """Length-weighted graph after a contraction, keeping original edge lengths.

    An edge into the contracted vertex keeps the length it had in the original
    graph (the shortest, where several merge) rather than the distance to the
    contracted vertex's new position. Graph distance from the contracted vertex
    is then distance from the *surface* of the contracted region. Measured from
    its centre instead, every vertex just outside a soma would be at least a
    soma radius away, beyond the reach of invalidation, and each would become
    its own stub branch.
    """
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    mapped = np.sort(vertex_map[edges], axis=1)
    keep = mapped[:, 0] != mapped[:, 1]
    mapped, lengths = mapped[keep], lengths[keep]

    unique_edges, inverse = np.unique(mapped, axis=0, return_inverse=True)
    weights = np.full(len(unique_edges), np.inf)
    np.minimum.at(weights, inverse.ravel(), lengths)

    n = int(vertex_map.max()) + 1
    return sparse.csr_matrix(
        (
            np.concatenate([weights, weights]),
            (
                np.concatenate([unique_edges[:, 0], unique_edges[:, 1]]),
                np.concatenate([unique_edges[:, 1], unique_edges[:, 0]]),
            ),
        ),
        shape=(n, n),
    )


def _soma_ball(
    vertices: np.ndarray,
    csgraph: sparse.csr_matrix,
    soma_point: np.ndarray,
    soma_radius: float,
) -> np.ndarray:
    """Mask of the vertices to collapse for a spherical soma.

    Starts from the vertex nearest ``soma_point`` (always included, so the
    mask is never empty) and keeps only the vertices within ``soma_radius``
    that are joined to it without leaving the ball. A process that merely
    passes back through the ball is left alone, since collapsing it would
    short-circuit the arbor through the soma.
    """
    distance = np.linalg.norm(vertices - soma_point, axis=1)
    seed = int(np.argmin(distance))
    in_ball = distance <= soma_radius
    in_ball[seed] = True

    ball_index = np.flatnonzero(in_ball)
    _, component = sparse.csgraph.connected_components(
        csgraph[ball_index][:, ball_index], directed=False
    )
    seed_component = component[np.searchsorted(ball_index, seed)]

    mask = np.zeros(len(vertices), dtype=bool)
    mask[ball_index[component == seed_component]] = True
    return mask


def _extract_paths(
    csgraph: sparse.csr_matrix, root: int, invalidation: float
) -> np.ndarray:
    """Core TEASAR loop: choose which graph vertices form the skeleton.

    Repeatedly takes the farthest vertex from the root that is still valid,
    walks it back toward the root until it meets the skeleton, and invalidates
    everything within ``invalidation`` (along the graph) of the new path.

    The walk follows the shortest route to the root only while it is in
    uncovered territory. As soon as it reaches a vertex an earlier path has
    already covered, it turns and takes the shortest route to that path
    instead. Without this, a thick process represented by several parallel
    chains of vertices gets one skeleton path per chain, because each branch
    beyond it has its own equally short route home along a different chain.

    Parameters
    ----------
    csgraph : sparse.csr_matrix
        Symmetric, connected, length-weighted graph.
    root : int
        Root vertex.
    invalidation : float
        Graph distance around each path within which vertices are covered.

    Returns
    -------
    np.ndarray
        ``(N,)`` parent of each skeleton vertex, ``-1`` for the root and
        ``-2`` for vertices that are not part of the skeleton.
    """
    n = csgraph.shape[0]
    root_distance, predecessor = sparse.csgraph.dijkstra(
        csgraph, directed=False, indices=root, return_predecessors=True
    )

    parent = np.full(n, -2, dtype=np.int64)
    parent[root] = -1
    valid = np.ones(n, dtype=bool)
    valid[root] = False
    # For each covered vertex: distance to the skeleton path that covers it
    # most closely, and the next vertex on the way there.
    cover_distance = np.full(n, np.inf)
    toward_skeleton = np.full(n, -1, dtype=np.int64)

    # Root distances never change, so one sort replaces an argmax per path.
    for target in np.argsort(-root_distance, kind="stable"):
        if not valid[target]:
            continue
        path = []
        node = int(target)
        while parent[node] == -2:
            path.append(node)
            if np.isfinite(cover_distance[node]):
                step = int(toward_skeleton[node])
            else:
                step = int(predecessor[node])
            parent[node] = step
            node = step
        # Invalidate around the vertex the path joins as well, so that the
        # root covers its own surroundings rather than only its neighbours.
        path.append(node)

        distance, toward, _ = sparse.csgraph.dijkstra(
            csgraph,
            directed=False,
            indices=path,
            limit=invalidation,
            min_only=True,
            return_predecessors=True,
        )
        # Strictly closer only: on a tie the earlier path keeps the vertex,
        # which keeps every chain of ``toward_skeleton`` steps getting closer.
        closer = distance < cover_distance
        cover_distance[closer] = distance[closer]
        toward_skeleton[closer] = toward[closer]
        valid[np.isfinite(distance)] = False

    return parent


def _soma_core(
    vertices: np.ndarray,
    edges: np.ndarray,
    radius: np.ndarray,
    seed: int,
    large_sphere_fraction: float,
) -> np.ndarray:
    """Vertices that are provably inside the same blob as ``seed``.

    Every vertex carries a sphere known to lie inside the cell. Two vertices
    belong together when either

    - they share a graph edge shorter than the larger of their two radii, so
      one centre sits inside the other's sphere, or
    - both spheres are at least ``large_sphere_fraction`` of the seed's and
      they overlap. This is what holds together a soma that chunk boundaries
      have cut into several large pieces, none big enough to swallow the next.

    Vertices along a neurite satisfy neither: they are spaced much farther
    apart than they are thick. The core is the group containing ``seed``.
    """
    n = len(vertices)
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    engulfed = length < np.maximum(radius[edges[:, 0]], radius[edges[:, 1]])
    first, second = [edges[engulfed, 0]], [edges[engulfed, 1]]

    large = np.flatnonzero(radius >= large_sphere_fraction * radius[seed])
    if len(large) > 1 and radius[seed] > 0:
        pairs = spatial.cKDTree(vertices[large]).query_pairs(
            2 * radius[seed], output_type="ndarray"
        )
        if len(pairs) > 0:
            a, b = large[pairs[:, 0]], large[pairs[:, 1]]
            overlap = np.linalg.norm(vertices[a] - vertices[b], axis=1) < (
                radius[a] + radius[b]
            )
            first.append(a[overlap])
            second.append(b[overlap])

    first, second = np.concatenate(first), np.concatenate(second)
    _, component = sparse.csgraph.connected_components(
        sparse.csr_matrix(
            (np.ones(len(first), dtype=bool), (first, second)), shape=(n, n)
        ),
        directed=False,
    )
    return component == component[seed]


@dataclass
class _SomaTree:
    """A skeleton grown from a contracted soma, in contracted-graph indices."""

    root: int
    parent: np.ndarray  # -1 for the root, -2 off the skeleton
    children: dict  # skeleton vertex -> list of child skeleton vertices
    cable: np.ndarray  # length of the subtree hanging from each vertex,
    # including the edge to its parent
    nearest: np.ndarray  # skeleton vertex for each vertex of the original graph


def _soma_tree(
    vertices: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    soma_point: np.ndarray,
    invalidation: float,
) -> _SomaTree:
    """Contract ``mask`` to a root at ``soma_point`` and skeletonize from it."""
    work_vertices, _, vertex_map = contract_vertices(
        vertices, edges, mask, position=soma_point
    )
    csgraph = _contracted_csgraph(vertices, edges, vertex_map)
    root = len(work_vertices) - 1
    parent = _extract_paths(csgraph, root, invalidation)

    nodes = np.flatnonzero(parent != -2)
    _, _, nearest = sparse.csgraph.dijkstra(
        csgraph, directed=False, indices=nodes, min_only=True, return_predecessors=True
    )

    children: dict = {}
    for node in nodes[nodes != root]:
        children.setdefault(int(parent[node]), []).append(int(node))
    order = [root]
    position = 0
    while position < len(order):
        order.extend(children.get(order[position], []))
        position += 1

    cable = np.zeros(len(work_vertices))
    below = nodes[nodes != root]
    cable[below] = np.linalg.norm(
        work_vertices[below] - work_vertices[parent[below]], axis=1
    )
    for node in order[:0:-1]:
        if parent[node] != root:
            cable[parent[node]] += cable[node]

    return _SomaTree(root, parent, children, cable, nearest[vertex_map])


def _subtree(tree: _SomaTree, start: int) -> list:
    nodes = [start]
    position = 0
    while position < len(nodes):
        nodes.extend(tree.children.get(nodes[position], []))
        position += 1
    return nodes


def find_soma_point(
    vertices: np.ndarray,
    edges: np.ndarray,
    vertex_radius: np.ndarray,
    vertex_volume: np.ndarray,
    *,
    min_peak_radius: float = 2200.0,
    min_soma_radius: float = 3000.0,
    large_sphere_fraction: float = 0.5,
) -> Optional[np.ndarray]:
    """Guess where the soma is from the graph alone, if there is one.

    The thickest point of a neuron is its soma. This takes the vertex with
    the largest inscribed sphere, gathers the blob around it (see
    :func:`detect_soma`), and returns the blob's centre of volume, provided
    the blob is big enough to be a cell body.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 3)`` vertex positions.
    edges : np.ndarray
        ``(E, 2)`` positional edges.
    vertex_radius : np.ndarray
        ``(N,)`` inscribed radius of each vertex (``max_dt_nm``).
    vertex_volume : np.ndarray
        ``(N,)`` volume of each vertex (``size_nm3``).
    min_peak_radius : float
        The largest inscribed sphere must have at least this radius. Default
        2200 (nm).
    min_soma_radius : float
        The blob must have at least the volume of a sphere of this radius.
        Default 3000 (nm).
    large_sphere_fraction : float
        As for :func:`detect_soma`. Default 0.5.

    Returns
    -------
    Optional[np.ndarray]
        ``(3,)`` estimated soma centre, or None if nothing in the graph is
        large enough to be a soma.

    Notes
    -----
    The thresholds come from 80 proofread cortical neurons in two datasets.
    Their somas had a largest sphere of at least 2.7 um and a blob radius of
    at least 4.0 um, while nothing more than 25 um from a nucleus had a sphere
    over 1.7 um. On those cells the estimate fell a median of 1.2 um, and at
    most 2.7 um, from the nucleus centroid. Other cell types and species were
    not checked.

    An object holding several somas yields only the largest.

    Examples
    --------
    >>> point = find_soma_point(vertices, edges, max_dt_nm, size_nm3)
    >>> if point is not None:
    ...     mask = detect_soma(vertices, edges, max_dt_nm, size_nm3, point)
    """
    vertices = _as_vertices(vertices)
    n = len(vertices)
    edges = _as_edges(edges, n)
    radius = np.nan_to_num(np.asarray(vertex_radius, dtype=float))
    volume = np.nan_to_num(np.asarray(vertex_volume, dtype=float))
    if radius.shape != (n,) or volume.shape != (n,):
        raise ValueError(
            "vertex_radius and vertex_volume must each have one value per vertex "
            f"({n}), got shapes {radius.shape} and {volume.shape}."
        )
    if n == 0:
        return None

    seed = int(np.argmax(radius))
    if radius[seed] < min_peak_radius:
        return None
    core = _soma_core(vertices, edges, radius, seed, large_sphere_fraction)
    core_volume = volume[core].sum()
    if core_volume < 4 / 3 * np.pi * min_soma_radius**3:
        return None
    return (vertices[core] * volume[core, np.newaxis]).sum(axis=0) / core_volume


def detect_soma(
    vertices: np.ndarray,
    edges: np.ndarray,
    vertex_radius: np.ndarray,
    vertex_volume: np.ndarray,
    soma_point: Union[np.ndarray, Sequence[float]],
    *,
    invalidation: float = 7500.0,
    min_branch_length: float = 20000.0,
    seed_distance: float = 15000.0,
    large_sphere_fraction: float = 0.5,
    base_thickness_fraction: float = 0.25,
    base_distance_factor: float = 1.5,
) -> np.ndarray:
    """Find the graph vertices that make up the soma, from the graph alone.

    The aim is the smallest set of vertices that, collapsed into one root at
    ``soma_point``, leaves every real process leaving the soma as its own
    branch and nothing else. It works in two passes.

    **First pass: the core.** Each vertex has a sphere of radius
    ``vertex_radius`` known to lie inside the cell. Starting from the largest
    sphere near ``soma_point``, the core gathers the vertices whose spheres
    engulf their graph neighbours, plus large spheres that overlap one another.
    Only something as thick as a soma does either.

    **Second pass: branches.** The graph is skeletonized from that core, and
    the branches leaving the root are examined:

    - a branch with less than ``min_branch_length`` of cable is not a process
      at all but a leftover piece of soma surface, and is added to the soma;
    - where a branch forks close to the soma into two branches that each pass
      that test, and the stretch before the fork is thick, that stretch is a
      bulge of the soma rather than a shared trunk. It is added to the soma,
      so the two branches leave the root separately.

    This repeats until nothing more is added.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 3)`` vertex positions.
    edges : np.ndarray
        ``(E, 2)`` positional edges of a connected graph.
    vertex_radius : np.ndarray
        ``(N,)`` radius of the largest sphere centred in each vertex that fits
        inside the cell, in the units of ``vertices``. For a level-2 graph
        this is ``max_dt_nm``. Missing values count as zero.
    vertex_volume : np.ndarray
        ``(N,)`` volume of each vertex in the cubed units of ``vertices``
        (``size_nm3``). Used to estimate the soma radius. Missing values count
        as zero.
    soma_point : array-like
        Approximate centre of the soma, for example a nucleus centroid.
    invalidation : float
        Invalidation distance used for the trial skeletons. Use the value the
        final skeleton will be built with. Default 7500.
    min_branch_length : float
        Cable a branch leaving the soma must have to count as a process.
        Default 20000.
    seed_distance : float
        The core is seeded at the largest sphere within this distance of
        ``soma_point``, or at the nearest vertex if there is none. Default
        15000.
    large_sphere_fraction : float
        Spheres at least this fraction of the seed's radius are joined to the
        core when they overlap it. Default 0.5.
    base_thickness_fraction : float
        A stretch before a fork counts as thick when its typical radius is at
        least this fraction of the seed's radius. Default 0.25.
    base_distance_factor : float
        A fork counts as close when it lies within this many soma radii of
        ``soma_point``. The soma radius is that of a sphere with the volume of
        the core. Default 1.5.

    Returns
    -------
    np.ndarray
        ``(N,)`` boolean mask of soma vertices, suitable for the ``soma_mask``
        argument of :func:`skeletonize_graph`.

    Examples
    --------
    >>> mask = detect_soma(vertices, edges, max_dt_nm, size_nm3, nucleus_point)
    >>> result = skeletonize_graph(
    ...     vertices, edges, soma_mask=mask, soma_point=nucleus_point
    ... )
    """
    vertices = _as_vertices(vertices)
    n = len(vertices)
    edges = _as_edges(edges, n)
    soma_point = np.asarray(soma_point, dtype=float).reshape(3)
    radius = np.nan_to_num(np.asarray(vertex_radius, dtype=float))
    volume = np.nan_to_num(np.asarray(vertex_volume, dtype=float))
    if radius.shape != (n,) or volume.shape != (n,):
        raise ValueError(
            "vertex_radius and vertex_volume must each have one value per vertex "
            f"({n}), got shapes {radius.shape} and {volume.shape}."
        )

    soma_distance = np.linalg.norm(vertices - soma_point, axis=1)
    candidates = np.flatnonzero(soma_distance < seed_distance)
    if len(candidates) > 0:
        seed = int(candidates[np.argmax(radius[candidates])])
    else:
        seed = int(np.argmin(soma_distance))

    mask = _soma_core(vertices, edges, radius, seed, large_sphere_fraction)
    soma_radius = (3 * volume[mask].sum() / (4 * np.pi)) ** (1 / 3)
    thick = base_thickness_fraction * radius[seed]
    close = base_distance_factor * soma_radius

    # Each round changes the soma, and with it the skeleton, so re-grow and
    # look again. It settles in two or three rounds; the cap is a backstop.
    for _ in range(8):
        tree = _soma_tree(vertices, edges, mask, soma_point, invalidation)
        outside = ~mask
        # Thickness of the cell at each skeleton vertex: the largest sphere
        # among the graph vertices that map to it.
        calibre = np.zeros(len(tree.parent))
        np.maximum.at(calibre, tree.nearest[outside], radius[outside])
        distance = np.append(soma_distance[outside], 0.0)

        absorbed = []
        frontier = list(tree.children.get(tree.root, []))
        while frontier:
            branch = frontier.pop()
            if tree.cable[branch] < min_branch_length:
                absorbed.extend(_subtree(tree, branch))
                continue
            # Follow the branch to its first real fork.
            base = [branch]
            node = branch
            while True:
                onward = [
                    child
                    for child in tree.children.get(node, [])
                    if tree.cable[child] >= min_branch_length
                ]
                if len(onward) != 1:
                    break
                node = onward[0]
                base.append(node)
            if (
                len(onward) >= 2
                and np.median(calibre[base]) >= thick
                and distance[node] <= close
            ):
                for vertex in base:
                    absorbed.append(vertex)
                    for child in tree.children.get(vertex, []):
                        if tree.cable[child] < min_branch_length:
                            absorbed.extend(_subtree(tree, child))
                frontier.extend(onward)

        if not absorbed:
            break
        is_absorbed = np.zeros(len(tree.parent), dtype=bool)
        is_absorbed[absorbed] = True
        mask = mask | is_absorbed[tree.nearest]

    return mask


def skeletonize_graph(
    vertices: np.ndarray,
    edges: np.ndarray,
    *,
    invalidation: float = 7500.0,
    root: Optional[int] = None,
    soma_point: Optional[Union[np.ndarray, Sequence[float]]] = None,
    soma_radius: Optional[float] = None,
    soma_mask: Optional[np.ndarray] = None,
    vertex_radius: Optional[np.ndarray] = None,
    vertex_volume: Optional[np.ndarray] = None,
    method: Literal["default", "pcg_skel"] = "default",
) -> SkeletonResult:
    """Skeletonize a connected, spatially embedded graph given as arrays.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 3)`` vertex positions.
    edges : np.ndarray
        ``(E, 2)`` positional edges. Direction, repeats and self-loops are
        ignored. The graph must be a single connected component.
    invalidation : float
        Distance along the graph, in the units of ``vertices``, around each
        skeleton path within which graph vertices count as already covered.
        Side branches shorter than this are absorbed into their parent path.
        Default 7500 (nm).
    root : int, optional
        Graph vertex to root the skeleton at. Cannot be combined with a soma
        collapse, where the root is the collapsed vertex.
    soma_point : array-like, optional
        Location of the soma. What it does depends on the other arguments:

        - alone: root the skeleton at the nearest graph vertex;
        - with ``soma_radius`` or ``soma_mask``: the position of the new root
          vertex that the soma region is collapsed to.
    soma_radius : float, optional
        Collapse the graph vertices within this distance of ``soma_point``
        into a single root vertex at ``soma_point``. Only vertices joined to
        the soma without leaving the ball are collapsed, and the vertex
        nearest ``soma_point`` always is. Requires ``soma_point``.
    soma_mask : np.ndarray, optional
        Boolean mask over graph vertices to collapse into the root instead of
        a ball. Used exactly as given, connected or not. The root is placed at
        ``soma_point``, or the centroid of the masked vertices without one.
        Takes precedence over ``soma_radius``.
    vertex_radius, vertex_volume : np.ndarray, optional
        Per-vertex inscribed radius and volume (``max_dt_nm`` and ``size_nm3``
        for a level-2 graph). Giving both, with ``soma_point``, finds the soma
        from the graph with :func:`detect_soma` and collapses it. This is the
        recommended way to handle a soma; a ball both merges too much and
        splits or leaves stubs depending on its size. Without a ``soma_point``
        the soma is located with :func:`find_soma_point`, and if nothing in
        the graph is large enough to be one the skeleton is built without a
        soma. Cannot be combined with ``soma_radius`` or ``soma_mask``.
    method : {"default", "pcg_skel"}
        ``"pcg_skel"`` reproduces the skeletons of the CAVE SkeletonService
        (``pcg_skel`` on top of MeshParty) instead of the algorithm described
        here. See the notes for what changes. Default ``"default"``.

    Returns
    -------
    SkeletonResult
        Skeleton vertices and edges, the root, and the graph-to-skeleton map.

    Raises
    ------
    ValueError
        If the graph is not a single connected component, or the arguments
        are inconsistent.

    Notes
    -----
    Without ``root`` or any soma argument the root is an extremity of the
    graph, found by hopping between mutually farthest vertices.

    The result is always a tree. Loops in the graph are cut where the two
    routes back to the root are equally long, and are not reported.

    With ``method="pcg_skel"`` the same graph and parameters give the same
    vertices, edges and graph-to-skeleton map as the SkeletonService, where
    ``invalidation`` is its ``invalidation_d``, ``soma_point`` its root point
    in nanometers and ``soma_radius`` its ``collapse_radius``. Passing both
    soma arguments corresponds to ``collapse_soma=True``. ``root`` and
    ``soma_mask`` have no equivalent there and raise. That method differs from
    the default in several ways:

    - the skeleton is grown from an extremity of the graph even when a soma
      is given, and the soma is collapsed on the finished skeleton;
    - the collapse takes skeleton vertices inside the sphere and those one
      edge outside it;
    - a branch joins the skeleton at the nearest skeleton vertex, not along
      its route to the root;
    - a graph vertex maps to the nearest skeleton vertex among the paths that
      invalidated it, which is not always the nearest overall;
    - repeated edges count double, and skeleton vertices joined by a
      zero-length edge are merged.

    Examples
    --------
    >>> vertices = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    >>> edges = np.array([[0, 1], [1, 2], [2, 3]])
    >>> result = skeletonize_graph(vertices, edges, invalidation=5, root=0)
    >>> result.graph_to_skeleton
    array([0, 1, 2, 3])
    """
    vertices = _as_vertices(vertices)
    n_graph = len(vertices)
    if n_graph == 0:
        raise ValueError("Cannot skeletonize a graph with no vertices.")
    raw_edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = _as_edges(raw_edges, n_graph)
    if method not in ("default", "pcg_skel"):
        raise ValueError(f'method must be "default" or "pcg_skel", got {method!r}.')

    if invalidation < 0:
        raise ValueError(f"invalidation must be non-negative, got {invalidation}.")
    if soma_radius is not None and soma_point is None and soma_mask is None:
        raise ValueError("soma_radius requires a soma_point to measure from.")
    if soma_point is not None:
        soma_point = np.asarray(soma_point, dtype=float).reshape(3)

    adaptive = vertex_radius is not None or vertex_volume is not None
    if adaptive:
        if vertex_radius is None or vertex_volume is None:
            raise ValueError(
                "vertex_radius and vertex_volume must be given together to "
                "detect the soma."
            )
        if soma_radius is not None or soma_mask is not None:
            raise ValueError(
                "vertex_radius/vertex_volume detect the soma themselves and "
                "cannot be combined with soma_radius or soma_mask."
            )
        if method == "pcg_skel":
            raise ValueError('Soma detection is not available with method="pcg_skel".')

    if n_graph == 1 and method == "default":
        # Nothing to trace: the skeleton is the vertex itself.
        return SkeletonResult(
            vertices=vertices.copy(),
            edges=np.empty((0, 2), dtype=np.int64),
            root=0,
            graph_to_skeleton=np.zeros(1, dtype=np.int64),
            source_index=np.zeros(1, dtype=np.int64),
        )

    csgraph = _build_csgraph(vertices, edges)
    n_components, _ = sparse.csgraph.connected_components(csgraph, directed=False)
    if n_components != 1:
        raise ValueError(
            f"The graph must be a single connected component, found {n_components}."
        )

    if method == "pcg_skel":
        if root is not None or soma_mask is not None:
            raise ValueError(
                'root and soma_mask are not available with method="pcg_skel", '
                "which has no equivalent of either."
            )
        if n_graph < 2:
            raise ValueError('method="pcg_skel" needs at least two graph vertices.')
        # Repeated edges are passed through untouched: they change the result.
        return SkeletonResult(
            *pcg_skel_skeleton(
                vertices, raw_edges, invalidation, soma_point, soma_radius
            )
        )

    # Decide what, if anything, collapses into the root.
    collapse_mask = None
    if adaptive and soma_point is None:
        soma_point = find_soma_point(vertices, edges, vertex_radius, vertex_volume)
        adaptive = soma_point is not None
    if adaptive:
        collapse_mask = detect_soma(
            vertices,
            edges,
            vertex_radius,
            vertex_volume,
            soma_point,
            invalidation=invalidation,
        )
    elif soma_mask is not None:
        collapse_mask = _as_mask(soma_mask, n_graph)
        if not collapse_mask.any():
            raise ValueError("soma_mask is empty: it selects no graph vertices.")
    elif soma_radius is not None:
        collapse_mask = _soma_ball(vertices, csgraph, soma_point, soma_radius)

    if collapse_mask is not None:
        if root is not None:
            raise ValueError(
                "root cannot be combined with a soma collapse: the collapsed "
                "soma vertex is the root."
            )
        work_vertices, _, vertex_map = contract_vertices(
            vertices, edges, collapse_mask, position=soma_point
        )
        csgraph = _contracted_csgraph(vertices, edges, vertex_map)
        work_root = len(work_vertices) - 1
        # Graph vertex behind each working vertex; the contracted one has none.
        work_source = np.append(np.flatnonzero(~collapse_mask), -1)
    else:
        work_vertices = vertices
        vertex_map = np.arange(n_graph)
        work_source = np.arange(n_graph)
        if root is not None:
            if not 0 <= root < n_graph:
                raise ValueError(
                    f"root {root} is not a vertex of the graph (0 to {n_graph - 1})."
                )
            work_root = int(root)
        elif soma_point is not None:
            work_root = int(np.argmin(np.linalg.norm(vertices - soma_point, axis=1)))
        else:
            work_root = int(utils.find_far_points_graph(csgraph)[0])

    parent = _extract_paths(csgraph, work_root, invalidation)

    # Number the skeleton vertices in graph order.
    skeleton_nodes = np.flatnonzero(parent != -2)
    skeleton_index = np.full(len(work_vertices), -1, dtype=np.int64)
    skeleton_index[skeleton_nodes] = np.arange(len(skeleton_nodes))

    children = skeleton_nodes[skeleton_nodes != work_root]
    skeleton_edges = np.stack(
        [skeleton_index[children], skeleton_index[parent[children]]], axis=1
    )

    # Each graph vertex goes to the skeleton vertex nearest along the graph.
    _, _, nearest = sparse.csgraph.dijkstra(
        csgraph,
        directed=False,
        indices=skeleton_nodes,
        min_only=True,
        return_predecessors=True,
    )

    return SkeletonResult(
        vertices=work_vertices[skeleton_nodes],
        edges=skeleton_edges,
        root=int(skeleton_index[work_root]),
        graph_to_skeleton=skeleton_index[nearest][vertex_map],
        source_index=work_source[skeleton_nodes],
    )


def skeletonize(
    cell: Cell,
    *,
    invalidation: float = 7500.0,
    root: Optional[int] = None,
    root_as_positional: bool = False,
    soma_point: Optional[Union[np.ndarray, Sequence[float]]] = None,
    soma_radius: Optional[float] = None,
    soma_mask: Optional[Union[str, np.ndarray]] = None,
    detect_soma: bool = False,
    radius_feature: str = "max_dt_nm",
    volume_feature: str = "size_nm3",
    method: Literal["default", "pcg_skel"] = "default",
    overwrite: bool = False,
) -> Cell:
    """Skeletonize a cell's graph layer and add the result as its skeleton.

    The skeleton is linked to the graph, so features and annotations can be
    mapped between the two layers as for any other linked skeleton.

    Parameters
    ----------
    cell : Cell
        Cell with a graph layer that is a single connected component.
    invalidation : float
        Distance along the graph around each skeleton path within which graph
        vertices count as already covered. Side branches shorter than this
        are absorbed. Default 7500 (nm).
    root : int, optional
        Graph vertex to root the skeleton at, as a vertex index (see
        ``root_as_positional``). Cannot be combined with a soma collapse.
    root_as_positional : bool
        Whether ``root`` is a positional index into the graph vertices rather
        than a vertex index. Default False.
    soma_point : array-like, optional
        Location of the soma. Alone, the skeleton is rooted at the nearest
        graph vertex. With ``soma_radius`` or ``soma_mask`` it is the position
        of the new root vertex.
    soma_radius : float, optional
        Collapse graph vertices within this distance of ``soma_point``, and
        connected to it within that ball, into a new root vertex at
        ``soma_point``.
    soma_mask : str or np.ndarray, optional
        Graph vertices to collapse into the root instead of a ball: a boolean
        array over graph vertices, or the name of a boolean graph feature.
    detect_soma : bool
        Find the soma from the graph itself and collapse it into a root at
        ``soma_point``, instead of using a ball or a mask. Needs the two graph
        features named below. This is the recommended choice for level-2
        graphs. See :func:`ossify.skeletonization.detect_soma`. Without a
        ``soma_point`` the soma is located from the graph too, and the
        skeleton is built without one if nothing is large enough. Default
        False.
    radius_feature : str
        Graph feature holding each vertex's inscribed radius, used when
        ``detect_soma`` is True. Default ``"max_dt_nm"``.
    volume_feature : str
        Graph feature holding each vertex's volume, used when ``detect_soma``
        is True. Default ``"size_nm3"``.
    method : {"default", "pcg_skel"}
        ``"pcg_skel"`` reproduces the skeleton the CAVE SkeletonService would
        build from this graph, given the same ``invalidation``, ``soma_point``
        (in nanometers) and ``soma_radius``. It does not accept ``root`` or
        ``soma_mask``. See :func:`skeletonize_graph` for how it differs.
    overwrite : bool
        Replace an existing skeleton. Default False, which raises if the cell
        already has one. Replacing a skeleton also drops any links that other
        layers and annotations had to the old one.

    Returns
    -------
    Cell
        The same cell, with a skeleton layer added.

    Raises
    ------
    ValueError
        If the cell has no graph, already has a skeleton and ``overwrite`` is
        False, or the graph is not a single connected component.

    Examples
    --------
    >>> cell = ossify.load_cell("my_cell.osy")
    >>> ossify.skeletonize(
    ...     cell, soma_point=[766848, 547712, 899480], detect_soma=True
    ... )
    >>> cell.graph.map_index_to_layer("skeleton")  # skeleton vertex per graph vertex
    """
    graph = cell.graph
    if graph is None:
        raise ValueError("The cell has no graph layer to skeletonize.")
    if cell.skeleton is not None and not overwrite:
        raise ValueError(
            "The cell already has a skeleton. Pass overwrite=True to replace it."
        )

    if root is not None and not root_as_positional:
        matches = np.flatnonzero(graph.vertex_index == root)
        if len(matches) == 0:
            raise ValueError(f"root {root} is not a vertex index of the graph.")
        root = int(matches[0])

    if isinstance(soma_mask, str):
        soma_mask = np.asarray(graph.get_feature(soma_mask), dtype=bool)

    vertex_radius = vertex_volume = None
    if detect_soma:
        missing = [
            name
            for name in (radius_feature, volume_feature)
            if name not in graph.feature_names
        ]
        if missing:
            raise ValueError(
                f"detect_soma needs the graph feature(s) {missing}, which this "
                "graph does not have. Level-2 graphs carry them when loaded "
                "with their properties."
            )
        vertex_radius = np.asarray(graph.get_feature(radius_feature), dtype=float)
        vertex_volume = np.asarray(graph.get_feature(volume_feature), dtype=float)

    result = skeletonize_graph(
        graph.vertices,
        graph.edges_positional,
        invalidation=invalidation,
        root=root,
        soma_point=soma_point,
        soma_radius=soma_radius,
        soma_mask=soma_mask,
        vertex_radius=vertex_radius,
        vertex_volume=vertex_volume,
        method=method,
    )

    if cell.skeleton is not None:
        cell.remove_layer(cell.SKEL_LN)

    return cell.add_skeleton(
        vertices=result.vertices,
        edges=result.edges,
        root=result.root,
        root_as_positional=True,
        linkage=Link(
            mapping=result.graph_to_skeleton,
            source=cell.GRAPH_LN,
            map_value_is_index=False,
        ),
    )


def add_skeleton_radius(
    cell: Cell,
    *,
    radius_feature: str = "max_dt_nm",
    volume_feature: str = "size_nm3",
) -> Cell:
    """Add radius estimates to a skeleton from the graph it was built on.

    Two skeleton features are added:

    ``radius``
        The equivalent radius of each segment, as ``pcg_skel`` and the CAVE
        skeleton service report it: the radius of a cylinder with the
        segment's volume and length. Every vertex of a segment shares one
        value. The segment holding the root is treated as a sphere of its
        volume instead, which for a collapsed soma is the soma radius. If the
        root is not a branch point, that segment is a whole process and the
        sphere is meaningless there; this is kept to match ``pcg_skel``.
    ``radius_local``
        The largest inscribed radius among the graph vertices that map to
        each skeleton vertex. It follows the process along its length, but
        reads low where a chunk boundary has split a thick process lengthwise.

    Parameters
    ----------
    cell : Cell
        Cell with a skeleton linked to its graph.
    radius_feature : str
        Graph feature holding each vertex's inscribed radius. Default
        ``"max_dt_nm"``.
    volume_feature : str
        Graph feature holding each vertex's volume. Default ``"size_nm3"``.

    Returns
    -------
    Cell
        The same cell, with the two features added to its skeleton.

    Examples
    --------
    >>> ossify.skeletonize(cell, soma_point=nucleus_nm, detect_soma=True)
    >>> ossify.add_skeleton_radius(cell)
    >>> cell.skeleton.get_feature("radius")
    """
    graph, skel = cell.graph, cell.skeleton
    if graph is None or skel is None:
        raise ValueError("The cell needs both a graph and a skeleton.")
    missing = [
        name
        for name in (radius_feature, volume_feature)
        if name not in graph.feature_names
    ]
    if missing:
        raise ValueError(f"The graph has no feature(s) {missing}.")

    n = skel.n_vertices
    to_skeleton = np.asarray(
        graph.map_index_to_layer(cell.SKEL_LN, as_positional=True), dtype=np.int64
    )
    inscribed = np.nan_to_num(
        np.asarray(graph.get_feature(radius_feature), dtype=float)
    )
    volume = np.nan_to_num(np.asarray(graph.get_feature(volume_feature), dtype=float))

    local = np.zeros(n)
    np.maximum.at(local, to_skeleton, inscribed)
    vertex_volume = np.bincount(to_skeleton, weights=volume, minlength=n)

    # Length of the edge from each vertex to its parent, zero at the root.
    edges = skel.edges_positional
    edge_length = np.zeros(n)
    if len(edges) > 0:
        edge_length[edges[:, 0]] = np.linalg.norm(
            skel.vertices[edges[:, 0]] - skel.vertices[edges[:, 1]], axis=1
        )

    root = skel.root_positional
    equivalent = np.zeros(n)
    for segment in skel.segments_positional:
        segment_volume = vertex_volume[segment].sum()
        # A segment's length runs up to the vertex it hangs from.
        segment_length = edge_length[segment].sum()
        if root in segment:
            value = (3 * segment_volume / (4 * np.pi)) ** (1 / 3)
        elif segment_length > 0:
            value = np.sqrt(segment_volume / (np.pi * segment_length))
        else:
            value = local[segment].max()
        equivalent[segment] = value

    skel.add_feature(equivalent, "radius")
    skel.add_feature(local, "radius_local")
    return cell
