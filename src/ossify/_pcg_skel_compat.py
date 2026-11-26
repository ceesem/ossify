"""Reimplementation of the skeletonization used by the CAVE SkeletonService.

The service builds its skeletons with ``pcg_skel``, which wraps
``meshparty.skeletonize.skeletonize_mesh`` with a fixed set of options
(``cc_vertex_thresh=0``, ``remove_zero_length_edges=True``, sphere soma
collapse, no smoothing). This module follows that code path step by step so
that the same graph and parameters give the same skeleton: the same vertices
in the same order, the same edges and the same graph-to-skeleton map.

It deliberately keeps behaviours that :mod:`ossify.skeletonization` does
differently, because matching them is the point:

* Edge weights are float32, and repeated edges add their lengths together.
* The skeleton is always grown from an extremity of the graph, even when a
  soma point is given. The soma only enters afterwards.
* A new branch joins the skeleton at the nearest skeleton vertex found by a
  search from the branch tip, not along the route back to the root.
* Each graph vertex maps to the skeleton vertex nearest it among the paths
  that invalidated it, which is not always the nearest one overall.
* The soma is collapsed on the finished skeleton: skeleton vertices inside the
  sphere, *plus the vertices one edge outside it*, are merged into a new vertex
  at the soma point.
* Skeleton vertices joined by a zero-length edge are merged.

Two things are intentionally not reproduced, both in places where the original
would fail outright or produce a broken skeleton: a run of three or more
coincident skeleton vertices is merged fully here (the original leaves a
doubled edge behind), and a branch whose bounded search misses the skeleton
through floating point round-off is retried without the bound (the original
raises).
"""

from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from . import utils

__all__ = ["pcg_skel_skeleton"]


def _mesh_csgraph(vertices: np.ndarray, edges: np.ndarray) -> sparse.csr_matrix:
    """The graph ``meshparty.trimesh_io.Mesh`` builds from ``link_edges``.

    Every edge is entered in both directions and the matrix is built without
    de-duplicating, so an edge listed twice carries twice its length.
    """
    symmetric = np.vstack([edges, edges[:, ::-1]])
    return utils.build_csgraph(
        vertices, symmetric, euclidean_weight=True, directed=True
    )


def _teasar(
    csgraph: sparse.csr_matrix, invalidation: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``mesh_teasar`` rooted by ``setup_root`` with no soma information.

    Returns
    -------
    edges : np.ndarray
        ``(K, 2)`` skeleton edges in graph indices, in the order the paths
        were found and running from each branch tip toward the skeleton.
    nearest : np.ndarray
        ``(N,)`` graph index of the skeleton vertex each graph vertex maps to.
    """
    n = csgraph.shape[0]
    root, _, root_predecessor, _, root_distance = utils.find_far_points_graph(
        csgraph, start_ind=0
    )
    root = int(root)

    valid = np.ones(n, dtype=bool)
    valid[root] = False
    visited = [root]
    is_visited = np.zeros(n, dtype=bool)
    is_visited[root] = True

    nearest = np.full(n, -1, dtype=np.int64)
    nearest_distance = np.full(n, np.inf)
    path_edges = []

    while valid.any():
        # Farthest vertex from the root that is still valid (first on ties).
        target = int(np.argmax(np.where(valid, root_distance, -1.0)))

        # The route back to the root bounds how far away the skeleton can be.
        max_branch = target
        while not is_visited[max_branch]:
            max_branch = int(root_predecessor[max_branch])
        limit = root_distance[target] - root_distance[max_branch]

        distance, predecessor = sparse.csgraph.dijkstra(
            csgraph, False, target, limit=limit, return_predecessors=True
        )
        branch = visited[int(np.argmin(distance[visited]))]
        if not np.isfinite(distance[branch]):
            distance, predecessor = sparse.csgraph.dijkstra(
                csgraph, False, target, return_predecessors=True
            )
            branch = visited[int(np.argmin(distance[visited]))]

        # Path from the target to where it joins the skeleton.
        path = [branch]
        node = branch
        while node != target:
            node = int(predecessor[node])
            path.append(node)
        path.reverse()

        visited.extend(path[:-1])
        is_visited[path[:-1]] = True
        path_edges.append(np.stack([path[:-1], path[1:]], axis=1))

        covered, _, source = sparse.csgraph.dijkstra(
            csgraph,
            False,
            path,
            limit=invalidation,
            min_only=True,
            return_predecessors=True,
        )
        reached = np.isfinite(covered)
        closer = reached & (covered < nearest_distance)
        nearest[closer] = source[closer]
        nearest_distance[closer] = covered[closer]
        valid[reached] = False

    return np.vstack(path_edges).astype(np.int64), nearest


def _collapse_zero_length_edges(
    vertices: np.ndarray,
    edges: np.ndarray,
    graph_map: np.ndarray,
    source_index: np.ndarray,
    root: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int]]:
    """Merge skeleton vertices joined by a zero-length edge.

    For each such edge ``[a, b]`` vertex ``a`` is removed and ``b`` kept.
    """
    zero_length = (
        np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) == 0
    )
    if not zero_length.any():
        return vertices, edges, graph_map, source_index, root

    merge_into = {int(a): int(b) for a, b in edges[zero_length]}

    def resolve(vertex: int) -> int:
        seen = set()
        while vertex in merge_into and vertex not in seen:
            seen.add(vertex)
            vertex = merge_into[vertex]
        return vertex

    keep = np.ones(len(vertices), dtype=bool)
    keep[list(merge_into)] = False
    new_index = np.full(len(vertices), -1, dtype=np.int64)
    new_index[keep] = np.arange(keep.sum())
    for removed in merge_into:
        new_index[removed] = new_index[resolve(removed)]

    new_edges = new_index[edges]
    new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]
    new_root = None if root is None else int(new_index[root])
    return (
        vertices[keep],
        new_edges,
        new_index[graph_map],
        source_index[keep],
        new_root,
    )


def _soma_via_sphere(
    vertices: np.ndarray, edges: np.ndarray, soma_point: np.ndarray, radius: float
) -> np.ndarray:
    """Skeleton vertices ``soma_via_sphere`` selects for collapse.

    The original keeps an edge when either end is inside the sphere, then
    takes the connected piece holding the vertex nearest the soma. The result
    is therefore the connected region inside the sphere plus the vertices one
    edge beyond it.
    """
    distance = np.linalg.norm(vertices - soma_point, axis=1)
    closest = int(np.argmin(distance))
    close = distance < radius

    kept = edges[close[edges[:, 0]] | close[edges[:, 1]]]
    n = len(vertices)
    graph = sparse.csr_matrix(
        (np.ones(len(kept), dtype=bool), (kept[:, 0], kept[:, 1])), shape=(n, n)
    )
    _, component = sparse.csgraph.connected_components(graph, directed=False)
    return np.flatnonzero(component == component[closest])


def _collapse_soma(
    vertices: np.ndarray,
    edges: np.ndarray,
    graph_map: np.ndarray,
    source_index: np.ndarray,
    soma_vertices: np.ndarray,
    soma_point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """``collapse_soma_skeleton``: merge ``soma_vertices`` into a new last vertex."""
    n = len(vertices)
    soma_index = n
    all_vertices = np.vstack([vertices, soma_point[np.newaxis, :]])

    merged_edges = edges.copy()
    merged_edges[np.isin(edges, soma_vertices)] = soma_index
    used = np.unique(merged_edges)
    new_edges = np.searchsorted(used, merged_edges)
    new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]

    keep = np.ones(n + 1, dtype=bool)
    keep[soma_vertices] = False
    new_index = np.full(n + 1, -1, dtype=np.int64)
    new_index[keep] = np.arange(keep.sum())
    new_index[soma_vertices] = new_index[soma_index]

    return (
        all_vertices[used],
        new_edges,
        new_index[graph_map],
        np.append(source_index, -1)[used],
        int(new_index[soma_index]),
    )


def _orient_toward_root(edges: np.ndarray, n_vertices: int, root: int) -> np.ndarray:
    """Flip edges, keeping their order, so each reads ``[child, parent]``."""
    graph = sparse.csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices),
    )
    hops = sparse.csgraph.dijkstra(graph, directed=False, indices=root, unweighted=True)
    child_first = hops[edges[:, 0]] > hops[edges[:, 1]]
    return np.where(child_first[:, np.newaxis], edges, edges[:, ::-1])


def pcg_skel_skeleton(
    vertices: np.ndarray,
    edges: np.ndarray,
    invalidation: float,
    soma_point: Optional[np.ndarray],
    soma_radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Skeletonize a connected graph the way the SkeletonService does.

    Parameters
    ----------
    vertices : np.ndarray
        ``(N, 3)`` graph vertex positions, ``N >= 2``.
    edges : np.ndarray
        ``(E, 2)`` positional graph edges, exactly as the service would see
        them: repeated edges are not removed.
    invalidation : float
        TEASAR invalidation distance (``invalidation_d``).
    soma_point : np.ndarray, optional
        Soma location (``root_point``, in the units of ``vertices``).
    soma_radius : float, optional
        Soma collapse radius (``collapse_radius``). The soma is collapsed only
        when both this and ``soma_point`` are given (``collapse_soma=True``).

    Returns
    -------
    vertices : np.ndarray
        ``(S, 3)`` skeleton vertices.
    edges : np.ndarray
        ``(S - 1, 2)`` skeleton edges as ``[child, parent]``.
    root : int
        Root index into the skeleton vertices.
    graph_to_skeleton : np.ndarray
        ``(N,)`` skeleton vertex for each graph vertex (``mesh_to_skel_map``).
    source_index : np.ndarray
        ``(S,)`` graph vertex behind each skeleton vertex (``mesh_index``),
        ``-1`` for the collapsed soma.
    """
    edges = edges[edges[:, 0] != edges[:, 1]]
    csgraph = _mesh_csgraph(vertices, edges)
    path_edges, nearest = _teasar(csgraph, invalidation)

    # Skeleton vertices are numbered in graph order.
    source_index = np.unique(path_edges)
    skel_vertices = vertices[source_index]
    skel_edges = np.searchsorted(source_index, path_edges)
    graph_map = np.searchsorted(source_index, nearest)

    if soma_point is not None and soma_radius is not None:
        skel_vertices, skel_edges, graph_map, source_index, _ = (
            _collapse_zero_length_edges(
                skel_vertices, skel_edges, graph_map, source_index, None
            )
        )
        soma_vertices = _soma_via_sphere(
            skel_vertices, skel_edges, soma_point, soma_radius
        )
        skel_vertices, skel_edges, graph_map, source_index, root = _collapse_soma(
            skel_vertices,
            skel_edges,
            graph_map,
            source_index,
            soma_vertices,
            soma_point,
        )
    else:
        if soma_point is not None:
            root = int(np.argmin(np.linalg.norm(skel_vertices - soma_point, axis=1)))
        else:
            root = int(
                utils.find_far_points_graph(
                    utils.build_csgraph(skel_vertices, skel_edges)
                )[0]
            )
        skel_vertices, skel_edges, graph_map, source_index, root = (
            _collapse_zero_length_edges(
                skel_vertices, skel_edges, graph_map, source_index, root
            )
        )

    skel_edges = _orient_toward_root(skel_edges, len(skel_vertices), root)
    return skel_vertices, skel_edges, root, graph_map, source_index
