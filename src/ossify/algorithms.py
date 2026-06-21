from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from .base import Cell, SkeletonLayer

__all__ = [
    "strahler_number",
    "branch_order",
    "smooth_features",
    "label_axon_from_synapse_flow",
    "label_axon_from_spectral_split",
    "synapse_betweenness",
    "segregation_index",
    "segment_aggregate",
    "subtree_aggregate",
    "path_to_root_aggregate",
    "root_connected_mask",
]


def _as_skeleton(cell: Union[Cell, SkeletonLayer]) -> SkeletonLayer:
    """Return the SkeletonLayer from a Cell or pass a SkeletonLayer through."""
    if isinstance(cell, Cell):
        skel = cell.skeleton
        if skel is None:
            raise ValueError("Cell does not have a skeleton.")
        return skel
    return cell


def segment_aggregate(
    cell: Union[Cell, SkeletonLayer],
    values: np.ndarray,
    agg: Union[str, callable] = "mean",
) -> np.ndarray:
    """Replace each vertex value with an aggregate over its skeleton segment.

    A segment is an unbranched span between topological points. This computes a
    single aggregate per segment and broadcasts it back to every vertex in that
    segment, producing a piecewise-constant per-vertex array.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The skeleton (or a Cell with one).
    values : np.ndarray
        Per-vertex values in skeleton positional order.
    agg : Union[str, callable], optional
        Aggregation applied within each segment. A string is looked up as a numpy
        reduction (e.g. "mean", "max", "median", "sum"); a callable is applied to
        each segment's value array. Default "mean".

    Returns
    -------
    np.ndarray
        Per-vertex array (positional) with each vertex set to its segment aggregate.
    """
    skel = _as_skeleton(cell)
    values = np.asarray(values)
    fn = agg if callable(agg) else getattr(np, agg)
    seg_vals = np.array([fn(values[seg]) for seg in skel.segments_positional])
    return seg_vals[skel.segment_map]


def root_connected_mask(
    cell: Union[Cell, SkeletonLayer],
    cut_vertices: np.ndarray,
    as_positional: bool = False,
) -> np.ndarray:
    """Boolean mask of vertices still connected to the root after cutting edges.

    Each vertex in ``cut_vertices`` is severed from its parent, then the mask
    marks every vertex in the same connected component as the root. Useful for
    restricting a candidate region to the part that remains attached to the root.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The skeleton (or a Cell with one).
    cut_vertices : np.ndarray
        Vertices to cut from their parents.
    as_positional : bool, optional
        Whether ``cut_vertices`` are positional indices. Default False.

    Returns
    -------
    np.ndarray
        Boolean array (positional) marking vertices in the root's component.
    """
    skel = _as_skeleton(cell)
    G = skel.cut_graph(
        cut_vertices,
        directed=False,
        euclidean_weight=False,
        as_positional=as_positional,
    )
    _, comp_labels = sparse.csgraph.connected_components(G, directed=False)
    return comp_labels == comp_labels[skel.root_positional]


def _strahler_path(baseline):
    out = np.full(len(baseline), -1, dtype=np.int64)
    last_val = 1
    for ii in np.arange(len(out)):
        if baseline[ii] > last_val:
            last_val = baseline[ii]
        elif baseline[ii] == last_val:
            last_val += 1
        out[ii] = last_val
    return out


def _laplacian_offset(
    skeleton: Cell,
) -> sparse.csr_matrix:
    """Compute the degree-normalized adjacency matrix part of the Laplacian matrix.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Neuron object

    Returns
    -------
    sparse.spmatrix
        Degree-normalized adjacency matrix in sparse format.
    """
    Amat = skeleton.csgraph_binary_undirected
    deg = np.array(Amat.sum(axis=0)).squeeze()
    Dmat = sparse.diags_array(1 / np.sqrt(deg))
    Lmat = Dmat @ Amat @ Dmat
    return Lmat


def smooth_features(
    cell: Union[Cell, SkeletonLayer],
    feature: np.ndarray,
    alpha: float = 0.90,
) -> np.ndarray:
    """Computes a smoothed feature spreading that is akin to steady-state solutions to the heat equation on the skeleton graph.

    Parameters
    ----------
    cell : Cell
        Neuron object
    feature : np.ndarray
        The initial feature array. Must be Nxm, where N is the number of skeleton vertices
    alpha : float, optional
        A neighborhood influence parameter between 0 and 1. Higher values give more influence to neighbors, by default 0.90.

    Returns
    -------
    np.ndarray
        The smoothed feature array
    """
    if isinstance(cell, SkeletonLayer):
        skel = cell
    else:
        skel = cell.skeleton
    Smat = _laplacian_offset(skel)
    Imat = sparse.eye(Smat.shape[0])
    invertLap = Imat - alpha * Smat
    feature = np.atleast_2d(feature).reshape(Smat.shape[0], -1)
    F = sparse.linalg.spsolve(invertLap, feature)
    return np.squeeze((1 - alpha) * F)


def strahler_number(cell: Union[Cell, SkeletonLayer]) -> np.ndarray:
    """Compute Strahler number on a skeleton, starting at 1 for each tip.
    Returns a feature suitable for a SkeletonLayer.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The skeleton to compute the Strahler number on.
        For convenience, you can pass a Cell object, but note
        that the return feature is always for the skeleton.

    Returns
    -------
    np.ndarray
        The Strahler number for each vertex in the skeleton.
    """
    if isinstance(cell, Cell):
        skel: SkeletonLayer = cell.skeleton
        if skel is None:
            raise ValueError("Cell is does not have a skeleton.")
    else:
        skel: SkeletonLayer = cell
    strahler_number = np.full(skel.n_vertices, -1, dtype=np.int32)
    for pth in skel.cover_paths_positional[::-1]:
        pth_vals = _strahler_path(strahler_number[pth])
        strahler_number[pth] = pth_vals
        pind = skel.parent_node_array[pth[-1]]
        if pind >= 0:
            if strahler_number[pth[-1]] > strahler_number[pind]:
                strahler_number[pind] = strahler_number[pth[-1]]
            elif strahler_number[pth[-1]] == strahler_number[pind]:
                strahler_number[pind] += 1
    return strahler_number


def subtree_aggregate(
    cell: Union[Cell, SkeletonLayer],
    values: np.ndarray,
    agg: str = "sum",
) -> np.ndarray:
    """Aggregate a per-vertex value over each vertex's downstream subtree.

    For each vertex, combines its own value with the values of every vertex
    downstream of it (toward the tips), inclusive -- the classic post-order
    accumulation, in one O(n) pass.

    Parameters
    ----------
    cell : Cell or SkeletonLayer
    values : np.ndarray
        Per-vertex values in skeleton positional order.
    agg : {"sum", "max", "min"}
        Associative reducer combining a vertex with its descendants. For a
        (length-)weighted subtree mean, use sum-then-derive: aggregate ``x * w``
        and ``w`` with ``"sum"`` and divide.

    Returns
    -------
    np.ndarray
        Per-vertex (positional) subtree aggregate.
    """
    skel = _as_skeleton(cell)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != skel.n_vertices:
        raise ValueError("values must have one entry per skeleton vertex.")
    out = values.copy()
    parent = skel.parent_node_array
    # Leaves first: a child always has more hops to root than its parent, so by
    # the time a vertex is visited its full subtree has already folded into it.
    order = np.argsort(skel.hops_to_root(as_positional=True), kind="stable")[::-1]
    if agg == "sum":
        for v in order:
            p = parent[v]
            if p >= 0:
                out[p] += out[v]
    elif agg == "max":
        for v in order:
            p = parent[v]
            if p >= 0 and out[v] > out[p]:
                out[p] = out[v]
    elif agg == "min":
        for v in order:
            p = parent[v]
            if p >= 0 and out[v] < out[p]:
                out[p] = out[v]
    else:
        raise ValueError(f"Unsupported agg {agg!r}; use 'sum', 'max', or 'min'.")
    return out


def path_to_root_aggregate(
    cell: Union[Cell, SkeletonLayer],
    values: np.ndarray,
    agg: str = "sum",
    inclusive: bool = True,
) -> np.ndarray:
    """Aggregate a per-vertex value along each vertex's path to the root.

    The upstream analog of :func:`subtree_aggregate`: upstream on a rooted tree
    is a single path (not a branching subtree), so this combines values over the
    path from each vertex up to the root, in one O(n) root-first pass.

    Parameters
    ----------
    cell : Cell or SkeletonLayer
    values : np.ndarray
        Per-vertex values in skeleton positional order.
    agg : {"sum", "max", "min"}
        Associative reducer combining a vertex with its ancestors.
    inclusive : bool
        Include the vertex's own value (True) or aggregate only its ancestors
        (False). For ``False`` the root gets 0 (sum) or NaN (max/min).

    Returns
    -------
    np.ndarray
        Per-vertex (positional) path-to-root aggregate.
    """
    skel = _as_skeleton(cell)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != skel.n_vertices:
        raise ValueError("values must have one entry per skeleton vertex.")
    parent = skel.parent_node_array
    # Root first: a parent always has fewer hops to root than its child, so its
    # full path-to-root is complete before the child is visited.
    order = np.argsort(skel.hops_to_root(as_positional=True), kind="stable")
    if agg == "sum":
        combine, default = np.add, 0.0
    elif agg == "max":
        combine, default = np.maximum, np.nan
    elif agg == "min":
        combine, default = np.minimum, np.nan
    else:
        raise ValueError(f"Unsupported agg {agg!r}; use 'sum', 'max', or 'min'.")
    inc = values.copy()
    for v in order:
        p = parent[v]
        if p >= 0:
            inc[v] = combine(inc[v], inc[p])
    if inclusive:
        return inc
    excl = np.full_like(inc, default)
    mask = parent >= 0
    excl[mask] = inc[parent[mask]]
    return excl


def branch_order(cell: Union[Cell, SkeletonLayer]) -> np.ndarray:
    """Number of branch-point ancestors on each vertex's path to root.

    Increments by one each time the path from the root crosses a branch point.
    This is the from-root topological order that complements
    :func:`strahler_number` (the from-tips order).

    Parameters
    ----------
    cell : Cell or SkeletonLayer

    Returns
    -------
    np.ndarray
        Per-vertex (positional) branch order, starting at 0 at the root.
    """
    skel = _as_skeleton(cell)
    is_branch = np.zeros(skel.n_vertices, dtype=float)
    is_branch[skel.branch_points_positional] = 1.0
    inclusive = path_to_root_aggregate(skel, is_branch, agg="sum", inclusive=True)
    return (inclusive - is_branch).astype(np.int64)


def _distribution_entropy(counts: np.ndarray) -> float:
    """Compute the distribution entropy of a Nx2 set of synapse counts per compartment."""
    if np.sum(counts) == 0:
        return 0
    ps = np.divide(
        counts,
        np.sum(counts, axis=1)[:, np.newaxis],
        where=np.sum(counts, axis=1)[:, np.newaxis] > 0,
    )
    Hpart = np.sum(np.multiply(ps, np.log2(ps, where=ps > 0)), axis=1)
    Hws = np.sum(counts, axis=1) / np.sum(counts)
    Htot = -np.sum(Hpart * Hws)
    return Htot


def segregation_index(
    axon_pre: int,
    axon_post: int,
    dendrite_pre: int,
    dendrite_post: int,
) -> float:
    """Compute the segregation index between pre and post-synaptic compartments relative a compartment-free neuron.
    Values close to 1 indicate strong segregation, values close to 0 indicate no segregation.

    Parameters
    ----------
    axon_pre : int
        The number of pre-synaptic axon compartments.
    axon_post : int
        The number of post-synaptic axon compartments.
    dendrite_pre : int
        The number of pre-synaptic dendrite compartments.
    dendrite_post : int
        The number of post-synaptic dendrite compartments.

    Returns
    -------
    float
        The segregation index, between 0 and 1.
    """
    if axon_pre + dendrite_pre == 0 or axon_post + dendrite_post == 0:
        return 0

    counts = np.array([[axon_pre, axon_post], [dendrite_pre, dendrite_post]])
    observed_ent = _distribution_entropy(counts)

    unsplit_ent = _distribution_entropy(
        [[axon_pre + dendrite_pre, axon_post + dendrite_post]]
    )

    return 1 - observed_ent / (unsplit_ent + 1e-10)


def label_axon_from_synapse_flow(
    cell: Union[Cell, SkeletonLayer],
    pre_syn: Union[str, np.ndarray] = "pre_syn",
    post_syn: Union[str, np.ndarray] = "post_syn",
    extend_feature_to_segment: bool = False,
    ntimes: int = 1,
    return_segregation_index: bool = False,
    segregation_index_threshold: float = 0,
    as_positional: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Split a neuron into axon and dendrite compartments using synapse locations.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The neuron to split.
    pre_syn : Union[str, np.ndarray], optional
        The annotation associated with presynaptic sites or a list of skeleton vertex ids (see as_positional).
    post_syn : Union[str, np.ndarray], optional
        The annotation associated with postsynaptic sites or a list of skeleton vertex ids (see as_positional).
    extend_feature_to_segment : bool, optional
        Whether to propagate the is_axon feature to the whole segment, rather than a specific vertex.
        This is likely more biologically accurate, but potentially a less optimal split.
    segregation_index_threshold : float, optional
        The minimum segregation index required to accept a split. If the best split has a segregation index
        below this threshold, no split is performed and all vertices are featureed as dendrite.
    as_positional : bool, optional
        If True, assumes the pre_syn and post_syn arrays are positional indices into the skeleton vertex array.
        If False, assumes they are the vertex indices (i.e. cell.skeleton.vertex_indices).

    Returns
    -------
    is_axon:
        A boolean array on Skeleton vertices with True for the axon compartments, False for the dendrite compartments.
    """
    if isinstance(cell, Cell):
        skel = cell.skeleton
    else:
        skel = cell
    if isinstance(pre_syn, str):
        if isinstance(cell, SkeletonLayer):
            raise ValueError("If passing a SkeletonLayer, pre_syn must be an array.")
        pre_syn_inds = cell.annotations[pre_syn].map_index_to_layer(
            "skeleton", as_positional=True
        )
    else:
        if not as_positional:
            pre_syn_inds = np.array(
                [np.flatnonzero(skel.vertex_index == vid)[0] for vid in pre_syn]
            )
        else:
            pre_syn_inds = np.asarray(pre_syn)
    if isinstance(post_syn, str):
        if isinstance(cell, SkeletonLayer):
            raise ValueError("If passing a SkeletonLayer, post_syn must be an array.")
        post_syn_inds = cell.annotations[post_syn].map_index_to_layer(
            "skeleton", as_positional=True
        )
    else:
        if not as_positional:
            post_syn_inds = np.array(
                [np.flatnonzero(skel.vertex_index == vid)[0] for vid in post_syn]
            )
        else:
            post_syn_inds = np.asarray(post_syn)
    is_axon, Hsplit = _label_axon_synapse_flow(
        skel, pre_syn_inds, post_syn_inds, extend_feature_to_segment
    )
    if Hsplit < segregation_index_threshold:
        is_axon = np.full(skel.n_vertices, False)
    if ntimes > 1:
        if not isinstance(cell, Cell):
            raise ValueError(
                "Multiple iterations (ntimes > 1) requires a Cell object, not SkeletonLayer"
            )

        for _ in range(ntimes - 1):
            with cell.mask_context("skeleton", ~is_axon) as masked_cell:
                # Recalculate synapse indices for the masked skeleton
                if isinstance(pre_syn, str):
                    masked_pre_syn_inds = masked_cell.annotations[
                        pre_syn
                    ].map_index_to_layer("skeleton", as_positional=True)
                else:
                    # Map original indices to masked skeleton indices
                    masked_pre_syn_inds = []
                    for idx in pre_syn_inds:
                        if (
                            idx < len(is_axon) and ~is_axon[idx]
                        ):  # This vertex is still in masked skeleton
                            # Find its position in the masked skeleton
                            masked_idx = np.sum(~is_axon[: idx + 1]) - 1
                            masked_pre_syn_inds.append(masked_idx)
                    masked_pre_syn_inds = np.array(masked_pre_syn_inds)

                if isinstance(post_syn, str):
                    masked_post_syn_inds = masked_cell.annotations[
                        post_syn
                    ].map_index_to_layer("skeleton", as_positional=True)
                else:
                    # Map original indices to masked skeleton indices
                    masked_post_syn_inds = []
                    for idx in post_syn_inds:
                        if (
                            idx < len(is_axon) and ~is_axon[idx]
                        ):  # This vertex is still in masked skeleton
                            # Find its position in the masked skeleton
                            masked_idx = np.sum(~is_axon[: idx + 1]) - 1
                            masked_post_syn_inds.append(masked_idx)
                    masked_post_syn_inds = np.array(masked_post_syn_inds)

                is_axon_sub, Hsplit_sub = _label_axon_synapse_flow(
                    masked_cell.skeleton,
                    masked_pre_syn_inds,
                    masked_post_syn_inds,
                    extend_feature_to_segment,
                )
            if Hsplit_sub < segregation_index_threshold:
                break
            is_axon[~is_axon] = is_axon_sub
    return (is_axon, Hsplit) if return_segregation_index else is_axon


def _split_direction_and_quality(
    split_idx, skeleton, pre_inds, post_inds
) -> Tuple[bool, float]:
    """Evaluate a split at a given positional index with pre and post syn indices in positional indices.
    Returns True if the downstream compartment has higher fraction of pre than post, False otherwise, and then the segregation index of the split.
    """
    downstream_inds = skeleton.downstream_vertices(
        split_idx, inclusive=True, as_positional=True
    )
    n_pre_ds = np.sum(np.isin(pre_inds, downstream_inds))
    n_post_ds = np.sum(np.isin(post_inds, downstream_inds))
    n_pre_us = len(pre_inds) - n_pre_ds
    n_post_us = len(post_inds) - n_post_ds
    seg_index = segregation_index(
        n_pre_ds,
        n_post_ds,
        n_pre_us,
        n_post_us,
    )

    ds_fraction_pre = n_pre_ds / (n_post_ds + n_pre_ds + 0.0001)
    us_fraction_pre = n_pre_us / (n_post_us + n_pre_us + 0.0001)
    return ds_fraction_pre >= us_fraction_pre, seg_index


def _label_axon_synapse_flow(
    skeleton: SkeletonLayer,
    pre_syn_inds: np.ndarray,
    post_syn_inds: np.ndarray,
    extend_feature_to_segment: bool,
) -> Tuple[np.ndarray, float]:
    """feature an axon compartment by synapse betweenness. All parameters are as positional indices."""
    syn_btw = synapse_betweenness(skeleton, pre_syn_inds, post_syn_inds)
    high_vinds = np.flatnonzero(syn_btw == max(syn_btw))
    close_vind = high_vinds[np.argmin(skeleton.distance_to_root(high_vinds))]
    if extend_feature_to_segment:
        relseg = skeleton.segment_map[close_vind]
        min_ind = np.argmin(skeleton.distance_to_root(skeleton.segments[relseg]))
        axon_split_ind = skeleton.segments[relseg][min_ind]
    else:
        axon_split_ind = close_vind
    downstream_inds = skeleton.downstream_vertices(
        axon_split_ind, inclusive=True, as_positional=True
    )
    axon_is_ds, Hsplit = _split_direction_and_quality(
        axon_split_ind, skeleton, pre_syn_inds, post_syn_inds
    )
    if axon_is_ds:
        is_axon = np.full(skeleton.n_vertices, False)
        is_axon[downstream_inds] = True
    else:
        is_axon = np.full(skeleton.n_vertices, True)
        is_axon[downstream_inds] = False
    return is_axon, Hsplit


def label_axon_from_spectral_split(
    cell: Union[Cell, SkeletonLayer],
    pre_syn: str = "pre_syn",
    post_syn: str = "post_syn",
    aggregation_distance: float = 1,
    smoothing_alpha: float = 0.99,
    axon_bias: float = 0,
    raw_split: bool = False,
    extend_feature_to_segment: bool = True,
    max_times: Optional[int] = None,
    segregation_index_threshold: float = 0.5,
    return_segregation_index: bool = False,
) -> np.ndarray:
    if isinstance(cell, SkeletonLayer):
        skel = cell
    else:
        skel = cell.skeleton
    if skel is None:
        raise ValueError("Cell is does not have a skeleton.")
    pre_density = (
        skel.map_annotations_to_feature(
            pre_syn,
            distance_threshold=aggregation_distance,
            agg="count",
        )
        + axon_bias
    )
    post_density = skel.map_annotations_to_feature(
        post_syn,
        distance_threshold=aggregation_distance,
        agg="count",
    )
    syn_density = np.vstack([pre_density, post_density]).T
    smoothed_feature = smooth_features(
        skel,
        feature=syn_density,
        alpha=smoothing_alpha,
    )

    is_axon = smoothed_feature[:, 0] > smoothed_feature[:, 1]

    split_edges = np.flatnonzero(
        is_axon[skel.edges_positional[:, 0]] != is_axon[skel.edges_positional[:, 1]]
    )
    if len(split_edges) == 0 or raw_split:
        return is_axon
    # If not doing a raw split, treat each split edge as a candidate split point, and evaluate the segregation index of each split.

    split_parents = skel.edges_positional[split_edges, 1]
    if extend_feature_to_segment:
        split_parent_segments = skel.segment_map[split_parents]
        split_parents = np.array(
            [
                skel.segments_positional[seg][
                    np.argmin(
                        skel.distance_to_root(
                            skel.segments_positional[seg], as_positional=True
                        )
                    )
                ]
                for seg in split_parent_segments
            ]
        )
    split_parents = np.unique(split_parents)

    is_axon_final = np.full(skel.n_vertices, False)
    best_split = 1
    ntimes = 0
    split_values = {}
    while max_times is None or ntimes < max_times:
        with cell.mask_context(layer="skeleton", mask=~is_axon_final) as masked_cell:
            Hsplits = np.zeros(len(split_parents), dtype=np.float32)
            pre_idx_masked = masked_cell.annotations[pre_syn].map_index_to_layer(
                "skeleton", as_positional=True
            )
            post_idx_masked = masked_cell.annotations[post_syn].map_index_to_layer(
                "skeleton", as_positional=True
            )
            for ii, parent in enumerate(split_parents):
                split_vert_ind = cell.skeleton.vertex_index[parent]
                if split_vert_ind in masked_cell.skeleton.vertex_index:
                    local_parent_index = np.flatnonzero(
                        masked_cell.skeleton.vertex_index == split_vert_ind
                    )[0]
                    Hsplits[ii] = _split_direction_and_quality(
                        local_parent_index,
                        masked_cell.skeleton,
                        pre_idx_masked,
                        post_idx_masked,
                    )[1]
            if np.all(np.asarray(Hsplits) <= segregation_index_threshold):
                break
        best_split_index = split_parents[np.argmax(Hsplits)]
        best_split = np.max(Hsplits)
        if best_split <= segregation_index_threshold:
            break
        split_values[best_split_index] = best_split
        ds_split = skel.downstream_vertices(
            best_split_index, inclusive=True, as_positional=True
        )
        is_axon_final[ds_split] = True
        ntimes += 1
    return (
        is_axon_final if not return_segregation_index else (is_axon_final, split_values)
    )


def _precompute_synapse_inds(skel: SkeletonLayer, syn_inds: np.ndarray) -> tuple:
    Nsyn = len(syn_inds)
    n_syn = np.zeros(skel.n_vertices, dtype=int)
    for ind in syn_inds:
        n_syn[ind] += 1
    return Nsyn, n_syn


def synapse_betweenness(
    skel: SkeletonLayer,
    pre_inds: np.ndarray,
    post_inds: np.ndarray,
) -> np.ndarray:
    """Compute synapse betweenness, the number of paths from all post indices to all pre indices along the graph. Vertices can be included multiple times, indicating multiple paths

    Parameters
    ----------
    skel : SkeletonLayer
        Skeleton to measure.
    pre_inds : list or array
        Collection of skeleton vertex indices, each representing one output synapse (i.e. target of a path).
    post_inds : list or array
        Collection of skeleton vertex indices, each representing one input synapse (i.e. source of a path).

    Returns
    -------
    synapse_betweenness : np.array
        Array with a value for each skeleton vertex, with the number of all paths from source to target vertices passing through that vertex.
    """
    Npre, n_pre = _precompute_synapse_inds(skel, pre_inds)
    Npost, n_post = _precompute_synapse_inds(skel, post_inds)

    syn_btwn = np.zeros(skel.n_vertices, dtype=np.int64)
    cov_paths_rev = skel.cover_paths_positional[::-1]
    for path in cov_paths_rev:
        downstream_pre = 0
        downstream_post = 0
        for ind in path:
            downstream_pre += n_pre[ind]
            downstream_post += n_post[ind]
            syn_btwn[ind] = (
                downstream_pre * (Npost - downstream_post)
                + (Npre - downstream_pre) * downstream_post
            )
        # Deposit each branch's synapses at the branch point.
        bp_ind = skel.parent_node_array[path[-1]]
        if bp_ind != -1:
            n_pre[bp_ind] += downstream_pre
            n_post[bp_ind] += downstream_post
    return syn_btwn
