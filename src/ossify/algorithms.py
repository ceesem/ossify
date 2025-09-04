from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Cell, SkeletonLayer


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


def smooth_labels(
    cell: Union[Cell, SkeletonLayer],
    label: np.ndarray,
    alpha: float = 0.90,
) -> np.ndarray:
    """Computes a smoothed label spreading that is akin to steady-state solutions to the heat equation on the skeleton graph.

    Parameters
    ----------
    cell : Cell
        Neuron object
    label : np.ndarray
        The initial label array. Must be Nxm, where N is the number of skeleton vertices
    alpha : float, optional
        A neighborhood influence parameter between 0 and 1. Higher values give more influence to neighbors, by default 0.90.

    Returns
    -------
    np.ndarray
        The smoothed label array
    """
    if isinstance(cell, SkeletonLayer):
        skel = cell
    else:
        skel = cell.skeleton
    Smat = _laplacian_offset(skel)
    Imat = sparse.eye(Smat.shape[0])
    invertLap = Imat - alpha * Smat
    label = np.atleast_2d(label).reshape(Smat.shape[0], -1)
    F = sparse.linalg.spsolve(invertLap, label)
    return np.squeeze((1 - alpha) * F)


def strahler_number(cell: Union[Cell, SkeletonLayer]) -> np.ndarray:
    """Compute Strahler number on a skeleton, starting at 1 for each tip.
    Returns a label suitable for a SkeletonLayer.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The skeleton to compute the Strahler number on.
        For convenience, you can pass a Cell object, but note
        that the return label is always for the skeleton.

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


def _distribution_entropy(counts: np.ndarray) -> float:
    """Compute the distribution entropy of a 2x2 set of synapse counts per compartment."""
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


def label_axon_from_synapses(
    cell: Union[Cell, SkeletonLayer],
    pre_syn: Union[str, np.ndarray] = "pre_syn",
    post_syn: Union[str, np.ndarray] = "post_syn",
    how: Literal["synapse_flow", "spectral"] = "synapse_flow",
    n_splits: int = 1,
    label_to_segment: bool = False,
):
    """Split a neuron into axon and dendrite compartments using synapse locations.

    Parameters
    ----------
    cell : Union[Cell, SkeletonLayer]
        The neuron to split.
    pre_syn : Union[str, np.ndarray], optional
        The annotation associated with presynaptic sites or a list of skeleton ids.
    post_syn : Union[str, np.ndarray], optional
        The annotation associated with postsynaptic sites or a list of skeleton ids.
    how : Literal["synapse_flow", "spectral"], optional
        The method to use for splitting.
    n_splits : int, optional
        The number of splits to perform. Only applies to the "synapse_flow" method.
    label_to_segment : bool, optional
        Whether to propagate the is_axon label to the whole segment, rather than the precise vertex.
        This is likely more biologically accurate, but potentially a less optimal split.

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
            "skeleton", positional=True
        )
    else:
        pre_syn_inds = np.array(pre_syn)
    if isinstance(post_syn, str):
        if isinstance(cell, SkeletonLayer):
            raise ValueError("If passing a SkeletonLayer, post_syn must be an array.")
        post_syn_inds = cell.annotations[post_syn].map_index_to_layer(
            "skeleton", positional=True
        )
    else:
        post_syn_inds = np.array(post_syn)
    match how:
        case "synapse_flow":
            return _label_axon_synapse_flow(
                skel, pre_syn_inds, post_syn_inds, n_splits, label_to_segment
            )
        case "spectral":
            raise NotImplementedError("Spectral method not yet implemented.")
            return _label_axon_spectral(
                skel, pre_syn_inds, post_syn_inds, label_to_segment
            )


def _label_axon_synapse_flow(
    skeleton: SkeletonLayer,
    pre_syn_inds: np.ndarray,
    post_syn_inds: np.ndarray,
    extend_label_to_segment: bool,
) -> np.ndarray:
    """Label an axon compartment by synapse betweenness. All parameters are as positional indices."""
    syn_btw = synapse_betweenness(skeleton, pre_syn_inds, post_syn_inds)
    high_vinds = np.flatnonzero(syn_btw == max(syn_btw))
    close_vind = high_vinds[np.argmin(skeleton.distance_to_root(high_vinds))]
    if extend_label_to_segment:
        relseg = skeleton.segment_map[close_vind]
        min_ind = np.argmin(skeleton.distance_to_root(skeleton.segments[relseg]))
        axon_split_ind = skeleton.segments[relseg][min_ind]
    else:
        axon_split_ind = close_vind
    downstream_inds = skeleton.downstream_vertices(
        axon_split_ind, inclusive=True, as_positional=True
    )
    n_pre_ds = np.sum(np.isin(pre_syn_inds, downstream_inds))
    n_post_ds = np.sum(np.isin(post_syn_inds, downstream_inds))
    n_pre_us = len(pre_syn_inds) - n_pre_ds
    n_post_us = len(post_syn_inds) - n_post_ds
    if (n_pre_ds / (n_post_ds + n_pre_ds + 1)) >= (
        n_pre_us / (n_post_us + n_pre_us + 1)
    ):
        is_axon = np.full(skeleton.n_vertices, False)
        is_axon[downstream_inds] = True
    else:
        is_axon = np.full(skeleton.n_vertices, True)
        is_axon[downstream_inds] = False
    return is_axon


def _label_axon_spectral(
    skeleton: SkeletonLayer,
    pre_syn_inds: np.ndarray,
    post_syn_inds: np.ndarray,
    label_to_segment: bool,
):
    pass


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
    sk : Skeleton
        Skeleton to measure
    pre_inds : list or array
        Collection of skeleton vertex indices, each representing one output synapse (i.e. target of a path).
    post_inds : list or array
        Collection of skeleton certex indices, each representing one input synapse (i.e. source of a path).
    use_entropy : bool, optional
        If True, also returns the entropic segregatation index if one were to cut at a given vertex, by default False

    Returns
    -------
    synapse_betweenness : np.array
        Array with a value for each skeleton vertex, with the number of all paths from source to target vertices passing through that vertex.
    segregation_index : np.array (optional)
        Array with a value for each skeleton vertex, with the segregatio index if the cut were to happen at that vertex. Only returned if `use_entropy=True`.
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
