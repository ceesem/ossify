from enum import IntEnum
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import fastremap
import numpy as np
import pandas as pd

from ossify.data_layers import SkeletonLayer

from .base import Cell, Link
from .utils import get_l2id_column, get_supervoxel_column, suppress_output

__all__ = [
    "load_cell_from_client",
    "load_cell_batch_from_client",
    "fetch_frames_batch",
    "SWCCompartment",
]

if TYPE_CHECKING:
    import datetime

    from caveclient import CAVEclientFull as CAVEclient


# Per-call cap for cached bulk-skeleton downloads via ``client.skeleton.fetch_skeletons``
# (caveclient.skeletonservice.MAX_BULK_CACHED_SKELETONS); longer lists are silently truncated
# by the endpoint, so callers must chunk at this size.
_BULK_SKELETON_CHUNK = 500


class SWCCompartment(IntEnum):
    """Standard SWC compartment labels. See https://swc-specification.readthedocs.io/en/latest/swc.html#id-type-x-y-z-radius-parent"""

    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    DENDRITE = 3
    APICAL_DENDRITE = 4


def _other_side(side: Literal["pre", "post"]) -> Literal["pre", "post"]:
    return "post" if side == "pre" else "pre"


def _query_reference_table(
    client: "CAVEclient",
    ref_table: str,
    target_ids: list,
    timestamp: "datetime.datetime",
) -> pd.DataFrame:
    "Live-query a reference table for the given synapse target ids."
    with suppress_output():
        ref_df = client.materialize.live_live_query(
            ref_table,
            filter_in_dict={ref_table: {"target_id": list(target_ids)}},
            timestamp=timestamp,
            metadata=False,
            desired_resolution=[1, 1, 1],
        ).drop(columns=["created", "valid"], errors="ignore")
    return ref_df


def _merge_reference_frame(
    syn_df: pd.DataFrame,
    ref_table: str,
    ref_df: pd.DataFrame,
    reference_suffixes: dict,
) -> pd.DataFrame:
    "Merge a reference-table frame into a synapse frame (rename/suffix rules)."
    suffix = reference_suffixes.get(ref_table, ref_table)
    return syn_df.merge(
        ref_df.rename(columns={"target_id": "id", "id": f"id_{suffix}"}),
        how="left",
        on="id",
        suffixes=("", f"_{suffix}"),
    )


def _finalize_synapse_frame(
    syn_df: pd.DataFrame,
    side: Literal["pre", "post"],
    columns: dict,
    l2_ids,
    *,
    reference_tables: Optional[list[str]] = None,
    reference_frames: Optional[dict] = None,
    reference_suffixes: Optional[dict] = None,
    drop_other_side: bool = True,
) -> pd.DataFrame:
    """Shape a (autapse-filtered) synapse frame identically for the single and batch paths.

    Assigns the L2-id column from ``l2_ids`` (positionally aligned to the rows of
    ``syn_df``), optionally drops the partner root-id column, and merges any
    already-fetched reference frames. Pure DataFrame operations only — no network I/O.
    """
    side_column = columns[side]
    other_column = columns[_other_side(side)]
    l2_column = get_l2id_column(side_column)

    syn_df = syn_df.copy()
    syn_df[l2_column] = np.asarray(l2_ids)
    if drop_other_side:
        syn_df.drop(columns=other_column, inplace=True)

    if reference_tables:
        reference_suffixes = reference_suffixes or {}
        for ref_table in reference_tables:
            syn_df = _merge_reference_frame(
                syn_df, ref_table, reference_frames[ref_table], reference_suffixes
            )
    return syn_df


def _process_synapse_table(
    root_id: int,
    table_name: str,
    client: "CAVEclient",
    side: Literal["pre", "post"],
    columns: dict,
    timestamp: "datetime.datetime",
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: dict = dict(),
    drop_other_side: bool = True,
    omit_autapses: bool = True,
) -> pd.DataFrame:
    "Perform a synapse query and get the l2 ids for the given root_id."
    side_column = columns[side]
    other_column = columns[_other_side(side)]

    with suppress_output():
        syn_df = client.materialize.tables[table_name](
            **{side_column: root_id}
        ).live_query(
            desired_resolution=[1, 1, 1],
            split_positions=True,
            timestamp=timestamp,
            metadata=False,
        )
    if omit_autapses:
        syn_df = syn_df[syn_df[side_column] != syn_df[other_column]]

    svid_column = get_supervoxel_column(side_column)
    l2_ids = client.chunkedgraph.get_roots(
        syn_df[svid_column], stop_layer=2, timestamp=timestamp
    )

    reference_frames = None
    if reference_tables is not None:
        reference_frames = {
            ref_table: _query_reference_table(
                client, ref_table, syn_df["id"].tolist(), timestamp
            )
            for ref_table in reference_tables
        }

    return _finalize_synapse_frame(
        syn_df,
        side,
        columns,
        l2_ids,
        reference_tables=reference_tables,
        reference_frames=reference_frames,
        reference_suffixes=reference_suffixes,
        drop_other_side=drop_other_side,
    )


def load_cell_from_client(
    root_id: int,
    client: "CAVEclient",
    *,
    synapses: bool = False,
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
    restore_graph: bool = False,
    restore_properties: bool = True,
    synapse_spatial_point: str = "ctr_pt_position",
    include_partner_root_id: bool = False,
    timestamp: Optional["datetime.datetime"] = None,
    omit_self_synapses: bool = True,
    skeleton_version: int = 4,
    pre_syn_df: Optional[pd.DataFrame] = None,
    post_syn_df: Optional[pd.DataFrame] = None,
    l2_df: Optional[pd.DataFrame] = None,
    skeleton: Optional[dict] = None,
    assume_valid: bool = False,
) -> Cell:
    """Import an "L2" skeleton and spatial graph using the CAVE skeleton service.

    Parameters
    ----------
    root_id: int
        The root ID of the cell to import.
    client: CAVEclient
        The CAVE client to use for data retrieval.
    synapses: bool
        Whether to include synapse information in the imported cell. Default is False.
    reference_tables: Optional[list[str]]
        A list of table names to include as reference tables for synapse annotation.
        These will be merged into the synapse DataFrame if synapses=True.
    restore_graph: bool
        Whether to restore the complete spatial graph for the imported cell. Default is False. Setting to True will include all graph edges, but can take longer to process.
    restore_properties: bool
        Whether to restore all graph vertex properties of the imported cell. Default is False.
    synapse_spatial_point: str
        The spatial point column name for synapses. Default is "ctr_pt_position".
    include_partner_root_id: bool
        Whether to include the synaptic partner root ID from the imported cell. Default is False.
        If including partner root id, you are encouraged to set a timestamp to ensure consistent results.
        Otherwise, querying different cells at different points in time can result in different results for partner root ids.
    timestamp : Optional[datetime.datetime]
        The timestamp to use for the query. If not provided, the latest timestamp the root id is valid will be used.
    omit_self_synapses: bool
        Whether to omit self-synapses from the imported cell. Default is True, since most are false detections.
    skeleton_version: int
        The skeleton service data version to use for the query. Default is 4.
    pre_syn_df: Optional[pd.DataFrame]
        Pre-fetched pre-synapse frame. When provided (and ``synapses=True``), the internal
        pre-synapse query is skipped and this frame is used as-is. Must be in the exact shape
        the internal fetch produces: autapses omitted, ``drop_other_side``/
        ``include_partner_root_id`` applied, the ``pre_pt_l2_id`` column present, and any
        reference columns already merged. Used by :func:`fetch_frames_batch`.
    post_syn_df: Optional[pd.DataFrame]
        Pre-fetched post-synapse frame; same contract as ``pre_syn_df`` (``post_pt_l2_id``).
    l2_df: Optional[pd.DataFrame]
        Pre-fetched L2 property frame. When provided, the internal ``get_l2data_table`` call is
        skipped. Must already be ``reset_index()``'d (``l2_id`` as a column), row-ordered to
        match ``sk["lvl2_ids"]``, and contain the attribute set implied by ``restore_properties``.
    skeleton: Optional[dict]
        Pre-fetched skeleton dict (as returned by ``client.skeleton.get_skeleton(...,
        output_format="dict")``). When provided, the internal ``get_skeleton`` call is skipped.
        Used by :func:`load_cell_batch_from_client` to avoid re-fetching a skeleton already
        pulled to obtain ``lvl2_ids``.
    assume_valid: bool
        When True, skip the timestamp/validity round trip and use ``timestamp`` directly (which
        must be provided). Set by :func:`load_cell_batch_from_client` after validating the whole
        batch once. With ``skeleton``, ``l2_df`` and the synapse frames all injected, this makes
        assembly fully network-free.

    Returns
    -------
    Cell
        The imported cell object.
    """
    if skeleton is None:
        sk = client.skeleton.get_skeleton(
            root_id, skeleton_version=skeleton_version, output_format="dict"
        )
    else:
        sk = skeleton
    if reference_suffixes is None:
        reference_suffixes = {}
    if assume_valid:
        if timestamp is None:
            raise ValueError("assume_valid=True requires an explicit timestamp.")
        ts = timestamp
    elif timestamp is None:
        ts = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    else:
        is_valid = client.chunkedgraph.is_latest_roots(root_id, timestamp=timestamp)[0]
        if not is_valid:
            raise ValueError(f"Root id {root_id} is not valid at the given timestamp.")
        ts = timestamp

    if synapses:
        synapse_columns = {
            "pre": "pre_pt_root_id",
            "post": "post_pt_root_id",
        }
        synapse_table = client.materialize.synapse_table
        if pre_syn_df is None:
            pre_syn_df = _process_synapse_table(
                root_id,
                synapse_table,
                client,
                "pre",
                synapse_columns,
                ts,
                drop_other_side=not include_partner_root_id,
                omit_autapses=omit_self_synapses,
                reference_tables=reference_tables,
                reference_suffixes=reference_suffixes,
            )
        if post_syn_df is None:
            post_syn_df = _process_synapse_table(
                root_id,
                synapse_table,
                client,
                "post",
                synapse_columns,
                ts,
                drop_other_side=not include_partner_root_id,
                omit_autapses=omit_self_synapses,
                reference_tables=reference_tables,
                reference_suffixes=reference_suffixes,
            )

    l2ids = sk["lvl2_ids"]
    l2_spatial_columns = [
        "rep_coord_nm_x",
        "rep_coord_nm_y",
        "rep_coord_nm_z",
    ]
    if l2_df is None:
        if restore_properties:
            l2_df = client.l2cache.get_l2data_table(l2ids)
        else:
            l2_df = client.l2cache.get_l2data_table(l2ids, attributes=["rep_coord_nm"])
        l2_df = l2_df.reset_index()

    if restore_graph:
        l2_graph = client.chunkedgraph.level2_chunk_graph(root_id)
        l2_map = {v: k for k, v in l2_df["l2_id"].to_dict().items()}

        edges = fastremap.remap(
            l2_graph,
            l2_map,
        )
    else:
        edges = []

    nrn = (
        Cell(
            name=root_id,
            meta={
                "source": f"SkeletonService({client.local_server})",
                "timestamp": ts,
                "datastack": client.datastack_name,
                "root_id": root_id,
            },
        )
        .add_graph(
            vertices=l2_df,
            spatial_columns=l2_spatial_columns,
            edges=edges,
            vertex_index="l2_id",
        )
        .add_skeleton(
            vertices=np.array(sk["vertices"]),
            edges=np.array(sk["edges"]),
            features={"radius": sk["radius"], "compartment": sk["compartment"]},
            linkage=Link(
                mapping=sk["mesh_to_skel_map"], source="graph", map_value_is_index=False
            ),
        )
    )
    if synapses:
        nrn = nrn.add_point_annotations(
            "pre_syn",
            vertices=pre_syn_df,
            spatial_columns=synapse_spatial_point,
            vertex_index="id",
            linkage=Link(mapping="pre_pt_l2_id", target="graph"),
        ).add_point_annotations(
            "post_syn",
            vertices=post_syn_df,
            spatial_columns=synapse_spatial_point,
            vertex_index="id",
            linkage=Link(mapping="post_pt_l2_id", target="graph"),
        )

    return nrn


def _fetch_synapse_frames_batch(
    root_ids: list[int],
    client: "CAVEclient",
    side: Literal["pre", "post"],
    columns: dict,
    timestamp: "datetime.datetime",
    *,
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
    drop_other_side: bool = True,
    omit_autapses: bool = True,
    row_limit: int = 500_000,
) -> dict[int, pd.DataFrame]:
    """Pooled synapse fetch for one side across many root ids.

    Issues a single ``live_live_query`` filtered on the root-id list, one pooled
    ``get_roots`` over all supervoxels, and one query per reference table over the union of
    synapse ids, then finalizes each per-root frame with the exact same logic as the
    single-cell path (:func:`_finalize_synapse_frame`).

    The pooled query runs inside ``suppress_output()``, which silences caveclient's
    server-side row-limit warning; ``row_limit`` re-adds an explicit guard so a truncated
    (silently incomplete) result raises instead of corrupting the per-root split. Set to 0 to
    disable.
    """
    side_column = columns[side]
    other_column = columns[_other_side(side)]
    svid_column = get_supervoxel_column(side_column)
    synapse_table = client.materialize.synapse_table

    # 1. Pooled synapse query (one round trip for all roots on this side).
    with suppress_output():
        pooled = client.materialize.live_live_query(
            synapse_table,
            filter_in_dict={synapse_table: {side_column: [int(r) for r in root_ids]}},
            timestamp=timestamp,
            split_positions=True,
            metadata=False,
            desired_resolution=[1, 1, 1],
        )
    if row_limit and len(pooled) >= row_limit:
        raise RuntimeError(
            f"Pooled {side}-synapse query for {len(root_ids)} roots returned {len(pooled)} rows, "
            f"at/above the server row limit ({row_limit}); results are likely truncated. "
            f"Reduce the batch size."
        )
    if omit_autapses:
        pooled = pooled[pooled[side_column] != pooled[other_column]]

    # Group per root; roots with no synapses get an empty frame with the same columns.
    groups = {int(r): g for r, g in pooled.groupby(side_column)}
    empty = pooled.iloc[0:0]
    per_root = {int(r): groups.get(int(r), empty).copy() for r in root_ids}

    # 2. Pooled supervoxel -> L2 resolution, scattered back in input order.
    svid_arrays = [per_root[int(r)][svid_column].to_numpy() for r in root_ids]
    lengths = [len(a) for a in svid_arrays]
    if sum(lengths) > 0:
        all_l2 = client.chunkedgraph.get_roots(
            np.concatenate(svid_arrays), stop_layer=2, timestamp=timestamp
        )
    else:
        all_l2 = np.array([], dtype=np.int64)
    l2_by_root: dict[int, np.ndarray] = {}
    offset = 0
    for r, n in zip(root_ids, lengths):
        l2_by_root[int(r)] = all_l2[offset : offset + n]
        offset += n

    # 3. Pooled reference-table queries over the union of synapse ids.
    reference_frames = None
    if reference_tables is not None:
        all_ids: list = []
        for r in root_ids:
            all_ids.extend(per_root[int(r)]["id"].tolist())
        reference_frames = {
            ref_table: _query_reference_table(client, ref_table, all_ids, timestamp)
            for ref_table in reference_tables
        }

    return {
        int(r): _finalize_synapse_frame(
            per_root[int(r)],
            side,
            columns,
            l2_by_root[int(r)],
            reference_tables=reference_tables,
            reference_frames=reference_frames,
            reference_suffixes=reference_suffixes,
            drop_other_side=drop_other_side,
        )
        for r in root_ids
    }


def fetch_frames_batch(
    root_ids: list[int],
    client: "CAVEclient",
    *,
    synapses: bool = True,
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
    include_partner_root_id: bool = False,
    omit_self_synapses: bool = True,
    restore_properties: bool = True,
    timestamp: Optional["datetime.datetime"] = None,
    lvl2_ids_by_root: Optional[dict] = None,
    row_limit: int = 500_000,
) -> dict[int, dict]:
    """Batch-fetch the poolable per-cell frames for many root ids in a few queries.

    Collapses the two dominant per-cell fetches of :func:`load_cell_from_client` — synapse
    queries and L2 property lookups — into a handful of pooled requests, then slices the
    results back per root id. The returned frames are byte-identical to what the single-cell
    path produces and can be handed straight to ``load_cell_from_client`` via its
    ``pre_syn_df`` / ``post_syn_df`` / ``l2_df`` injection params.

    All roots share a single ``timestamp`` (pass an explicit value or pin ``client.version``);
    the pooled ``filter_in`` queries require one consistent materialization for the batch.

    Parameters
    ----------
    root_ids: list[int]
        Root ids to fetch. Order is preserved in the scatter-back.
    client: CAVEclient
        The CAVE client to use.
    synapses: bool
        Whether to fetch synapse frames. Default True. When False, only ``l2_df`` is populated
        (requires ``lvl2_ids_by_root``).
    reference_tables, reference_suffixes, include_partner_root_id, omit_self_synapses:
        Same meaning as in :func:`load_cell_from_client`; applied identically per root.
    restore_properties: bool
        When True, fetch all L2 attributes; otherwise only ``rep_coord_nm``.
    timestamp: Optional[datetime.datetime]
        The single shared timestamp for the batch.
    lvl2_ids_by_root: Optional[dict]
        Mapping ``{root_id: sk["lvl2_ids"]}`` from the caller's skeleton dicts. Required to
        fetch L2 data; the pooled ``get_l2data_table`` runs over the union and each root's
        frame is sliced back in ``lvl2_ids`` order (dropping cache-missing ids), matching the
        single-cell path exactly. If omitted, ``l2_df`` is left ``None`` for every root.

    Returns
    -------
    dict[int, dict]
        ``{root_id: {"pre_syn_df": ..., "post_syn_df": ..., "l2_df": ...}}``. Entries that were
        not fetched (e.g. synapses off) are ``None``.
    """
    root_ids = [int(r) for r in root_ids]
    if reference_suffixes is None:
        reference_suffixes = {}
    results: dict[int, dict] = {
        r: {"pre_syn_df": None, "post_syn_df": None, "l2_df": None} for r in root_ids
    }

    if synapses:
        synapse_columns = {"pre": "pre_pt_root_id", "post": "post_pt_root_id"}
        drop_other_side = not include_partner_root_id
        for side in ("pre", "post"):
            frames = _fetch_synapse_frames_batch(
                root_ids,
                client,
                side,
                synapse_columns,
                timestamp,
                reference_tables=reference_tables,
                reference_suffixes=reference_suffixes,
                drop_other_side=drop_other_side,
                omit_autapses=omit_self_synapses,
                row_limit=row_limit,
            )
            for r in root_ids:
                results[r][f"{side}_syn_df"] = frames[r]

    # 4. Pooled L2 data over the union of all L2 ids, sliced per root in lvl2_ids order.
    if lvl2_ids_by_root is not None:
        union: list = []
        seen: set = set()
        for r in root_ids:
            for i in lvl2_ids_by_root[r]:
                if i not in seen:
                    seen.add(i)
                    union.append(i)
        if restore_properties:
            pooled_l2 = client.l2cache.get_l2data_table(union)
        else:
            pooled_l2 = client.l2cache.get_l2data_table(
                union, attributes=["rep_coord_nm"]
            )
        available = set(pooled_l2.index)
        for r in root_ids:
            ids = [i for i in lvl2_ids_by_root[r] if i in available]
            results[r]["l2_df"] = pooled_l2.loc[ids].reset_index()

    return results


def _get_bulk_skeletons(
    client: "CAVEclient",
    root_ids: list[int],
    skeleton_version: int,
    method: Literal["gcs", "server"] = "gcs",
) -> dict[int, dict]:
    """Download cached skeletons in bulk, chunked at the cached-bulk cap.

    Uses ``client.skeleton.fetch_skeletons``, which retrieves only already-cached skeletons and
    skips the per-root chunkedgraph validation of the older ``get_bulk_skeletons`` path. With
    ``method="gcs"`` (the default) skeleton H5 files are downloaded directly from the storage
    bucket via a short-lived downscoped token, bypassing the service for data transfer — this
    avoids the request rate limits that throttle bulk loads through the server. ``method="server"``
    routes the download through the skeleton service instead.

    Skeletons are never generated inline (``generate_missing_skeletons=False`` always). Returns
    ``{int root_id: skeleton_dict}``; roots whose skeleton is not yet cached (async/not-generated)
    are omitted — assuming a prior ``generate_bulk_skeletons_async`` pre-pass, these simply don't
    appear.
    """
    out: dict[int, dict] = {}
    for i in range(0, len(root_ids), _BULK_SKELETON_CHUNK):
        chunk = root_ids[i : i + _BULK_SKELETON_CHUNK]
        with suppress_output():
            res = client.skeleton.fetch_skeletons(
                chunk,
                skeleton_version=skeleton_version,
                output_format="dict",
                method=method,
                generate_missing_skeletons=False,
            )
        for k, v in res.items():
            out[int(k)] = v
    return out


def load_cell_batch_from_client(
    root_ids: list[int],
    client: "CAVEclient",
    *,
    synapses: bool = False,
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
    restore_graph: bool = False,
    restore_properties: bool = True,
    synapse_spatial_point: str = "ctr_pt_position",
    include_partner_root_id: bool = False,
    timestamp: Optional["datetime.datetime"] = None,
    omit_self_synapses: bool = True,
    skeleton_version: int = 4,
    skip_invalid: bool = False,
    skeleton_download_method: Literal["gcs", "server"] = "gcs",
    row_limit: int = 500_000,
) -> dict[int, Cell]:
    """Load many cells with the poolable fetches batched into a few queries.

    Equivalent to calling :func:`load_cell_from_client` on each root id, but the synapse and
    L2-property fetches are pooled across the whole batch (see :func:`fetch_frames_batch`) and
    each cell is then assembled network-free — no skeleton is fetched twice, and the batch is
    validated with a single round trip. All roots share one ``timestamp``.

    All parameters not listed below match :func:`load_cell_from_client` and are applied
    identically to every cell in the batch.

    Parameters
    ----------
    root_ids: list[int]
        Root ids to load.
    timestamp: Optional[datetime.datetime]
        Single shared timestamp for the batch. If None, the current materialization timestamp
        (``client.materialize.get_timestamp()``) is used — pin ``client.version`` for a
        reproducible batch.
    skip_invalid: bool
        If True, roots that are not valid at ``timestamp`` — or whose skeleton is not available
        — are dropped from the result instead of raising.
    skeleton_download_method: "gcs" or "server"
        How ``fetch_skeletons`` retrieves cached skeletons. ``"gcs"`` (default) downloads H5 files
        directly from the storage bucket via a downscoped token, bypassing the service for data
        transfer and avoiding its request rate limits — preferred for bulk loads. ``"server"``
        routes the download through the skeleton service instead.
    row_limit: int
        Passed to :func:`fetch_frames_batch`; guards against a silently-truncated pooled synapse
        query at/above the server row limit (default 500,000). Set to 0 to disable.

    Returns
    -------
    dict[int, Cell]
        Mapping of root id to the assembled :class:`~ossify.base.Cell`. Invalid roots are absent
        when ``skip_invalid=True``.
    """
    root_ids = [int(r) for r in root_ids]
    if reference_suffixes is None:
        reference_suffixes = {}

    # 1. One shared timestamp + one validity round trip for the whole batch.
    ts = timestamp if timestamp is not None else client.materialize.get_timestamp()
    valid_mask = client.chunkedgraph.is_latest_roots(root_ids, timestamp=ts)
    valid_ids = [r for r, ok in zip(root_ids, valid_mask) if ok]
    invalid_ids = [r for r, ok in zip(root_ids, valid_mask) if not ok]
    if invalid_ids and not skip_invalid:
        raise ValueError(
            f"{len(invalid_ids)} root id(s) not valid at {ts}: {invalid_ids[:5]}"
            + (" ..." if len(invalid_ids) > 5 else "")
        )

    # 2. Skeletons: one cached bulk download (chunked at the cached-bulk cap) to obtain lvl2_ids,
    #    then injected into assembly. Assumes a prior generate_bulk_skeletons_async pre-pass.
    sk_by_root = _get_bulk_skeletons(
        client,
        valid_ids,
        skeleton_version,
        method=skeleton_download_method,
    )
    no_skeleton = [r for r in valid_ids if r not in sk_by_root]
    if no_skeleton and not skip_invalid:
        raise ValueError(
            f"{len(no_skeleton)} root id(s) have no cached skeleton "
            f"(run generate_bulk_skeletons_async first, or pass skip_invalid=True): "
            f"{no_skeleton[:5]}" + (" ..." if len(no_skeleton) > 5 else "")
        )
    valid_ids = [r for r in valid_ids if r in sk_by_root]
    lvl2_ids_by_root = {r: sk_by_root[r]["lvl2_ids"] for r in valid_ids}

    # 3. Pooled synapse + L2 fetch.
    frames = fetch_frames_batch(
        valid_ids,
        client,
        synapses=synapses,
        reference_tables=reference_tables,
        reference_suffixes=reference_suffixes,
        include_partner_root_id=include_partner_root_id,
        omit_self_synapses=omit_self_synapses,
        restore_properties=restore_properties,
        timestamp=ts,
        lvl2_ids_by_root=lvl2_ids_by_root,
        row_limit=row_limit,
    )

    # 4. Network-free per-cell assembly (skeleton injected, validity assumed).
    cells: dict[int, Cell] = {}
    for r in valid_ids:
        cells[r] = load_cell_from_client(
            r,
            client,
            synapses=synapses,
            reference_tables=reference_tables,
            reference_suffixes=reference_suffixes,
            restore_graph=restore_graph,
            restore_properties=restore_properties,
            synapse_spatial_point=synapse_spatial_point,
            include_partner_root_id=include_partner_root_id,
            timestamp=ts,
            omit_self_synapses=omit_self_synapses,
            skeleton_version=skeleton_version,
            pre_syn_df=frames[r]["pre_syn_df"],
            post_syn_df=frames[r]["post_syn_df"],
            l2_df=frames[r]["l2_df"],
            skeleton=sk_by_root[r],
            assume_valid=True,
        )
    return cells


# def resample(
#         sk: Union[Cell, SkeletonLayer],
#         spacing, kind="linear",
#         tip_length_ratio=0.5,
#         avoid_root=True
#     ) -> Tuple[SkeletonLayer, np.ndarray]:
#     """Resample a skeleton's vertices

#     Parameters
#     ----------
#     sk : Skeleton
#         Input skeleton file with a skeleton
#     spacing : numeric
#         Desired spacing in nanometers
#     kind : str, optional
#         Type of interpolation to use when resampling. Options follow scipy.interpolate.interp1d. By default "linear"
#     tip_length_ratio : float, optional
#         The ratio of spacing to branch tip length that a branch tip must have in order to be included in the final skeleton
#         for example: spacing is 10 and branch length is 8. The branch tip will be included if tip_length_ratio is .8 or lower,
#         but excluded if tip_length_ratio is greater than 0.8.

#     Returns
#     -------
#     Skeleton
#         New skeleton with resampled vertices.

#     resample_map
#         Array where the ith index corresponds to the ith vertex of the resampled skeleton and the value
#         is the associated index in the original skeleton. To assign vertices, we assign a "domain" to each
#         vertex in the original skeleton that is halfway between the vertex and its neighbors. Resampled
#         vertices that fall within that domain (based on topology and distance-to-root) are then associated
#         with the original vertex.
#     """
#     path_counter = 0
#     branch_d = {}
#     vert_list = []
#     edge_list = []
#     output_map_list = []

#     for path in sk.cover_paths:
#         new_verts, new_edges, output_map_path, branch_d = resample_path(
#             path,
#             sk,
#             path_counter,
#             spacing,
#             kind,
#             tip_length_ratio,
#             branch_d,
#             avoid_root,
#         )
#         vert_list.append(new_verts)
#         edge_list.append(new_edges)
#         output_map_list.append(output_map_path)
#         path_counter += len(new_verts)

#     new_verts = np.vstack(vert_list)
#     new_edges = np.vstack(edge_list)
#     resample_map = np.concatenate(output_map_list)

#     return (
#         Skeleton(
#             new_verts,
#             new_edges,
#             root=branch_d[int(sk.root)],
#             remove_zero_length_edges=False,
#         ),
#         resample_map,
#     )

# def export_swc_dataframe(
#     cell: Cell,
#     resample_distance: Optional[float] = None,
#     compartment: Optional[Union[str, list]] = None,
#     compartment_mapping: Optional[dict] = None,
#     radius: Optional[Union[str, list]] = None,
#     rescale: Optional[float] = None,
#     rescale_radius: bool = True,
#     default_compartment_label: int = 0,
#     default_radius: float = 1.0,
# ) -> pd.DataFrame:
#     """Export the skeleton layer of a Cell as a SWC-format DataFrame. See https://swc-specification.readthedocs.io/en/latest/swc.html for SWC format details.

#     Parameters
#     ----------
#     cell : Cell
#         The Cell object containing the skeleton to export.
#     resample_distance : Optional[float]
#         If provided, resample the skeleton vertices to be approximately this distance apart.
#     compartment : Optional[Union[str, list]]
#         If provided, the name(s) of annotation(s) to use for compartment labels. Can be a single annotation name or a list of names (if multiple annotations contain compartment info).
#     compartment_mapping : Optional[dict]
#         If provided, a mapping from original compartment labels to desired output labels. Only applied if `compartment` is provided.
#     radius : Optional[Union[str, list]]
#         If provided, the name(s) of feature(s) to use for radius values. Can be a single feature name or a list of names (if multiple features contain radius info).
#     rescale : Optional[float]
#         If provided, rescale all coordinates by this factor (e.g., to convert from nm to um).
#     rescale_radius : bool
#         Whether to apply the same rescaling factor to radius values (if `radius` is provided).
#     default_compartment_label : int
#         The default label to use for compartments if no compartment annotation is provided.
#     default_radius : float
#         The default radius to use if no radius annotation is provided.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame in SWC format with columns: ["id", "type", "x", "y", "z", "radius", "parent"].
#     """
