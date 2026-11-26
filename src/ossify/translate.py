import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ossify.data_layers import SkeletonLayer

from ._sync.base import canonicalize_ids
from .base import Cell, Link
from .skeletonization import add_skeleton_radius, find_soma_point, skeletonize
from .utils import get_l2id_column, get_supervoxel_column, suppress_output

__all__ = [
    "load_cell_from_client",
    "load_cell_from_l2graph",
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


class CompartmentLabeler(Protocol):
    """Anything that labels a cell's skeleton vertices by compartment.

    :class:`ossify.compartments.StructuredLabeler` and
    :class:`ossify.compartments.SkeletonVertexModel` both qualify.
    """

    def predict(self, cell: Cell) -> np.ndarray:
        """One label per skeleton vertex, as compartment names or SWC codes."""
        ...


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
        # The chunkedgraph gives edges as l2_id pairs, which is what the graph
        # is indexed by, so hand them over as-is rather than round-tripping
        # them through positional indices and straight back again.
        edges = client.chunkedgraph.level2_chunk_graph(root_id)
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
            edges_as_positional=False,
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


# Attributes needed to place level-2 vertices and to find the soma from them.
_L2_SKELETON_ATTRIBUTES = ["rep_coord_nm", "max_dt_nm", "size_nm3"]
_L2_SPATIAL_COLUMNS = ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]
# The L2 cache answers at a steady rate per id, so one large request is slow
# and several smaller ones side by side are not. Below this size a request is
# quick enough that splitting it further only adds overhead.
_L2_MIN_CHUNK = 1000


def _get_l2data_table_chunked(
    client: "CAVEclient",
    l2_ids: np.ndarray,
    attributes: Optional[list[str]],
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch the L2 property table as several concurrent requests.

    Parameters
    ----------
    client : CAVEclient
        Client whose ``l2cache`` serves the table.
    l2_ids : np.ndarray
        Level-2 ids to look up.
    attributes : Optional[list[str]]
        L2 cache attributes to fetch, or None for all of them.
    max_workers : int
        Most requests to run at once. Default 8.

    Returns
    -------
    pd.DataFrame
        One row per id in ``l2_ids``, in that order, indexed by ``l2_id``. Ids
        the cache has no entry for are present with missing values.
    """
    l2_ids = np.asarray(l2_ids)
    n_chunks = int(
        np.clip(np.ceil(len(l2_ids) / _L2_MIN_CHUNK), 1, max(max_workers, 1))
    )
    chunks = np.array_split(l2_ids, n_chunks)

    def fetch(chunk: np.ndarray) -> pd.DataFrame:
        return client.l2cache.get_l2data_table(chunk, attributes=attributes)

    if n_chunks == 1:
        frames = [fetch(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=n_chunks) as executor:
            frames = list(executor.map(fetch, chunks))

    table = pd.concat(frames)
    table.index = canonicalize_ids(table.index, name="l2_id")
    table = table.reindex(canonicalize_ids(l2_ids, name="l2_id"))
    table.index.name = "l2_id"
    return table


def _blank_origin_positions(positions: np.ndarray) -> np.ndarray:
    """Mark positions of exactly ``(0, 0, 0)`` as unknown.

    The level-2 cache can answer an id it has not computed with a location at
    the origin and no other features, instead of leaving it out. Left alone,
    such a vertex sits hundreds of micrometres from its neighbours and drags a
    skeleton branch out to it.

    Parameters
    ----------
    positions : np.ndarray
        ``(N, 3)`` positions.

    Returns
    -------
    np.ndarray
        A copy with NaN in every row that was exactly at the origin.
    """
    positions = np.array(positions, dtype=float)
    positions[(positions == 0).all(axis=1)] = np.nan
    return positions


def _fill_missing_positions(
    positions: np.ndarray, edges: np.ndarray, max_rounds: int = 20
) -> np.ndarray:
    """Give vertices with no position the average position of their neighbours.

    Level-2 ids that the cache has not computed yet come back without a
    location. Each round places every missing vertex that has at least one
    placed neighbour, so a run of missing vertices fills in from both ends.

    Parameters
    ----------
    positions : np.ndarray
        ``(N, 3)`` positions, NaN where unknown.
    edges : np.ndarray
        ``(E, 2)`` positional edges.
    max_rounds : int
        Rounds to try before giving up. Default 20.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` positions with no missing values.

    Raises
    ------
    ValueError
        If some vertices still have no position after ``max_rounds``.
    """
    positions = np.array(positions, dtype=float)
    missing = np.isnan(positions).any(axis=1)
    if not missing.any():
        return positions

    both_ways = np.vstack([edges, edges[:, ::-1]])
    for _ in range(max_rounds):
        usable = missing[both_ways[:, 0]] & ~missing[both_ways[:, 1]]
        if not usable.any():
            break
        target, source = both_ways[usable, 0], both_ways[usable, 1]
        total = np.zeros_like(positions)
        count = np.zeros(len(positions))
        np.add.at(total, target, positions[source])
        np.add.at(count, target, 1)
        placed = count > 0
        positions[placed] = total[placed] / count[placed, np.newaxis]
        missing[placed] = False
        if not missing.any():
            break

    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} level-2 vertices have no position and no "
            "neighbour to take one from. The L2 cache may not be populated for "
            "this cell yet."
        )
    return positions


def _lookup_soma_point(
    root_id: int,
    client: "CAVEclient",
    soma_table: Optional[Union[str, Sequence[str]]],
    timestamp: "datetime.datetime",
) -> Optional[np.ndarray]:
    """Location of the cell's nucleus in nanometers, if it has exactly one."""
    if soma_table is None:
        soma_table = client.info.get_datastack_info().get("soma_table")
    if soma_table is None:
        return None
    tables = [soma_table] if isinstance(soma_table, str) else list(soma_table)

    points = []
    for table in tables:
        with suppress_output():
            found = client.materialize.tables[table](pt_root_id=root_id).live_query(
                desired_resolution=[1, 1, 1],
                split_positions=False,
                timestamp=timestamp,
                metadata=False,
            )
        points.extend(np.asarray(point, dtype=float) for point in found["pt_position"])

    if len(points) == 1:
        return points[0]
    if len(points) > 1:
        warnings.warn(
            f"Root {root_id} has {len(points)} entries in {tables}, so its nucleus "
            "is ambiguous. The soma is taken from the graph instead, which "
            "picks the largest. Pass soma_point to choose.",
            UserWarning,
            stacklevel=3,
        )
    return None


def _resolve_timestamp(
    root_id: int, client: "CAVEclient", timestamp: Optional["datetime.datetime"]
) -> "datetime.datetime":
    """The timestamp to query at: the one given, checked, or the root's latest."""
    if timestamp is None:
        return client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    if not client.chunkedgraph.is_latest_roots(root_id, timestamp=timestamp)[0]:
        raise ValueError(f"Root id {root_id} is not valid at the given timestamp.")
    return timestamp


def _fetch_l2_graph(
    root_id: int,
    client: "CAVEclient",
    attributes: Optional[list[str]],
    max_workers: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Fetch a root's level-2 vertices and edges.

    Returns
    -------
    l2_df : pd.DataFrame
        One row per level-2 id, sorted by id, with an ``l2_id`` column, the
        requested attributes, and a position for every vertex.
    edges : np.ndarray
        ``(E, 2)`` edges as pairs of level-2 ids. Empty for an object made of
        a single level-2 vertex.
    """
    edges = np.asarray(client.chunkedgraph.level2_chunk_graph(root_id))
    if edges.size == 0:
        # A single level-2 vertex has no edges to name it, so ask for it.
        edges = np.empty((0, 2), dtype=np.int64)
        l2_ids = np.unique(
            canonicalize_ids(
                np.asarray(client.chunkedgraph.get_leaves(root_id, stop_layer=2)),
                name="l2_id",
            )
        )
    else:
        edges = np.asarray(canonicalize_ids(edges.ravel(), name="l2_id")).reshape(-1, 2)
        l2_ids = np.unique(edges)

    l2_df = _get_l2data_table_chunked(client, l2_ids, attributes, max_workers)
    l2_df[_L2_SPATIAL_COLUMNS] = _fill_missing_positions(
        _blank_origin_positions(l2_df[_L2_SPATIAL_COLUMNS].to_numpy(dtype=float)),
        np.searchsorted(l2_ids, edges),
    )
    return l2_df.reset_index(), edges


def _resolve_soma_point(
    root_id: int,
    client: "CAVEclient",
    l2_df: pd.DataFrame,
    edges: np.ndarray,
    timestamp: "datetime.datetime",
    *,
    soma_point: Optional[Union[np.ndarray, Sequence[float]]],
    soma_table: Optional[Union[str, Sequence[str]]],
    lookup: bool,
    guess: bool,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Decide where the soma is, trying each enabled source in turn.

    Returns
    -------
    point : Optional[np.ndarray]
        Soma location in nanometers, or None.
    source : Optional[str]
        ``"given"``, ``"nucleus"``, ``"graph"`` or None.
    """
    if soma_point is not None:
        return np.asarray(soma_point, dtype=float).reshape(3), "given"
    if lookup:
        point = _lookup_soma_point(root_id, client, soma_table, timestamp)
        if point is not None:
            return point, "nucleus"
    if guess:
        point = find_soma_point(
            l2_df[_L2_SPATIAL_COLUMNS].to_numpy(dtype=float),
            np.searchsorted(l2_df["l2_id"].to_numpy(), edges),
            l2_df["max_dt_nm"].to_numpy(dtype=float),
            l2_df["size_nm3"].to_numpy(dtype=float),
        )
        if point is not None:
            return point, "graph"
    return None, None


def _add_synapse_annotations(
    cell: Cell,
    root_id: int,
    client: "CAVEclient",
    timestamp: "datetime.datetime",
    *,
    reference_tables: Optional[list[str]],
    reference_suffixes: dict,
    synapse_spatial_point: str,
    include_partner_root_id: bool,
    omit_self_synapses: bool,
) -> Cell:
    """Query a root's synapses and attach them to its graph as annotations."""
    synapse_columns = {"pre": "pre_pt_root_id", "post": "post_pt_root_id"}
    synapse_table = client.materialize.synapse_table
    for side in ("pre", "post"):
        syn_df = _process_synapse_table(
            root_id,
            synapse_table,
            client,
            side,
            synapse_columns,
            timestamp,
            drop_other_side=not include_partner_root_id,
            omit_autapses=omit_self_synapses,
            reference_tables=reference_tables,
            reference_suffixes=reference_suffixes,
        )
        cell.add_point_annotations(
            f"{side}_syn",
            vertices=syn_df,
            spatial_columns=synapse_spatial_point,
            vertex_index="id",
            linkage=Link(mapping=f"{side}_pt_l2_id", target="graph"),
        )
    return cell


def load_cell_from_l2graph(
    root_id: int,
    client: "CAVEclient",
    *,
    synapses: bool = False,
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
    restore_properties: bool = True,
    synapse_spatial_point: str = "ctr_pt_position",
    include_partner_root_id: bool = False,
    timestamp: Optional["datetime.datetime"] = None,
    omit_self_synapses: bool = True,
    build_skeleton: bool = True,
    soma_point: Optional[Union[np.ndarray, Sequence[float]]] = None,
    soma_table: Optional[Union[str, Sequence[str]]] = None,
    lookup_soma: bool = True,
    guess_soma: bool = True,
    detect_soma: bool = True,
    invalidation: float = 7500.0,
    skeleton_kwargs: Optional[dict] = None,
    skeleton_radius: bool = True,
    compartments: bool = True,
    compartment_labeler: Optional["CompartmentLabeler"] = None,
    l2_fetch_workers: int = 8,
) -> Cell:
    """Import a cell's level-2 graph from CAVE and skeletonize it locally.

    Unlike :func:`load_cell_from_client` this does not use the skeleton
    service, so it works for cells the service has not skeletonized or
    refuses to. The graph and its properties come from the chunkedgraph and
    the L2 cache, and the skeleton is built with
    :func:`ossify.skeletonization.skeletonize`.

    Every step after fetching the graph is optional. With the defaults the
    result is laid out like a cell from :func:`load_cell_from_client`.

    Parameters
    ----------
    root_id : int
        The root ID of the cell to import.
    client : CAVEclient
        The CAVE client to use for data retrieval.
    synapses : bool
        Whether to include pre- and postsynaptic sites as annotations linked
        to the graph. Default False.
    reference_tables : Optional[list[str]]
        Tables to merge into the synapse annotations when ``synapses`` is True.
    reference_suffixes : Optional[dict]
        Column suffixes for the reference tables, keyed by table name.
    restore_properties : bool
        Whether to keep every L2 cache attribute as a graph feature. If False
        only the position, ``max_dt_nm`` and ``size_nm3`` are fetched, which
        is what skeletonization needs. Default True.
    synapse_spatial_point : str
        The spatial point column to use for synapses. Default
        ``"ctr_pt_position"``.
    include_partner_root_id : bool
        Whether to keep the synaptic partner's root ID. Default False.
    timestamp : Optional[datetime.datetime]
        Timestamp at which ``root_id`` is valid. Defaults to the latest
        timestamp at which it is.
    omit_self_synapses : bool
        Whether to drop synapses a cell makes onto itself. Default True.
    build_skeleton : bool
        Whether to skeletonize the graph. If False the cell has a graph and,
        if requested, synapses, and none of the soma or skeleton options
        below apply. Default True.
    soma_point : array-like, optional
        Soma location in nanometers. If given it is used as is and the soma
        is not looked up or guessed.
    soma_table : str or sequence of str, optional
        Annotation table(s) to look the nucleus up in, by ``pt_root_id``.
        Defaults to the datastack's ``soma_table``. The nucleus is used only
        when exactly one entry is found across them.
    lookup_soma : bool
        Whether to look the nucleus up when no ``soma_point`` is given.
        Default True.
    guess_soma : bool
        Whether to locate the soma from the graph when it is neither given
        nor found by lookup: the centre of the thickest region, if that is
        large enough to be a cell body (see
        :func:`ossify.skeletonization.find_soma_point`). Default True.
    detect_soma : bool
        Whether to work out which graph vertices are soma and collapse them
        into a root at the soma point (see
        :func:`ossify.skeletonization.detect_soma`). If False the skeleton is
        just rooted at the graph vertex nearest the soma point, unless
        ``skeleton_kwargs`` asks for something else. Default True.
    invalidation : float
        Skeletonization invalidation distance in nanometers. Default 7500.
    skeleton_kwargs : Optional[dict]
        Further arguments for :func:`ossify.skeletonization.skeletonize`,
        for example ``{"soma_radius": 7500}`` for a ball collapse (with
        ``detect_soma=False``) or ``{"method": "pcg_skel"}``. These take
        precedence over the arguments above.
    skeleton_radius : bool
        Whether to add ``radius`` and ``radius_local`` features to the
        skeleton (see :func:`ossify.skeletonization.add_skeleton_radius`).
        Default True.
    compartments : bool
        Whether to add a ``compartment`` feature to the skeleton, labelling
        axon, dendrite and soma by the synapse-flow split the skeleton service
        uses. It needs the synapses, so it has no effect unless ``synapses``
        is True. ``cell.meta["compartment_source"]`` records how the labels
        were made. Default True.
    compartment_labeler : Optional[CompartmentLabeler]
        A model to label compartments with in place of the synapse-flow split:
        anything with a ``predict(cell)`` method that returns one label per
        skeleton vertex, such as an
        :class:`ossify.compartments.StructuredLabeler`. Labels may be
        compartment names (``"axon"``, ``"dendrite"``, ``"soma"``,
        ``"apical_dendrite"``) or SWC codes. The root is marked soma when the
        cell has one. The bundled models read the synapses and the L2 surface
        area, so they need ``synapses=True`` and ``restore_properties=True``.
        Ignored when ``compartments`` is False. Default None.
    l2_fetch_workers : int
        Number of concurrent requests the L2 property table is split across.
        The L2 cache is several times faster this way than with one large
        request. Default 8.

    Returns
    -------
    Cell
        The cell, with a graph layer of level-2 vertices and their edges and,
        as requested, a skeleton linked to the graph and ``pre_syn`` and
        ``post_syn`` annotations linked to the graph.

    Raises
    ------
    ValueError
        If ``root_id`` is not valid at ``timestamp``.

    Notes
    -----
    ``cell.meta["soma_source"]`` records where the soma came from: ``"given"``,
    ``"nucleus"``, ``"graph"`` or None, with its location in
    ``cell.meta["soma_point"]``. Without a soma the skeleton is rooted at an
    extremity of the graph.

    An object made of a single level-2 vertex gives a one-vertex skeleton.

    Examples
    --------
    >>> from caveclient import CAVEclient
    >>> client = CAVEclient("minnie65_public")
    >>> cell = ossify.load_cell_from_l2graph(864691135336055529, client)
    >>> cell.skeleton.get_feature("radius")

    Only the graph and its synapses, to skeletonize some other way:

    >>> cell = ossify.load_cell_from_l2graph(
    ...     864691135336055529, client, synapses=True, build_skeleton=False
    ... )
    >>> ossify.skeletonize(cell, soma_point=my_point, soma_radius=10000)
    """
    ts = _resolve_timestamp(root_id, client, timestamp)
    attributes = None if restore_properties else _L2_SKELETON_ATTRIBUTES
    l2_df, edges = _fetch_l2_graph(root_id, client, attributes, l2_fetch_workers)

    nrn = Cell(
        name=root_id,
        meta={
            "source": f"L2Graph({client.local_server})",
            "timestamp": ts,
            "datastack": client.datastack_name,
            "root_id": root_id,
        },
    ).add_graph(
        vertices=l2_df,
        spatial_columns=_L2_SPATIAL_COLUMNS,
        edges=edges,
        vertex_index="l2_id",
        edges_as_positional=False,
    )

    has_soma = False
    if build_skeleton:
        soma_point, soma_source = _resolve_soma_point(
            root_id,
            client,
            l2_df,
            edges,
            ts,
            soma_point=soma_point,
            soma_table=soma_table,
            lookup=lookup_soma,
            guess=guess_soma,
        )
        has_soma = soma_point is not None
        nrn.meta["soma_source"] = soma_source
        nrn.meta["soma_point"] = soma_point.tolist() if has_soma else None

        arguments = dict(
            invalidation=invalidation,
            soma_point=soma_point,
            detect_soma=detect_soma and has_soma,
        )
        arguments.update(skeleton_kwargs or {})
        skeletonize(nrn, **arguments)
        if skeleton_radius:
            add_skeleton_radius(nrn)

    if synapses:
        _add_synapse_annotations(
            nrn,
            root_id,
            client,
            ts,
            reference_tables=reference_tables,
            reference_suffixes=reference_suffixes or {},
            synapse_spatial_point=synapse_spatial_point,
            include_partner_root_id=include_partner_root_id,
            omit_self_synapses=omit_self_synapses,
        )

    if build_skeleton and compartments:
        if compartment_labeler is not None:
            labels, source = (
                _labeler_compartments(nrn, compartment_labeler, has_soma=has_soma),
                "labeler",
            )
        elif synapses:
            labels, source = (
                _synapse_flow_compartments(nrn, has_soma=has_soma),
                "synapse_flow",
            )
        else:
            labels, source = None, None
        if labels is not None:
            nrn.skeleton.add_feature(labels, "compartment")
            nrn.meta["compartment_source"] = source

    return nrn


def _labeler_compartments(
    cell: Cell, labeler: "CompartmentLabeler", has_soma: bool
) -> np.ndarray:
    """SWC compartment codes from a model's per-vertex labels.

    Parameters
    ----------
    cell : Cell
        Cell with a skeleton and whatever layers the labeler reads.
    labeler : CompartmentLabeler
        Object whose ``predict(cell)`` returns one label per skeleton vertex,
        as compartment names or SWC codes.
    has_soma : bool
        Whether the skeleton is rooted at a soma. If so the root is marked
        soma, as the synapse-flow labels do.

    Returns
    -------
    np.ndarray
        ``(n_vertices,)`` ``uint8`` SWC codes.

    Raises
    ------
    ValueError
        If the labeler cannot run on this cell, returns the wrong number of
        labels, or returns a label that is not an SWC compartment.
    """
    skel = cell.skeleton
    try:
        predicted = np.asarray(labeler.predict(cell))
    except (KeyError, AttributeError) as error:
        raise ValueError(
            "The compartment labeler could not find what it needs on this cell "
            f"({error!r}). The bundled models read the synapses and the L2 "
            "surface area: pass synapses=True and restore_properties=True."
        ) from error
    if predicted.shape != (skel.n_vertices,):
        raise ValueError(
            f"The compartment labeler returned labels of shape {predicted.shape}; "
            f"expected one per skeleton vertex ({skel.n_vertices},)."
        )

    if np.issubdtype(predicted.dtype, np.integer):
        valid = {int(member) for member in SWCCompartment}
        unknown = sorted(set(np.unique(predicted).tolist()) - valid)
        if unknown:
            raise ValueError(f"Labels {unknown} are not SWC compartment codes.")
        labels = predicted.astype(np.uint8)
    else:
        names = {member.name.lower(): int(member) for member in SWCCompartment}
        found = pd.unique(predicted)
        unknown = [name for name in found if str(name).lower() not in names]
        if unknown:
            raise ValueError(
                f"Labels {unknown} are not SWC compartments; expected one of "
                f"{sorted(names)}."
            )
        labels = np.zeros(skel.n_vertices, dtype=np.uint8)
        for name in found:
            labels[predicted == name] = names[str(name).lower()]

    if has_soma:
        labels[skel.root_positional] = int(SWCCompartment.SOMA)
    return labels


def _synapse_flow_compartments(cell: Cell, has_soma: bool) -> np.ndarray:
    """Axon, dendrite and soma labels as the skeleton service assigns them.

    The service splits axon from dendrite by synapse flow, accepts the split
    only if its segregation index reaches 0.5, extends it to whole segments,
    and then marks the root as soma.
    """
    skel = cell.skeleton
    labels = np.full(skel.n_vertices, int(SWCCompartment.DENDRITE), dtype=np.uint8)
    n_pre = len(cell.annotations["pre_syn"])
    n_post = len(cell.annotations["post_syn"])
    if n_pre > 0 and n_post > 0 and skel.n_vertices > 1:
        # Imported here: algorithms pulls in optional machinery translate
        # otherwise has no need of.
        from .algorithms import label_axon_from_synapse_flow

        is_axon = label_axon_from_synapse_flow(
            cell,
            extend_feature_to_segment=True,
            ntimes=1,
            segregation_index_threshold=0.5,
        )
        labels[np.asarray(is_axon, dtype=bool)] = int(SWCCompartment.AXON)
    if has_soma:
        labels[skel.root_positional] = int(SWCCompartment.SOMA)
    return labels


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
    if not root_ids:
        # Nothing to fetch. Never issue a pooled query with an empty root-id filter: the
        # materialization server rejects it with a misleading 500 ("ALL root id filters
        # passed ([]) were invalid").
        return {}
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
    if not valid_ids:
        return {}

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
    if not valid_ids:
        # Every valid root lacked a cached skeleton (only reachable with skip_invalid=True).
        return {}
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
