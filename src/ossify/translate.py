from typing import TYPE_CHECKING, Literal, Optional

import fastremap
import numpy as np
import pandas as pd

from .base import Cell, Link
from .utils import get_l2id_column, get_supervoxel_column

__all__ = ["cell_from_client"]

if TYPE_CHECKING:
    import datetime

    from caveclient import CAVEclientFull as CAVEclient


def _process_synapse_table(
    root_id: int,
    table_name: str,
    client: "CAVEclient",
    side: Literal["pre", "post"],
    columns: dict,
    timestamp: "datetime.datetime",
    reference_tables: Optional[list[str]] = None,
    drop_other_side: bool = True,
    omit_autapses: bool = True,
) -> pd.DataFrame:
    "Perform a synapse query and get the l2 ids for the given root_id."
    if side == "pre":
        other_side = "post"
    else:
        other_side = "pre"
    side_column = columns[side]
    other_column = columns[other_side]

    syn_df = client.materialize.tables[table_name](**{side_column: root_id}).live_query(
        desired_resolution=[1, 1, 1],
        split_positions=True,
        timestamp=timestamp,
    )
    if omit_autapses:
        syn_df.query(f"{side_column} != {other_column}", inplace=True)

    svid_column = get_supervoxel_column(side_column)
    l2_ids = client.chunkedgraph.get_roots(
        syn_df[svid_column], stop_layer=2, timestamp=timestamp
    )
    l2_column = get_l2id_column(side_column)
    syn_df[l2_column] = l2_ids
    if drop_other_side:
        syn_df.drop(columns=other_column, inplace=True)

    return syn_df


def cell_from_client(
    root_id: int,
    client: "CAVEclient",
    *,
    synapses: bool = False,
    restore_graph: bool = False,
    restore_properties: bool = True,
    synapse_spatial_point: str = "ctr_pt_position",
    include_partner_root_id: bool = False,
    timestamp: Optional["datetime.datetime"] = None,
    omit_self_synapses: bool = True,
    skeleton_version: int = 4,
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

    Returns
    -------
    Cell
        The imported cell object.
    """
    sk = client.skeleton.get_skeleton(
        root_id, skeleton_version=skeleton_version, output_format="dict"
    )
    if timestamp is None:
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
        pre_syn_df = _process_synapse_table(
            root_id,
            synapse_table,
            client,
            "pre",
            synapse_columns,
            ts,
            drop_other_side=not include_partner_root_id,
            omit_autapses=omit_self_synapses,
        )
        post_syn_df = _process_synapse_table(
            root_id,
            synapse_table,
            client,
            "post",
            synapse_columns,
            ts,
            drop_other_side=not include_partner_root_id,
            omit_autapses=omit_self_synapses,
        )

    l2ids = sk["lvl2_ids"]
    l2_spatial_columns = [
        "rep_coord_nm_x",
        "rep_coord_nm_y",
        "rep_coord_nm_z",
    ]
    if restore_properties:
        l2_df = client.l2cache.get_l2data_table(l2ids)
    else:
        l2_df = client.l2cache.get_l2data_table(l2ids, attributes=["rep_coord_nm"])
    l2_df.reset_index(inplace=True)

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
            labels={"radius": sk["radius"], "compartment": sk["compartment"]},
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
