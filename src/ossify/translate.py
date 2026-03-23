from enum import IntEnum
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import fastremap
import numpy as np
import pandas as pd

from ossify.data_layers import SkeletonLayer

from .base import Cell, Link
from .utils import get_l2id_column, get_supervoxel_column, suppress_output

__all__ = ["load_cell_from_client", "SWCCompartment"]

if TYPE_CHECKING:
    import datetime

    from caveclient import CAVEclientFull as CAVEclient


class SWCCompartment(IntEnum):
    """Standard SWC compartment labels. See https://swc-specification.readthedocs.io/en/latest/swc.html#id-type-x-y-z-radius-parent"""

    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    DENDRITE = 3
    APICAL_DENDRITE = 4


def _process_synapse_table(
    root_id: int,
    table_name: str,
    client: "CAVEclient",
    side: Literal["pre", "post"],
    columns: dict,
    timestamp: "datetime.datetime",
    reference_tables: Optional[list[str]] = None,
    reference_suffixes: Optional[dict] = None,
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
    l2_column = get_l2id_column(side_column)
    syn_df[l2_column] = l2_ids
    if drop_other_side:
        syn_df.drop(columns=other_column, inplace=True)

    if reference_tables is not None:
        for ref_table in reference_tables:
            with suppress_output():
                ref_df = client.materialize.live_live_query(
                    ref_table,
                    filter_in_dict={ref_table: {"target_id": syn_df["id"].tolist()}},
                    timestamp=timestamp,
                    metadata=False,
                    desired_resolution=[1, 1, 1],
                ).drop(columns=["created", "valid"], errors="ignore")
            syn_df = syn_df.merge(
                ref_df.rename(
                    columns={
                        "target_id": "id",
                        "id": f"id_{reference_suffixes.get(ref_table, ref_table)}",
                    }
                ),
                how="left",
                on="id",
                suffixes=("", f"_{reference_suffixes.get(ref_table, ref_table)}"),
            )
    return syn_df


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
            reference_tables=reference_tables,
            reference_suffixes=reference_suffixes,
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
            reference_tables=reference_tables,
            reference_suffixes=reference_suffixes,
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
