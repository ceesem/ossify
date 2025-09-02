import fastremap
import numpy as np

from ossify import Cell, Link


def test_create_meshwork(
    root_id,
    skel_dict,
    l2_graph,
    l2_df,
    pre_syn_df,
    post_syn_df,
    synapse_spatial_columns,
    l2_spatial_columns,
):
    l2_map = {v: k for k, v in l2_df["l2_id"].to_dict().items()}
    edges = fastremap.remap(
        l2_graph,
        l2_map,
    )

    nrn = (
        Cell(
            name=root_id,
        )
        .add_graph(
            vertices=l2_df,
            spatial_columns=l2_spatial_columns,
            edges=edges,
            vertex_index="l2_id",
        )
        .add_skeleton(
            vertices=np.array(skel_dict["vertices"]),
            edges=np.array(skel_dict["edges"]),
            labels={
                "radius": skel_dict["radius"],
                "compartment": skel_dict["compartment"],
            },
            linkage=Link(
                mapping=skel_dict["mesh_to_skel_map"],
                source="graph",
                map_value_is_index=False,
            ),
        )
        .add_point_annotations(
            "pre_syn",
            vertices=pre_syn_df,
            spatial_columns=synapse_spatial_columns,
            vertex_index="id",
            linkage=Link(mapping="pre_pt_l2_id", target="graph"),
        )
        .add_point_annotations(
            "post_syn",
            vertices=post_syn_df,
            spatial_columns=synapse_spatial_columns,
            vertex_index="id",
            linkage=Link(mapping="post_pt_l2_id", target="graph"),
        )
    )
    assert len(nrn.layers) == 2


def test_morphology_loading(nrn):
    assert nrn is not None
    assert isinstance(nrn, Cell)
    assert np.isclose(nrn.skeleton.distance_to_root()[0], 501790.10060501)
