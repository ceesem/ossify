"""Tests for ``load_cell_from_l2graph``: local skeletonization of a CAVE L2 graph."""

import datetime
import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ossify.translate import (
    _blank_origin_positions,
    _fill_missing_positions,
    _get_l2data_table_chunked,
    load_cell_from_l2graph,
)

TIMESTAMP = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
XYZ = ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]


def make_client(l2_table: pd.DataFrame, l2_graph, nuclei: pd.DataFrame) -> MagicMock:
    """A CAVE client stand-in serving one cell's L2 graph, L2 table and nucleus."""
    client = MagicMock()
    client.local_server = "test-server"
    client.datastack_name = "test_stack"
    client.chunkedgraph.get_root_timestamps.return_value = [TIMESTAMP]
    client.chunkedgraph.is_latest_roots.return_value = [True]
    client.chunkedgraph.level2_chunk_graph.return_value = l2_graph
    client.info.get_datastack_info.return_value = {"soma_table": "nucleus_table"}

    # The loader calls this from several threads at once. A real client builds
    # a fresh table per response; this one reads a shared frame, and pandas
    # index lookups are not safe to run concurrently, so take turns.
    lock = threading.Lock()

    def get_l2data_table(l2_ids, attributes=None):
        with lock:
            table = l2_table.loc[l2_table.index.intersection(np.asarray(l2_ids))]
        if attributes is not None:
            keep = [
                column
                for column in table.columns
                if any(column.startswith(name) for name in attributes)
            ]
            table = table[keep]
        return table.copy()

    client.l2cache.get_l2data_table.side_effect = get_l2data_table
    client.materialize.tables.__getitem__.return_value.return_value.live_query.return_value = nuclei
    return client


@pytest.fixture
def l2_table(l2_df) -> pd.DataFrame:
    return l2_df.set_index("l2_id")


@pytest.fixture
def nucleus(skel_dict) -> pd.DataFrame:
    meta = skel_dict["meta"]
    point = np.array([meta["soma_pt_x"], meta["soma_pt_y"], meta["soma_pt_z"]])
    return pd.DataFrame({"id": [1], "pt_position": [point]})


class TestLoadCellFromL2Graph:
    def test_builds_graph_and_local_skeleton(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client)

        assert cell.graph.n_vertices == len(l2_table)
        skel = cell.skeleton
        assert len(skel.edges) == skel.n_vertices - 1
        assert np.allclose(skel.root_location, nucleus.pt_position.iloc[0])
        assert cell.graph.mapping_coverage("skeleton") == 1.0
        assert cell.meta["root_id"] == root_id
        assert cell.meta["timestamp"] == TIMESTAMP

        # The skeleton service is never involved.
        client.skeleton.get_skeleton.assert_not_called()
        # The nucleus was looked up in the datastack's soma table for this root.
        client.materialize.tables.__getitem__.assert_called_with("nucleus_table")
        client.materialize.tables.__getitem__.return_value.assert_called_with(
            pt_root_id=root_id
        )

    def test_l2_table_is_fetched_in_concurrent_chunks(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        load_cell_from_l2graph(root_id, client, l2_fetch_workers=8)

        calls = client.l2cache.get_l2data_table.call_args_list
        assert len(calls) == 8
        requested = np.concatenate([np.asarray(call.args[0]) for call in calls])
        assert np.array_equal(np.sort(requested), np.sort(l2_table.index.to_numpy()))

    def test_restore_properties_controls_attributes(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client, restore_properties=False)
        assert {"max_dt_nm", "size_nm3"} <= set(cell.graph.feature_names)
        assert "area_nm2" not in cell.graph.feature_names
        for call in client.l2cache.get_l2data_table.call_args_list:
            assert call.kwargs["attributes"] == [
                "rep_coord_nm",
                "max_dt_nm",
                "size_nm3",
            ]

        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client, restore_properties=True)
        assert "area_nm2" in cell.graph.feature_names

    def test_explicit_soma_point_skips_the_lookup(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        point = nucleus.pt_position.iloc[0] + np.array([500.0, 0.0, 0.0])
        cell = load_cell_from_l2graph(root_id, client, soma_point=point)
        assert np.allclose(cell.skeleton.root_location, point)
        client.materialize.tables.__getitem__.assert_not_called()

    def test_no_nucleus_falls_back_to_the_graph(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus.iloc[:0])
        cell = load_cell_from_l2graph(root_id, client)
        assert cell.meta["soma_source"] == "graph"
        # The guess lands within a few microns of the real nucleus.
        offset = np.linalg.norm(
            cell.skeleton.root_location - nucleus.pt_position.iloc[0]
        )
        assert offset < 5000
        assert np.allclose(cell.meta["soma_point"], cell.skeleton.root_location)

    def test_no_soma_like_region_gives_a_skeleton_without_a_soma(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        # With nothing thick in the graph there is no soma to find.
        thin = l2_table.copy()
        thin["max_dt_nm"] = np.minimum(thin["max_dt_nm"], 400)
        client = make_client(thin, l2_graph, nucleus.iloc[:0])
        cell = load_cell_from_l2graph(root_id, client)
        skel = cell.skeleton
        assert cell.meta["soma_source"] is None
        assert cell.meta["soma_point"] is None
        assert len(skel.edges) == skel.n_vertices - 1
        # Rooted on a graph vertex rather than a new soma vertex.
        distance = np.linalg.norm(cell.graph.vertices - skel.root_location, axis=1)
        assert distance.min() == 0

    def test_several_nuclei_warn_and_use_the_graph(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, pd.concat([nucleus, nucleus]))
        with pytest.warns(UserWarning, match="2 entries"):
            cell = load_cell_from_l2graph(root_id, client)
        assert cell.meta["soma_source"] == "graph"

    def test_no_soma_table_configured(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        client.info.get_datastack_info.return_value = {}
        cell = load_cell_from_l2graph(root_id, client)
        assert cell.meta["soma_source"] == "graph"
        client.materialize.tables.__getitem__.assert_not_called()

    def test_soma_source_is_recorded(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        assert load_cell_from_l2graph(root_id, client).meta["soma_source"] == "nucleus"
        given = load_cell_from_l2graph(root_id, client, soma_point=[7.6e5, 5.4e5, 9e5])
        assert given.meta["soma_source"] == "given"

    def test_skeleton_has_radius_features(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        skel = load_cell_from_l2graph(root_id, client).skeleton
        assert {"radius", "radius_local"} <= set(skel.feature_names)
        radius = skel.get_feature("radius")
        local = skel.get_feature("radius_local")
        assert np.isfinite(radius).all() and (radius > 0).all()
        assert np.isfinite(local).all()
        # The soma is by far the thickest thing in the cell on both measures.
        assert radius[skel.root_positional] == radius.max()
        assert local[skel.root_positional] == local.max()
        assert 3000 < radius[skel.root_positional] < 12000
        # Compartments need synapses, which were not requested.
        assert "compartment" not in skel.feature_names

    def test_single_l2_object_gives_a_trivial_skeleton(
        self, root_id, l2_table, nucleus
    ):
        only = l2_table.index.to_numpy()[:1]
        client = make_client(l2_table, [], nucleus.iloc[:0])
        client.chunkedgraph.get_leaves.return_value = only
        cell = load_cell_from_l2graph(root_id, client)

        client.chunkedgraph.get_leaves.assert_called_with(root_id, stop_layer=2)
        assert cell.graph.n_vertices == 1
        assert len(cell.graph.edges) == 0
        skel = cell.skeleton
        assert skel.n_vertices == 1
        assert len(skel.edges) == 0
        assert np.allclose(skel.root_location, cell.graph.vertices[0])
        assert cell.graph.mapping_coverage("skeleton") == 1.0
        assert cell.meta["soma_source"] is None
        assert np.isfinite(skel.get_feature("radius")).all()

    def test_missing_l2_entries_are_filled_from_neighbours(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        # Twenty L2 ids the cache has nothing for, and five it has no position for.
        rng = np.random.default_rng(0)
        absent = rng.choice(l2_table.index.to_numpy(), size=25, replace=False)
        partial = l2_table.drop(index=absent[:20])
        partial.loc[absent[20:], XYZ] = np.nan
        client = make_client(partial, l2_graph, nucleus)

        cell = load_cell_from_l2graph(root_id, client)
        assert cell.graph.n_vertices == len(l2_table)
        assert np.isfinite(cell.graph.vertices).all()
        assert cell.graph.mapping_coverage("skeleton") == 1.0

    def test_origin_positions_are_treated_as_missing(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        # The cache can answer an id it has not computed with a position of
        # exactly (0, 0, 0) and no other features, rather than NaN.
        rng = np.random.default_rng(1)
        unplaced = rng.choice(l2_table.index.to_numpy(), size=5, replace=False)
        partial = l2_table.copy()
        partial.loc[unplaced, XYZ] = 0.0
        partial.loc[unplaced, ["max_dt_nm", "size_nm3"]] = np.nan
        client = make_client(partial, l2_graph, nucleus)

        cell = load_cell_from_l2graph(root_id, client)
        assert not (cell.graph.vertices == 0).all(axis=1).any()
        placed = cell.graph.vertices[np.isin(l2_table.index.to_numpy(), unplaced)]
        expected = l2_table.loc[np.sort(unplaced), XYZ].to_numpy(dtype=float)
        # Filled from neighbours, so each lands within a few chunks of where it belongs.
        assert (np.linalg.norm(placed - expected, axis=1) < 20_000).all()

    def test_invalid_root_at_timestamp_raises(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        client.chunkedgraph.is_latest_roots.return_value = [False]
        with pytest.raises(ValueError, match="not valid"):
            load_cell_from_l2graph(root_id, client, timestamp=TIMESTAMP)


class TestLoaderOptions:
    """Each step after fetching the graph can be switched off or redirected."""

    def test_graph_only(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client, build_skeleton=False)
        assert cell.skeleton is None
        assert cell.graph.n_vertices == len(l2_table)
        assert len(cell.graph.edges) > 0
        assert "soma_source" not in cell.meta
        # Nothing soma-related was asked of the server.
        client.materialize.tables.__getitem__.assert_not_called()

    def test_no_lookup(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client, lookup_soma=False)
        assert cell.meta["soma_source"] == "graph"
        client.materialize.tables.__getitem__.assert_not_called()

    def test_no_guess(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus.iloc[:0])
        cell = load_cell_from_l2graph(root_id, client, guess_soma=False)
        assert cell.meta["soma_source"] is None
        assert cell.skeleton is not None

    def test_no_lookup_and_no_guess(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(
            root_id, client, lookup_soma=False, guess_soma=False
        )
        assert cell.meta["soma_source"] is None

    def test_no_soma_detection_roots_at_nearest_vertex(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(root_id, client, detect_soma=False)
        assert cell.meta["soma_source"] == "nucleus"
        point = nucleus.pt_position.iloc[0]
        distance = np.linalg.norm(cell.graph.vertices - point, axis=1)
        assert np.allclose(
            cell.skeleton.root_location, cell.graph.vertices[np.argmin(distance)]
        )

    def test_no_radius(self, root_id, l2_table, l2_graph, nucleus):
        client = make_client(l2_table, l2_graph, nucleus)
        skel = load_cell_from_l2graph(root_id, client, skeleton_radius=False).skeleton
        assert "radius" not in skel.feature_names
        assert "radius_local" not in skel.feature_names

    def test_skeleton_kwargs_reach_the_skeletonizer(
        self, root_id, l2_table, l2_graph, nucleus, skel_dict
    ):
        # Asking for the service's method and its ball reproduces the
        # service's skeleton, radius included.
        client = make_client(l2_table, l2_graph, nucleus)
        cell = load_cell_from_l2graph(
            root_id,
            client,
            detect_soma=False,
            skeleton_kwargs={"method": "pcg_skel", "soma_radius": 7500},
        )
        skel = cell.skeleton
        assert np.array_equal(skel.vertices, np.array(skel_dict["vertices"]))
        assert np.allclose(skel.get_feature("radius"), skel_dict["radius"], rtol=1e-6)

    def test_skeleton_kwargs_override_named_arguments(
        self, root_id, l2_table, l2_graph, nucleus
    ):
        client = make_client(l2_table, l2_graph, nucleus)
        coarse = load_cell_from_l2graph(
            root_id, client, invalidation=7500, skeleton_kwargs={"invalidation": 30000}
        )
        fine = load_cell_from_l2graph(root_id, client, invalidation=7500)
        assert coarse.skeleton.n_vertices < fine.skeleton.n_vertices


class TestLoaderSynapses:
    @pytest.fixture
    def synapse_client(
        self, monkeypatch, l2_table, l2_graph, nucleus, pre_syn_df, post_syn_df
    ):
        """A client whose synapse queries return the stored test frames."""
        frames = {"pre": pre_syn_df, "post": post_syn_df}
        calls = []

        def fake_process(root_id, table_name, client, side, columns, timestamp, **kw):
            calls.append((side, kw))
            return frames[side].copy()

        monkeypatch.setattr("ossify.translate._process_synapse_table", fake_process)
        client = make_client(l2_table, l2_graph, nucleus)
        client.synapse_calls = calls
        return client

    def test_synapses_are_attached_to_the_graph(self, root_id, synapse_client):
        cell = load_cell_from_l2graph(
            root_id,
            synapse_client,
            synapses=True,
            synapse_spatial_point="ctr_pt_position",
        )
        assert set(cell.annotations.names) == {"pre_syn", "post_syn"}
        assert len(cell.annotations.pre_syn) > 0
        # Reachable from the skeleton through the graph.
        on_skeleton = cell.annotations.post_syn.map_index_to_layer(
            "skeleton", as_positional=True
        )
        assert len(on_skeleton) == len(cell.annotations.post_syn)

    def test_synapse_options_are_passed_through(self, root_id, synapse_client):
        load_cell_from_l2graph(
            root_id,
            synapse_client,
            synapses=True,
            include_partner_root_id=True,
            omit_self_synapses=False,
            reference_tables=["ref"],
            reference_suffixes={"ref": "_r"},
        )
        assert [side for side, _ in synapse_client.synapse_calls] == ["pre", "post"]
        for _, options in synapse_client.synapse_calls:
            assert options["drop_other_side"] is False
            assert options["omit_autapses"] is False
            assert options["reference_tables"] == ["ref"]
            assert options["reference_suffixes"] == {"ref": "_r"}

    def test_compartments_follow_the_service_convention(self, root_id, synapse_client):
        cell = load_cell_from_l2graph(root_id, synapse_client, synapses=True)
        skel = cell.skeleton
        compartment = skel.get_feature("compartment")
        assert set(np.unique(compartment)) <= {1, 2, 3}
        assert compartment[skel.root_positional] == 1
        assert (compartment == 1).sum() == 1
        # This cell has a real axon and a real dendrite.
        assert (compartment == 2).sum() > 100
        assert (compartment == 3).sum() > 100

    def test_compartments_can_be_switched_off(self, root_id, synapse_client):
        cell = load_cell_from_l2graph(
            root_id, synapse_client, synapses=True, compartments=False
        )
        assert "compartment" not in cell.skeleton.feature_names
        assert "pre_syn" in cell.annotations.names

    def test_compartment_labeler_replaces_synapse_flow(self, root_id, synapse_client):
        seen = {}

        class EverythingAxon:
            def predict(self, cell):
                seen["annotations"] = list(cell.annotations.names)
                seen["graph_features"] = list(cell.graph.feature_names)
                return np.full(cell.skeleton.n_vertices, "axon", dtype=object)

        cell = load_cell_from_l2graph(
            root_id, synapse_client, synapses=True, compartment_labeler=EverythingAxon()
        )
        skel = cell.skeleton
        compartment = skel.get_feature("compartment")
        # The labeler sees the synapses and every L2 property, as the bundled models need.
        assert {"pre_syn", "post_syn"} <= set(seen["annotations"])
        assert "area_nm2" in seen["graph_features"]
        # Its labels are stored as SWC codes, and the root is still marked soma.
        assert compartment[skel.root_positional] == 1
        assert (np.delete(compartment, skel.root_positional) == 2).all()
        assert cell.meta["compartment_source"] == "labeler"

    def test_compartment_source_is_recorded_for_synapse_flow(
        self, root_id, synapse_client
    ):
        cell = load_cell_from_l2graph(root_id, synapse_client, synapses=True)
        assert cell.meta["compartment_source"] == "synapse_flow"

    def test_compartment_labeler_accepts_swc_codes(self, root_id, synapse_client):
        class Codes:
            def predict(self, cell):
                return np.full(cell.skeleton.n_vertices, 3)

        cell = load_cell_from_l2graph(
            root_id, synapse_client, synapses=True, compartment_labeler=Codes()
        )
        assert set(np.unique(cell.skeleton.get_feature("compartment"))) == {1, 3}

    def test_compartment_labeler_unknown_label_raises(self, root_id, synapse_client):
        class Odd:
            def predict(self, cell):
                return np.full(cell.skeleton.n_vertices, "spine", dtype=object)

        with pytest.raises(ValueError, match="spine"):
            load_cell_from_l2graph(
                root_id, synapse_client, synapses=True, compartment_labeler=Odd()
            )

    def test_compartment_labeler_failure_names_what_it_needs(
        self, root_id, synapse_client
    ):
        class NeedsSynapses:
            def predict(self, cell):
                return cell.annotations["pre_syn"]

        # No synapses were loaded, so the labeler cannot find its inputs.
        with pytest.raises(ValueError, match="synapses=True"):
            load_cell_from_l2graph(
                root_id, synapse_client, compartment_labeler=NeedsSynapses()
            )

    def test_compartment_labeler_is_skipped_when_compartments_off(
        self, root_id, synapse_client
    ):
        class Boom:
            def predict(self, cell):
                raise AssertionError("should not be called")

        cell = load_cell_from_l2graph(
            root_id,
            synapse_client,
            synapses=True,
            compartments=False,
            compartment_labeler=Boom(),
        )
        assert "compartment" not in cell.skeleton.feature_names

    def test_synapses_without_a_skeleton(self, root_id, synapse_client):
        cell = load_cell_from_l2graph(
            root_id, synapse_client, synapses=True, build_skeleton=False
        )
        assert cell.skeleton is None
        assert set(cell.annotations.names) == {"pre_syn", "post_syn"}


class TestChunkedL2Table:
    def test_small_requests_are_not_split(self, l2_table):
        client = make_client(l2_table, [], pd.DataFrame())
        ids = l2_table.index.to_numpy()[:300]
        table = _get_l2data_table_chunked(client, ids, None, max_workers=8)
        assert client.l2cache.get_l2data_table.call_count == 1
        assert np.array_equal(table.index.to_numpy(), ids)

    def test_rows_follow_the_requested_order(self, l2_table):
        client = make_client(l2_table, [], pd.DataFrame())
        ids = l2_table.index.to_numpy()[::-1][:5000]
        table = _get_l2data_table_chunked(client, ids, None, max_workers=4)
        assert client.l2cache.get_l2data_table.call_count == 4
        assert np.array_equal(table.index.to_numpy(), ids)
        assert table.index.name == "l2_id"


class TestFillMissingPositions:
    def test_fills_from_neighbours_in_rounds(self):
        # A chain 0-1-2-3-4 with the middle three missing.
        positions = np.array(
            [[0.0, 0, 0], [np.nan] * 3, [np.nan] * 3, [np.nan] * 3, [40.0, 0, 0]]
        )
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        filled = _fill_missing_positions(positions, edges)
        assert np.isfinite(filled).all()
        assert np.allclose(filled[[0, 4]], positions[[0, 4]])
        assert np.allclose(filled[1], [0.0, 0, 0])
        assert np.allclose(filled[3], [40.0, 0, 0])
        assert np.allclose(filled[2], [20.0, 0, 0])

    def test_origin_rows_are_blanked(self):
        positions = np.array([[0.0, 0, 0], [0.0, 5, 0], [np.nan] * 3, [3.0, 4, 5]])
        blanked = _blank_origin_positions(positions)
        assert np.isnan(blanked[0]).all() and np.isnan(blanked[2]).all()
        assert np.array_equal(blanked[[1, 3]], positions[[1, 3]])
        assert positions[0, 0] == 0.0  # the input is not modified

    def test_complete_positions_are_untouched(self):
        positions = np.arange(12, dtype=float).reshape(4, 3)
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        assert np.array_equal(_fill_missing_positions(positions, edges), positions)

    def test_unreachable_missing_vertex_raises(self):
        positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [np.nan] * 3])
        with pytest.raises(ValueError, match="position"):
            _fill_missing_positions(positions, np.array([[0, 1]]))
