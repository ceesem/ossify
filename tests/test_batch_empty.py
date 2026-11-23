"""Regression: batch loading must never issue a pooled query with an empty root-id filter.

The materialization server rejects ``filter_in_dict={...: []}`` with a misleading 500
("ALL root id filters passed ([]) were invalid"), which used to sink whole batches whose
roots all lacked cached skeletons.
"""

from unittest.mock import MagicMock

from ossify.translate import fetch_frames_batch, load_cell_batch_from_client


def _client(valid_mask, skeletons):
    client = MagicMock()
    client.materialize.get_timestamp.return_value = "ts"
    client.chunkedgraph.is_latest_roots.return_value = valid_mask
    client.skeleton.fetch_skeletons.return_value = skeletons
    client.materialize.live_live_query.side_effect = AssertionError("queried")
    client.l2cache.get_l2data_table.side_effect = AssertionError("queried")
    return client


def test_fetch_frames_batch_empty_issues_no_query():
    client = _client([], {})
    assert fetch_frames_batch([], client, lvl2_ids_by_root={}) == {}
    client.materialize.live_live_query.assert_not_called()


def test_batch_with_no_cached_skeletons_returns_empty():
    client = _client([True, True], {})
    out = load_cell_batch_from_client(
        [1, 2], client, synapses=True, timestamp="ts", skip_invalid=True
    )
    assert out == {}
    client.materialize.live_live_query.assert_not_called()


def test_batch_with_no_valid_roots_returns_empty():
    client = _client([False, False], {})
    out = load_cell_batch_from_client(
        [1, 2], client, synapses=True, timestamp="ts", skip_invalid=True
    )
    assert out == {}
    client.skeleton.fetch_skeletons.assert_not_called()
    client.materialize.live_live_query.assert_not_called()
