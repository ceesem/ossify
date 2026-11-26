import numpy as np
import pandas as pd
import pytest

from ossify import utils


class TestMajorityAgg:
    """Tests for majority_agg utility function."""

    def test_majority_aggregation_numeric(self):
        """Test majority vote aggregation with numeric data."""
        data = [1, 1, 2, 1, 3]

        majority_func = utils.majority_agg()
        result = majority_func(data)

        assert result == 1


class TestProcessVertices:
    """Tests for process_vertices utility function."""

    def test_process_vertices_with_dataframe(self, spatial_columns):
        """Test processing vertices when input is already a DataFrame."""
        # Create test DataFrame
        vertices_df = pd.DataFrame(
            {
                "x": [0, 1, 2],
                "y": [0, 0, 0],
                "z": [0, 0, 0],
                "feature1": ["a", "b", "c"],
                "feature2": [1, 2, 3],
            }
        )

        features = {"extra_feature": [10, 20, 30]}

        result_df, spatial_cols, feature_cols = utils.process_vertices(
            vertices=vertices_df,
            spatial_columns=spatial_columns,
            features=features,
            vertex_index=None,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert spatial_cols == spatial_columns
        assert "feature1" in feature_cols
        assert "feature2" in feature_cols
        assert "extra_feature" in feature_cols
        assert len(result_df) == 3

    def test_process_vertices_with_numpy_array(self, spatial_columns):
        """Test processing vertices when input is numpy array."""
        vertices_array = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

        features = {"compartment": ["soma", "dendrite", "axon"]}

        result_df, spatial_cols, feature_cols = utils.process_vertices(
            vertices=vertices_array,
            spatial_columns=spatial_columns,
            features=features,
            vertex_index=None,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert spatial_cols == spatial_columns
        assert "compartment" in feature_cols
        assert len(result_df) == 3
        assert result_df[spatial_columns[0]].tolist() == [0, 1, 2]


class TestProcessSpatialColumns:
    """Tests for process_spatial_columns utility function."""

    def test_process_spatial_columns_string_input(self):
        """Test processing spatial columns with string input."""
        result = utils.process_spatial_columns("position")

        assert result == ["position_x", "position_y", "position_z"]

    def test_process_spatial_columns_list_input(self):
        """Test processing spatial columns with list input."""
        input_cols = ["x", "y", "z"]
        result = utils.process_spatial_columns(input_cols)

        assert result == input_cols

    def test_process_spatial_columns_default(self):
        """Test processing spatial columns with default input."""
        result = utils.process_spatial_columns()

        assert result == ["x", "y", "z"]


class TestSinglePathLength:
    """Tests for single_path_length utility function."""

    def test_path_length_linear(self):
        """Test path length calculation on linear path."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

        edges = np.array([[0, 1], [1, 2], [2, 3]])

        path = np.array([0, 1, 2, 3])

        length = utils.single_path_length(path, vertices, edges)

        # Should be 3.0 (unit steps)
        assert np.isclose(length, 3.0)

    def test_path_length_single_vertex(self):
        """Test path length calculation on single vertex."""
        vertices = np.array([[0, 0, 0]])
        edges = np.array([])
        path = np.array([0])

        length = utils.single_path_length(path, vertices, edges)

        # Should be 0.0
        assert np.isclose(length, 0.0)


class TestRemapVerticesAndEdges:
    """Tests for remap_vertices_and_edges utility function."""

    def test_remap_basic(self):
        """Test basic remapping functionality."""
        id_list = np.array([100, 200, 300])
        edgelist = np.array([[100, 200], [200, 300]])

        id_map, new_edges = utils.remap_vertices_and_edges(id_list, edgelist)

        # Check mapping
        assert id_map[100] == 0
        assert id_map[200] == 1
        assert id_map[300] == 2

        # Check remapped edges
        expected_edges = np.array([[0, 1], [1, 2]])
        np.testing.assert_array_equal(new_edges, expected_edges)


class TestGetColumnMappings:
    """Tests for column mapping utility functions."""

    def test_get_supervoxel_column(self):
        """Test supervoxel column name generation."""
        result = utils.get_supervoxel_column("synapse_root_id")
        assert result == "synapse_supervoxel_id"

    def test_get_l2id_column(self):
        """Test l2id column name generation."""
        result = utils.get_l2id_column("synapse_root_id")
        assert result == "synapse_l2_id"


class TestSuppressOutputThreads:
    """suppress_output swaps process-wide streams, so overlapping uses must nest."""

    def test_overlapping_threads_restore_streams(self):
        import sys
        import threading
        import time

        from ossify.utils import suppress_output

        before_out, before_err = sys.stdout, sys.stderr
        errors = []
        start = threading.Barrier(8)

        def work(delay: float) -> None:
            try:
                start.wait()
                for _ in range(20):
                    with suppress_output():
                        time.sleep(delay)
                        print("hidden")
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=work, args=(0.0005 * (i + 1),)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert sys.stdout is before_out
        assert sys.stderr is before_err

    def test_first_in_first_out_restores_streams(self):
        import sys
        import threading

        from ossify.utils import suppress_output

        before_out, before_err = sys.stdout, sys.stderr
        a_inside, b_inside, a_done = (threading.Event() for _ in range(3))
        release_b = threading.Event()

        def first() -> None:
            with suppress_output():
                a_inside.set()
                b_inside.wait()
            a_done.set()

        def second() -> None:
            a_inside.wait()
            with suppress_output():
                b_inside.set()
                a_done.wait()
                release_b.wait()
                print("still hidden")  # must not hit a closed file

        threads = [
            threading.Thread(target=first, daemon=True),
            threading.Thread(target=second, daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            assert a_done.wait(5)
            # One thread is still inside: output must still be suppressed.
            still_suppressed = sys.stdout is not before_out
        finally:
            release_b.set()
            for t in threads:
                t.join(5)
            after_out, after_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = before_out, before_err
        assert still_suppressed
        assert after_out is before_out
        assert after_err is before_err

    def test_logger_level_restored_after_overlap(self):
        import logging
        import threading

        from ossify.utils import suppress_output

        logger = logging.getLogger()
        level = logger.level
        inside = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with suppress_output():
                inside.set()
                release.wait()

        t = threading.Thread(target=hold)
        t.start()
        inside.wait()
        with suppress_output():
            pass
        # The other thread is still inside, so output stays suppressed.
        assert logger.level == logging.CRITICAL
        release.set()
        t.join()
        assert logger.level == level
