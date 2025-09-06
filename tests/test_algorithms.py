import numpy as np
import pandas as pd
import pytest

from ossify import Cell, Link, algorithms


class TestStrahlerNumber:
    """Tests for strahler_number algorithm."""

    def test_strahler_linear_skeleton(self, simple_skeleton_data, spatial_columns):
        """Test Strahler number calculation on linear skeleton."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Calculate Strahler numbers
        strahler_nums = algorithms.strahler_number(cell.skeleton)

        # For a linear skeleton, all should have Strahler number 1
        assert len(strahler_nums) == 5
        assert all(num == 1 for num in strahler_nums)

    def test_strahler_branched_skeleton(self, branched_skeleton_data, spatial_columns):
        """Test Strahler number calculation on branched skeleton."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[1]],  # 103 -> 101 (branch)
                [vertex_indices[4], vertex_indices[1]],  # 104 -> 101 (branch)
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Calculate Strahler numbers
        strahler_nums = algorithms.strahler_number(cell.skeleton)

        # Should have proper Strahler numbers reflecting branch structure
        assert len(strahler_nums) == 5
        assert all(num >= 1 for num in strahler_nums)  # All should be at least 1

    def test_strahler_with_cell_input(self, simple_skeleton_data, spatial_columns):
        """Test that strahler_number works with Cell input."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Test with Cell input
        strahler_nums_cell = algorithms.strahler_number(cell)

        # Test with SkeletonLayer input
        strahler_nums_skel = algorithms.strahler_number(cell.skeleton)

        # Should give same result
        np.testing.assert_array_equal(strahler_nums_cell, strahler_nums_skel)


class TestSmoothLabels:
    """Tests for smooth_labels algorithm."""

    def test_smooth_labels_basic(self, simple_skeleton_data, spatial_columns):
        """Test basic label smoothing functionality."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Create initial label with sharp boundaries
        initial_label = np.array([1.0, 1.0, 0.0, 0.0, 0.0])

        # Smooth the labels
        smoothed_labels = algorithms.smooth_labels(cell.skeleton, initial_label)

        # Should return same length as input
        assert len(smoothed_labels) == len(initial_label)

        # Smoothed labels should be between 0 and 1
        assert all(0 <= label <= 1 for label in smoothed_labels)

    def test_smooth_labels_with_cell_input(self, simple_skeleton_data, spatial_columns):
        """Test that smooth_labels works with Cell input."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Create initial label
        initial_label = np.array([1.0, 0.0, 0.0, 0.0, 1.0])

        # Test with Cell input
        smoothed_cell = algorithms.smooth_labels(cell, initial_label)

        # Test with SkeletonLayer input
        smoothed_skel = algorithms.smooth_labels(cell.skeleton, initial_label)

        # Should give same result
        np.testing.assert_array_almost_equal(smoothed_cell, smoothed_skel)

    def test_smooth_labels_alpha_parameter(self, simple_skeleton_data, spatial_columns):
        """Test that alpha parameter affects smoothing."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[2]],  # 103 -> 102
                [vertex_indices[4], vertex_indices[3]],  # 104 -> 103
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Create initial label
        initial_label = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # Test with different alpha values
        smoothed_low = algorithms.smooth_labels(cell.skeleton, initial_label, alpha=0.1)
        smoothed_high = algorithms.smooth_labels(
            cell.skeleton, initial_label, alpha=0.9
        )

        # Should give different results
        assert not np.array_equal(smoothed_low, smoothed_high)

        # Both should have same length as input
        assert len(smoothed_low) == len(initial_label)
        assert len(smoothed_high) == len(initial_label)


class TestSegregationIndex:
    """Tests for segregation_index algorithm."""

    def test_segregation_index_perfect_segregation(self):
        """Test segregation index with perfect segregation."""
        # Perfect segregation: all pre on axon, all post on dendrite
        seg_index = algorithms.segregation_index(
            axon_pre=10, axon_post=0, dendrite_pre=0, dendrite_post=10
        )

        # Should be close to 1 (perfect segregation)
        assert 0.8 <= seg_index <= 1.0

    def test_segregation_index_no_segregation(self):
        """Test segregation index with no segregation."""
        # No segregation: equal pre and post on both compartments
        seg_index = algorithms.segregation_index(
            axon_pre=5, axon_post=5, dendrite_pre=5, dendrite_post=5
        )

        # Should be close to 0 (no segregation)
        assert 0.0 <= seg_index <= 0.2

    def test_segregation_index_partial_segregation(self):
        """Test segregation index with partial segregation."""
        # Partial segregation: more pre on axon, more post on dendrite
        seg_index = algorithms.segregation_index(
            axon_pre=8, axon_post=2, dendrite_pre=2, dendrite_post=8
        )

        # Should be somewhere in middle (adjust range based on actual behavior)
        assert 0.2 <= seg_index <= 0.8

    def test_segregation_index_zero_synapses(self):
        """Test segregation index with zero synapses."""
        # Edge case: no synapses at all
        seg_index = algorithms.segregation_index(
            axon_pre=0, axon_post=0, dendrite_pre=0, dendrite_post=0
        )

        # Should handle gracefully (likely return 0 or some default)
        assert isinstance(seg_index, (int, float))
        assert not np.isnan(seg_index)


class TestAlgorithmIntegration:
    """Integration tests for algorithms working together."""

    def test_strahler_and_smooth_integration(
        self, branched_skeleton_data, spatial_columns
    ):
        """Test using Strahler numbers as input to smoothing."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices instead of positional indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],  # 101 -> 100
                [vertex_indices[2], vertex_indices[1]],  # 102 -> 101
                [vertex_indices[3], vertex_indices[1]],  # 103 -> 101 (branch)
                [vertex_indices[4], vertex_indices[1]],  # 104 -> 101 (branch)
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Get Strahler numbers
        strahler_nums = algorithms.strahler_number(cell.skeleton)

        # Use as input to smoothing
        smoothed_strahler = algorithms.smooth_labels(
            cell.skeleton, strahler_nums.astype(float)
        )

        # Should work without errors
        assert len(smoothed_strahler) == len(strahler_nums)
        assert all(isinstance(val, (int, float)) for val in smoothed_strahler)

    def test_algorithms_with_real_data(self, nrn):
        """Test algorithms with real neuronal data if available."""
        if nrn is None:
            pytest.skip("Real neuronal data not available")

        if nrn.skeleton is None:
            pytest.skip("Real neuronal data has no skeleton")

        # Test Strahler calculation on real data
        strahler_nums = algorithms.strahler_number(nrn)

        assert len(strahler_nums) == nrn.skeleton.n_vertices
        assert all(num >= 1 for num in strahler_nums)  # All should be at least 1

        # Test smoothing on real data
        # Create a simple binary label for testing
        n_vertices = nrn.skeleton.n_vertices
        binary_label = np.zeros(n_vertices)
        binary_label[: min(10, n_vertices)] = 1.0  # Label first 10 vertices

        smoothed = algorithms.smooth_labels(nrn.skeleton, binary_label)

        assert len(smoothed) == n_vertices
        assert all(0 <= val <= 1 for val in smoothed)
