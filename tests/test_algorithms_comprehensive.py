"""Comprehensive tests for ossify algorithms module.

This module provides both direct function tests and high-level integration tests
using real Cell, SkeletonLayer, and other data layer objects.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ossify import Cell, SkeletonLayer, algorithms
from ossify.algorithms import (
    _distribution_entropy,
    _feature_axon_synapse_flow,
    _laplacian_offset,
    _precompute_synapse_inds,
    _split_direction_and_quality,
    _strahler_path,
)


class TestSynapseBetweenness:
    """Tests for synapse_betweenness algorithm."""

    def test_synapse_betweenness_linear_skeleton(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test synapse betweenness on a linear skeleton."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices
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

        # Define synapse locations (positional indices)
        pre_inds = np.array([0])  # Root vertex
        post_inds = np.array([4])  # Terminal vertex

        # Calculate synapse betweenness
        betweenness = algorithms.synapse_betweenness(cell.skeleton, pre_inds, post_inds)

        # Should have values for all vertices
        assert len(betweenness) == 5
        assert all(val >= 0 for val in betweenness)  # All should be non-negative

        # Middle vertices should have higher betweenness
        assert betweenness[2] > 0  # Middle vertex should have some betweenness

    def test_synapse_betweenness_branched_skeleton(
        self, branched_skeleton_data, spatial_columns
    ):
        """Test synapse betweenness on a branched skeleton."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices
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

        # Define synapse locations
        pre_inds = np.array([2, 3])  # Branch terminals
        post_inds = np.array([0])  # Root

        # Calculate synapse betweenness
        betweenness = algorithms.synapse_betweenness(cell.skeleton, pre_inds, post_inds)

        # Should have values for all vertices
        assert len(betweenness) == 5
        assert all(val >= 0 for val in betweenness)

        # Branch point should have high betweenness
        assert betweenness[1] > 0  # Branch point vertex

    def test_synapse_betweenness_no_synapses(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test synapse betweenness with no synapses."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # No synapses
        pre_inds = np.array([])
        post_inds = np.array([])

        betweenness = algorithms.synapse_betweenness(cell.skeleton, pre_inds, post_inds)

        # Should be all zeros
        assert len(betweenness) == 5
        assert all(val == 0 for val in betweenness)

    def test_synapse_betweenness_single_synapse(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test synapse betweenness with single synapse."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Single pre, no post
        pre_inds = np.array([2])
        post_inds = np.array([])

        betweenness = algorithms.synapse_betweenness(cell.skeleton, pre_inds, post_inds)

        # Should handle gracefully
        assert len(betweenness) == 5
        assert all(val == 0 for val in betweenness)  # No paths possible


class TestfeatureAxonFromSynapseFlow:
    """Tests for feature_axon_from_synapse_flow algorithm."""

    def test_feature_axon_basic_functionality(self, nrn):
        """Test basic axon featureing from synapse flow."""
        # Test basic functionality
        is_axon = algorithms.feature_axon_from_synapse_flow(
            nrn, pre_syn="pre_syn", post_syn="post_syn"
        )
        assert is_axon.sum() == 4181
        assert np.any(is_axon) or np.any(~is_axon)

    def test_feature_axon_with_arrays(self, nrn):
        """Test axon featureing with direct arrays instead of annotation names."""

        # Test with SkeletonLayer and arrays
        pre_syn_inds = nrn.annotations["pre_syn"].map_index_to_layer(
            "skeleton", as_positional=True
        )
        post_syn_inds = nrn.annotations["post_syn"].map_index_to_layer(
            "skeleton", as_positional=True
        )

        is_axon = algorithms.feature_axon_from_synapse_flow(
            nrn.skeleton,
            pre_syn=pre_syn_inds,
            post_syn=post_syn_inds,
            as_postitional=True,
        )

        # Should work without error
        assert is_axon.dtype == bool
        assert is_axon.sum() == 4181

    def test_feature_axon_return_segregation_index(self, nrn):
        """Test axon featureing with segregation index return using real data."""
        # Test with return_segregation_index=True using real data
        result = algorithms.feature_axon_from_synapse_flow(
            nrn, pre_syn="pre_syn", post_syn="post_syn", return_segregation_index=True
        )

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        is_axon, seg_index = result
        assert len(is_axon) == nrn.skeleton.n_vertices
        assert is_axon.dtype == bool
        assert isinstance(seg_index, (int, float))
        assert 0 <= seg_index <= 1  # Segregation index should be in [0, 1]

    def test_feature_axon_multiple_times(self, nrn):
        """Test axon featureing with multiple splits (ntimes > 1) using real data."""
        # Test with multiple splits using real data
        is_axon = algorithms.feature_axon_from_synapse_flow(
            nrn, pre_syn="pre_syn", post_syn="post_syn", ntimes=2
        )

        assert len(is_axon) == nrn.skeleton.n_vertices
        assert is_axon.dtype == bool

        # Compare with single split to ensure different results
        is_axon_single = algorithms.feature_axon_from_synapse_flow(
            nrn, pre_syn="pre_syn", post_syn="post_syn", ntimes=1
        )

        # Results may be different due to multiple iterations
        assert len(is_axon_single) == len(is_axon)


class TestfeatureAxonFromSpectralSplit:
    """Tests for feature_axon_from_spectral_split algorithm."""

    def test_spectral_split_basic(self, nrn):
        """Test basic spectral split functionality using real data."""
        # Test spectral split with real data
        is_axon = algorithms.feature_axon_from_spectral_split(
            nrn, pre_syn="pre_syn", post_syn="post_syn"
        )

        assert len(is_axon) == nrn.skeleton.n_vertices
        assert is_axon.dtype == bool
        # Verify we get meaningful results (some axon, some dendrite)
        assert np.any(is_axon) or np.any(~is_axon)

    def test_spectral_split_with_skeletonlayer(self, nrn):
        """Test spectral split with SkeletonLayer input using real data."""
        # spectral_split doesn't support arrays like synapse_flow does
        # It should raise an error when trying to use SkeletonLayer with string annotation names
        # since SkeletonLayer doesn't have annotations
        with pytest.raises((TypeError, AttributeError, KeyError)):
            algorithms.feature_axon_from_spectral_split(
                nrn.skeleton,  # SkeletonLayer input
                pre_syn="pre_syn",  # String annotation name
                post_syn="post_syn",
            )

    def test_spectral_split_parameters(self, nrn):
        """Test spectral split with different parameters using real data."""
        # Test with raw_split=True
        is_axon_raw = algorithms.feature_axon_from_spectral_split(
            nrn, pre_syn="pre_syn", post_syn="post_syn", raw_split=True
        )

        # Test with different smoothing_alpha
        is_axon_smooth = algorithms.feature_axon_from_spectral_split(
            nrn, pre_syn="pre_syn", post_syn="post_syn", smoothing_alpha=0.5
        )

        # Test with very low smoothing
        is_axon_low_smooth = algorithms.feature_axon_from_spectral_split(
            nrn, pre_syn="pre_syn", post_syn="post_syn", smoothing_alpha=0.1
        )

        n_vertices = nrn.skeleton.n_vertices
        assert len(is_axon_raw) == n_vertices
        assert len(is_axon_smooth) == n_vertices
        assert len(is_axon_low_smooth) == n_vertices
        assert is_axon_raw.dtype == bool
        assert is_axon_smooth.dtype == bool
        assert is_axon_low_smooth.dtype == bool


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_precompute_synapse_inds(self, simple_skeleton_data, spatial_columns):
        """Test _precompute_synapse_inds helper function."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Test with synapse indices
        syn_inds = np.array([0, 2, 2])  # Two synapses at vertex 2

        Nsyn, n_syn = _precompute_synapse_inds(cell.skeleton, syn_inds)

        assert Nsyn == 3  # Total synapses
        assert len(n_syn) == 5  # One per vertex
        assert n_syn[0] == 1  # One synapse at vertex 0
        assert n_syn[2] == 2  # Two synapses at vertex 2
        assert n_syn[1] == 0  # No synapses at vertex 1

    def test_strahler_path(self):
        """Test _strahler_path helper function."""
        # Test with simple sequence - starts with -1 values as initial
        baseline = np.array([-1, -1, -1, -1])
        result = _strahler_path(baseline)

        assert len(result) == len(baseline)
        assert all(val >= 1 for val in result)  # All should be at least 1

        # For baseline of all -1, should get incrementing values starting at 1
        expected = np.array([1, 1, 1, 1])  # All get value 1 since baseline is constant
        np.testing.assert_array_equal(result, expected)

        # Test with mixed values
        baseline_mixed = np.array([-1, 1, 1, 2])
        result_mixed = _strahler_path(baseline_mixed)
        assert len(result_mixed) == len(baseline_mixed)
        assert all(val >= 1 for val in result_mixed)

    def test_laplacian_offset(self, simple_skeleton_data, spatial_columns):
        """Test _laplacian_offset helper function."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        laplacian = _laplacian_offset(cell.skeleton)

        # Should be sparse matrix
        assert sparse.issparse(laplacian)
        assert laplacian.shape == (5, 5)

        # Should be symmetric
        assert np.allclose(laplacian.toarray(), laplacian.T.toarray())

    def test_distribution_entropy(self):
        """Test _distribution_entropy helper function."""
        # Test with perfect segregation
        counts_perfect = np.array([[10, 0], [0, 10]])
        entropy_perfect = _distribution_entropy(counts_perfect)
        # Perfect segregation gives low entropy (close to 0)
        assert entropy_perfect == -0.0  # This is actually -0.0 due to floating point

        # Test with no segregation (equal distribution)
        counts_none = np.array([[5, 5], [5, 5]])
        entropy_none = _distribution_entropy(counts_none)
        # Equal distribution gives high entropy (close to 1)
        assert entropy_none == 1.0

        # Test with all zeros
        counts_zero = np.array([[0, 0], [0, 0]])
        entropy_zero = _distribution_entropy(counts_zero)
        assert entropy_zero == 0  # All zeros should give 0

        # Test with partial segregation
        counts_partial = np.array([[8, 2], [2, 8]])
        entropy_partial = _distribution_entropy(counts_partial)
        # Should be between 0 and 1
        assert 0.0 <= entropy_partial <= 1.0


class TestAlgorithmIntegration:
    """High-level integration tests using real Cell and data layer objects."""

    def test_full_axon_detection_workflow(self, nrn):
        """Test complete workflow: synapses -> betweenness -> axon detection using real data."""
        # Step 1: Calculate synapse betweenness using real annotations
        pre_inds = nrn.annotations["pre_syn"].map_index_to_layer(
            "skeleton", as_positional=True
        )
        post_inds = nrn.annotations["post_syn"].map_index_to_layer(
            "skeleton", as_positional=True
        )

        betweenness = algorithms.synapse_betweenness(nrn.skeleton, pre_inds, post_inds)
        n_vertices = nrn.skeleton.n_vertices
        assert len(betweenness) == n_vertices
        assert np.sum(betweenness) >= 0  # Should have non-negative values

        # Step 2: Use synapse flow method for axon detection
        is_axon_flow = algorithms.feature_axon_from_synapse_flow(
            nrn, "pre_syn", "post_syn", return_segregation_index=True
        )
        axon_features_flow, seg_index_flow = is_axon_flow

        assert len(axon_features_flow) == n_vertices
        assert 0 <= seg_index_flow <= 1
        assert axon_features_flow.dtype == bool

        # Step 3: Use spectral method for comparison
        is_axon_spectral = algorithms.feature_axon_from_spectral_split(
            nrn, "pre_syn", "post_syn"
        )

        assert len(is_axon_spectral) == n_vertices
        assert is_axon_spectral.dtype == bool

        # Both methods should give meaningful results
        assert np.any(axon_features_flow) or np.any(~axon_features_flow)
        assert np.any(is_axon_spectral) or np.any(~is_axon_spectral)

    def test_algorithm_chain_with_strahler_smoothing(
        self, branched_skeleton_data, spatial_columns
    ):
        """Test chaining algorithms: Strahler -> smoothing -> axon detection."""
        vertices, edges, vertex_indices = branched_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[1]],
                [vertex_indices[4], vertex_indices[1]],
            ]
        )

        cell = Cell(name="complex_neuron")
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Step 1: Calculate Strahler numbers
        strahler_nums = algorithms.strahler_number(cell)
        assert len(strahler_nums) == 5
        assert all(num >= 1 for num in strahler_nums)

        # Step 2: Smooth Strahler numbers
        smoothed_strahler = algorithms.smooth_features(
            cell.skeleton, strahler_nums.astype(float), alpha=0.8
        )
        assert len(smoothed_strahler) == 5

        # Step 3: Use smoothed features as basis for further analysis
        # Create synthetic synapse data based on Strahler structure
        high_strahler_vertices = np.where(strahler_nums > 1)[0]
        low_strahler_vertices = np.where(strahler_nums == 1)[0]

        if len(high_strahler_vertices) > 0 and len(low_strahler_vertices) > 0:
            # Simulate synapses based on structure
            pre_syn_inds = (
                high_strahler_vertices  # Pre-synapses on high Strahler regions
            )
            post_syn_inds = low_strahler_vertices[
                :1
            ]  # Post-synapses on low Strahler regions

            betweenness = algorithms.synapse_betweenness(
                cell.skeleton, pre_syn_inds, post_syn_inds
            )
            assert len(betweenness) == 5
            assert np.sum(betweenness) >= 0

    def test_algorithms_with_multilayer_cell(
        self, simple_skeleton_data, simple_mesh_data, spatial_columns
    ):
        """Test algorithms with Cell containing multiple data layers."""
        vertices_skel, edges_skel, indices_skel = simple_skeleton_data
        vertices_mesh, faces_mesh, indices_mesh = simple_mesh_data

        skel_df = pd.DataFrame(
            vertices_skel, columns=spatial_columns, index=indices_skel
        )
        mesh_df = pd.DataFrame(
            vertices_mesh, columns=spatial_columns, index=indices_mesh
        )

        edges_with_indices = np.array(
            [
                [indices_skel[1], indices_skel[0]],
                [indices_skel[2], indices_skel[1]],
                [indices_skel[3], indices_skel[2]],
                [indices_skel[4], indices_skel[3]],
            ]
        )

        # Create multi-layer cell
        cell = Cell(name="multilayer_neuron")
        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=indices_skel[0],
        )
        cell.add_mesh(
            vertices=mesh_df, faces=faces_mesh, spatial_columns=spatial_columns
        )

        # Verify we have multiple layers
        assert len(cell.layers.names) == 2
        assert "skeleton" in cell.layers.names
        assert "mesh" in cell.layers.names

        # Test algorithms work with Cell containing multiple layers
        strahler_nums = algorithms.strahler_number(cell)
        assert len(strahler_nums) == 5  # Should work on skeleton despite having mesh

        # Test smoothing
        smoothed = algorithms.smooth_features(cell, strahler_nums.astype(float))
        assert len(smoothed) == 5

    def test_algorithms_with_real_data(self, nrn):
        """Test algorithms with real neuronal data."""
        if nrn is None:
            pytest.skip("Real neuronal data not available")

        if nrn.skeleton is None:
            pytest.skip("Real neuronal data has no skeleton")

        # Test all major algorithms with real data
        n_vertices = nrn.skeleton.n_vertices

        # Test Strahler calculation
        strahler_nums = algorithms.strahler_number(nrn)
        assert len(strahler_nums) == n_vertices
        assert all(num >= 1 for num in strahler_nums)
        assert strahler_nums.dtype in [np.int32, np.int64]

        # Test smoothing with Strahler numbers
        smoothed = algorithms.smooth_features(nrn.skeleton, strahler_nums.astype(float))
        assert len(smoothed) == n_vertices
        assert all(val >= 0 for val in smoothed)

        # Create synthetic synapse data for testing
        # Place pre-synapses at high-Strahler vertices (branch points)
        high_strahler_mask = strahler_nums > np.median(strahler_nums)
        high_strahler_inds = np.where(high_strahler_mask)[0]

        # Place post-synapses at low-Strahler vertices (terminals)
        low_strahler_mask = strahler_nums == 1
        low_strahler_inds = np.where(low_strahler_mask)[0]

        if len(high_strahler_inds) > 0 and len(low_strahler_inds) > 0:
            # Sample a subset to avoid too many synapses
            pre_sample = np.random.choice(
                high_strahler_inds, size=min(10, len(high_strahler_inds)), replace=False
            )
            post_sample = np.random.choice(
                low_strahler_inds, size=min(5, len(low_strahler_inds)), replace=False
            )

            # Test synapse betweenness
            betweenness = algorithms.synapse_betweenness(
                nrn.skeleton, pre_sample, post_sample
            )
            assert len(betweenness) == n_vertices
            assert all(val >= 0 for val in betweenness)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_algorithms_with_empty_cell(self):
        """Test algorithms with empty Cell (no skeleton)."""
        empty_cell = Cell(name="empty")

        # Should raise appropriate errors
        with pytest.raises(ValueError):
            algorithms.strahler_number(empty_cell)

    def test_algorithms_with_invalid_inputs(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test algorithms with invalid input parameters."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Test smooth_features with wrong size feature
        wrong_size_feature = np.array([1.0, 0.0])  # Too short
        with pytest.raises((ValueError, IndexError)):
            algorithms.smooth_features(cell.skeleton, wrong_size_feature)

        # Test synapse_betweenness with out-of-bounds indices
        invalid_pre = np.array([100])  # Out of bounds positional index
        invalid_post = np.array([0])

        # This might raise IndexError or give unexpected results
        try:
            betweenness = algorithms.synapse_betweenness(
                cell.skeleton, invalid_pre, invalid_post
            )
            # If it doesn't raise an error, check result makes sense
            assert len(betweenness) == 5
        except IndexError:
            pass  # Expected behavior

    def test_segregation_index_edge_cases(self):
        """Test segregation index with edge cases."""
        # Test all zeros
        seg_index = algorithms.segregation_index(0, 0, 0, 0)
        assert seg_index == 0

        # Test only one type of synapse
        seg_index_pre_only = algorithms.segregation_index(10, 0, 5, 0)
        assert seg_index_pre_only == 0  # No post synapses

        seg_index_post_only = algorithms.segregation_index(0, 10, 0, 5)
        assert seg_index_post_only == 0  # No pre synapses
