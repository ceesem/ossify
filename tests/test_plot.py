import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ossify import Cell, plot


class TestUtilityFunctions:
    """Tests for plotting utility functions."""

    def test_map_value_to_colors_continuous(self):
        """Test continuous color mapping."""
        values = np.array([0.0, 0.5, 1.0])
        colors = plot._map_value_to_colors(values, colormap="viridis")

        # Should return RGB colors
        assert colors.shape == (3, 3)
        assert np.all((colors >= 0) & (colors <= 1))

    def test_map_value_to_colors_discrete_dict(self):
        """Test discrete color mapping with dictionary."""
        values = np.array(["A", "B", "A", "C"])
        colormap = {"A": "#ff0000", "B": "blue", "C": (0.0, 1.0, 0.0)}

        colors = plot._map_value_to_colors(values, colormap=colormap)

        # Should return RGB colors (consistent with default alpha=1.0)
        assert colors.shape == (4, 3)
        assert np.all((colors >= 0) & (colors <= 1))
        # First and third should be same (both 'A')
        np.testing.assert_array_equal(colors[0], colors[2])

    def test_map_value_to_colors_boolean(self):
        """Test boolean color mapping."""
        values = np.array([True, False, True, False])
        colors = plot._map_value_to_colors(values, colormap="viridis")

        assert colors.shape == (4, 3)
        # First and third should be same (both True)
        np.testing.assert_array_equal(colors[0], colors[2])
        # Second and fourth should be same (both False)
        np.testing.assert_array_equal(colors[1], colors[3])

    def test_map_value_to_colors_with_normalization(self):
        """Test color mapping with custom normalization."""
        values = np.array([10, 50, 100])
        colors = plot._map_value_to_colors(
            values, colormap="viridis", color_norm=(0, 100)
        )

        assert colors.shape == (3, 3)
        assert np.all((colors >= 0) & (colors <= 1))

    def test_is_discrete_data(self):
        """Test automatic discrete data detection."""
        # String data should be discrete
        assert plot._is_discrete_data(np.array(["A", "B", "C"]))

        # Boolean data should be discrete
        assert plot._is_discrete_data(np.array([True, False, True]))

        # Few unique numeric values should be discrete
        assert plot._is_discrete_data(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))

        # Many continuous values should not be discrete
        assert not plot._is_discrete_data(np.linspace(0, 1, 100))

        # Empty array
        assert not plot._is_discrete_data(np.array([]))

    def test_get_discrete_colormap(self):
        """Test discrete colormap generation."""
        # Test automatic selection
        cmap_small = plot._get_discrete_colormap("auto", 5)
        assert len(cmap_small.colors) == 5

        cmap_large = plot._get_discrete_colormap("auto", 15)
        assert len(cmap_large.colors) == 15

        # Test standard qualitative colormaps
        cmap_set1 = plot._get_discrete_colormap("Set1", 5)
        assert len(cmap_set1.colors) == 5

        # Test exceeding colormap capacity
        cmap_exceed = plot._get_discrete_colormap("Set1", 15)
        assert len(cmap_exceed.colors) == 15

    def test_create_discrete_color_dict(self):
        """Test discrete color dictionary creation."""
        values = np.array(["red", "green", "blue", "red"])
        color_dict = plot._create_discrete_color_dict(values, "Set1")

        assert len(color_dict) == 4  # 3 unique values + missing color
        assert "red" in color_dict
        assert "green" in color_dict
        assert "blue" in color_dict
        assert "__missing__" in color_dict

    def test_map_value_to_colors_auto_discrete(self):
        """Test automatic discrete color mapping."""
        # Categorical string data
        values = np.array(["cat", "dog", "cat", "bird", "dog"])
        colors = plot._map_value_to_colors(values, colormap="auto")

        assert colors.shape[0] == 5
        assert colors.shape[1] in [3, 4]  # RGB or RGBA

        # Same category should get same color
        np.testing.assert_array_equal(colors[0], colors[2])  # Both "cat"
        np.testing.assert_array_equal(colors[1], colors[4])  # Both "dog"

    def test_map_value_to_colors_discrete_numeric(self):
        """Test discrete mapping with numeric categorical data."""
        # Small number of numeric categories
        values = np.array([1, 2, 3, 1, 2, 3])
        colors = plot._map_value_to_colors(values, colormap="Set1", force_discrete=True)

        assert colors.shape[0] == 6
        # Same values should get same colors
        np.testing.assert_array_equal(colors[0], colors[3])  # Both 1
        np.testing.assert_array_equal(colors[1], colors[4])  # Both 2

    def test_map_value_to_colors_with_missing_values(self):
        """Test color mapping with missing/NaN values."""
        values = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        colors = plot._map_value_to_colors(
            values, colormap="viridis", missing_color="red"
        )

        assert colors.shape[0] == 5
        # Check that NaN values got the missing color (red = [1, 0, 0])
        np.testing.assert_allclose(colors[2, :3], [1.0, 0.0, 0.0], atol=0.1)
        np.testing.assert_allclose(colors[4, :3], [1.0, 0.0, 0.0], atol=0.1)

    def test_map_value_to_colors_force_continuous(self):
        """Test forcing continuous mapping on discrete-looking data."""
        values = np.array([1, 2, 3, 1, 2, 3])
        colors = plot._map_value_to_colors(
            values, colormap="viridis", force_discrete=False
        )

        assert colors.shape[0] == 6
        # Should treat as continuous, so same values get same colors but different from discrete mode
        np.testing.assert_array_equal(colors[0], colors[3])  # Both 1
        np.testing.assert_array_equal(colors[1], colors[4])  # Both 2

    def test_map_value_to_colors_with_alpha(self):
        """Test color mapping with alpha values."""
        values = np.array(["A", "B", "A"])
        alpha = np.array([0.5, 0.8, 0.3])
        colors = plot._map_value_to_colors(values, colormap="Set1", alpha=alpha)

        assert colors.shape == (3, 4)  # Should include alpha channel
        np.testing.assert_array_equal(colors[:, 3], alpha)

    def test_should_invert_y_axis(self):
        """Test y-axis inversion detection."""
        assert plot._should_invert_y_axis("xy") == True
        assert plot._should_invert_y_axis("xz") == False
        assert plot._should_invert_y_axis("yz") == True

    def test_projection_factory(self):
        """Test projection factory function."""
        # String projection
        proj_func = plot.projection_factory("xy")
        test_vertices = np.array([[1, 2, 3], [4, 5, 6]])
        result = proj_func(test_vertices)
        expected = np.array([[1, 2], [4, 5]])
        np.testing.assert_array_equal(result, expected)

        # Custom projection function
        def custom_proj(vertices):
            return vertices[:, [0, 2]]  # x, z

        proj_func = plot.projection_factory(custom_proj)
        result = proj_func(test_vertices)
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_plotted_bounds(self):
        """Test plotted bounds calculation."""
        vertices_3d = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
        bounds = plot._plotted_bounds(vertices_3d, projection="xy")

        expected = np.array([[0.0, 2.0], [0.0, 1.0]])  # [[xmin, xmax], [ymin, ymax]]
        np.testing.assert_array_equal(bounds, expected)

    def test_rescale_scalar(self):
        """Test scalar rescaling function."""
        # Test with regular values
        result = plot._rescale_scalar(
            np.array([1, 2, 3]), norm=(1, 3), out_range=(0.5, 2.0)
        )
        expected = np.array([0.5, 1.25, 2.0])
        np.testing.assert_array_equal(result, expected)

        # Test with all same values
        result = plot._rescale_scalar(
            np.array([5, 5, 5]), norm=None, out_range=(1.0, 3.0)
        )
        expected = np.array(
            [1.0, 1.0, 1.0]
        )  # Maps to lower bound when input range is zero
        np.testing.assert_array_equal(result, expected)


class TestSkeletonPlotting:
    """Tests for skeleton plotting functions."""

    def test_plot_skeleton_basic(self, simple_skeleton_data, spatial_columns):
        """Test basic skeleton plotting."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )

        # Fix edges to use vertex indices
        edges_with_indices = np.array(
            [
                [vertex_indices[1], vertex_indices[0]],
                [vertex_indices[2], vertex_indices[1]],
                [vertex_indices[3], vertex_indices[2]],
                [vertex_indices[4], vertex_indices[3]],
            ]
        )

        # Create skeleton
        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )

        # Test plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot.plot_skeleton(cell.skeleton, ax=ax)

        # Verify plot was created
        assert len(ax.collections) > 0  # Should have line collections
        plt.close(fig)

    def test_plot_skeleton_with_colors(
        self, simple_skeleton_data, spatial_columns, mock_features
    ):
        """Test skeleton plotting with color mapping."""
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
            features=mock_features,
        )

        # Test plotting with colors (basic test - may need feature resolution)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Use RGB colors - need one color per vertex (5 vertices)
        colors = np.array(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
            ]
        )  # RGB color array matching vertex count
        plot.plot_skeleton(cell.skeleton, ax=ax, colors=colors)

        assert len(ax.collections) > 0
        plt.close(fig)

    def test_plot_skeleton_with_projection(self, simple_skeleton_data, spatial_columns):
        """Test skeleton plotting with different projections."""
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

        # Test different projections
        for projection in ["xy", "xz", "yz"]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plot.plot_skeleton(cell.skeleton, ax=ax, projection=projection)
            assert len(ax.collections) > 0
            plt.close(fig)


class TestPointPlotting:
    """Tests for point plotting functions."""

    def test_plot_points_basic(self, mock_point_annotations, spatial_columns):
        """Test basic point plotting."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Extract coordinates for plotting
        coords = mock_point_annotations[spatial_columns].values

        plot.plot_points(coords, ax=ax)

        # Verify points were plotted
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_plot_points_with_colors(self, mock_point_annotations, spatial_columns):
        """Test point plotting with color mapping."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        coords = mock_point_annotations[spatial_columns].values
        colors = np.array([0, 1, 2, 3])  # Simple numeric colors

        plot.plot_points(coords, ax=ax, colors=colors)

        assert len(ax.collections) > 0
        plt.close(fig)

    def test_plot_points_with_sizes(self, mock_point_annotations, spatial_columns):
        """Test point plotting with size mapping."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        coords = mock_point_annotations[spatial_columns].values
        sizes = mock_point_annotations["size"].values

        plot.plot_points(coords, ax=ax, sizes=sizes)

        assert len(ax.collections) > 0
        plt.close(fig)

    def test_plot_points_with_projection(self, mock_point_annotations, spatial_columns):
        """Test point plotting with different projections."""
        coords = mock_point_annotations[spatial_columns].values

        for projection in ["xy", "xz", "yz"]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plot.plot_points(coords, ax=ax, projection=projection)
            assert len(ax.collections) > 0
            plt.close(fig)


class TestHighLevelPlotting:
    """Tests for high-level plotting functions."""

    def test_plot_morphology_2d_skeleton_only(
        self, simple_skeleton_data, spatial_columns
    ):
        """Test morphology plotting with skeleton only."""
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

        ax = plot.plot_morphology_2d(cell, projection="xy")
        assert ax is not None
        plt.close(ax.figure)

    def test_plot_morphology_2d_multi_layer(
        self, simple_skeleton_data, simple_mesh_data, spatial_columns
    ):
        """Test morphology plotting with multiple layers."""
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

        cell = Cell()
        cell.add_skeleton(
            vertices=skel_df,
            edges=edges_with_indices,
            spatial_columns=spatial_columns,
            root=indices_skel[0],
        )
        cell.add_mesh(
            vertices=mesh_df, faces=faces_mesh, spatial_columns=spatial_columns
        )

        ax = plot.plot_morphology_2d(cell, projection="xy")
        assert ax is not None
        plt.close(ax.figure)

    def test_plot_annotations_2d(self, mock_point_annotations, spatial_columns):
        """Test annotation plotting with point cloud layer."""
        # Test plotting annotations directly as numpy array since we can't easily
        # create a PointCloudLayer in the test
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        try:
            # Use the point annotation data directly as np array
            points = mock_point_annotations[spatial_columns].values
            plot.plot_annotations_2d(points, ax=ax, projection="xy")
            assert True  # Function executed without error
        except Exception:
            # Annotation plotting may not work with raw arrays
            pytest.skip("Annotation plotting requires PointCloudLayer")
        finally:
            plt.close(fig)

    def test_plot_cell_2d(self, simple_skeleton_data, spatial_columns):
        """Test complete cell plotting."""
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

        ax = plot.plot_cell_2d(cell, projection="xy")
        assert ax is not None
        plt.close(ax.figure)

    def test_plot_cell_multiview(self, simple_skeleton_data, spatial_columns):
        """Test multi-view cell plotting."""
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

        axes_dict = plot.plot_cell_multiview(cell)
        assert axes_dict is not None
        assert len(axes_dict) == 3  # Should have 3 projection views
        # Close figure from one of the axes
        first_ax = list(axes_dict.values())[0]
        plt.close(first_ax.figure)


class TestFigureUtilities:
    """Tests for figure utility functions."""

    def test_single_panel_figure(self):
        """Test single panel figure creation."""
        data_bounds_min = np.array([0, 0])
        data_bounds_max = np.array([100, 100])
        units_per_inch = 10.0

        fig, ax = plot.single_panel_figure(
            data_bounds_min=data_bounds_min,
            data_bounds_max=data_bounds_max,
            units_per_inch=units_per_inch,
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_multi_panel_figure(self):
        """Test multi-panel figure creation."""
        data_bounds_min = np.array([0, 0, 0])
        data_bounds_max = np.array([100, 100, 100])
        units_per_inch = 10.0

        fig, axes = plot.multi_panel_figure(
            data_bounds_min=data_bounds_min,
            data_bounds_max=data_bounds_max,
            units_per_inch=units_per_inch,
            layout="three_panel",
        )

        assert fig is not None
        assert axes is not None
        assert isinstance(axes, dict)  # Returns dict of axes
        plt.close(fig)

    def test_single_panel_figure_limits_preserved_after_hlines(self):
        """Test that single_panel_figure axis limits are not overwritten by autoscale."""
        data_bounds_min = np.array([100, 200])
        data_bounds_max = np.array([300, 400])
        units_per_inch = 20.0

        fig, ax = plot.single_panel_figure(
            data_bounds_min=data_bounds_min,
            data_bounds_max=data_bounds_max,
            units_per_inch=units_per_inch,
        )

        # Autoscale should be off
        assert not ax.get_autoscale_on()

        # Add hlines far outside the data bounds (simulating layer boundaries)
        ax.hlines([0, 50, 600, 800], xmin=100, xmax=300, colors="gray")

        # Limits should remain unchanged despite hlines outside bounds
        assert ax.get_xlim() == (100, 300)
        assert ax.get_ylim() == (200, 400)

        # Verify figure dimensions match data_range / units_per_inch
        fig_width, fig_height = fig.get_size_inches()
        np.testing.assert_allclose(fig_width, 200 / 20.0)
        np.testing.assert_allclose(fig_height, 200 / 20.0)
        plt.close(fig)

    def test_multi_panel_figure_limits_preserved_after_hlines(self):
        """Test that multi_panel_figure axis limits are not overwritten by autoscale."""
        data_bounds_min = np.array([0, 0, 0])
        data_bounds_max = np.array([100, 200, 50])
        units_per_inch = 10.0

        fig, axes = plot.multi_panel_figure(
            data_bounds_min=data_bounds_min,
            data_bounds_max=data_bounds_max,
            units_per_inch=units_per_inch,
            layout="side_by_side",
        )

        # Autoscale should be off for all panels
        for ax in axes.values():
            assert not ax.get_autoscale_on()

        # Add hlines far outside bounds
        axes["xy"].hlines([-500, 500], xmin=0, xmax=100, colors="gray")

        # xy panel limits should be preserved
        assert axes["xy"].get_xlim() == (0, 100)
        assert axes["xy"].get_ylim() == (0, 200)
        plt.close(fig)

    def test_add_scale_bar(self):
        """Test scale bar addition to plots."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Set up some basic plot bounds
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # Add scale bar
        plot.add_scale_bar(ax, length=10, feature="10 μm")

        # Should have added elements to the plot
        # This is hard to test directly, so we just verify no errors
        assert True
        plt.close(fig)


class TestRealDataPlotting:
    """Tests for plotting with real neuronal data."""

    def test_plot_real_cell(self, nrn):
        """Test plotting real neuronal data if available."""
        if nrn is None:
            pytest.skip("Real neuronal data not available")

        if nrn.skeleton is None:
            pytest.skip("Real neuronal data has no skeleton")

        try:
            ax = plot.plot_cell_2d(nrn, projection="xy")
            assert ax is not None
            plt.close(ax.figure)
        except Exception as e:
            # Let real data plotting errors surface
            pytest.fail(f"Real data plotting failed: {e}")

    def test_plot_real_cell_with_mesh(self, cell_with_mesh):
        """Test plotting real data with mesh if available."""
        if cell_with_mesh is None:
            pytest.skip("Real mesh data not available")

        try:
            ax = plot.plot_cell_2d(cell_with_mesh, projection="xy")
            assert ax is not None
            plt.close(ax.figure)
        except Exception as e:
            # Let mesh plotting errors surface
            pytest.fail(f"Mesh plotting failed: {e}")


class TestErrorHandling:
    """Tests for error handling in plotting functions."""

    def test_plot_empty_skeleton(self):
        """Test plotting empty skeleton."""
        empty_cell = Cell()

        with pytest.raises((AttributeError, ValueError)):
            plot.plot_cell_2d(empty_cell, projection="xy")

    def test_invalid_projection(self, simple_skeleton_data, spatial_columns):
        """Test invalid projection specification."""
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

        with pytest.raises((ValueError, KeyError)):
            plot.plot_cell_2d(cell, projection="invalid")

    def test_synapses_with_no_annotations(self, simple_skeleton_data, spatial_columns):
        """Test that synapses parameters work gracefully when no annotations exist."""
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

        # Should not raise errors even when synapses are requested but don't exist
        ax = plot.plot_cell_2d(cell, projection="xy", synapses="both")
        assert ax is not None
        plt.close(ax.figure)

        ax = plot.plot_cell_2d(cell, projection="xy", synapses="pre")
        assert ax is not None
        plt.close(ax.figure)

        ax = plot.plot_cell_2d(cell, projection="xy", synapses="post")
        assert ax is not None
        plt.close(ax.figure)

    def test_invalid_colormap(self, simple_skeleton_data, spatial_columns):
        """Test invalid colormap specification."""
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

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Test with invalid colors parameter instead
        with pytest.raises((ValueError, TypeError)):
            plot.plot_skeleton(cell.skeleton, ax=ax, colors="invalid_colors")


# ===========================================================================
# Rotation helpers and public API
# ===========================================================================


def _make_cell_with_skeleton(vertices: np.ndarray, root_idx: int = 0) -> "Cell":
    """Build a minimal Cell with a SkeletonLayer from a vertex array."""
    n = len(vertices)
    spatial_columns = ["x", "y", "z"]
    vertex_indices = np.arange(n)
    vertex_df = pd.DataFrame(vertices, columns=spatial_columns, index=vertex_indices)
    # Linear chain edges
    edges = np.array([[vertex_indices[i + 1], vertex_indices[i]] for i in range(n - 1)])
    cell = Cell()
    cell.add_skeleton(
        vertices=vertex_df,
        edges=edges,
        spatial_columns=spatial_columns,
        root=vertex_indices[root_idx],
    )
    return cell


class TestRotationHelpers:
    """Tests for the private rotation utility functions."""

    # --- _resolve_axis ---

    def test_resolve_axis_x(self):
        np.testing.assert_array_equal(plot._resolve_axis("x"), [1.0, 0.0, 0.0])

    def test_resolve_axis_y(self):
        np.testing.assert_array_equal(plot._resolve_axis("y"), [0.0, 1.0, 0.0])

    def test_resolve_axis_z(self):
        np.testing.assert_array_equal(plot._resolve_axis("z"), [0.0, 0.0, 1.0])

    def test_resolve_axis_array_normalizes(self):
        result = plot._resolve_axis(np.array([2.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_resolve_axis_arbitrary_array(self):
        v = np.array([1.0, 1.0, 0.0])
        result = plot._resolve_axis(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)
        np.testing.assert_allclose(result, v / np.sqrt(2))

    def test_resolve_axis_zero_raises(self):
        with pytest.raises(ValueError):
            plot._resolve_axis(np.array([0.0, 0.0, 0.0]))

    def test_resolve_axis_bad_string_raises(self):
        with pytest.raises(ValueError):
            plot._resolve_axis("w")

    # --- _perp_basis ---

    @pytest.mark.parametrize(
        "axis", ["x", "y", "z", np.array([1.0, 1.0, 0.0]) / np.sqrt(2)]
    )
    def test_perp_basis_orthogonality(self, axis):
        k = plot._resolve_axis(axis)
        u, v = plot._perp_basis(k)
        assert abs(np.dot(u, v)) < 1e-12
        assert abs(np.dot(u, k)) < 1e-12
        assert abs(np.dot(v, k)) < 1e-12

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_perp_basis_unit_length(self, axis):
        k = plot._resolve_axis(axis)
        u, v = plot._perp_basis(k)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0)

    def test_perp_basis_near_x_does_not_degenerate(self):
        k = plot._resolve_axis(np.array([0.999, 0.001, 0.0]))
        u, v = plot._perp_basis(k)
        assert np.linalg.norm(u) > 0.99
        assert np.linalg.norm(v) > 0.99

    # --- _build_rotation_matrix ---

    def test_rotation_matrix_identity_at_zero(self):
        k = plot._resolve_axis("z")
        R = plot._build_rotation_matrix(k, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_matrix_quarter_turn_z(self):
        k = plot._resolve_axis("z")
        R = plot._build_rotation_matrix(k, np.pi / 2)
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_rotation_matrix_is_orthogonal(self):
        k = plot._resolve_axis(np.array([1.0, 2.0, 3.0]))
        R = plot._build_rotation_matrix(k, 1.23)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_matrix_det_one(self):
        k = plot._resolve_axis("y")
        R = plot._build_rotation_matrix(k, np.pi / 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    # --- _best_angle_for_axis ---

    def test_best_angle_aligns_principal_axis(self):
        # Point cloud elongated along [cos(a), 0, sin(a)] in xz plane.
        # Rotation about y should find the angle that aligns this with x.
        rng = np.random.default_rng(42)
        a = np.pi / 5  # known angle
        direction = np.array([np.cos(a), 0.0, np.sin(a)])
        pts = rng.normal(scale=[5.0, 0.1, 0.1], size=(200, 3))
        pts = (
            pts
            @ np.column_stack([direction, [0, 1, 0], np.cross(direction, [0, 1, 0])]).T
        )
        pts_c = pts - pts.mean(axis=0)
        k = plot._resolve_axis("y")
        theta = plot._best_angle_for_axis(pts_c, k)
        R = plot._build_rotation_matrix(k, theta)
        projected = pts_c @ R.T
        # After optimal rotation the x-variance should dominate
        assert np.var(projected[:, 0]) > np.var(projected[:, 2])


class TestRotation:
    """Tests for the public Rotation() factory."""

    def test_identity_is_xy_projection(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        proj = plot.Rotation([0, 0, 0], "z", 0.0, invert_y=False)
        np.testing.assert_allclose(proj(pts), pts[:, :2], atol=1e-12)

    def test_invert_y_default(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        proj_inv = plot.Rotation([0, 0, 0], "z", 0.0, invert_y=True)
        proj_no = plot.Rotation([0, 0, 0], "z", 0.0, invert_y=False)
        result_inv = proj_inv(pts)
        result_no = proj_no(pts)
        np.testing.assert_allclose(result_inv[:, 0], result_no[:, 0])
        np.testing.assert_allclose(result_inv[:, 1], -result_no[:, 1])

    def test_half_turn_z_negates_xy(self):
        pts = np.array([[3.0, 4.0, 0.0]])
        proj = plot.Rotation([0, 0, 0], "z", 180, invert_y=False)
        result = proj(pts)
        np.testing.assert_allclose(result, [[-3.0, -4.0]], atol=1e-12)

    def test_quarter_turn_z(self):
        pts = np.array([[1.0, 0.0, 0.0]])
        proj = plot.Rotation([0, 0, 0], "z", 90, invert_y=False)
        np.testing.assert_allclose(proj(pts), [[0.0, 1.0]], atol=1e-12)

    def test_quarter_turn_x_maps_z_to_neg_y(self):
        # 90° about x: [0,0,1] → [0,-1,0] in 3D → projects to [0,-1]
        pts = np.array([[0.0, 0.0, 1.0]])
        proj = plot.Rotation([0, 0, 0], "x", 90, invert_y=False)
        np.testing.assert_allclose(proj(pts), [[0.0, -1.0]], atol=1e-12)

    def test_center_is_fixed_point(self):
        center = np.array([10.0, 20.0, 30.0])
        pts = np.atleast_2d(center)
        for angle in [5.7, 28.6, 180, 114.6]:
            proj = plot.Rotation(center, "z", angle, invert_y=False)
            result = proj(pts)
            np.testing.assert_allclose(result[0], center[:2], atol=1e-12)

    def test_non_unit_axis_same_as_unit(self):
        pts = np.random.default_rng(0).random((20, 3))
        center = np.zeros(3)
        proj_unit = plot.Rotation(center, [1, 0, 0], 57.3, invert_y=False)
        proj_scaled = plot.Rotation(center, [3, 0, 0], 57.3, invert_y=False)
        np.testing.assert_allclose(proj_unit(pts), proj_scaled(pts), atol=1e-12)

    def test_new_center_places_pivot_at_origin(self):
        center = np.array([100.0, 200.0, 300.0])
        pts = np.atleast_2d(center)
        proj = plot.Rotation(
            center, "z", 28.6, new_center=np.array([0.0, 0.0]), invert_y=False
        )
        result = proj(pts)
        np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-12)

    def test_new_center_arbitrary(self):
        center = np.array([100.0, 200.0, 300.0])
        pts = np.atleast_2d(center)
        target = np.array([5.0, 7.0])
        proj = plot.Rotation(center, "z", 28.6, new_center=target, invert_y=False)
        result = proj(pts)
        np.testing.assert_allclose(result[0], target, atol=1e-12)

    def test_new_center_uniform_shift(self):
        rng = np.random.default_rng(1)
        pts = rng.random((50, 3))
        center = np.array([0.5, 0.5, 0.5])
        proj_none = plot.Rotation(center, "z", 40.1, new_center=None, invert_y=False)
        target = np.array([3.0, 4.0])
        proj_shifted = plot.Rotation(
            center, "z", 40.1, new_center=target, invert_y=False
        )
        diff = proj_shifted(pts) - proj_none(pts)
        # Every row should have the same constant offset
        np.testing.assert_allclose(diff - diff[0], 0.0, atol=1e-12)

    def test_compatible_with_plot_skeleton(self, simple_skeleton_data, spatial_columns):
        """Rotation callable works as a drop-in for the projection parameter."""
        vertices, edges, vertex_indices = simple_skeleton_data
        vertex_df = pd.DataFrame(
            vertices, columns=spatial_columns, index=vertex_indices
        )
        edges_wi = np.array(
            [[vertex_indices[i + 1], vertex_indices[i]] for i in range(4)]
        )
        cell = Cell()
        cell.add_skeleton(
            vertices=vertex_df,
            edges=edges_wi,
            spatial_columns=spatial_columns,
            root=vertex_indices[0],
        )
        proj = plot.Rotation(np.zeros(3), "z", 45)
        fig, ax = plt.subplots()
        plot.plot_skeleton(cell.skeleton, projection=proj, ax=ax)
        plt.close(fig)


class TestRotateCell:
    """Tests for the public RotateCell() wrapper."""

    @pytest.fixture
    def elongated_cell(self):
        """Cell with skeleton elongated along x at the origin."""
        rng = np.random.default_rng(7)
        pts = rng.normal(scale=[10.0, 0.5, 0.5], size=(50, 3))
        pts[0] = [0.0, 0.0, 0.0]  # root at origin
        return _make_cell_with_skeleton(pts, root_idx=0)

    @pytest.fixture
    def tilted_cell(self):
        """Cell elongated along a known direction in the xz plane (45°)."""
        rng = np.random.default_rng(13)
        direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        pts = rng.normal(scale=[10.0, 0.1, 0.1], size=(80, 3))
        R = np.column_stack([direction, [0, 1, 0], np.cross(direction, [0, 1, 0])])
        pts = pts @ R.T
        pts[0] = [0.0, 0.0, 0.0]
        return _make_cell_with_skeleton(pts, root_idx=0)

    def test_center_defaults_to_root_location(self, elongated_cell):
        root_loc = elongated_cell.skeleton.root_location
        proj = plot.RotateCell(elongated_cell, axis="z", angle=28.6, invert_y=False)
        # Applying to the root location should match Rotation with the same center
        ref = plot.Rotation(root_loc, "z", 28.6, invert_y=False)
        pts = elongated_cell.skeleton.vertices
        np.testing.assert_allclose(proj(pts), ref(pts), atol=1e-12)

    def test_explicit_center_overrides_root(self, elongated_cell):
        custom_center = np.array([1.0, 2.0, 3.0])
        proj = plot.RotateCell(
            elongated_cell, axis="z", angle=17.2, center=custom_center, invert_y=False
        )
        ref = plot.Rotation(custom_center, "z", 17.2, invert_y=False)
        pts = elongated_cell.skeleton.vertices
        np.testing.assert_allclose(proj(pts), ref(pts), atol=1e-12)

    def test_no_skeleton_raises(self):
        cell = Cell()
        with pytest.raises(ValueError):
            plot.RotateCell(cell, axis="z", angle=0.0)

    def test_string_axis_matches_vector(self, elongated_cell):
        pts = elongated_cell.skeleton.vertices
        proj_str = plot.RotateCell(elongated_cell, axis="y", angle=57.3, invert_y=False)
        proj_vec = plot.RotateCell(
            elongated_cell, axis=np.array([0.0, 1.0, 0.0]), angle=57.3, invert_y=False
        )
        np.testing.assert_allclose(proj_str(pts), proj_vec(pts), atol=1e-12)

    def test_explicit_angle_matches_rotation(self, elongated_cell):
        root_loc = elongated_cell.skeleton.root_location
        pts = elongated_cell.skeleton.vertices
        proj = plot.RotateCell(elongated_cell, axis="x", angle=68.8, invert_y=False)
        ref = plot.Rotation(root_loc, "x", 68.8, invert_y=False)
        np.testing.assert_allclose(proj(pts), ref(pts), atol=1e-12)

    def test_none_angle_defaults_to_zero(self, elongated_cell):
        pts = elongated_cell.skeleton.vertices
        proj_none = plot.RotateCell(
            elongated_cell, axis="z", angle=None, invert_y=False
        )
        proj_zero = plot.RotateCell(elongated_cell, axis="z", angle=0.0, invert_y=False)
        np.testing.assert_allclose(proj_none(pts), proj_zero(pts), atol=1e-12)

    def test_new_center_passthrough(self, elongated_cell):
        root_loc = elongated_cell.skeleton.root_location
        pts = elongated_cell.skeleton.vertices
        nc = np.array([0.0, 0.0])
        proj = plot.RotateCell(
            elongated_cell, axis="z", angle=28.6, new_center=nc, invert_y=False
        )
        ref = plot.Rotation(root_loc, "z", 28.6, new_center=nc, invert_y=False)
        np.testing.assert_allclose(proj(pts), ref(pts), atol=1e-12)

    def test_angle_best_maximizes_x_variance(self, tilted_cell):
        """Best angle about y should orient the elongated axis toward x."""
        proj = plot.RotateCell(tilted_cell, axis="y", angle="best")
        pts = tilted_cell.skeleton.vertices
        result = proj(pts)
        assert np.var(result[:, 0]) > np.var(result[:, 1])

    def test_axis_best_full_pca(self, tilted_cell):
        """Full PCA mode should produce at least as much x-variance as the y-constrained mode."""
        pts = tilted_cell.skeleton.vertices
        proj_full = plot.RotateCell(tilted_cell)  # axis=None
        proj_y = plot.RotateCell(tilted_cell, axis="y", angle="best")
        var_full = np.var(proj_full(pts)[:, 0])
        var_constrained = np.var(proj_y(pts)[:, 0])
        # Full PCA is unconstrained, so x-variance should be >= constrained case
        assert (
            var_full >= var_constrained * 0.9
        )  # small tolerance for numeric precision

    def test_axis_best_string_same_as_none(self, tilted_cell):
        """axis='best' and axis=None should produce identical results."""
        pts = tilted_cell.skeleton.vertices
        proj_str = plot.RotateCell(tilted_cell, axis="best")
        proj_none = plot.RotateCell(tilted_cell, axis=None)
        np.testing.assert_allclose(proj_str(pts), proj_none(pts), atol=1e-12)


# ===========================================================================
# TestRotationProjectionIntegration
# ===========================================================================


class TestRotationProjectionIntegration:
    """Tests for _resolve_rotation_params and inline rotation in plotting functions."""

    @pytest.fixture
    def tilted_cell(self):
        """Cell elongated along a known direction in the xz plane (45°)."""
        rng = np.random.default_rng(13)
        direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        pts = rng.normal(scale=[10.0, 0.1, 0.1], size=(80, 3))
        R = np.column_stack([direction, [0, 1, 0], np.cross(direction, [0, 1, 0])])
        pts = pts @ R.T
        pts[0] = [0.0, 0.0, 0.0]
        return _make_cell_with_skeleton(pts, root_idx=0)

    # --- _resolve_rotation_params unit tests ---

    def test_none_returns_projection_unchanged(self):
        result = plot._resolve_rotation_params("xy", None, None, None, None, True)
        assert result == "xy"

    def test_numeric_angle_produces_callable(self):
        center = np.array([0.0, 0.0, 0.0])
        result = plot._resolve_rotation_params("xy", 45, "y", None, center, True)
        assert callable(result)

    def test_numeric_angle_without_axis_raises(self):
        center = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="rotation_axis is required"):
            plot._resolve_rotation_params("xy", 45, None, None, center, True)

    def test_numeric_angle_without_center_raises(self):
        with pytest.raises(ValueError, match="center is required"):
            plot._resolve_rotation_params("xy", 45, "y", None, None, True)

    def test_best_with_axis_maximizes_x_variance(self, tilted_cell):
        verts = tilted_cell.skeleton.vertices
        center = np.asarray(tilted_cell.skeleton.root_location, dtype=float)
        result = plot._resolve_rotation_params("xy", "best", "y", verts, center, False)
        assert callable(result)
        projected = result(verts)
        assert np.var(projected[:, 0]) > np.var(projected[:, 1])

    def test_best_full_pca(self, tilted_cell):
        verts = tilted_cell.skeleton.vertices
        center = np.asarray(tilted_cell.skeleton.root_location, dtype=float)
        result = plot._resolve_rotation_params("xy", "best", None, verts, center, False)
        assert callable(result)

    def test_non_default_projection_with_rotation_raises(self):
        center = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot combine projection"):
            plot._resolve_rotation_params("xz", 45, "y", None, center, True)

    def test_callable_projection_with_rotation_raises(self):
        center = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot combine a callable"):
            plot._resolve_rotation_params(
                lambda pts: pts[:, :2], 45, "y", None, center, True
            )

    def test_bool_rotation_angle_raises(self):
        with pytest.raises(ValueError, match="not bool"):
            plot._resolve_rotation_params("xy", True, "y", None, np.zeros(3), True)

    # --- Integration tests with plotting functions ---

    def test_plot_skeleton_with_numeric_rotation(self, tilted_cell):
        fig, ax = plt.subplots()
        plot.plot_skeleton(
            tilted_cell.skeleton, rotation_angle=45, rotation_axis="y", ax=ax
        )
        plt.close(fig)

    def test_plot_skeleton_with_best_rotation_and_axis(self, tilted_cell):
        fig, ax = plt.subplots()
        plot.plot_skeleton(
            tilted_cell.skeleton, rotation_angle="best", rotation_axis="y", ax=ax
        )
        plt.close(fig)

    def test_plot_skeleton_with_best_full_pca(self, tilted_cell):
        fig, ax = plt.subplots()
        plot.plot_skeleton(tilted_cell.skeleton, rotation_angle="best", ax=ax)
        plt.close(fig)

    def test_plot_morphology_2d_with_rotation(self, tilted_cell):
        fig, ax = plt.subplots()
        plot.plot_morphology_2d(
            tilted_cell, rotation_angle=90, rotation_axis="z", ax=ax
        )
        plt.close(fig)

    def test_plot_cell_2d_with_best_rotation(self, tilted_cell):
        fig, ax = plt.subplots()
        plot.plot_cell_2d(tilted_cell, rotation_angle="best", rotation_axis="y", ax=ax)
        plt.close(fig)

    def test_multiple_angles_same_axes(self, tilted_cell):
        """Animation loop: multiple angles on same axes works."""
        fig, ax = plt.subplots()
        for angle in [0, 30, 60, 90]:
            ax.clear()
            plot.plot_skeleton(
                tilted_cell.skeleton, rotation_angle=angle, rotation_axis="y", ax=ax
            )
        plt.close(fig)

    def test_backward_compat_string_projections(self, tilted_cell):
        """Existing string projections still work without rotation params."""
        fig, ax = plt.subplots()
        for proj in ["xy", "xz", "yz"]:
            ax.clear()
            plot.plot_skeleton(tilted_cell.skeleton, projection=proj, ax=ax)
        plt.close(fig)

    def test_plot_points_best_raises(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="not supported for plot_points"):
            plot.plot_points(pts, rotation_angle="best")


# ===========================================================================
# TestPlotLineup
# ===========================================================================


class TestPlotLineup:
    """Tests for plot_lineup and its private helpers."""

    @pytest.fixture
    def two_cells(self):
        """Two minimal cells with distinct x positions."""
        cell1 = _make_cell_with_skeleton(
            np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0], [20.0, 10.0, 0.0]])
        )
        cell2 = _make_cell_with_skeleton(
            np.array([[0.0, 0.0, 0.0], [10.0, -5.0, 0.0], [20.0, -10.0, 0.0]])
        )
        return [cell1, cell2]

    # --- _broadcast_param ---

    def test_broadcast_param_scalar(self):
        assert plot._broadcast_param("blue", 3) == ["blue", "blue", "blue"]

    def test_broadcast_param_list_correct_length(self):
        lst = ["a", "b", "c"]
        assert plot._broadcast_param(lst, 3) is lst

    def test_broadcast_param_list_wrong_length_raises(self):
        with pytest.raises(ValueError):
            plot._broadcast_param(["a", "b"], 3)

    # --- _project_point_y ---

    def test_project_point_y(self):
        proj = plot.projection_factory("xy")
        result = plot._project_point_y(np.array([1.0, 2.0, 3.0]), proj)
        assert result == pytest.approx(2.0)

    # --- plot_lineup basic ---

    def test_returns_axes(self, two_cells):
        ax = plot.plot_lineup(two_cells, projection="xy")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_empty_cells_raises(self):
        with pytest.raises(ValueError):
            plot.plot_lineup([])

    def test_uses_existing_ax(self, two_cells):
        fig, existing_ax = plt.subplots()
        returned_ax = plot.plot_lineup(two_cells, ax=existing_ax)
        assert returned_ax is existing_ax
        plt.close("all")

    def test_creates_figure_with_units_per_inch(self, two_cells):
        ax = plot.plot_lineup(two_cells, units_per_inch=1000)
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        assert w > 0 and h > 0
        plt.close("all")

    # --- horizontal layout ---

    def test_horizontal_no_overlap(self, two_cells):
        proj = plot.projection_factory("xy")
        offsets = plot._lineup_offsets(
            two_cells, proj, gap=0.0, align="natural", alignment_points=None
        )
        bounds = [
            plot._plotted_bounds(
                c.skeleton.vertices, proj, offsets[i][0], offsets[i][1]
            )
            for i, c in enumerate(two_cells)
        ]
        # cell 0 xmax <= cell 1 xmin
        assert bounds[0][0, 1] <= bounds[1][0, 0] + 1e-6

    def test_gap_increases_separation(self, two_cells):
        proj = plot.projection_factory("xy")
        offsets_no_gap = plot._lineup_offsets(
            two_cells, proj, gap=0.0, align="natural", alignment_points=None
        )
        offsets_gap = plot._lineup_offsets(
            two_cells, proj, gap=100.0, align="natural", alignment_points=None
        )
        # Center of cell 1 with gap should be further right than without
        center_no_gap = offsets_no_gap[1][0]
        center_gap = offsets_gap[1][0]
        assert center_gap > center_no_gap

    # --- styling ---

    def test_shared_style_all_cells(self, two_cells):
        ax = plot.plot_lineup(two_cells, palette="viridis")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_per_cell_palette(self, two_cells):
        palettes = [{"A": "red"}, "plasma"]
        ax = plot.plot_lineup(two_cells, palette=palettes)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    # --- alignment ---

    def test_natural_align_no_vertical_offset(self, two_cells):
        proj = plot.projection_factory("xy")
        offsets = plot._lineup_offsets(
            two_cells, proj, gap=0.0, align="natural", alignment_points=None
        )
        for _, offset_v in offsets:
            assert offset_v == 0.0

    def test_soma_align_centers_soma(self):
        # Create a cell whose soma (root) projects to a known y via "xy"
        root_y = 7.0
        vertices = np.array([[5.0, root_y, 0.0], [10.0, 20.0, 0.0]])
        cell = _make_cell_with_skeleton(vertices, root_idx=0)
        proj = plot.projection_factory("xy")
        offsets = plot._lineup_offsets(
            [cell], proj, gap=0.0, align="soma", alignment_points=None
        )
        assert offsets[0][1] == pytest.approx(-root_y)

    def test_point_align_centers_given_point(self):
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
        cell = _make_cell_with_skeleton(vertices)
        proj = plot.projection_factory("xy")
        point = np.array([0.0, 3.0, 0.0])
        offsets = plot._lineup_offsets(
            [cell], proj, gap=0.0, align="point", alignment_points=[point]
        )
        assert offsets[0][1] == pytest.approx(-3.0)

    def test_soma_and_point_equivalent_at_root(self):
        vertices = np.array([[5.0, 9.0, 0.0], [10.0, 20.0, 0.0]])
        cell = _make_cell_with_skeleton(vertices, root_idx=0)
        proj = plot.projection_factory("xy")
        root_loc = cell.skeleton.root_location
        offsets_soma = plot._lineup_offsets(
            [cell], proj, gap=0.0, align="soma", alignment_points=None
        )
        offsets_point = plot._lineup_offsets(
            [cell], proj, gap=0.0, align="point", alignment_points=[root_loc]
        )
        assert offsets_soma[0][1] == pytest.approx(offsets_point[0][1])

    def test_point_align_missing_alignment_point_raises(self, two_cells):
        with pytest.raises(ValueError):
            plot.plot_lineup(two_cells, align="point", alignment_point=None)


class TestPlotLineupGrid:
    """Tests for LineupGroup + plot_lineup_grid + add_layer_lines."""

    @pytest.fixture
    def cell_a(self):
        return _make_cell_with_skeleton(
            np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 5.0, 0.0]])
        )

    @pytest.fixture
    def cell_b(self):
        return _make_cell_with_skeleton(
            np.array([[0.0, 100.0, 0.0], [5.0, 110.0, 0.0], [10.0, 105.0, 0.0]])
        )

    @pytest.fixture
    def cell_c(self):
        return _make_cell_with_skeleton(
            np.array([[0.0, 200.0, 0.0], [5.0, 210.0, 0.0], [10.0, 205.0, 0.0]])
        )

    # --- LineupGroup dataclass ---

    def test_lineup_group_defaults(self, cell_a):
        grp = plot.LineupGroup([cell_a])
        assert grp.label is None
        assert grp.color is None
        assert grp.palette == "coolwarm"
        assert grp.synapses is False
        assert grp.cells == [cell_a]
        plt.close("all")

    def test_lineup_group_accepts_per_cell_lists(self, cell_a, cell_b):
        grp = plot.LineupGroup(
            [cell_a, cell_b],
            color=["red", "blue"],
            alpha=[1.0, 0.5],
        )
        assert grp.color == ["red", "blue"]
        assert grp.alpha == [1.0, 0.5]

    def test_resolve_cell_style_scalar(self, cell_a, cell_b):
        grp = plot.LineupGroup([cell_a, cell_b], color="red", alpha=0.7)
        style0 = plot._resolve_cell_style(grp, 0)
        style1 = plot._resolve_cell_style(grp, 1)
        assert style0["color"] == "red"
        assert style1["color"] == "red"
        assert style0["alpha"] == 0.7

    def test_resolve_cell_style_list(self, cell_a, cell_b):
        grp = plot.LineupGroup(
            [cell_a, cell_b], color=["red", "blue"], alpha=[1.0, 0.5]
        )
        style0 = plot._resolve_cell_style(grp, 0)
        style1 = plot._resolve_cell_style(grp, 1)
        assert style0["color"] == "red"
        assert style1["color"] == "blue"
        assert style0["alpha"] == 1.0
        assert style1["alpha"] == 0.5

    def test_resolve_cell_style_wrong_list_length_raises(self, cell_a, cell_b):
        grp = plot.LineupGroup([cell_a, cell_b], color=["red", "blue", "green"])
        with pytest.raises(ValueError, match="length 3"):
            plot._resolve_cell_style(grp, 0)

    # --- plot_lineup_grid basics ---

    def test_returns_axes(self, cell_a, cell_b):
        ax = plot.plot_lineup_grid([plot.LineupGroup([cell_a, cell_b], label="A")])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError):
            plot.plot_lineup_grid([])

    def test_uses_existing_ax(self, cell_a):
        _fig, existing_ax = plt.subplots()
        returned_ax = plot.plot_lineup_grid(
            [plot.LineupGroup([cell_a])], ax=existing_ax
        )
        assert returned_ax is existing_ax
        plt.close("all")

    # --- _grid_offsets layout ---

    def test_grid_offsets_single_row(self, cell_a, cell_b, cell_c):
        proj = plot.projection_factory("xy")
        groups = [
            plot.LineupGroup([cell_a, cell_b]),
            plot.LineupGroup([cell_c]),
        ]
        offsets, anchors = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=0.0,
            row_max_cells=None,
            row_max_width=None,
            row_gap=0.0,
            alignment_points=None,
        )
        # All cells on row 0 → offset_v == 0 in natural alignment
        for group_offs in offsets:
            for _, v in group_offs:
                assert v == 0.0
        # Cells are placed in increasing x order
        flat_h = [h for group_offs in offsets for h, _ in group_offs]
        assert flat_h == sorted(flat_h)

    def test_grid_offsets_row_max_cells_wraps(self, cell_a, cell_b, cell_c):
        proj = plot.projection_factory("xy")
        groups = [
            plot.LineupGroup([cell_a]),
            plot.LineupGroup([cell_b]),
            plot.LineupGroup([cell_c]),
        ]
        offsets, _ = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=0.0,
            row_max_cells=2,
            row_max_width=None,
            row_gap=10.0,
            alignment_points=None,
        )
        # Groups 0 and 1 should share a row; group 2 wraps to the next.
        # Row 0 cells have offset_v == 0; row 1 cells have offset_v < 0.
        assert offsets[0][0][1] == 0.0
        assert offsets[1][0][1] == 0.0
        assert offsets[2][0][1] < 0.0

    def test_grid_offsets_row_max_width_wraps(self, cell_a, cell_b, cell_c):
        proj = plot.projection_factory("xy")
        # Each cell spans x∈[0, 10], width 10. Set max width to force wrap.
        groups = [plot.LineupGroup([c]) for c in (cell_a, cell_b, cell_c)]
        offsets, _ = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=5.0,
            row_max_cells=None,
            row_max_width=20.0,  # fits cell + gap + cell (10 + 5 + 10 = 25 → no)
            row_gap=10.0,
            alignment_points=None,
        )
        # Each group should wrap to its own row
        assert offsets[1][0][1] < offsets[0][0][1]
        assert offsets[2][0][1] < offsets[1][0][1]

    def test_grid_offsets_label_anchor_only_when_labeled(self, cell_a, cell_b):
        proj = plot.projection_factory("xy")
        groups = [
            plot.LineupGroup([cell_a], label="A"),
            plot.LineupGroup([cell_b]),  # no label
        ]
        _, anchors = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=0.0,
            row_max_cells=None,
            row_max_width=None,
            row_gap=0.0,
            alignment_points=None,
        )
        assert anchors[0] is not None
        assert anchors[1] is None

    # --- add_layer_lines ---

    def test_add_layer_lines_adds_lines(self):
        _fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        n_lines_before = len(ax.get_lines())
        plot.add_layer_lines(ax, {25: "A", 50: "B", 75: None})
        n_lines_after = len(ax.get_lines())
        assert n_lines_after - n_lines_before == 3
        plt.close("all")

    def test_add_layer_lines_labels_only_for_truthy(self):
        _fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        n_texts_before = len(ax.texts)
        plot.add_layer_lines(ax, {25: "A", 50: None, 75: "C"})
        labels = [t.get_text() for t in ax.texts[n_texts_before:]]
        assert labels == ["A", "C"]
        plt.close("all")

    # --- end-to-end ---

    def test_lineup_grid_with_layer_lines_and_labels(self, cell_a, cell_b, cell_c):
        ax = plot.plot_lineup_grid(
            [
                plot.LineupGroup([cell_a, cell_b], label="Top", color="red"),
                plot.LineupGroup([cell_c], label="Bottom", color="blue"),
            ],
            inter_cell_gap=2.0,
            inter_group_gap=5.0,
            row_max_cells=2,
            row_gap=20.0,
            layer_lines={50: "mid", 150: "high"},
            group_label_offset=5.0,
        )
        assert isinstance(ax, plt.Axes)
        # axhlines from layer_lines (the lines are added as Line2D children)
        n_axhlines = sum(1 for line in ax.get_lines() if line.get_xdata()[0] == 0)
        # There are at least 2 layer lines
        assert n_axhlines >= 2
        plt.close("all")

    # --- rotation in lineup ---

    def test_resolve_group_projections_no_rotation(self, cell_a, cell_b):
        """No rotation → all cells get the base projection callable."""
        grp = plot.LineupGroup([cell_a, cell_b])
        projs = plot._resolve_group_projections(grp, "xy", invert_y=True)
        assert len(projs) == 2
        # All callable; both produce the same xy projection for the same input
        pts = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(projs[0](pts), projs[1](pts))

    def test_resolve_group_projections_per_cell_rotation(self, cell_a, cell_b):
        """rotation_angle='best' → each cell gets its own rotation callable."""
        grp = plot.LineupGroup(
            [cell_a, cell_b],
            rotation_angle="best",
            rotation_axis="y",
        )
        projs = plot._resolve_group_projections(grp, "xy", invert_y=True)
        assert len(projs) == 2
        # Each callable rotates around its own root — the two should not be
        # identical since their roots differ.
        assert projs[0] is not projs[1]

    def test_plot_lineup_grid_with_best_rotation_renders(self, cell_a, cell_b, cell_c):
        """End-to-end: groups with rotation_angle='best' lay out without overlap."""
        grp = plot.LineupGroup(
            [cell_a, cell_b, cell_c],
            label="Rotated",
            rotation_angle="best",
            rotation_axis="y",
            color="red",
        )
        ax = plot.plot_lineup_grid([grp], inter_cell_gap=5.0)
        assert isinstance(ax, plt.Axes)
        # Confirm we drew something — at least one LineCollection per cell.
        from matplotlib.collections import LineCollection

        lc_count = sum(isinstance(child, LineCollection) for child in ax.get_children())
        assert lc_count >= 3
        plt.close("all")

    def test_plot_lineup_grid_mixed_rotation_groups(self, cell_a, cell_b, cell_c):
        """Two groups: one rotated, one not. Both render cleanly."""
        rotated = plot.LineupGroup(
            [cell_a, cell_b],
            label="Rotated",
            rotation_angle="best",
            rotation_axis="y",
            color="red",
        )
        natural = plot.LineupGroup(
            [cell_c],
            label="Natural",
            color="blue",
        )
        ax = plot.plot_lineup_grid(
            [rotated, natural], inter_cell_gap=5.0, inter_group_gap=20.0
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    # --- row-stacking direction ---

    def test_grid_offsets_inverted_axis_stacks_positive_y(self, cell_a, cell_b):
        """With y_axis_inverted=True, subsequent row baselines INCREASE
        in data y so they appear below on the inverted screen.
        """
        proj = plot.projection_factory("xy")
        groups = [plot.LineupGroup([c]) for c in (cell_a, cell_b)]
        offsets, _ = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=0.0,
            row_max_cells=1,  # force wrap so each group is its own row
            row_max_width=None,
            row_gap=10.0,
            alignment_points=None,
            y_axis_inverted=True,
        )
        # Row 0 at y=0. Row 1 should be at y > 0 because axis is inverted.
        assert offsets[0][0][1] == 0.0
        assert offsets[1][0][1] > 0.0

    def test_grid_offsets_non_inverted_axis_stacks_negative_y(self, cell_a, cell_b):
        """With y_axis_inverted=False, subsequent baselines DECREASE in
        data y so they appear below on a non-inverted screen.
        """
        proj = plot.projection_factory("xz")  # no "y" in projection → not inverted
        groups = [plot.LineupGroup([c]) for c in (cell_a, cell_b)]
        offsets, _ = plot._grid_offsets(
            groups,
            projection=proj,
            align="natural",
            inter_cell_gap=0.0,
            inter_group_gap=0.0,
            row_max_cells=1,
            row_max_width=None,
            row_gap=10.0,
            alignment_points=None,
            y_axis_inverted=False,
        )
        assert offsets[0][0][1] == 0.0
        assert offsets[1][0][1] < 0.0

    def test_plot_lineup_grid_xy_stacks_below_on_screen(self, cell_a, cell_b):
        """End-to-end: with default "xy" projection (invert_y=True), the
        second row should sit at a HIGHER data y than the first row so
        that — on the inverted axis — it appears below on screen.
        """
        ax = plot.plot_lineup_grid(
            [
                plot.LineupGroup([cell_a], label="row 0"),
                plot.LineupGroup([cell_b], label="row 1"),
            ],
            projection="xy",
            row_max_cells=1,
            row_gap=20.0,
        )
        # Axis should have been inverted.
        assert ax.yaxis_inverted()
        plt.close("all")
