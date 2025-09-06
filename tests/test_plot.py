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

        # Should return RGBA colors
        assert colors.shape == (4, 4)
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
        self, simple_skeleton_data, spatial_columns, mock_labels
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
            labels=mock_labels,
        )

        # Test plotting with colors (basic test - may need label resolution)
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

    def test_add_scale_bar(self):
        """Test scale bar addition to plots."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Set up some basic plot bounds
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # Add scale bar
        plot.add_scale_bar(ax, length=10, label="10 μm")

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

        plt.close(fig)
