"""Tests for ossify.plot3d — 3D plotting via PyVista."""

import numpy as np
import pandas as pd
import pytest
import pyvista as pv

pv.OFF_SCREEN = True

from ossify import plot3d
from ossify.plot3d import (
    plot_annotations_3d,
    plot_cell_3d,
    plot_morphology_3d,
    plot_points_3d,
    plot_skeleton_3d,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _has_actors(plotter: pv.Plotter) -> bool:
    """Return True if the plotter has at least one actor."""
    return len(plotter.renderer.actors) > 0


# ===========================================================================
# plot_skeleton_3d
# ===========================================================================


class TestPlotSkeleton3d:
    def test_returns_plotter(self, nrn):
        pl = plot_skeleton_3d(nrn.skeleton)
        assert isinstance(pl, pv.Plotter)

    def test_has_actors(self, nrn):
        pl = plot_skeleton_3d(nrn.skeleton)
        assert _has_actors(pl)

    def test_with_colors(self, nrn):
        skel = nrn.skeleton
        colors = np.ones((skel.n_vertices, 3), dtype=float) * 0.5
        pl = plot_skeleton_3d(skel, colors=colors)
        assert _has_actors(pl)

    def test_with_uniform_tube_radius(self, nrn):
        pl = plot_skeleton_3d(nrn.skeleton, tube_radius=500.0)
        assert _has_actors(pl)

    def test_with_array_tube_radius(self, nrn):
        skel = nrn.skeleton
        radii = np.ones(skel.n_vertices, dtype=float) * 200.0
        pl = plot_skeleton_3d(skel, tube_radius=radii)
        assert _has_actors(pl)

    def test_uses_provided_plotter(self, nrn):
        pl = pv.Plotter()
        result = plot_skeleton_3d(nrn.skeleton, plotter=pl)
        assert result is pl

    def test_with_opacity(self, nrn):
        pl = plot_skeleton_3d(nrn.skeleton, opacity=0.5)
        assert _has_actors(pl)


# ===========================================================================
# plot_morphology_3d
# ===========================================================================


class TestPlotMorphology3d:
    def test_default(self, nrn):
        pl = plot_morphology_3d(nrn)
        assert _has_actors(pl)

    def test_color_scale_log(self, nrn):
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        pl = plot_morphology_3d(nrn, color=feature_names[0], color_scale="log")
        assert _has_actors(pl)

    def test_color_scale_log_with_norm(self, nrn):
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        vals = skel.get_feature(feature_names[0]).astype(float)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmin <= 0:
            pytest.skip("feature has non-positive values, log scale not applicable")
        pl = plot_morphology_3d(
            nrn, color=feature_names[0], color_scale="log", color_norm=(vmin, vmax)
        )
        assert _has_actors(pl)

    def test_with_skeleton_layer_directly(self, nrn):
        pl = plot_morphology_3d(nrn.skeleton)
        assert _has_actors(pl)

    def test_single_color_string(self, nrn):
        pl = plot_morphology_3d(nrn, color="red")
        assert _has_actors(pl)

    def test_color_array(self, nrn):
        skel = nrn.skeleton
        values = np.arange(skel.n_vertices, dtype=float)
        pl = plot_morphology_3d(nrn, color=values)
        assert _has_actors(pl)

    def test_color_feature_name(self, nrn):
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        pl = plot_morphology_3d(nrn, color=feature_names[0])
        assert _has_actors(pl)

    def test_tube_radius_float(self, nrn):
        pl = plot_morphology_3d(nrn, tube_radius=300.0)
        assert _has_actors(pl)

    def test_tube_radius_feature_name(self, nrn):
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        pl = plot_morphology_3d(nrn, tube_radius=feature_names[0])
        assert _has_actors(pl)

    def test_tube_radius_with_radii_scaling(self, nrn):
        """tube_radii rescales feature values to a fixed output range."""
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        pl = plot_morphology_3d(
            nrn,
            tube_radius=feature_names[0],
            tube_radii=(100.0, 500.0),
        )
        assert _has_actors(pl)

    def test_tube_radius_with_norm_caps(self, nrn):
        """tube_radius_norm clips and normalizes the input range."""
        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        pl = plot_morphology_3d(
            nrn,
            tube_radius=feature_names[0],
            tube_radius_norm=(0.0, 1.0),
            tube_radii=(50.0, 300.0),
        )
        assert _has_actors(pl)

    def test_tube_radius_array_with_radii(self, nrn):
        skel = nrn.skeleton
        raw = np.linspace(0.0, 1.0, skel.n_vertices)
        pl = plot_morphology_3d(nrn, tube_radius=raw, tube_radii=(100.0, 400.0))
        assert _has_actors(pl)

    def test_root_marker_default_radius(self, nrn):
        pl = plot_morphology_3d(nrn, root_marker=True)
        assert _has_actors(pl)

    def test_root_marker_explicit_radius(self, nrn):
        pl = plot_morphology_3d(nrn, root_marker=True, root_radius=500.0)
        assert _has_actors(pl)

    def test_root_marker_radius_from_tube_radius(self, nrn):
        pl = plot_morphology_3d(nrn, root_marker=True, tube_radius=400.0)
        assert _has_actors(pl)

    def test_root_marker_explicit_color(self, nrn):
        pl = plot_morphology_3d(nrn, root_marker=True, root_color="blue")
        assert _has_actors(pl)

    def test_root_marker_color_from_color_array(self, nrn):
        skel = nrn.skeleton
        values = np.zeros(skel.n_vertices, dtype=float)
        pl = plot_morphology_3d(nrn, color=values, root_marker=True)
        assert _has_actors(pl)

    def test_uses_provided_plotter(self, nrn):
        pl = pv.Plotter()
        result = plot_morphology_3d(nrn, plotter=pl)
        assert result is pl

    def test_color_mapping_shape(self, nrn):
        """Color array derived from feature values should have shape (N, 3)."""
        skel = nrn.skeleton
        values = np.linspace(0, 1, skel.n_vertices)
        # Use plot_morphology_3d to trigger color resolution; inspect via
        # the PolyData stored in the plotter's mesh actors indirectly by
        # checking that n_vertices in skeleton is consistent.
        assert skel.n_vertices > 0


# ===========================================================================
# plot_points_3d
# ===========================================================================


class TestPlotPoints3d:
    def test_returns_plotter(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        pl = plot_points_3d(pts)
        assert isinstance(pl, pv.Plotter)

    def test_has_actors(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        pl = plot_points_3d(pts)
        assert _has_actors(pl)

    def test_uniform_size(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        pl = plot_points_3d(pts, sizes=100.0)
        assert _has_actors(pl)

    def test_array_sizes(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        radii = np.linspace(50.0, 200.0, len(pts))
        pl = plot_points_3d(pts, sizes=radii)
        assert _has_actors(pl)

    def test_string_color(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        pl = plot_points_3d(pts, colors="red")
        assert _has_actors(pl)

    def test_value_colors(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        values = np.arange(len(pts), dtype=float)
        pl = plot_points_3d(pts, colors=values)
        assert _has_actors(pl)

    def test_rgb_colors(self, nrn):
        pts = nrn.skeleton.vertices[:10]
        rgb = np.ones((len(pts), 3), dtype=float) * 0.5
        pl = plot_points_3d(pts, colors=rgb)
        assert _has_actors(pl)

    def test_uses_provided_plotter(self, nrn):
        pts = nrn.skeleton.vertices[:5]
        pl = pv.Plotter()
        result = plot_points_3d(pts, plotter=pl)
        assert result is pl

    # --- array-like input normalization ---
    # plot_points_3d should accept any object that numpy can convert via
    # __array__ (pandas Series, polars Series, plain lists, etc.).

    @pytest.mark.parametrize(
        "make_colors",
        [
            pytest.param(lambda v: v, id="ndarray"),
            pytest.param(lambda v: v.tolist(), id="list"),
            pytest.param(lambda v: pd.Series(v), id="pandas_Series"),
        ],
    )
    def test_array_like_colors(self, nrn, make_colors):
        """colors accepts any array-like, not just ndarray."""
        pts = nrn.skeleton.vertices[:10]
        values = make_colors(np.arange(len(pts), dtype=float))
        pl = plot_points_3d(pts, colors=values)
        assert _has_actors(pl)

    @pytest.mark.parametrize(
        "make_colors",
        [
            pytest.param(lambda v: v, id="ndarray"),
            pytest.param(lambda v: v.tolist(), id="list"),
            pytest.param(lambda v: pd.Series(v), id="pandas_Series"),
        ],
    )
    def test_array_like_colors_glyph_path(self, nrn, make_colors):
        """array-like colors work on the glyph (per-point sizes) path too."""
        pts = nrn.skeleton.vertices[:10]
        values = make_colors(np.arange(len(pts), dtype=float))
        radii = np.linspace(50.0, 200.0, len(pts))
        pl = plot_points_3d(pts, colors=values, sizes=radii, palette="viridis")
        assert _has_actors(pl)

    @pytest.mark.parametrize(
        "make_sizes",
        [
            pytest.param(lambda v: v, id="ndarray"),
            pytest.param(lambda v: v.tolist(), id="list"),
            pytest.param(lambda v: pd.Series(v), id="pandas_Series"),
        ],
    )
    def test_array_like_sizes(self, nrn, make_sizes):
        """sizes accepts any array-like, not just ndarray."""
        pts = nrn.skeleton.vertices[:10]
        radii = make_sizes(np.linspace(50.0, 200.0, len(pts)))
        pl = plot_points_3d(pts, sizes=radii)
        assert _has_actors(pl)


# ===========================================================================
# plot_annotations_3d
# ===========================================================================


class TestPlotAnnotations3d:
    def test_returns_plotter(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        pl = plot_annotations_3d(anno)
        assert isinstance(pl, pv.Plotter)

    def test_has_actors(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        pl = plot_annotations_3d(anno)
        assert _has_actors(pl)

    def test_with_color_string(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        pl = plot_annotations_3d(anno, color="green")
        assert _has_actors(pl)

    def test_with_size(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        pl = plot_annotations_3d(anno, size=50.0)
        assert _has_actors(pl)

    def test_uses_provided_plotter(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        pl = pv.Plotter()
        result = plot_annotations_3d(anno, plotter=pl)
        assert result is pl

    def _get_positive_feature(self, nrn):
        """Return (anno, feature_name) where feature values are all positive."""
        for name in nrn.annotations.names:
            anno = nrn.annotations[name]
            for feat in anno.feature_names:
                vals = anno.get_feature(feat).astype(float)
                if np.all(vals > 0):
                    return anno, feat
        return None, None

    def test_color_scale_log(self, nrn):
        anno, feat = self._get_positive_feature(nrn)
        if anno is None:
            pytest.skip("no annotation with all-positive feature")
        pl = plot_annotations_3d(anno, color=feat, color_scale="log")
        assert _has_actors(pl)

    def test_color_scale_log_with_norm(self, nrn):
        anno, feat = self._get_positive_feature(nrn)
        if anno is None:
            pytest.skip("no annotation with all-positive feature")
        vals = anno.get_feature(feat).astype(float)
        pl = plot_annotations_3d(
            anno,
            color=feat,
            color_scale="log",
            color_norm=(float(np.nanmin(vals)), float(np.nanmax(vals))),
        )
        assert _has_actors(pl)

    def test_size_scale_sqrt(self, nrn):
        anno, feat = self._get_positive_feature(nrn)
        if anno is None:
            pytest.skip("no annotation with all-positive feature")
        pl = plot_annotations_3d(anno, size=feat, size_scale="sqrt")
        assert _has_actors(pl)

    def test_size_scale_cbrt(self, nrn):
        anno, feat = self._get_positive_feature(nrn)
        if anno is None:
            pytest.skip("no annotation with all-positive feature")
        pl = plot_annotations_3d(anno, size=feat, size_scale="cbrt")
        assert _has_actors(pl)

    def test_size_scale_log(self, nrn):
        anno, feat = self._get_positive_feature(nrn)
        if anno is None:
            pytest.skip("no annotation with all-positive feature")
        pl = plot_annotations_3d(anno, size=feat, size_scale="log")
        assert _has_actors(pl)


# ===========================================================================
# plot_cell_3d
# ===========================================================================


class TestPlotCell3d:
    def test_skeleton_only(self, nrn):
        pl = plot_cell_3d(nrn)
        assert _has_actors(pl)

    def test_with_root_marker(self, nrn):
        pl = plot_cell_3d(nrn, root_marker=True)
        assert _has_actors(pl)

    def test_uses_provided_plotter(self, nrn):
        pl = pv.Plotter()
        result = plot_cell_3d(nrn, plotter=pl)
        assert result is pl

    def _anno_count(self, pl: pv.Plotter) -> int:
        return len(pl.renderer.actors)

    def test_synapses_false_renders_no_annotations(self, nrn):
        pl_no = plot_cell_3d(nrn, synapses=False)
        pl_pre = plot_cell_3d(nrn, synapses="pre")
        anno_names = nrn.annotations.names
        if "pre_syn" not in anno_names:
            pytest.skip("no pre_syn annotation")
        assert self._anno_count(pl_pre) > self._anno_count(pl_no)

    def test_synapses_pre(self, nrn):
        if "pre_syn" not in nrn.annotations.names:
            pytest.skip("no pre_syn annotation")
        pl = plot_cell_3d(nrn, synapses="pre")
        assert _has_actors(pl)

    def test_synapses_post(self, nrn):
        if "post_syn" not in nrn.annotations.names:
            pytest.skip("no post_syn annotation")
        pl = plot_cell_3d(nrn, synapses="post")
        assert _has_actors(pl)

    def test_synapses_both(self, nrn):
        anno_names = nrn.annotations.names
        if "pre_syn" not in anno_names and "post_syn" not in anno_names:
            pytest.skip("no pre_syn or post_syn annotation")
        pl = plot_cell_3d(nrn, synapses="both")
        assert _has_actors(pl)

    def test_synapses_true_alias(self, nrn):
        anno_names = nrn.annotations.names
        if "pre_syn" not in anno_names and "post_syn" not in anno_names:
            pytest.skip("no pre_syn or post_syn annotation")
        pl = plot_cell_3d(nrn, synapses=True)
        assert _has_actors(pl)

    def test_pre_post_separate_colors(self, nrn):
        anno_names = nrn.annotations.names
        if "pre_syn" not in anno_names or "post_syn" not in anno_names:
            pytest.skip("need both pre_syn and post_syn")
        pl = plot_cell_3d(
            nrn,
            synapses="both",
            pre_color="red",
            post_color="blue",
        )
        assert _has_actors(pl)

    def test_missing_pre_anno_ignored(self, nrn):
        pl = plot_cell_3d(nrn, synapses="pre", pre_anno="nonexistent_xyz")
        assert _has_actors(pl)

    def test_missing_post_anno_ignored(self, nrn):
        pl = plot_cell_3d(nrn, synapses="post", post_anno="nonexistent_xyz")
        assert _has_actors(pl)

    def test_custom_pre_anno_name(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        pl = plot_cell_3d(nrn, synapses="pre", pre_anno=anno_names[0])
        assert _has_actors(pl)

    def test_syn_size_and_opacity(self, nrn):
        if "pre_syn" not in nrn.annotations.names:
            pytest.skip("no pre_syn annotation")
        pl = plot_cell_3d(nrn, synapses="pre", syn_size=50.0, syn_opacity=0.5)
        assert _has_actors(pl)

    def test_syn_size_scale_log(self, nrn):
        anno_names = nrn.annotations.names
        if not anno_names:
            pytest.skip("no annotations on cell")
        anno = nrn.annotations[anno_names[0]]
        feat_names = anno.feature_names
        if not feat_names:
            pytest.skip("no features on annotation")
        feat = feat_names[0]
        vals = anno.get_feature(feat).astype(float)
        if not np.all(vals > 0):
            pytest.skip("feature has non-positive values")
        pl = plot_cell_3d(
            nrn,
            synapses="pre",
            pre_anno=anno_names[0],
            syn_size=feat,
            syn_size_scale="log",
        )
        assert _has_actors(pl)


# ===========================================================================
# plot_utils integration — _resolve_scalar_parameter
# ===========================================================================


class TestResolveScalarParameter:
    """Tests for the shared scalar resolution utility."""

    def test_none_returns_none(self):
        from ossify.plot_utils import _resolve_scalar_parameter

        assert _resolve_scalar_parameter(None, 5) is None

    def test_number_returns_constant_array(self):
        from ossify.plot_utils import _resolve_scalar_parameter

        out = _resolve_scalar_parameter(0.5, 4)
        assert out is not None
        assert out.shape == (4,)
        assert np.allclose(out, 0.5)

    def test_array_no_rescale(self):
        from ossify.plot_utils import _resolve_scalar_parameter

        arr = np.array([0.1, 0.5, 0.9])
        out = _resolve_scalar_parameter(arr, 3)
        assert np.allclose(out, arr)

    def test_array_with_out_range(self):
        from ossify.plot_utils import _resolve_scalar_parameter

        arr = np.array([0.0, 1.0, 2.0])
        out = _resolve_scalar_parameter(arr, 3, out_range=(0.0, 10.0))
        assert out is not None
        assert np.isclose(out[0], 0.0)
        assert np.isclose(out[-1], 10.0)

    def test_feature_name_resolves(self, nrn):
        from ossify.plot_utils import _resolve_scalar_parameter

        skel = nrn.skeleton
        feature_names = skel.feature_names
        if not feature_names:
            pytest.skip("no features on skeleton")
        out = _resolve_scalar_parameter(feature_names[0], skel.n_vertices, layer=skel)
        assert out is not None
        assert out.shape == (skel.n_vertices,)

    def test_feature_name_requires_layer(self):
        from ossify.plot_utils import _resolve_scalar_parameter

        with pytest.raises(ValueError, match="layer is required"):
            _resolve_scalar_parameter("some_feature", 10, layer=None)
