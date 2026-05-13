# Changelog

## 0.1.0

### New features

- **3D plotting** (`ossify.plot3d`, available behind the `viz` extra): a
  PyVista-based rendering backend with `plot_cell_3d`, `plot_morphology_3d`,
  `plot_mesh_3d`, `plot_graph_3d`, `plot_annotations_3d`,
  `plot_skeleton_3d`, and `plot_points_3d`. Utilities `add_colorbar_3d` and
  `orbit_3d` support colorbars and orbit-style animations.
  Install with `pip install ossify[viz]`.
- **Skeleton resampling**: `SkeletonLayer.resample(spacing, ...)` returns a
  new skeleton with approximately uniform vertex spacing along each
  unbranched segment. Topological points (root, branch points, end
  points) are preserved; feature values are remapped by nearest neighbor
  or by user-supplied pandas aggregations.
- **SWC export improvements**: `export_swc` now accepts a `SkeletonLayer`
  directly (not just a `Cell`) and supports a `resample_distance` parameter
  for resampling on the fly.
- **Projection helpers**: `ossify.plot.Rotation` and
  `ossify.plot.RotateCell` build callable projections for rotated views,
  including PCA-based "best angle" and full-PCA modes.

### Improvements

- **2D plot fixes**: corrected annotation color mapping, point-value
  mapping, and default sizing for cleaner output across unit systems.
- **Better `CellFiles` ergonomics** for batched cell I/O.
- **Documentation**: new conceptual pages, expanded user guide for
  skeleton resampling, 3D visualization, and rotation helpers.

### Internal

- The masking and linking machinery formerly provided by the external
  `morphsync` dependency has been vendored into `ossify._sync`. The
  public API is unchanged; the package now has one fewer external
  dependency.

## 0.0.4

Initial release on PyPI.