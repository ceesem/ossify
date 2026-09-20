# Changelog

## 0.2.8

### Bug fixes

- **`SkeletonLayer.cover_paths_specific` always raised.** It called a
  `graph_functions` name that does not exist, so every call ended in
  `AttributeError`. It had no test coverage.

- **Branch points and end points were wrong for skeletons whose edges were
  written `[parent, child]`.** Edges are reoriented to `[child, parent]` at
  construction, but the cached graph had already been built from the original
  orientation and was not invalidated, so it disagreed with `edges_positional`
  afterwards. Everything read off the directed graph followed it: a branching
  skeleton reported no branch points and a single end point, and segments and
  cover paths inherited that. `parent_node_array` was correct throughout,
  which is why it went unnoticed.

- **`"majority"` aggregation failed inside a per-feature dict.**
  `map_features_to_layer` translated ossify's own `"majority"` reducer only
  when it was the entire `agg` argument. Passed per feature — the documented
  way to mix aggregations — the string reached pandas untouched and raised.

- **`vertex_index` as an array never worked.** The array form is part of the
  signature of `add_skeleton`, `add_graph` and `add_mesh`, but the
  implementation tested it for truthiness, which raises on any array of more
  than one element. Only the string-column form was reachable.

- **Root inference returned a positional index where a vertex index is
  required.** Any skeleton whose vertices are not indexed `0..n-1` failed at
  construction with an opaque `IndexError` whenever `root` was left to be
  inferred — the normal case for data keyed by segment id. A `root` that is
  not one of the layer's vertices now raises an error naming the convention.

- **Plotting no longer warns on every `root_marker=True` figure.** A lone
  RGB/RGBA colour was passed to matplotlib's `c`, where it is ambiguous with
  one scalar value per point.

### API

- **`edges_as_positional` / `faces_as_positional`** on `add_skeleton`,
  `add_graph` and `add_mesh` declare whether connectivity is written as
  positional indices into `vertices` or as vertex indices. The default,
  `None`, keeps the previous rule — positional exactly when `vertex_index` is
  supplied — so existing callers are unaffected. The flag matters because that
  rule cannot disambiguate a vertex index whose values are also valid
  positions: there, the inferred reading silently builds a different graph.
  Connectivity outside the positional range now raises an error naming the
  convention rather than a `KeyError` from inside `fastremap`.

- The convention itself is now documented on the `edges`, `faces`,
  `vertex_index` and `root` parameters of all three methods, including the one
  asymmetry: `root` is a vertex index while `edges` in the same call may be
  positional.

### Documentation

- Ten of the thirteen user-guide pages now run end to end, and
  `tests/test_docs.py` executes them so they cannot silently rot. Most pages
  previously began mid-stream using a `cell` no page defined, so nothing on
  them could be run as written. The three remaining pages are external- or
  optional-dependency bound rather than broken; their API usage was checked
  separately and corrected.

### Internal

- The rule for interpreting edges and faces lives in one place,
  `PointMixin._index_connectivity`, instead of being repeated per layer.
- `translate.py` no longer remaps chunkedgraph edges from `l2_id` to
  positional indices purely so that `add_graph` can remap them back.

## 0.2.7

### Bug fixes

- **`distance_to_root()` and `hops_to_root()` returned values attached to the
  wrong vertices on masked skeletons.** Both methods measure through the
  *original, unmasked* graph — that is deliberate, so that a masked axon still
  reports its distance back to the soma even when the soma is masked out. Their
  results are therefore indexed by position in the base skeleton, and the
  vertex selection has to be converted into that space. It was not always
  converted, so on a masked layer entry *i* could belong to base vertex *i*
  rather than to masked vertex *i*.

    The array had the right length and nothing raised, so the corruption was
    silent. The usual symptom was implausible values, such as negative path
    distances.

    The two methods failed differently:

    - `distance_to_root()` was wrong **only in its no-argument form**. The
      explicit forms — `distance_to_root(vertices, as_positional=True)` and
      `distance_to_root(vertex_ids)` — were already correct.
    - `hops_to_root()` was wrong in **every** call form, including by-vertex-id.
      Because all three were wrong in the same way, they agreed with each other,
      which is why the problem was hard to spot.

    Both now guarantee that `f()` returns exactly one value per vertex of the
    current layer, in the current positional order, and always equals
    `f(np.arange(n), as_positional=True)` and `f(layer.vertex_index)`.

    **Unmasked skeletons were never affected**, and their values are unchanged.
    If you cached or saved a feature computed with either method on a *masked*
    skeleton, recompute it.

- **Knock-on effects of the `hops_to_root` bug.** It is used as a topological
  sort key, so anything ordering by it was also wrong on a masked skeleton:

    - `segments` / `segments_positional`: the **ordering of vertices within each
      segment** (distal → proximal). Segment *membership* and `segment_map` were
      never affected. Consumers that read a segment's endpoints — `segments_plus`,
      `segments_capped`, `segment_graph`, `expand_to_segment` and `resample` —
      inherited the bad ordering.
    - `algorithms.subtree_aggregate`, `algorithms.path_to_root_aggregate`, and
      `algorithms.branch_order` (built on the latter) returned wrong numbers
      rather than raising.

- **`algorithms.label_axon_from_synapse_flow(ntimes > 1)`** ran its inner split
  inside a mask context and mixed vertex ids with positional indices. With
  `extend_feature_to_segment=True` this raised `IndexError`; otherwise it could
  silently pick the wrong split vertex. Skeletons whose `vertex_index` is not
  simply `0..n-1` raised `KeyError` even with `ntimes=1`.

- **`PointCloudLayer.distance_to_root(as_positional=True)`** raised
  `IndexError` when called without explicit vertices; it passed vertex ids to a
  lookup that was reading them as positional indices.

- **Empty vertex selections raised.** `distance_to_root([])` failed with
  `IndexError: arrays used as indices must be of integer (or boolean) type`,
  because an empty Python list is `float64`. Empty selections now return empty
  results.

- **`plot_cell_multiview` produced a blank figure for a cell that is flat in
  one axis.** Panel sizes are derived from the data extent, so a zero extent
  (a planar reconstruction, or a skeleton stored with `z=0`) gave every panel a
  size of 0×0 and nothing was drawn. A degenerate axis is now padded to a small
  fraction of the largest extent and panel sizes are clamped to a minimum, as
  `single_panel_figure` already did, with a warning naming the flat axes.
  Because the panels keep an equal aspect ratio, a flat cell renders as a thin
  strip rather than being silently stretched. Cells with spread in all three
  axes are unaffected.

- **`import_legacy_meshwork(..., as_pcg_skel=True)`** emitted a spurious
  "`name` is ignored" warning: it passed a redundant column name alongside an
  already-named DataFrame. Imported features are unchanged.

### Internal

- `distance_to_root` and `hops_to_root` now share a single
  `SkeletonLayer._vertices_to_base_positional` converter, so the two cannot
  drift apart again.
- `DAGCache.distance_to_root` / `.hops_to_root` are renamed to
  `base_distance_to_root` / `base_hops_to_root`, making their index space
  explicit. The class docstring previously claimed all cached values were
  positional in the owning layer, which was untrue for exactly these two.
- New invariant tests in `tests/test_invariants.py` assert that every
  per-vertex accessor agrees across all equivalent call forms, on masked and
  unmasked skeletons alike, with pinned ground-truth values and a
  deliberately non-topological vertex ordering.

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