"""3D plotting for ossify using PyVista as the rendering backend."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from .base import Cell, GraphLayer, MeshLayer, PointCloudLayer, SkeletonLayer
from .plot_utils import (
    _map_value_to_colors,
    _resolve_color_to_array,
    _resolve_scalar_parameter,
    _resolve_size_with_transform,
)

try:
    import pyvista as pv
except ImportError as e:
    raise ImportError(
        "ossify.plot3d requires pyvista. Install with "
        "`pip install ossify[viz]` (or `uv add 'ossify[viz]'`)."
    ) from e

import matplotlib.colors as mcolors
from matplotlib.colors import Colormap

__all__ = [
    "plot_skeleton_3d",
    "plot_morphology_3d",
    "plot_points_3d",
    "plot_annotations_3d",
    "plot_mesh_3d",
    "plot_graph_3d",
    "plot_cell_3d",
    "add_colorbar_3d",
    "orbit_3d",
]


def _make_tube(
    poly: "pv.PolyData",
    radius: Union[float, np.ndarray],
    n_sides: int = 12,
) -> "pv.PolyData":
    """Build a tube from a polyline with scalar or per-vertex radius.

    For per-vertex arrays, VTK's tube filter maps a normalized scalar range
    ``[0, 1]`` to ``[radius, radius * radius_factor]``. We pick
    ``radius = min(radii)`` and ``radius_factor = max/min`` so the per-vertex
    radii are reproduced exactly.

    Parameters
    ----------
    poly : pv.PolyData
        Polyline geometry. Point data attached here will be propagated to
        the tube output points by VTK.
    radius : float or np.ndarray
        Scalar tube radius, or per-vertex radius array of length
        ``poly.n_points``.
    n_sides : int, default 12
        Number of sides on the tube cross-section.

    Returns
    -------
    pv.PolyData
        Triangulated tube mesh.
    """
    if isinstance(radius, np.ndarray):
        radii = radius.astype(float)
        r_min = float(np.min(radii))
        r_max = float(np.max(radii))
        if r_min <= 0:
            # Avoid degenerate zero-radius tubes
            r_min = max(r_max * 0.01, 1e-9)
            radii = np.clip(radii, r_min, None)
        if r_max <= r_min:
            return poly.tube(radius=r_min, n_sides=n_sides)
        poly.point_data["tube_radius"] = radii
        return poly.tube(
            radius=r_min,
            scalars="tube_radius",
            radius_factor=r_max / r_min,
            n_sides=n_sides,
        )
    return poly.tube(radius=float(radius), n_sides=n_sides)


def plot_skeleton_3d(
    skel: SkeletonLayer,
    colors: Optional[np.ndarray] = None,
    opacity: Optional[Union[float, np.ndarray]] = None,
    line_width: float = 2.0,
    tube_radius: Optional[Union[float, np.ndarray]] = None,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Render a skeleton as 3D lines or tubes using PyVista.

    This is the low-level skeleton renderer that accepts pre-computed arrays.
    For high-level rendering with feature-name resolution, use
    :func:`plot_morphology_3d`.

    Parameters
    ----------
    skel : SkeletonLayer
        Skeleton to render.
    colors : np.ndarray, optional
        Per-vertex color array, shape ``(N, 3)`` or ``(N, 4)`` (RGB or RGBA
        floats in ``[0, 1]``).
    opacity : float or np.ndarray, optional
        Overall opacity scalar, or per-vertex opacity array. Per-vertex
        arrays are interpolated by the tube filter when ``tube_radius`` is
        also a per-vertex array.
    line_width : float, default 2.0
        Line width in pixels. Used only when ``tube_radius`` is ``None``.
    tube_radius : float or np.ndarray, optional
        If a scalar float, renders the skeleton as tubes of that radius. If
        a per-vertex array (shape ``(N,)``), tube radius varies along the
        skeleton.
    plotter : pv.Plotter, optional
        Existing plotter to add actors to. A new plotter is created when
        ``None``.

    Returns
    -------
    pv.Plotter
        Plotter with the skeleton mesh added.
    """
    if plotter is None:
        plotter = pv.Plotter()

    # Build VTK polyline connectivity from cover_paths
    points = skel.vertices.astype(float)
    lines_connectivity: List[int] = []

    for path in skel.cover_paths_positional:
        path_end = path[-1]
        parent = skel.parent_node_array[path_end]
        if parent == -1:
            path_plus = path
        else:
            path_plus = np.concatenate([path, [parent]])
        lines_connectivity.append(len(path_plus))
        lines_connectivity.extend(path_plus.tolist())

    lines_array = np.array(lines_connectivity, dtype=np.intp)
    poly = pv.PolyData(points, lines=lines_array)

    mesh_kwargs: dict = {}

    # Attach per-vertex arrays BEFORE tube'ing so VTK's tube filter
    # interpolates them to its output points. This is the only way to keep
    # per-vertex opacity and color working under variable-radius tubes.
    if colors is not None:
        poly.point_data["colors"] = colors
        mesh_kwargs["scalars"] = "colors"
        mesh_kwargs["rgb"] = True
    if isinstance(opacity, np.ndarray):
        poly.point_data["opacity"] = opacity.astype(float)
        mesh_kwargs["opacity"] = "opacity"
    elif opacity is not None:
        mesh_kwargs["opacity"] = float(opacity)

    if tube_radius is not None:
        poly = _make_tube(poly, tube_radius)
        plotter.add_mesh(poly, **mesh_kwargs)
    else:
        plotter.add_mesh(poly, line_width=line_width, **mesh_kwargs)

    return plotter


def plot_morphology_3d(
    cell: Union[Cell, SkeletonLayer],
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    opacity: Optional[Union[str, np.ndarray, float]] = 1.0,
    line_width: float = 2.0,
    tube_radius: Optional[Union[float, str, np.ndarray]] = None,
    tube_radius_scale: Optional[float] = None,
    tube_radius_norm: Optional[Tuple[float, float]] = None,
    tube_radii: Optional[Tuple[float, float]] = None,
    root_marker: bool = False,
    root_radius: Optional[float] = None,
    root_color: Optional[Union[str, tuple]] = None,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Plot a skeleton in 3D with flexible color and radius styling.

    Parameters
    ----------
    cell : Cell or SkeletonLayer
        Cell or skeleton to render.
    color : str, np.ndarray, or tuple, optional
        Color specification. A string that matches a feature name is resolved
        to a per-vertex value array and mapped through *palette*. A string
        that does not match a feature is treated as a matplotlib color name.
        An array of shape ``(N,)`` is mapped through *palette*.
        An ``(N, 3)`` or ``(N, 4)`` array is used directly as RGB/RGBA.
    palette : str or dict, default "coolwarm"
        Colormap for mapping scalar color values to RGB. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed. A dict
        maps discrete values to colors.
    color_norm : tuple of float, optional
        ``(min, max)`` normalization range for continuous color mapping, in
        the original (pre-transform) value space.
    color_scale : {"log"} or None, optional
        Value transform applied before colormap projection. ``"log"``
        log-transforms the feature values so the colormap is distributed
        linearly in log-space. *color_norm* bounds are still specified in
        the original value space.
    opacity : str, np.ndarray, or float, default 1.0
        Opacity of the skeleton. Feature name, per-vertex array, or scalar.
    line_width : float, default 2.0
        Line width in pixels. Used only when ``tube_radius`` is ``None``.
    tube_radius : float, str, or np.ndarray, optional
        Render as tubes with this radius. A scalar float gives a uniform
        tube; a string resolves to a feature name; an array specifies
        per-vertex radii directly.
    tube_radius_scale : float, optional
        Multiplicative scale factor applied to all tube radius values after
        feature/array resolution.  Useful for unit conversion — e.g.
        ``tube_radius_scale=1/1000`` when the feature is in nm but the
        skeleton vertices are in µm.  Applied before *tube_radius_norm* and
        *tube_radii* rescaling.
    tube_radius_norm : tuple of float, optional
        ``(min, max)`` clip range for per-vertex tube radii. Values outside
        this range are clamped. When *tube_radii* is also given, the clipped
        values are further remapped to that output range; otherwise the
        original scale is preserved (clip only). Ignored when *tube_radius*
        is a scalar float.
    tube_radii : tuple of float, optional
        ``(min_radius, max_radius)`` output range for per-vertex tube radii
        after normalization. Ignored when *tube_radius* is a scalar float.
    root_marker : bool, default False
        If ``True``, place a sphere at the root vertex.
    root_radius : float, optional
        Radius for the root marker sphere. Falls back to *tube_radius* if it
        is a scalar float, then to ``1.0``.
    root_color : str or tuple, optional
        Color for the root marker sphere. Defaults to the root vertex's
        mapped color when ``None``.
    plotter : pv.Plotter, optional
        Existing plotter to add actors to.

    Returns
    -------
    pv.Plotter
        Plotter with the skeleton rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    if isinstance(cell, Cell):
        skel = cell.skeleton
    else:
        skel = cell

    # --- Resolve colors ---
    resolved = _resolve_color_to_array(
        color,
        skel,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
    )
    if resolved is None:
        colors_array = None
    elif isinstance(resolved, np.ndarray):
        colors_array = resolved
    else:
        # Single matplotlib color — tile across vertices so downstream
        # per-vertex consumers (root marker lookup, tube colors) work.
        single_color = mcolors.to_rgb(resolved)
        colors_array = np.tile(single_color, (skel.n_vertices, 1))

    # --- Resolve tube_radius ---
    # Three-step pipeline: (1) feature/array lookup, (2) unit scale, (3) norm+remap
    resolved_tube_radius: Optional[Union[float, np.ndarray]] = None
    if tube_radius is not None:
        if isinstance(tube_radius, (int, float)) and not isinstance(tube_radius, bool):
            resolved_tube_radius = float(tube_radius)
            if tube_radius_scale is not None:
                resolved_tube_radius *= tube_radius_scale
        else:
            # Step 1: resolve feature name or array to raw physical values.
            # Deliberately bypass _resolve_scalar_parameter here because that
            # function normalizes strings to [0, 1], which would break
            # tube_radius_scale (we need actual units, not normalized values).
            if isinstance(tube_radius, str):
                raw: Optional[np.ndarray] = np.asarray(
                    skel.get_feature(tube_radius), dtype=float
                )
            else:
                raw = np.asarray(tube_radius, dtype=float)
            # Step 2: unit conversion
            if tube_radius_scale is not None:
                raw = raw * tube_radius_scale
            # Step 3: optional norm + remap
            # When only tube_radius_norm is given (no tube_radii), treat it as
            # a clip range only — preserve the original scale rather than
            # collapsing to [0, 1], which would produce degenerate tube radii.
            if raw is not None and (
                tube_radius_norm is not None or tube_radii is not None
            ):
                effective_out_range = (
                    tube_radii if tube_radii is not None else tube_radius_norm
                )
                resolved_tube_radius = _resolve_scalar_parameter(
                    raw,
                    skel.n_vertices,
                    norm=tube_radius_norm,
                    out_range=effective_out_range,
                )
            else:
                resolved_tube_radius = raw

    # --- Resolve opacity ---
    # Numeric scalars stay scalar; feature names and array-likes resolve to a
    # per-vertex array that the tube filter will interpolate.
    opacity_out: Optional[Union[float, np.ndarray]] = None
    if isinstance(opacity, (int, float)) and not isinstance(opacity, bool):
        opacity_out = float(opacity)
    elif opacity is not None:
        opacity_out = _resolve_scalar_parameter(opacity, skel.n_vertices, layer=skel)

    # When root_marker is shown, replace the root vertex's tube radius with
    # the average of its immediate children so the tube doesn't bulge at the
    # soma (the sphere handles the visual root marker instead).
    if (
        root_marker
        and isinstance(resolved_tube_radius, np.ndarray)
        and skel.root_positional is not None
    ):
        root_pos = skel.root_positional
        child_mask = skel.parent_node_array == root_pos
        if np.any(child_mask):
            resolved_tube_radius = resolved_tube_radius.copy()
            resolved_tube_radius[root_pos] = float(
                np.mean(resolved_tube_radius[child_mask])
            )

    plotter = plot_skeleton_3d(
        skel=skel,
        colors=colors_array,
        opacity=opacity_out,
        line_width=line_width,
        tube_radius=resolved_tube_radius,
        plotter=plotter,
    )

    if root_marker and skel.root_location is not None:
        # Determine sphere radius
        if root_radius is not None:
            r = float(root_radius)
        elif isinstance(resolved_tube_radius, np.ndarray):
            r = float(resolved_tube_radius[skel.root_positional])
        elif isinstance(resolved_tube_radius, (int, float)):
            r = float(resolved_tube_radius)
        else:
            r = 1.0

        # Determine sphere color
        if root_color is not None:
            sphere_color = root_color
        elif colors_array is not None:
            root_rgb = colors_array[skel.root_positional]
            sphere_color = tuple(float(v) for v in root_rgb[:3])
        else:
            sphere_color = "white"

        sphere = pv.Sphere(radius=r, center=skel.root_location.tolist())
        plotter.add_mesh(sphere, color=sphere_color)

    return plotter


def plot_points_3d(
    points: np.ndarray,
    sizes: Optional[Union[float, np.ndarray]] = None,
    colors: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    opacity: float = 1.0,
    plotter: Optional["pv.Plotter"] = None,
    **kwargs,
) -> "pv.Plotter":
    """Render a point cloud as spheres in 3D.

    Per-point colors are pre-mapped to RGB before glyphing and attached
    to the source cloud's ``point_data``. The glyph filter propagates the
    array to every generated vertex, and ``add_mesh`` reads it by name.
    Avoids PyVista's ``cmap``/``clim`` forwarding, which can pick the
    wrong colormap when the glyph filter rebuilds active scalars.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape ``(N, 3)``.
    sizes : float or np.ndarray, optional
        Sphere radius. A scalar gives uniform radius; an array of shape
        ``(N,)`` gives per-point radius. Defaults to a bounding-box-relative
        radius when ``None`` (visible at any coordinate scale).
    colors : str, np.ndarray, or tuple, optional
        Color specification. May be:

        - A matplotlib color string or ``(r, g, b)`` tuple — uniform color
          for all points.
        - A 1-D array of shape ``(N,)`` — mapped through *palette*.
        - An ``(N, 3)`` or ``(N, 4)`` array — used directly as RGB(A).

        Discrete-label coloring is supported by passing the labels as a 1-D
        array and a dict *palette* mapping labels to colors.
    palette : str or dict, default "coolwarm"
        Colormap for scalar color mapping. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps from third-party packages
        (e.g. cmocean, colorcet, cmcrameri). A dict maps discrete values to
        colors.
    color_norm : tuple of float, optional
        ``(min, max)`` normalization range for continuous color mapping.
    opacity : float, default 1.0
        Overall opacity.
    plotter : pv.Plotter, optional
        Existing plotter to add actors to.
    **kwargs
        Additional keyword arguments forwarded to ``plotter.add_mesh``.

    Returns
    -------
    pv.Plotter
        Plotter with the point cloud rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    pts = np.asarray(points, dtype=float)
    cloud = pv.PolyData(pts)

    # Normalize array-like inputs to ndarray (pandas Series, lists, etc.).
    if sizes is not None and not isinstance(sizes, (int, float, np.ndarray)):
        sizes = np.asarray(sizes, dtype=float)
    if colors is not None and not isinstance(colors, (str, tuple, np.ndarray)):
        colors = np.asarray(colors)

    # Resolve colors to either a scalar matplotlib color or a pre-mapped
    # RGB(A) array. Scalar arrays go through the palette here so the glyph
    # filter sees per-point RGB and we never rely on cmap/clim forwarding.
    rgb_array: Optional[np.ndarray] = None
    scalar_color: Optional[Union[str, tuple]] = None
    if isinstance(colors, np.ndarray):
        if colors.ndim == 1:
            rgb_array = _map_value_to_colors(
                colors, colormap=palette, color_norm=color_norm
            )
        else:
            rgb_array = colors
    elif isinstance(colors, (str, tuple)):
        scalar_color = colors

    # Default sphere radius: ~0.2% of the smallest positive bounding-box
    # extent, so spheres are visible at any coordinate scale (nm/µm/vx).
    # Conservative on purpose — annotation clouds often have hundreds to
    # thousands of points, and a larger default crowds the rendering.
    # Callers can scale up via the ``sizes`` argument when warranted.
    if sizes is None:
        bbox = pts.max(0) - pts.min(0)
        positive_dims = bbox[bbox > 0]
        ref = float(positive_dims.min()) if len(positive_dims) > 0 else 1.0
        sizes = np.full(len(pts), ref * 0.002)
    elif not isinstance(sizes, np.ndarray):
        sizes = np.full(len(pts), float(sizes))

    mesh_kwargs: dict = {"opacity": opacity, **kwargs}

    # Build sphere glyphs in world units. We deliberately avoid PyVista's
    # render_points_as_spheres path: it takes pixel-unit `point_size`,
    # which would force every caller to know whether their data is nm or
    # µm. Glyphs interpret `sizes` as world-unit radii directly.
    cloud.point_data["radius"] = sizes.astype(float)
    if rgb_array is not None:
        # Attach colors before glyphing — VTK propagates point_data from
        # the source cloud to each generated glyph vertex (this is the
        # default behavior since vtkGlyph3D's PassPointData was made
        # on-by-default; we pin pyvista>=0.46 / VTK>=9.5 to guarantee it).
        cloud.point_data["colors"] = rgb_array
    # radius=1.0 makes the glyph scale factor equal the output radius
    # directly (default radius=0.5 would halve every size).
    geom = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
    glyphs = cloud.glyph(geom=geom, scale="radius", orient=False)

    if rgb_array is not None:
        mesh_kwargs["scalars"] = "colors"
        mesh_kwargs["rgb"] = True
    elif scalar_color is not None:
        mesh_kwargs["color"] = scalar_color

    plotter.add_mesh(glyphs, **mesh_kwargs)
    return plotter


def plot_annotations_3d(
    annotation: PointCloudLayer,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    opacity: float = 1.0,
    size: Optional[Union[str, np.ndarray, float]] = None,
    size_norm: Optional[Tuple[float, float]] = None,
    size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    sizes: Optional[Tuple[float, float]] = None,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Render a :class:`PointCloudLayer` annotation as 3D spheres.

    Parameters
    ----------
    annotation : PointCloudLayer
        Annotation layer to render.
    color : str, np.ndarray, or tuple, optional
        Color specification. A string matching a feature name resolves to a
        per-point value array mapped through *palette*. Otherwise treated as
        a matplotlib color.
    palette : str or dict, default "coolwarm"
        Colormap for scalar color mapping. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed. A dict
        maps discrete values to colors.
    color_norm : tuple of float, optional
        ``(min, max)`` clipping range for color mapping, in the original
        (pre-transform) value space.
    color_scale : {"log"} or None, optional
        Value transform applied before colormap projection. ``"log"``
        log-transforms the feature values so the colormap is distributed
        linearly in log-space. *color_norm* bounds are still specified in
        the original value space and are converted internally.
    opacity : float, default 1.0
        Overall opacity.
    size : str, np.ndarray, or float, optional
        Sphere radius specification. A string resolves to a feature name.
        An array or float is used directly or rescaled to *sizes*.
    size_norm : tuple of float, optional
        ``(min, max)`` clipping range for size mapping, in the original
        (pre-transform) value space.
    size_scale : {"log", "sqrt", "cbrt"} or None, optional
        Value transform applied before size normalization. ``"log"``
        log-transforms values (linear spacing in log-space); ``"sqrt"``
        takes the square root (useful when the feature is a cross-sectional
        area and radius ∝ √area); ``"cbrt"`` takes the cube root (useful
        when the feature is a volume and radius ∝ ∛volume). *size_norm*
        bounds are always specified in the original value space.
    sizes : tuple of float, optional
        ``(min_radius, max_radius)`` output range for size rescaling, in
        the same world units as the annotation vertices.  When ``None``
        (default), estimated from the annotation's bounding box
        (approximately 0.1 %–0.8 % of the smallest spatial extent) so
        spheres are visible regardless of coordinate scale (nm/µm/voxels).
        Pass an explicit tuple to make markers bolder.
    plotter : pv.Plotter, optional
        Existing plotter to add actors to.

    Returns
    -------
    pv.Plotter
        Plotter with the annotation rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    vertices = annotation.vertices

    # Auto-compute the output radius range from the annotation bounding
    # box when not provided. Conservative on purpose: synapse clouds are
    # often dense (hundreds to thousands of points), and larger defaults
    # crowd the rendering. Pass ``sizes`` explicitly when you want bolder
    # markers.
    if sizes is None:
        bbox = vertices.max(0) - vertices.min(0)
        positive_dims = bbox[bbox > 0]
        ref = float(positive_dims.min()) if len(positive_dims) > 0 else 1.0
        sizes = (ref * 0.001, ref * 0.008)

    # Resolve color through the shared pipeline. The result is either a
    # pre-mapped (N, 3/4) RGB(A) array, a scalar matplotlib color, or None;
    # plot_points_3d handles all three directly without needing palette
    # info, so we no longer forward palette/color_norm downstream.
    resolved_color = _resolve_color_to_array(
        color,
        annotation,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
    )

    resolved_size = _resolve_size_with_transform(
        size,
        len(vertices),
        scale=size_scale,
        norm=size_norm,
        out_range=sizes,
        layer=annotation,
    )

    return plot_points_3d(
        points=vertices,
        sizes=resolved_size,
        colors=resolved_color,
        opacity=opacity,
        plotter=plotter,
    )


def plot_mesh_3d(
    mesh: MeshLayer,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    opacity: float = 1.0,
    show_edges: bool = False,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Render a :class:`MeshLayer` as a 3D surface using PyVista.

    Parameters
    ----------
    mesh : MeshLayer
        Mesh to render.
    color : str, np.ndarray, or tuple, optional
        Color specification. May be:

        - A matplotlib color string (e.g. ``"steelblue"``) for a uniform surface color.
        - A tuple ``(r, g, b)`` or ``(r, g, b, a)`` for a uniform RGBA color.
        - A 1-D array of length ``n_vertices`` for per-vertex scalar coloring
          (mapped through *palette*).
        - A 2-D ``(n_vertices, 3)`` uint8 RGB array for direct per-vertex colors.
        - A feature name string present in the mesh to use those values for coloring.

        If ``None``, PyVista's default surface color is used.
    palette : str or dict, default "coolwarm"
        Colormap for scalar color mapping. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed.
    color_norm : tuple of float, optional
        ``(min, max)`` normalization range for scalar color mapping. Values
        outside this range are clamped.
    color_scale : {"log"} or None, optional
        Transform applied to scalar values before color mapping.
        ``"log"`` applies a natural-log transform; *color_norm* bounds are
        interpreted in the **original** (pre-transform) space and converted
        internally.
    opacity : float, default 1.0
        Surface opacity. Values less than 1.0 make the mesh semi-transparent,
        useful when rendering alongside a skeleton.
    show_edges : bool, default False
        If ``True``, draw the mesh wireframe edges on top of the surface.
    plotter : pv.Plotter, optional
        Existing plotter to add the mesh to. If ``None``, a new plotter is
        created.

    Returns
    -------
    pv.Plotter
        Plotter with the mesh surface rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    vertices = mesh.vertices.astype(float)
    faces_pos = mesh.faces_positional  # shape (M, 3)

    # Build VTK face connectivity array: [3, f0, f1, f2, 3, g0, g1, g2, ...]
    M = len(faces_pos)
    vtk_faces = np.empty(M * 4, dtype=np.intp)
    vtk_faces[0::4] = 3
    vtk_faces[1::4] = faces_pos[:, 0]
    vtk_faces[2::4] = faces_pos[:, 1]
    vtk_faces[3::4] = faces_pos[:, 2]
    poly = pv.PolyData(vertices, faces=vtk_faces)

    # --- Resolve colors ---
    resolved_color = _resolve_color_to_array(
        color,
        mesh,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
    )

    mesh_kwargs: dict = {"opacity": opacity, "show_edges": show_edges}

    if isinstance(resolved_color, np.ndarray):
        poly.point_data["colors"] = resolved_color
        mesh_kwargs["scalars"] = "colors"
        mesh_kwargs["rgb"] = True
    elif resolved_color is not None:
        mesh_kwargs["color"] = resolved_color

    plotter.add_mesh(poly, **mesh_kwargs)
    return plotter


def plot_graph_3d(
    graph: GraphLayer,
    # Node (vertex) styling
    node_color: Optional[Union[str, np.ndarray, tuple]] = None,
    node_palette: Union[str, dict] = "coolwarm",
    node_color_norm: Optional[Tuple[float, float]] = None,
    node_color_scale: Optional[Literal["log"]] = None,
    node_size: Optional[Union[str, np.ndarray, float]] = None,
    node_size_norm: Optional[Tuple[float, float]] = None,
    node_size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    node_sizes: Optional[Tuple[float, float]] = None,
    node_opacity: float = 1.0,
    show_nodes: bool = True,
    # Edge styling
    edge_color: Optional[Union[str, np.ndarray, tuple]] = None,
    edge_palette: Union[str, dict] = "coolwarm",
    edge_color_norm: Optional[Tuple[float, float]] = None,
    edge_color_scale: Optional[Literal["log"]] = None,
    edge_radius: Optional[Union[str, float]] = None,
    edge_radius_norm: Optional[Tuple[float, float]] = None,
    edge_radius_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    edge_radii: Optional[Tuple[float, float]] = None,
    edge_opacity: float = 1.0,
    line_width: float = 2.0,
    show_edges: bool = True,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Render a :class:`GraphLayer` as sphere glyphs at nodes with edges as
    lines or tubes.

    All color and size features are per-vertex.  When a feature name is given
    for an edge property, PyVista interpolates that value along each tube,
    creating a smooth gradient between the two endpoint values.

    Parameters
    ----------
    graph : GraphLayer
        Graph to render.
    node_color : str, np.ndarray, or tuple, optional
        Color specification for node spheres. A feature name string resolves
        to per-vertex values mapped through *node_palette*. Otherwise treated
        as a matplotlib color.
    node_palette : str or dict, default "coolwarm"
        Colormap for node scalar color mapping. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed. A dict
        maps discrete values to colors.
    node_color_norm : tuple of float, optional
        ``(min, max)`` clipping range for node color mapping, in the original
        (pre-transform) value space.
    node_color_scale : {"log"} or None, optional
        Value transform before node colormap projection. ``"log"``
        log-transforms values; *node_color_norm* stays in original space.
    node_size : str, np.ndarray, or float, optional
        Sphere radius for each node. A feature name string resolves to
        per-vertex values rescaled to *node_sizes*.
    node_size_norm : tuple of float, optional
        ``(min, max)`` clipping range for node size mapping, in the original
        (pre-transform) value space.
    node_size_scale : {"log", "sqrt", "cbrt"} or None, optional
        Value transform before node size normalization.
    node_sizes : tuple of float, optional
        ``(min_radius, max_radius)`` output range for node sphere radii, in
        the same world units as the vertex coordinates.  When ``None``
        (default), a sensible range is estimated from the graph's bounding
        box (approximately 0.5 %–3 % of the smallest spatial extent).
    node_opacity : float, default 1.0
        Opacity for node spheres.
    show_nodes : bool, default True
        If ``False``, suppress node sphere rendering.
    edge_color : str, np.ndarray, or tuple, optional
        Color specification for edges. A feature name string resolves to
        per-vertex values; each tube is then colored by smoothly
        interpolating between the two endpoint values. Otherwise treated as
        a uniform matplotlib color.
    edge_palette : str or dict, default "coolwarm"
        Colormap for edge scalar color mapping (see *node_palette*).
    edge_color_norm : tuple of float, optional
        ``(min, max)`` clipping range for edge color mapping.
    edge_color_scale : {"log"} or None, optional
        Value transform before edge colormap projection.
    edge_radius : str or float, optional
        Tube radius for edges. A scalar float gives a uniform tube radius.
        A feature name string resolves to per-vertex values rescaled to
        *edge_radii*; the tube radius interpolates between the two endpoint
        values along each edge. When ``None``, edges are rendered as lines
        of *line_width* pixels.
    edge_radius_norm : tuple of float, optional
        ``(min, max)`` clipping range for edge radius mapping.
    edge_radius_scale : {"log", "sqrt", "cbrt"} or None, optional
        Value transform before edge radius normalization.
    edge_radii : tuple of float, optional
        ``(min_radius, max_radius)`` output range for per-vertex edge radii,
        in world units.  When ``None`` (default), estimated from the graph's
        bounding box (approximately 0.1 %–1 % of the smallest extent).
    edge_opacity : float, default 1.0
        Opacity for edges.
    line_width : float, default 2.0
        Line width in pixels. Used only when *edge_radius* is ``None``.
    show_edges : bool, default True
        If ``False``, suppress edge rendering.
    plotter : pv.Plotter, optional
        Existing plotter to add actors to.

    Returns
    -------
    pv.Plotter
        Plotter with the graph rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    vertices = graph.vertices
    N = len(vertices)

    # Auto-compute size ranges from the bounding box when not provided.
    # This ensures defaults are appropriate regardless of coordinate scale
    # (nm, µm, voxels, etc.).
    if node_sizes is None or edge_radii is None:
        bbox = vertices.max(0) - vertices.min(0)
        positive_dims = bbox[bbox > 0]
        ref = float(positive_dims.min()) if len(positive_dims) > 0 else 1.0
        if node_sizes is None:
            node_sizes = (ref * 0.005, ref * 0.03)
        if edge_radii is None:
            edge_radii = (ref * 0.001, ref * 0.01)

    # ------------------------------------------------------------------ nodes
    if show_nodes:
        resolved_node_color = _resolve_color_to_array(
            node_color,
            graph,
            palette=node_palette,
            color_norm=node_color_norm,
            color_scale=node_color_scale,
        )
        resolved_node_size = _resolve_size_with_transform(
            node_size,
            N,
            scale=node_size_scale,
            norm=node_size_norm,
            out_range=node_sizes,
            layer=graph,
        )

        plotter = plot_points_3d(
            points=vertices,
            sizes=resolved_node_size,
            colors=resolved_node_color,
            opacity=node_opacity,
            plotter=plotter,
        )

    # ------------------------------------------------------------------ edges
    if show_edges:
        edge_idx = graph.edges_positional  # (M, 2)
        M = len(edge_idx)
        if M == 0:
            return plotter

        # Build VTK line connectivity: [2, i0, j0, 2, i1, j1, ...]
        lines = np.empty(M * 3, dtype=np.intp)
        lines[0::3] = 2
        lines[1::3] = edge_idx[:, 0]
        lines[2::3] = edge_idx[:, 1]
        poly = pv.PolyData(vertices.astype(float), lines=lines)

        # Resolve edge color — per-vertex values get pre-mapped to RGB so the
        # tube filter can interpolate them naturally between endpoints.
        edge_mesh_kwargs: dict = {"opacity": edge_opacity}
        resolved_edge_color = _resolve_color_to_array(
            edge_color,
            graph,
            palette=edge_palette,
            color_norm=edge_color_norm,
            color_scale=edge_color_scale,
        )
        if isinstance(resolved_edge_color, np.ndarray):
            poly.point_data["colors"] = resolved_edge_color
        elif isinstance(resolved_edge_color, (str, tuple)):
            edge_mesh_kwargs["color"] = resolved_edge_color

        # Resolve edge radius — scalar floats pass straight through; feature
        # names and arrays go through the transform pipeline.
        if isinstance(edge_radius, (int, float)) and not isinstance(edge_radius, bool):
            resolved_edge_radius: Optional[Union[float, np.ndarray]] = float(
                edge_radius
            )
        else:
            resolved_edge_radius = _resolve_size_with_transform(
                edge_radius,
                N,
                scale=edge_radius_scale,
                norm=edge_radius_norm,
                out_range=edge_radii,
                layer=graph,
            )

        # Render
        # Colors were attached to poly.point_data above; tube propagates them.
        if "colors" in poly.point_data:
            edge_mesh_kwargs["scalars"] = "colors"
            edge_mesh_kwargs["rgb"] = True

        if resolved_edge_radius is not None:
            poly = _make_tube(poly, resolved_edge_radius)
            plotter.add_mesh(poly, **edge_mesh_kwargs)
        else:
            plotter.add_mesh(poly, line_width=line_width, **edge_mesh_kwargs)

    return plotter


def plot_cell_3d(
    cell: Cell,
    # Skeleton styling
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    opacity: float = 1.0,
    line_width: float = 2.0,
    tube_radius: Optional[Union[float, str, np.ndarray]] = None,
    tube_radius_scale: Optional[float] = None,
    tube_radius_norm: Optional[Tuple[float, float]] = None,
    tube_radii: Optional[Tuple[float, float]] = None,
    root_marker: bool = False,
    # Mesh styling
    mesh: bool = False,
    mesh_color: Optional[Union[str, np.ndarray, tuple]] = None,
    mesh_palette: Union[str, dict] = "coolwarm",
    mesh_color_norm: Optional[Tuple[float, float]] = None,
    mesh_color_scale: Optional[Literal["log"]] = None,
    mesh_opacity: float = 0.3,
    mesh_show_edges: bool = False,
    # Synapse control
    synapses: Literal["pre", "post", "both", True, False] = False,
    # Pre-synaptic annotation styling
    pre_anno: str = "pre_syn",
    pre_color: Optional[Union[str, np.ndarray, tuple]] = None,
    pre_palette: Union[str, dict] = "coolwarm",
    pre_color_norm: Optional[Tuple[float, float]] = None,
    # Post-synaptic annotation styling
    post_anno: str = "post_syn",
    post_color: Optional[Union[str, np.ndarray, tuple]] = None,
    post_palette: Union[str, dict] = "coolwarm",
    post_color_norm: Optional[Tuple[float, float]] = None,
    # Shared synapse styling
    syn_opacity: float = 1.0,
    syn_color_scale: Optional[Literal["log"]] = None,
    syn_size: Optional[Union[str, np.ndarray, float]] = None,
    syn_size_norm: Optional[Tuple[float, float]] = None,
    syn_size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    syn_sizes: Optional[Tuple[float, float]] = None,
    plotter: Optional["pv.Plotter"] = None,
) -> "pv.Plotter":
    """Render a :class:`Cell` — skeleton and optional annotations — in 3D.

    Parameters
    ----------
    cell : Cell
        Cell to render.
    color : str, np.ndarray, or tuple, optional
        Skeleton color specification (see :func:`plot_morphology_3d`).
    palette : str or dict, default "coolwarm"
        Colormap for skeleton color mapping. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed. A dict
        maps discrete values to colors.
    color_norm : tuple of float, optional
        ``(min, max)`` normalization range for skeleton color.
    color_scale : {"log"} or None, optional
        Value transform applied before skeleton colormap projection (see
        :func:`plot_morphology_3d`).
    opacity : float, default 1.0
        Skeleton opacity.
    line_width : float, default 2.0
        Skeleton line width (used when ``tube_radius`` is ``None``).
    tube_radius : float, str, or np.ndarray, optional
        Skeleton tube radius (see :func:`plot_morphology_3d`).
    tube_radius_scale : float, optional
        Multiplicative unit-conversion factor (e.g. ``1/1000`` for nm → µm).
    tube_radius_norm : tuple of float, optional
        ``(min, max)`` input range / cap for per-vertex tube radii.
    tube_radii : tuple of float, optional
        ``(min_radius, max_radius)`` output range for per-vertex tube radii.
    root_marker : bool, default False
        If ``True``, mark the root vertex with a sphere.
    mesh : bool, default False
        If ``True`` and *cell* has a mesh layer, render the mesh surface.
    mesh_color : str, np.ndarray, or tuple, optional
        Color specification for the mesh surface (see :func:`plot_mesh_3d`).
    mesh_palette : str or dict, default "coolwarm"
        Colormap for mesh scalar color mapping (see *palette*).
    mesh_color_norm : tuple of float, optional
        ``(min, max)`` clipping range for mesh color mapping.
    mesh_color_scale : {"log"} or None, optional
        Value transform for mesh color mapping (see :func:`plot_mesh_3d`).
    mesh_opacity : float, default 0.3
        Opacity of the mesh surface. Defaults to semi-transparent so the
        skeleton remains visible underneath.
    mesh_show_edges : bool, default False
        If ``True``, draw wireframe edges on the mesh surface.
    synapses : {"pre", "post", "both"} or bool, default False
        Which synapse layers to render. ``False`` renders none; ``True`` or
        ``"both"`` renders both pre- and post-synaptic; ``"pre"``/``"post"``
        renders only that side.
    pre_anno : str, default "pre_syn"
        Name of the pre-synaptic annotation layer in *cell*.
    pre_color : str, np.ndarray, or tuple, optional
        Color specification for the pre-synaptic layer.
    pre_palette : str or dict, default "coolwarm"
        Colormap for pre-synaptic scalar color mapping (see *palette*).
    pre_color_norm : tuple of float, optional
        ``(min, max)`` clipping range for pre-synaptic color mapping.
    post_anno : str, default "post_syn"
        Name of the post-synaptic annotation layer in *cell*.
    post_color : str, np.ndarray, or tuple, optional
        Color specification for the post-synaptic layer.
    post_palette : str or dict, default "coolwarm"
        Colormap for post-synaptic scalar color mapping (see *palette*).
    post_color_norm : tuple of float, optional
        ``(min, max)`` clipping range for post-synaptic color mapping.
    syn_opacity : float, default 1.0
        Opacity for all synapse spheres.
    syn_color_scale : {"log"} or None, optional
        Value transform for synapse color mapping (see
        :func:`plot_annotations_3d`).
    syn_size : str, np.ndarray, or float, optional
        Sphere radius for all synapse layers.
    syn_size_norm : tuple of float, optional
        ``(min, max)`` clipping range for synapse size mapping.
    syn_size_scale : {"log", "sqrt", "cbrt"} or None, optional
        Value transform for synapse size mapping (see
        :func:`plot_annotations_3d`).
    syn_sizes : tuple of float, optional
        ``(min_radius, max_radius)`` output range for synapse sphere radii,
        in the same world units as the cell vertices. When ``None``
        (default), estimated from the synapse bounding box so spheres are
        visible regardless of coordinate scale (nm/µm/voxels).
    plotter : pv.Plotter, optional
        Existing plotter to add actors to.

    Returns
    -------
    pv.Plotter
        Plotter with the cell rendered.
    """
    if plotter is None:
        plotter = pv.Plotter()

    plotter = plot_morphology_3d(
        cell=cell,
        color=color,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
        opacity=opacity,
        line_width=line_width,
        tube_radius=tube_radius,
        tube_radius_scale=tube_radius_scale,
        tube_radius_norm=tube_radius_norm,
        tube_radii=tube_radii,
        root_marker=root_marker,
        plotter=plotter,
    )

    if mesh and cell.mesh is not None:
        plotter = plot_mesh_3d(
            cell.mesh,
            color=mesh_color,
            palette=mesh_palette,
            color_norm=mesh_color_norm,
            color_scale=mesh_color_scale,
            opacity=mesh_opacity,
            show_edges=mesh_show_edges,
            plotter=plotter,
        )

    if synapses is not False:
        _syn_kwargs: dict = dict(
            opacity=syn_opacity,
            color_scale=syn_color_scale,
            size=syn_size,
            size_norm=syn_size_norm,
            size_scale=syn_size_scale,
            sizes=syn_sizes,
            plotter=plotter,
        )

        if synapses in ("pre", "both") or synapses is True:
            if pre_anno in cell.annotations.names:
                plotter = plot_annotations_3d(
                    cell.annotations[pre_anno],
                    color=pre_color,
                    palette=pre_palette,
                    color_norm=pre_color_norm,
                    **_syn_kwargs,
                )
                _syn_kwargs["plotter"] = plotter

        if synapses in ("post", "both") or synapses is True:
            if post_anno in cell.annotations.names:
                plotter = plot_annotations_3d(
                    cell.annotations[post_anno],
                    color=post_color,
                    palette=post_palette,
                    color_norm=post_color_norm,
                    **_syn_kwargs,
                )

    return plotter


def add_colorbar_3d(
    plotter: "pv.Plotter",
    palette: Union[str, Colormap] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
    position_x: float = 0.85,
    position_y: float = 0.05,
    width: float = 0.1,
    height: float = 0.9,
    n_labels: int = 5,
    fmt: str = "%.3g",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    n_colors: int = 256,
    **scalar_bar_kwargs,
) -> "pv.Plotter":
    """Add a colorbar to a 3D plotter.

    Because ossify's plot functions pre-map scalars to RGB before they
    reach VTK, PyVista has no mapper to build a scalar bar from. This
    helper creates a :class:`vtkScalarBarActor` directly from a
    :class:`vtkLookupTable` populated by the matplotlib colormap, then
    attaches it as a 2D viewport overlay. The actor has no spatial
    bounds, so the scene camera is unaffected.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter to add the colorbar to.
    palette : str or Colormap, default "coolwarm"
        Colormap name (any matplotlib-registered name) or Colormap object.
    color_norm : tuple of float, optional
        ``(min, max)`` range for the colorbar.  When ``None``, the bar spans
        ``[0, 1]``.
    label : str, optional
        Title text displayed alongside the colorbar.
    position_x : float, default 0.85
        Horizontal position of the colorbar (0 = left edge, 1 = right edge).
    position_y : float, default 0.05
        Vertical position of the colorbar bottom (0 = bottom, 1 = top).
    width : float, default 0.1
        Width of the colorbar as a fraction of the window.
    height : float, default 0.9
        Height of the colorbar as a fraction of the window.
    n_labels : int, default 5
        Number of tick labels along the bar.
    fmt : str, default "%.3g"
        ``printf``-style format string for tick labels.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Bar orientation.
    n_colors : int, default 256
        Number of colors sampled from *palette* into the lookup table.
    **scalar_bar_kwargs
        Extra keyword arguments passed to :class:`vtkScalarBarActor`. Keys
        are converted from ``snake_case`` to ``SetPascalCase``; unknown
        keys are silently ignored.

    Returns
    -------
    pv.Plotter
        The same plotter, with the colorbar added.

    Examples
    --------
    >>> pl = plot_morphology_3d(cell, color="strahler_order", palette="viridis")
    >>> add_colorbar_3d(pl, palette="viridis", color_norm=(1, 7), label="Strahler order")
    """
    import matplotlib.pyplot as plt
    import vtk

    vmin, vmax = color_norm if color_norm is not None else (0.0, 1.0)
    cmap = plt.get_cmap(palette) if isinstance(palette, str) else palette

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    lut.SetRange(vmin, vmax)
    denom = max(n_colors - 1, 1)
    for i in range(n_colors):
        r, g, b, a = cmap(i / denom)
        lut.SetTableValue(i, r, g, b, a)
    lut.Build()

    sbar = vtk.vtkScalarBarActor()
    sbar.SetLookupTable(lut)
    sbar.SetTitle(label or "")
    sbar.SetNumberOfLabels(n_labels)
    sbar.SetLabelFormat(fmt)
    if orientation == "horizontal":
        sbar.SetOrientationToHorizontal()
    else:
        sbar.SetOrientationToVertical()

    sbar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    sbar.GetPositionCoordinate().SetValue(position_x, position_y)
    sbar.SetWidth(width)
    sbar.SetHeight(height)

    # Best-effort forwarding for any extra VTK setters the caller wants.
    for key, val in scalar_bar_kwargs.items():
        setter_name = "Set" + "".join(part.capitalize() for part in key.split("_"))
        setter = getattr(sbar, setter_name, None)
        if setter is not None:
            setter(val)

    plotter.add_actor(sbar, reset_camera=False)
    return plotter


def orbit_3d(
    plotter: "pv.Plotter",
    output: Optional[str] = None,
    n_frames: int = 60,
    elevation: float = 0.0,
    factor: float = 2.0,
    framerate: int = 24,
    viewup: Optional[Tuple[float, float, float]] = None,
    close: bool = True,
) -> "pv.Plotter":
    """Orbit the camera around the scene, optionally saving to a file.

    Generates a circular orbital path and either plays the orbit
    interactively or writes frames to a GIF / MP4 file.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter with actors already added.
    output : str, optional
        Path to write the animation to (``.gif`` or ``.mp4``).  Requires
        ``imageio`` to be installed.  When ``None``, the orbit is displayed
        interactively (or runs off-screen silently).
    n_frames : int, default 60
        Number of frames in the full orbit.
    elevation : float, default 0.0
        Camera elevation angle in degrees above the XY plane.
    factor : float, default 2.0
        Orbit radius factor relative to the scene bounding box.  Larger
        values move the camera further from the scene center.
    framerate : int, default 24
        Frames per second for file output.  Ignored when *output* is
        ``None``.
    viewup : tuple of float, optional
        Camera "up" direction as ``(x, y, z)``.  When ``None``, PyVista's
        default is used.
    close : bool, default True
        If ``True``, close the plotter after the orbit completes (default
        for one-shot animation rendering). Pass ``close=False`` to keep the
        plotter alive — e.g. to add more actors and orbit again, or to
        capture screenshots after the orbit.

    Returns
    -------
    pv.Plotter
        The plotter passed in. By default the plotter is closed and not
        reusable; pass ``close=False`` to keep it alive.

    Examples
    --------
    Interactive orbit:

    >>> pl = plot_cell_3d(cell, color="red", tube_radius=500)
    >>> orbit_3d(pl)

    Save to GIF:

    >>> pl = plot_cell_3d(cell, color="red", tube_radius=500)
    >>> orbit_3d(pl, output="neuron_orbit.gif", n_frames=90)
    """
    path_kwargs: dict = {"factor": factor, "n_points": n_frames}
    if viewup is not None:
        path_kwargs["viewup"] = viewup

    # Set camera elevation before generating the orbital path so the
    # path is computed at the desired viewing angle.
    if elevation != 0.0:
        plotter.camera.elevation = elevation

    path = plotter.generate_orbital_path(**path_kwargs)

    if output is not None:
        ext = output.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            plotter.open_gif(output)
        else:
            plotter.open_movie(output, framerate=framerate)
        plotter.orbit_on_path(path, write_frames=True)
    else:
        plotter.show(auto_close=False)
        plotter.orbit_on_path(path, write_frames=False)

    if close:
        plotter.close()

    return plotter
