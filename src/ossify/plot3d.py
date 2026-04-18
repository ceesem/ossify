"""3D plotting for ossify using PyVista as the rendering backend."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from .base import Cell, PointCloudLayer, SkeletonLayer
from .plot_utils import (
    _map_value_to_colors,
    _resolve_color_parameter,
    _resolve_scalar_parameter,
)

try:
    import pyvista as pv
except ImportError as e:
    raise ImportError(
        "pyvista is required for 3D plotting. Install it with: pip install ossify[viz]"
    ) from e

import matplotlib.colors as mcolors

__all__ = [
    "plot_skeleton_3d",
    "plot_morphology_3d",
    "plot_points_3d",
    "plot_annotations_3d",
    "plot_cell_3d",
]


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
        Overall opacity scalar, or per-vertex opacity array. Ignored when
        ``tube_radius`` is not ``None`` and colors are RGBA (alpha is baked
        into the color array).
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

    if colors is not None:
        poly.point_data["colors"] = colors

    mesh_kwargs: dict = {}

    if tube_radius is not None:
        if isinstance(tube_radius, np.ndarray):
            radii = tube_radius.astype(float)
            r_min = float(np.min(radii))
            r_max = float(np.max(radii))
            if r_min <= 0:
                # Avoid degenerate zero-radius tubes
                r_min = max(r_max * 0.01, 1e-9)
                radii = np.clip(radii, r_min, None)
            if r_max <= r_min:
                # All radii identical — skip scalar path
                poly = poly.tube(radius=r_min, n_sides=12)
            else:
                # VTK maps scalars to [radius, radius * radius_factor], so set
                # radius=r_min and radius_factor=r_max/r_min so that the
                # normalized [0, 1] scalar range maps exactly to [r_min, r_max].
                poly.point_data["tube_radius"] = radii
                poly = poly.tube(
                    radius=r_min,
                    scalars="tube_radius",
                    radius_factor=r_max / r_min,
                    n_sides=12,
                )
        else:
            poly = poly.tube(radius=float(tube_radius), n_sides=12)

    if colors is not None:
        # Re-attach colors after tube (tube interpolates point_data)
        if "colors" not in poly.point_data:
            # tube didn't propagate — attach by mapping original colors to
            # interpolated points.  This is a best-effort fallback; tube
            # rendering normally propagates point_data automatically.
            pass
        else:
            mesh_kwargs["scalars"] = "colors"
            mesh_kwargs["rgb"] = True

    if opacity is not None:
        if isinstance(opacity, np.ndarray):
            poly.point_data["opacity"] = opacity.astype(float)
            mesh_kwargs["opacity"] = "opacity"
        else:
            mesh_kwargs["opacity"] = float(opacity)

    if tube_radius is None:
        plotter.add_mesh(poly, line_width=line_width, **mesh_kwargs)
    else:
        plotter.add_mesh(poly, **mesh_kwargs)

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
    resolved_color = _resolve_color_parameter(color, skel)
    effective_color_norm = color_norm
    if (
        color_scale == "log"
        and isinstance(resolved_color, np.ndarray)
        and resolved_color.ndim == 1
    ):
        resolved_color = np.log(np.asarray(resolved_color, dtype=float))
        if color_norm is not None:
            effective_color_norm = (np.log(color_norm[0]), np.log(color_norm[1]))
    colors_array = None
    if resolved_color is not None:
        if isinstance(resolved_color, np.ndarray) and resolved_color.ndim == 1:
            colors_array = _map_value_to_colors(
                resolved_color, colormap=palette, color_norm=effective_color_norm
            )
        elif isinstance(resolved_color, np.ndarray) and resolved_color.ndim == 2:
            colors_array = resolved_color
        else:
            single_color = mcolors.to_rgb(resolved_color)
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
    opacity_out: Optional[Union[float, np.ndarray]] = None
    if opacity is not None:
        resolved_opacity = _resolve_scalar_parameter(
            opacity, skel.n_vertices, layer=skel
        )
        if resolved_opacity is not None:
            if np.ndim(resolved_opacity) == 0 or (
                isinstance(resolved_opacity, np.ndarray) and resolved_opacity.ndim == 0
            ):
                opacity_out = float(resolved_opacity)
            elif (
                isinstance(resolved_opacity, np.ndarray)
                and resolved_opacity.shape == (skel.n_vertices,)
                and np.allclose(resolved_opacity, resolved_opacity[0])
            ):
                opacity_out = float(resolved_opacity[0])
            else:
                opacity_out = resolved_opacity

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
        elif isinstance(resolved_tube_radius, float):
            r = resolved_tube_radius
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
    colors: Optional[Union[str, np.ndarray, dict]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    opacity: float = 1.0,
    plotter: Optional["pv.Plotter"] = None,
    **kwargs,
) -> "pv.Plotter":
    """Render a point cloud as spheres in 3D.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape ``(N, 3)``.
    sizes : float or np.ndarray, optional
        Sphere radius. A scalar gives uniform radius; an array of shape
        ``(N,)`` gives per-point radius. Defaults to a small fixed radius
        when ``None``.
    colors : str, np.ndarray, or dict, optional
        Color specification. A string is treated as a single matplotlib color.
        A 1D array of shape ``(N,)`` is mapped through *palette*. An
        ``(N, 3)`` array is used directly as RGB.  A dict maps discrete
        values to colors.
    palette : str or dict, default "coolwarm"
        Colormap for mapping scalar color values. Any name from the
        `matplotlib colormap registry
        <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        is accepted, including colormaps registered by third-party packages
        (e.g. cmocean, colorcet, cmcrameri) if they are installed. A dict
        maps discrete values to colors.
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

    # Normalize array-like inputs to ndarray so pandas Series, lists, etc.
    # pass all isinstance(…, np.ndarray) checks below.
    if sizes is not None and not isinstance(sizes, (int, float, np.ndarray)):
        sizes = np.asarray(sizes, dtype=float)
    if colors is not None and not isinstance(colors, (str, dict, np.ndarray)):
        colors = np.asarray(colors)

    # --- Resolve colors ---
    mesh_kwargs: dict = {"opacity": opacity, **kwargs}
    if colors is not None:
        if isinstance(colors, str):
            mesh_kwargs["color"] = colors
        elif isinstance(colors, np.ndarray):
            if colors.ndim == 1:
                # Raw scalar values: store on the cloud so the glyph path can
                # retrieve them for pre-mapping to RGB.
                if isinstance(palette, dict):
                    # Dict palette: pre-map to RGB since PyVista has no native
                    # support for arbitrary label→color mappings.
                    mapped = _map_value_to_colors(
                        colors, colormap=palette, color_norm=color_norm
                    )
                    cloud.point_data["colors"] = mapped
                    mesh_kwargs["scalars"] = "colors"
                    mesh_kwargs["rgb"] = True
                else:
                    # Use "pt_values" to avoid collision with VTK's internal
                    # "scalars" slot, which the glyph filter can overwrite.
                    cloud.point_data["pt_values"] = colors
                    mesh_kwargs["scalars"] = "pt_values"
                    if palette is not None:
                        mesh_kwargs["cmap"] = palette
                    if color_norm is not None:
                        mesh_kwargs["clim"] = list(color_norm)
            else:
                # Already (N, 3) or (N, 4) — use as pre-mapped RGB.
                cloud.point_data["colors"] = colors
                mesh_kwargs["scalars"] = "colors"
                mesh_kwargs["rgb"] = True
        elif isinstance(colors, dict):
            mesh_kwargs["scalars"] = "colors"
            mesh_kwargs["rgb"] = True

    # --- Render ---
    if sizes is not None and isinstance(sizes, np.ndarray):
        # Per-point radius via glyph
        cloud.point_data["radius"] = sizes.astype(float)
        # radius=1.0 ensures the glyph scale factor equals the output sphere
        # radius directly; the default radius=0.5 would halve all sizes.
        geom = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
        glyphs = cloud.glyph(geom=geom, scale="radius", orient=False)
        # VTK's vtkGlyph3D does not reliably propagate custom point_data arrays
        # (PassPointData=Off by default), so the named scalar key in mesh_kwargs
        # may not exist on the glyph output.  Pass the color data as an explicit
        # numpy array instead — PyVista's add_mesh accepts scalars=<array> and
        # uses it directly without needing the array to live in the mesh.
        glyph_kwargs: dict = {
            k: v
            for k, v in mesh_kwargs.items()
            if k not in ("scalars", "rgb", "cmap", "clim")
        }
        if "scalars" in mesh_kwargs:
            arr_name = mesh_kwargs["scalars"]
            src = cloud.point_data.get(arr_name)
            if src is not None and glyphs.n_points % cloud.n_points == 0:
                n_per = glyphs.n_points // cloud.n_points
                if src.ndim == 1 and not mesh_kwargs.get("rgb", False):
                    # Pre-map scalars → RGB so the correct palette is applied.
                    # Passing raw scalars + cmap= to add_mesh is unreliable
                    # across PyVista/VTK versions (often defaults to viridis).
                    _cmap = mesh_kwargs.get("cmap", "coolwarm")
                    _clim = mesh_kwargs.get("clim")
                    src = _map_value_to_colors(
                        src,
                        colormap=_cmap,
                        color_norm=tuple(_clim) if _clim is not None else None,
                    )
                    glyph_kwargs["rgb"] = True
                else:
                    for key in ("cmap", "clim", "rgb"):
                        if key in mesh_kwargs:
                            glyph_kwargs[key] = mesh_kwargs[key]
                glyph_kwargs["scalars"] = np.repeat(src, n_per, axis=0)
        plotter.add_mesh(glyphs, **glyph_kwargs)
    else:
        point_size = float(sizes) if sizes is not None else 5.0
        if colors is not None and isinstance(colors, np.ndarray):
            # render_points_as_spheres doesn't reliably apply per-point
            # coloring in all PyVista/VTK versions — the PolyData has no
            # cells, so VTK may ignore per-vertex scalars.  Use sphere
            # glyphs (actual mesh geometry) which always respect rgb=True.
            cloud.point_data["_r"] = np.full(len(pts), point_size)
            geom = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
            glyphs_uni = cloud.glyph(geom=geom, scale="_r", orient=False)
            u_kwargs: dict = {
                k: v
                for k, v in mesh_kwargs.items()
                if k not in ("scalars", "rgb", "cmap", "clim")
            }
            if glyphs_uni.n_points % cloud.n_points == 0:
                n_per = glyphs_uni.n_points // cloud.n_points
                if colors.ndim == 1:
                    mapped = _map_value_to_colors(
                        colors, colormap=palette, color_norm=color_norm
                    )
                else:
                    mapped = colors
                u_kwargs["scalars"] = np.repeat(mapped, n_per, axis=0)
                u_kwargs["rgb"] = True
            plotter.add_mesh(glyphs_uni, **u_kwargs)
        else:
            # No per-point colors — keep the lightweight render_points_as_spheres path.
            for key in ("scalars", "rgb", "cmap", "clim"):
                mesh_kwargs.pop(key, None)
            mesh_kwargs["point_size"] = point_size
            mesh_kwargs["render_points_as_spheres"] = True
            if isinstance(colors, str):
                mesh_kwargs["color"] = colors
            plotter.add_mesh(cloud, **mesh_kwargs)

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
    sizes: Optional[Tuple[float, float]] = (1, 30),
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
        ``(min_radius, max_radius)`` output range for size rescaling. Default
        ``(1, 30)``.
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

    # --- Resolve color ---
    resolved_color = _resolve_color_parameter(color, annotation)
    effective_color_norm = color_norm
    if (
        color_scale == "log"
        and isinstance(resolved_color, np.ndarray)
        and resolved_color.ndim == 1
    ):
        resolved_color = np.log(np.asarray(resolved_color, dtype=float))
        if color_norm is not None:
            effective_color_norm = (np.log(color_norm[0]), np.log(color_norm[1]))

    # --- Resolve size ---
    if (
        size_scale is not None
        and size is not None
        and not isinstance(size, (int, float, bool))
    ):
        # Fetch raw values, apply transform, convert norm bounds to match.
        if isinstance(size, str):
            raw_size = np.asarray(annotation.get_feature(size), dtype=float)
        else:
            raw_size = np.asarray(size, dtype=float)
        if size_scale == "log":
            fn = np.log
        elif size_scale == "sqrt":
            fn = np.sqrt
        elif size_scale == "cbrt":
            fn = np.cbrt
        transformed = fn(raw_size)
        transformed_norm = (
            (fn(size_norm[0]), fn(size_norm[1])) if size_norm is not None else None
        )
        resolved_size = _resolve_scalar_parameter(
            transformed, len(vertices), norm=transformed_norm, out_range=sizes
        )
    else:
        resolved_size = _resolve_scalar_parameter(
            size, len(vertices), norm=size_norm, out_range=sizes, layer=annotation
        )

    return plot_points_3d(
        points=vertices,
        sizes=resolved_size,
        colors=resolved_color,
        palette=palette,
        color_norm=effective_color_norm,
        opacity=opacity,
        plotter=plotter,
    )


def plot_cell_3d(
    cell: Cell,
    # Skeleton styling
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    opacity: float = 1.0,
    line_width: float = 2.0,
    tube_radius: Optional[Union[float, str, np.ndarray]] = None,
    tube_radius_scale: Optional[float] = None,
    tube_radius_norm: Optional[Tuple[float, float]] = None,
    tube_radii: Optional[Tuple[float, float]] = None,
    root_marker: bool = False,
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
    syn_sizes: Optional[Tuple[float, float]] = (1, 30),
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
    syn_sizes : tuple of float, default (1, 30)
        ``(min_radius, max_radius)`` output range for synapse sizes.
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
        opacity=opacity,
        line_width=line_width,
        tube_radius=tube_radius,
        tube_radius_scale=tube_radius_scale,
        tube_radius_norm=tube_radius_norm,
        tube_radii=tube_radii,
        root_marker=root_marker,
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
