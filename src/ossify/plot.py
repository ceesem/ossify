from numbers import Number
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize

from .base import Cell, GraphLayer, MeshLayer, PointCloudLayer, SkeletonLayer

__all__ = [
    "plot_cell_2d",
    "plot_morphology_2d",
    "plot_annotations_2d",
    "plot_cell_multiview",
    "plot_skeleton",
    "plot_points",
    "single_panel_figure",
    "multi_panel_figure",
]


def _map_value_to_colors(
    values: np.ndarray,
    colormap: Union[str, Colormap, Dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    values = np.asarray(values)
    if isinstance(colormap, dict):
        rgba_colors = np.zeros((len(values), 4))
        rgba_colors[:, 3] = alpha  # Set alpha channel
        for i, val in enumerate(values):
            if val in colormap:
                color = colormap[val]
                # Convert various color formats to RGB
                if isinstance(color, str):
                    if color.startswith("#"):
                        # Hex color
                        rgb = tuple(
                            int(color[j : j + 2], 16) / 255.0 for j in (1, 3, 5)
                        )
                    else:
                        # Named color - use matplotlib
                        rgb = cm.colors.to_rgb(color)
                else:
                    # Assume RGB tuple
                    rgb = color[:3]

                rgba_colors[i, :3] = rgb
            else:
                # Default to black for unmapped values
                rgba_colors[i, :3] = [0, 0, 0]

        return rgba_colors
    # Handle continuous mapping
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Apply normalization
    if color_norm is not None:
        vmin, vmax = color_norm
        norm = Normalize(vmin=vmin, vmax=vmax)
        normalized_values = norm(values)
    else:
        # Auto-normalize to [0, 1]
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmin == vmax:
            normalized_values = np.zeros_like(values)
        else:
            normalized_values = (values - vmin) / (vmax - vmin)
    # Map to colors
    rgba_colors = cmap(normalized_values)

    return rgba_colors[:, :3]


def _should_invert_y_axis(projection: Union[str, Callable]) -> bool:
    """Determine if y-axis should be inverted based on projection.

    Parameters
    ----------
    projection : str or Callable
        Projection specification

    Returns
    -------
    bool
        True if y-axis should be inverted (when 'y' is present in projection)
    """
    if isinstance(projection, str):
        return "y" in projection
    return False


def _apply_y_inversion_to_axes(
    ax: plt.Axes, projection: Union[str, Callable], invert_y: bool = True
) -> plt.Axes:
    """Apply y-axis inversion if needed based on projection.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to potentially invert
    projection : str or Callable
        Projection specification
    invert_y : bool, default True
        Whether to enable automatic y-axis inversion for projections containing 'y'

    Returns
    -------
    plt.Axes
        Axes with y-axis inverted if needed
    """
    if invert_y and _should_invert_y_axis(projection):
        # Only invert if not already inverted to avoid double-inversion
        if projection[1] == "y":
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
        elif projection[0] == "y":
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
    return ax


def projection_factory(
    proj: Literal["xy", "yx", "yz", "zy", "zx", "xz"],
) -> Callable:
    match proj:
        case "xy":
            return lambda pts: np.array(pts)[:, [0, 1]]
        case "yx":
            return lambda pts: np.array(pts)[:, [1, 0]]
        case "zx":
            return lambda pts: np.array(pts)[:, [2, 0]]
        case "xz":
            return lambda pts: np.array(pts)[:, [0, 2]]
        case "zy":
            return lambda pts: np.array(pts)[:, [2, 1]]
        case "yz":
            return lambda pts: np.array(pts)[:, [1, 2]]
    raise ValueError(
        f"Unknown projection {proj}, expected one of 'xy', 'yx', 'yz', 'zy', 'zx', or 'xz'"
    )


def _plotted_bounds(
    vertices: np.ndarray,
    projection: Union[str, Callable],
    offset_h: float = 0.0,
    offset_v: float = 0.0,
) -> np.ndarray:
    """Get the plotted bounds of the vertices after applying the projection.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of 3D points
    projection : Callable
        Projection function to apply to the points

    Returns
    -------
    np.ndarray
        (2, 2) array with [[xmin, xmax], [ymin, ymax]] of the projected points
    """
    if isinstance(projection, str):
        projection = projection_factory(proj=projection)
    projected = projection(vertices)
    projected[:, 0] += offset_h
    projected[:, 1] += offset_v
    xmin, xmax = projected[:, 0].min(), projected[:, 0].max()
    ymin, ymax = projected[:, 1].min(), projected[:, 1].max()
    return np.array([[xmin, xmax], [ymin, ymax]])


def plot_skeleton(
    skel: SkeletonLayer,
    projection: Union[str, Callable] = "xy",
    colors: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    linewidths: Optional[np.ndarray] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    zorder: int = 2,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot skeleton with explicit arrays for styling.

    Parameters
    ----------
    skel : SkeletonLayer
        SkeletonLayer to plot
    projection : str or Callable, default "xy"
        Projection function or string
    colors : np.ndarray, optional
        (N, 3) or (N, 4) RGB/RGBA color array for vertices
    alpha : np.ndarray, optional
        (N,) alpha values for vertices
    linewidths : np.ndarray, optional
        (N,) linewidth values for vertices
    offset_h : float, default 0.0
        Horizontal offset for projection
    offset_v : float, default 0.0
        Vertical offset for projection
    zorder : int, default 2
        Drawing order for line collection
    invert_y : bool, default True
        Whether to automatically invert y-axis for projections containing 'y'
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    plt.Axes
        Matplotlib axes with skeleton plotted
    """
    if ax is None:
        ax = plt.gca()

    # Store original projection for y-axis inversion detection
    orig_projection = projection
    if isinstance(projection, str):
        projection = projection_factory(proj=projection)

    for path in skel.cover_paths:
        match skel.parent_node_array[path[-1]]:
            case -1:
                path_plus = path
            case parent:
                path_plus = np.concat((path, [parent]))

        path_spatial = projection(skel.vertices[path_plus])
        path_spatial[:, 0] = path_spatial[:, 0] + offset_h
        path_spatial[:, 1] = path_spatial[:, 1] + offset_v
        path_segs = [
            (path_spatial[i], path_spatial[i + 1]) for i in range(len(path_spatial) - 1)
        ]

        # Extract styling for this path
        lc_kwargs = {}
        if colors is not None:
            lc_kwargs["colors"] = colors[path]
        if alpha is not None:
            lc_kwargs["alpha"] = alpha[path]
        if linewidths is not None:
            lc_kwargs["linewidths"] = linewidths[path]
        if zorder is not None:
            lc_kwargs["zorder"] = zorder

        lc = LineCollection(path_segs, capstyle="butt", joinstyle="round", **lc_kwargs)
        ax.add_collection(lc)

    ax.set_aspect("equal")

    # Apply y-axis inversion if needed
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)

    return ax


def _resolve_color_parameter(
    color_param: Union[str, np.ndarray, tuple, Any], skel: SkeletonLayer
) -> Union[np.ndarray, str, tuple, None]:
    """Resolve color parameter - try labels first, then matplotlib colors.

    Parameters
    ----------
    color_param : str, np.ndarray, tuple, or Any
        Color specification to resolve
    skel : SkeletonLayer
        Skeleton layer to look up labels from

    Returns
    -------
    np.ndarray, str, tuple, or None
        Resolved color parameter - array if label found, original value otherwise
    """
    if isinstance(color_param, str):
        # Try to get as label first
        try:
            return skel.get_label(color_param)
        except (KeyError, AttributeError):
            # Fall back to matplotlib color
            return color_param
    else:
        # Return as-is (array, tuple, etc.)
        return color_param


def plot_points(
    points: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    palette: Optional[Union[str, Dict]] = None,
    color_norm: Optional[Tuple[float, float]] = None,
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    zorder: int = 2,
    **scatter_kws,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    # Store original projection for y-axis inversion detection
    orig_projection = projection
    if isinstance(projection, str):
        projection = projection_factory(
            proj=projection,
        )
    points_proj = projection(points)
    points_proj[:, 0] = points_proj[:, 0] + offset_h
    points_proj[:, 1] = points_proj[:, 1] + offset_v
    if scatter_kws is None:
        scatter_kws = {}
    if "linewidths" not in scatter_kws:
        scatter_kws["linewidths"] = 0
    if isinstance(palette, str):
        scatter_kws["cmap"] = palette
        if color_norm is not None:
            scatter_kws["vmin"], scatter_kws["vmax"] = color_norm
    elif isinstance(palette, dict) and colors:
        colors = [palette[label] for label in colors]
    if colors:
        scatter_kws["c"] = colors

    ax.scatter(
        x=points_proj[:, 0],
        y=points_proj[:, 1],
        s=sizes,
        zorder=zorder,
        **scatter_kws,
    )

    # Apply y-axis inversion if needed
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)

    return ax


def _rescale_scalar(
    value: np.ndarray,
    norm: Optional[Tuple[float, float]],
    out_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Linearly rescale a scalar value to a new range with clipping

    Parameters
    ----------
    value : np.ndarray
        Value to rescale
    norm : tuple of float
        (min, max) tuple for normalization

    Returns
    -------
    np.ndarray
        Rescaled value
    """
    if norm is None:
        norm = (np.min(value), np.max(value))
    if out_range is None:
        out_range = (np.min(value), np.max(value))
    return (out_range[1] - out_range[0]) * np.asarray(
        Normalize(*norm, clip=True)(value)
    ) + out_range[0]


def plot_annotations_2d(
    annotation: PointCloudLayer,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: float = 1,
    size: Optional[Union[str, np.ndarray, float]] = None,
    size_norm: Optional[Tuple[float, float]] = None,
    sizes: Optional[np.ndarray] = (1, 30),
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if isinstance(annotation, PointCloudLayer):
        vertices = annotation.vertices
        if isinstance(color, str):
            color = color or annotation.get_label(color)
        if isinstance(size, str):
            size = annotation.get_label(size)
    else:
        vertices = np.asarray(annotation)

    if isinstance(size, Number) or size is None:
        sizes_out = size
    else:
        sizes_out = _rescale_scalar(size, size_norm, sizes)

    return plot_points(
        points=vertices,
        sizes=sizes_out,
        colors=color,
        palette=palette,
        color_norm=color_norm,
        projection=projection,
        offset_h=offset_h,
        offset_v=offset_v,
        invert_y=invert_y,
        alpha=alpha,
        ax=ax,
        **kwargs,
    )


def plot_morphology_2d(
    cell: Union[Cell, SkeletonLayer],
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    root_as_sphere: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    zorder: int = 2,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot 2D skeleton with flexible styling options.

    Parameters
    ----------
    cell : Cell or SkeletonLayer
        Cell or SkeletonLayer to plot
    projection : str or Callable, default "xy"
        Projection function or string mapping 3d points to a 2d projection.
    color : str, np.ndarray, or tuple, optional
        Color specification - can be label name, array of values, or matplotlib color
    palette : str or dict, default "plasma"
        Colormap for mapping array values to colors
    color_norm : tuple of float, optional
        (min, max) tuple for color normalization
    alpha : str, np.ndarray, or float, default 1.0
        Alpha specification - can be label name, array, or single value
    alpha_norm : tuple of float, optional
        (min, max) tuple for alpha normalization
    linewidth : str, np.ndarray, or float, default 1.0
        Linewidth specification - can be label name, array, or single value
    linewidth_norm : tuple of float, optional
        (min, max) tuple for linewidth normalization
    widths : tuple, optional
        (min, max) tuple for final linewidth scaling
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    plt.Axes
        Matplotlib axes with skeleton plotted
    """
    if isinstance(cell, Cell):
        skel = cell.skeleton
    else:
        skel = cell

    # Resolve color parameter
    resolved_color = _resolve_color_parameter(color, skel)

    # Process colors
    colors_array = None
    if resolved_color is not None:
        if isinstance(resolved_color, np.ndarray):
            # Array of values - map through colormap
            colors_array = _map_value_to_colors(
                resolved_color, colormap=palette, color_norm=color_norm
            )
        else:
            # Single color (string, tuple) - use matplotlib to convert
            import matplotlib.colors as mcolors

            single_color = mcolors.to_rgba(resolved_color)
            colors_array = np.tile(single_color, (skel.n_vertices, 1))

    # Process alpha (similar to existing logic)
    alpha_array = None
    if isinstance(alpha, str):
        alpha_values = skel.get_label(alpha)
        if alpha_norm is not None:
            alpha_array = np.asarray(Normalize(*alpha_norm, clip=True)(alpha_values))
        else:
            alpha_array = np.asarray(alpha_values)
    elif isinstance(alpha, (np.ndarray, list, pd.Series)):
        if alpha_norm is not None:
            alpha_array = np.asarray(Normalize(*alpha_norm, clip=True)(alpha))
        else:
            alpha_array = np.asarray(alpha)
    elif isinstance(alpha, Number):
        alpha_array = np.full(skel.n_vertices, alpha)

    # Process linewidth (similar to existing logic)
    linewidth_array = None
    if isinstance(linewidth, str):
        linewidth_values = skel.get_label(linewidth)
        if linewidth_norm is None:
            linewidth_norm = (np.min(linewidth_values), np.max(linewidth_values))
        normalized = Normalize(*linewidth_norm, clip=True)(linewidth_values)
        if widths is None:
            widths = (np.min(linewidth_values), np.max(linewidth_values))
        linewidth_array = widths[0] + (widths[1] - widths[0]) * normalized
    elif isinstance(linewidth, (np.ndarray, list, pd.Series)):
        if linewidth_norm is None:
            linewidth_norm = (np.min(linewidth), np.max(linewidth))
        normalized = Normalize(*linewidth_norm, clip=True)(linewidth)
        if widths is None:
            widths = (np.min(linewidth), np.max(linewidth))
        linewidth_array = widths[0] + (widths[1] - widths[0]) * normalized
    elif isinstance(linewidth, Number):
        linewidth_array = np.full(skel.n_vertices, linewidth)

    # Call the core plotting function
    ax = plot_skeleton(
        skel=skel,
        projection=projection,
        colors=colors_array,
        alpha=alpha_array,
        linewidths=linewidth_array,
        offset_h=offset_h,
        offset_v=offset_v,
        zorder=zorder,
        invert_y=invert_y,
        ax=ax,
    )
    if root_as_sphere:
        root_location = np.atleast_2d(skel.root_location)
        if root_color is None:
            root_color = (
                colors_array[skel.root_positional] if colors_array is not None else None
            )
        if root_color is not None:
            root_color = _resolve_color_parameter(root_color, skel)
        ax = plot_points(
            root_location,
            colors=np.atleast_2d([root_color]),
            sizes=[root_size],
            invert_y=invert_y,
            ax=ax,
            zorder=zorder + 1,
        )
    return ax


def plot_cell_2d(
    cell: Cell,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    root_as_sphere: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    synapses: Literal["pre", "post", "both", False] = "both",
    pre_anno: str = "pre_syn",
    pre_color: Optional[Union[str, tuple]] = None,
    pre_palette: Union[str, dict] = None,
    pre_color_norm: Optional[Tuple[float, float]] = None,
    syn_alpha: float = 1,
    syn_size: Optional[Union[str, np.ndarray, float]] = None,
    syn_size_norm: Optional[Tuple[float, float]] = None,
    syn_sizes: Optional[np.ndarray] = (1, 30),
    post_anno: str = "post_syn",
    post_color: Optional[Union[str, tuple]] = None,
    post_palette: Union[str, dict] = None,
    post_color_norm: Optional[Tuple[float, float]] = None,
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    units_per_inch: Optional[float] = None,
    dpi: Optional[float] = None,
    despine: bool = True,
    **syn_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    if units_per_inch is not None:
        bounds = _plotted_bounds(cell.skeleton.vertices, projection, offset_h, offset_v)
        _, ax = single_panel_figure(
            data_bounds_min=bounds[:, 0],
            data_bounds_max=bounds[:, 1],
            units_per_inch=units_per_inch,
            despine=despine,
            dpi=dpi,
        )

    ax = plot_morphology_2d(
        cell,
        color=color,
        palette=palette,
        color_norm=color_norm,
        alpha=alpha,
        alpha_norm=alpha_norm,
        linewidth=linewidth,
        linewidth_norm=linewidth_norm,
        widths=widths,
        root_as_sphere=root_as_sphere,
        root_size=root_size,
        root_color=root_color,
        projection=projection,
        offset_h=offset_h,
        offset_v=offset_v,
        invert_y=invert_y,
        ax=ax,
    )
    if synapses == "both" or synapses == "pre":
        ax = plot_annotations_2d(
            cell.annotations[pre_anno],
            color=pre_color,
            palette=pre_palette,
            color_norm=pre_color_norm,
            alpha=syn_alpha,
            size=syn_size,
            size_norm=syn_size_norm,
            sizes=syn_sizes,
            ax=ax,
            offset_h=offset_h,
            offset_v=offset_v,
            invert_y=invert_y,
            projection=projection,
            **syn_kwargs,
        )
    if synapses == "both" or synapses == "post":
        ax = plot_annotations_2d(
            cell.annotations[post_anno],
            color=post_color,
            palette=post_palette,
            color_norm=post_color_norm,
            alpha=syn_alpha,
            size=syn_size,
            size_norm=syn_size_norm,
            sizes=syn_sizes,
            ax=ax,
            offset_h=offset_h,
            offset_v=offset_v,
            invert_y=invert_y,
            projection=projection,
            **syn_kwargs,
        )
    return ax


def plot_cell_multiview(
    cell: Cell,
    layout: Literal["stacked", "side_by_side", "three_panel"] = "three_panel",
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    root_as_sphere: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    synapses: Literal["pre", "post", "both", False] = "both",
    pre_anno: str = "pre_syn",
    pre_color: Optional[Union[str, tuple]] = None,
    pre_palette: Union[str, dict] = None,
    pre_color_norm: Optional[Tuple[float, float]] = None,
    syn_alpha: float = 1,
    syn_size: Optional[Union[str, np.ndarray, float]] = None,
    syn_size_norm: Optional[Tuple[float, float]] = None,
    syn_sizes: Optional[np.ndarray] = (1, 30),
    post_anno: str = "post_syn",
    post_color: Optional[Union[str, tuple]] = None,
    post_palette: Union[str, dict] = None,
    post_color_norm: Optional[Tuple[float, float]] = None,
    invert_y: bool = True,
    despine: bool = True,
    units_per_inch: float = 100_000,
    dpi: Optional[float] = None,
    **syn_kwargs,
) -> Tuple[plt.Figure, dict]:
    fig, axes = multi_panel_figure(
        data_bounds_min=cell.skeleton.bbox[0],
        data_bounds_max=cell.skeleton.bbox[1],
        units_per_inch=units_per_inch,
        layout=layout,
        despine=despine,
        dpi=dpi,
    )
    for proj in axes:
        ax = proj["ax"]
        plot_cell_2d(
            cell,
            color=color,
            palette=palette,
            color_norm=color_norm,
            alpha=alpha,
            alpha_norm=alpha_norm,
            linewidth=linewidth,
            linewidth_norm=linewidth_norm,
            widths=widths,
            root_as_sphere=root_as_sphere,
            root_size=root_size,
            root_color=root_color,
            projection=proj,
            invert_y=invert_y,
            synapses=synapses,
            syn_alpha=syn_alpha,
            syn_size=syn_size,
            syn_size_norm=syn_size_norm,
            syn_sizes=syn_sizes,
            pre_anno=pre_anno,
            pre_color=pre_color,
            pre_palette=pre_palette,
            pre_color_norm=pre_color_norm,
            post_anno=post_anno,
            post_color=post_color,
            post_palette=post_palette,
            post_color_norm=post_color_norm,
            ax=ax,
            **syn_kwargs,
        )
    return fig, axes


def single_panel_figure(
    data_bounds_min: np.ndarray,
    data_bounds_max: np.ndarray,
    units_per_inch: float,
    despine: bool = True,
    dpi: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a single panel figure with precise unit-based sizing.

    Parameters
    ----------
    data_bounds_min : np.ndarray
        2-element array [x_min, y_min] of data bounds
    data_bounds_max : np.ndarray
        2-element array [x_max, y_max] of data bounds
    units_per_inch : float
        Number of data units per inch for scaling
    despine : bool, default True
        Whether to remove axis spines and ticks for clean appearance
    dpi : float, optional
        Dots per inch for figure resolution. If None, uses matplotlib default.

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
        Figure and axes objects with correct unit scaling

    Examples
    --------
    >>> bounds_min = np.array([0, 0])
    >>> bounds_max = np.array([100, 50])
    >>> fig, ax = create_single_panel_figure(bounds_min, bounds_max, 10)
    >>> # Creates 10" x 5" figure with 10 units per inch
    """
    data_bounds_min = np.asarray(data_bounds_min)
    data_bounds_max = np.asarray(data_bounds_max)

    # Calculate data extents
    data_width = data_bounds_max[0] - data_bounds_min[0]
    data_height = data_bounds_max[1] - data_bounds_min[1]

    # Convert to figure size in inches
    fig_width = data_width / units_per_inch
    fig_height = data_height / units_per_inch

    # Create figure and axis
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure

    # Set data limits and aspect ratio
    ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
    ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
    ax.set_aspect("equal")

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    return fig, ax


def multi_panel_figure(
    data_bounds_min: np.ndarray,
    data_bounds_max: np.ndarray,
    units_per_inch: float,
    layout: Literal["side_by_side", "stacked", "three_panel"],
    gap_inches: float = 0.5,
    despine: bool = True,
    dpi: Optional[float] = None,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create multi-panel figure with precise unit-based sizing and alignment.

    Parameters
    ----------
    data_bounds_min : np.ndarray
        3-element array [x_min, y_min, z_min] of data bounds
    data_bounds_max : np.ndarray
        3-element array [x_max, y_max, z_max] of data bounds
    units_per_inch : float
        Number of data units per inch for scaling
    layout : {"side_by_side", "stacked", "three_panel"}
        Layout configuration:
        - "side_by_side": xy | zy (horizontal)
        - "stacked": xz over xy (vertical)
        - "three_panel": L-shaped (xy bottom-left, xz top-left, zy bottom-right)
    gap_inches : float, default 0.5
        Gap between panels in inches
    despine : bool, default True
        Whether to remove axis spines and ticks for clean appearance
    dpi : float, optional
        Dots per inch for figure resolution. If None, uses matplotlib default.

    Returns
    -------
    tuple of (plt.Figure, dict of plt.Axes)
        Figure and dictionary of axes keyed by projection.
        - "side_by_side": {"xy": xy_ax, "zy": zy_ax}
        - "stacked": {"xz": xz_ax, "xy": xy_ax}
        - "three_panel": {"xy": xy_ax, "xz": xz_ax, "zy": zy_ax}

    Examples
    --------
    >>> bounds_min = np.array([0, 0, 0])
    >>> bounds_max = np.array([100, 50, 75])
    >>> fig, axes_dict = create_multi_panel_figure(bounds_min, bounds_max, 10, "side_by_side")
    >>> xy_ax, zy_ax = axes_dict["xy"], axes_dict["zy"]
    """
    data_bounds_min = np.asarray(data_bounds_min)
    data_bounds_max = np.asarray(data_bounds_max)

    # Calculate data extents for each dimension
    x_extent = data_bounds_max[0] - data_bounds_min[0]
    y_extent = data_bounds_max[1] - data_bounds_min[1]
    z_extent = data_bounds_max[2] - data_bounds_min[2]

    # Convert to sizes in inches
    x_inches = x_extent / units_per_inch
    y_inches = y_extent / units_per_inch
    z_inches = z_extent / units_per_inch

    if layout == "side_by_side":
        # xy | zy layout
        xy_width, xy_height = x_inches, y_inches
        zy_width, zy_height = z_inches, y_inches

        fig_width = xy_width + gap_inches + zy_width
        fig_height = max(xy_height, zy_height)

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # xy panel (left)
        xy_left = 0
        xy_bottom = (fig_height - xy_height) / 2  # Center vertically
        xy_ax = fig.add_axes(
            [
                xy_left / fig_width,
                xy_bottom / fig_height,
                xy_width / fig_width,
                xy_height / fig_height,
            ]
        )
        xy_ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
        xy_ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
        xy_ax.set_aspect("equal")

        # zy panel (right)
        zy_left = xy_width + gap_inches
        zy_bottom = (fig_height - zy_height) / 2  # Center vertically
        zy_ax = fig.add_axes(
            [
                zy_left / fig_width,
                zy_bottom / fig_height,
                zy_width / fig_width,
                zy_height / fig_height,
            ]
        )
        zy_ax.set_xlim(data_bounds_min[2], data_bounds_max[2])
        zy_ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
        zy_ax.set_aspect("equal")
        if despine:
            for ax in [xy_ax, zy_ax]:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

        return fig, {"xy": xy_ax, "zy": zy_ax}

    elif layout == "stacked":
        # xz over xy layout
        xy_width, xy_height = x_inches, y_inches
        xz_width, xz_height = x_inches, z_inches

        fig_width = max(xy_width, xz_width)
        fig_height = xy_height + gap_inches + xz_height

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # xz panel (top)
        xz_left = (fig_width - xz_width) / 2  # Center horizontally
        xz_bottom = xy_height + gap_inches
        xz_ax = fig.add_axes(
            [
                xz_left / fig_width,
                xz_bottom / fig_height,
                xz_width / fig_width,
                xz_height / fig_height,
            ]
        )
        xz_ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
        xz_ax.set_ylim(data_bounds_min[2], data_bounds_max[2])
        xz_ax.set_aspect("equal")

        # xy panel (bottom)
        xy_left = (fig_width - xy_width) / 2  # Center horizontally
        xy_bottom = 0
        xy_ax = fig.add_axes(
            [
                xy_left / fig_width,
                xy_bottom / fig_height,
                xy_width / fig_width,
                xy_height / fig_height,
            ]
        )
        xy_ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
        xy_ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
        xy_ax.set_aspect("equal")
        if despine:
            for ax in [xy_ax, xz_ax]:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

        return fig, {"xz": xz_ax, "xy": xy_ax}

    elif layout == "three_panel":
        # L-shaped: xy (bottom-left), xz (top-left), zy (bottom-right)
        xy_width, xy_height = x_inches, y_inches
        xz_width, xz_height = x_inches, z_inches
        zy_width, zy_height = z_inches, y_inches

        # Calculate figure dimensions
        left_width = max(xy_width, xz_width)
        fig_width = left_width + gap_inches + zy_width
        fig_height = xy_height + gap_inches + xz_height

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # xy panel (bottom-left)
        xy_left = (left_width - xy_width) / 2  # Center in left column
        xy_bottom = 0
        xy_ax = fig.add_axes(
            [
                xy_left / fig_width,
                xy_bottom / fig_height,
                xy_width / fig_width,
                xy_height / fig_height,
            ]
        )
        xy_ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
        xy_ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
        xy_ax.set_aspect("equal")

        # xz panel (top-left)
        xz_left = (left_width - xz_width) / 2  # Center in left column
        xz_bottom = xy_height + gap_inches
        xz_ax = fig.add_axes(
            [
                xz_left / fig_width,
                xz_bottom / fig_height,
                xz_width / fig_width,
                xz_height / fig_height,
            ]
        )
        xz_ax.set_xlim(data_bounds_min[0], data_bounds_max[0])
        xz_ax.set_ylim(data_bounds_min[2], data_bounds_max[2])
        xz_ax.set_aspect("equal")

        # zy panel (bottom-right, aligned with xy panel)
        zy_left = left_width + gap_inches
        zy_bottom = 0  # Align with xy panel bottom
        zy_ax = fig.add_axes(
            [
                zy_left / fig_width,
                zy_bottom / fig_height,
                zy_width / fig_width,
                zy_height / fig_height,
            ]
        )
        zy_ax.set_xlim(data_bounds_min[2], data_bounds_max[2])
        zy_ax.set_ylim(data_bounds_min[1], data_bounds_max[1])
        zy_ax.set_aspect("equal")
        if despine:
            for ax in [xy_ax, xz_ax, zy_ax]:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
        return fig, {"xy": xy_ax, "xz": xz_ax, "zy": zy_ax}

    else:
        raise ValueError(
            f"Unknown layout '{layout}'. Choose from 'side_by_side', 'stacked', or 'three_panel'."
        )
