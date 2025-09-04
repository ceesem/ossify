from numbers import Number
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize

from .base import Cell, GraphLayer, MeshLayer, PointCloudLayer, SkeletonLayer


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


def projection_factory(
    proj: Literal["xy", "yx", "yz", "zy", "zx", "xz"],
    offset_h: Optional[float] = 0.0,
    offset_v: Optional[float] = 0.0,
) -> Callable:
    translate = np.array([offset_h, offset_v])

    match proj:
        case "xy":
            return (
                lambda pts: np.array(pts)[:, [0, 1]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
        case "yx":
            return (
                lambda pts: np.array(pts)[:, [1, 0]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
        case "zx":
            return (
                lambda pts: np.array(pts)[:, [2, 0]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
        case "xz":
            return (
                lambda pts: np.array(pts)[:, [0, 2]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
        case "zy":
            return (
                lambda pts: np.array(pts)[:, [2, 1]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
        case "yz":
            return (
                lambda pts: np.array(pts)[:, [1, 2]] + translate
                if len(pts) > 0
                else np.empty((0, 2))
            )
    raise ValueError(
        f"Unknown projection {proj}, expected one of 'xy', 'yx', 'yz', 'zy', 'zx', or 'xz'"
    )


def plot_skeleton(
    skel: SkeletonLayer,
    projection: Union[str, Callable] = "xy",
    colors: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    linewidths: Optional[np.ndarray] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    zorder: int = 2,
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
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    plt.Axes
        Matplotlib axes with skeleton plotted
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(projection, str):
        projection = projection_factory(
            proj=projection, offset_h=offset_h, offset_v=offset_v
        )

    for path in skel.cover_paths:
        match skel.parent_node_array[path[-1]]:
            case -1:
                path_plus = path
            case parent:
                path_plus = np.concat((path, [parent]))

        path_spatial = projection(skel.vertices[path_plus])
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
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    ax: Optional[plt.Axes] = None,
    marker: Optional[str] = "o",
    zorder: int = 2,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if isinstance(projection, str):
        projection = projection_factory(
            proj=projection, offset_h=offset_h, offset_v=offset_v
        )
    points_proj = projection(points)
    ax.scatter(
        x=points_proj[:, 0],
        y=points_proj[:, 1],
        s=sizes,
        c=colors,
        zorder=zorder,
        marker=marker,
    )
    return ax


def plot_2d(
    cell: Union[Cell, SkeletonLayer],
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "plasma",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = None,
    projection: Union[str, Callable] = "xy",
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    root_as_sphere: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    zorder: int = 2,
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
            colors=[root_color],
            sizes=[root_size],
            ax=ax,
            zorder=zorder + 1,
        )
    return ax
