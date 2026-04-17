"""Renderer-agnostic color and scalar utilities shared by plot.py and plot3d.py."""

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, ListedColormap, Normalize

__all__ = [
    "_is_discrete_data",
    "_get_discrete_colormap",
    "_create_discrete_color_dict",
    "_map_value_to_colors",
    "_rescale_scalar",
    "_resolve_color_parameter",
    "_resolve_scalar_parameter",
]


def _is_discrete_data(
    values: np.ndarray, max_unique_ratio: float = 0.05, max_unique_count: int = 20
) -> bool:
    """Detect if data should be treated as discrete/categorical.

    Parameters
    ----------
    values : np.ndarray
        Array of values to analyze
    max_unique_ratio : float, default 0.05
        Maximum ratio of unique values to total values for discrete classification
    max_unique_count : int, default 20
        Maximum number of unique values for discrete classification

    Returns
    -------
    bool
        True if data appears to be discrete/categorical
    """
    values = np.asarray(values)

    # Always treat string/object data as discrete
    if values.dtype.kind in ["U", "S", "O"]:
        return True

    # Always treat boolean data as discrete
    if values.dtype == bool:
        return True

    # For numeric data, check uniqueness
    unique_vals = np.unique(values[~pd.isna(values)])
    n_unique = len(unique_vals)
    n_total = len(values)

    if n_total == 0:
        return False

    unique_ratio = n_unique / n_total

    # Consider discrete if few unique values OR low unique ratio
    return n_unique <= max_unique_count or unique_ratio <= max_unique_ratio


def _get_discrete_colormap(colormap_name: str, n_colors: int) -> ListedColormap:
    """Get a discrete colormap with specified number of colors.

    Parameters
    ----------
    colormap_name : str
        Name of the colormap or 'auto' for automatic selection
    n_colors : int
        Number of discrete colors needed

    Returns
    -------
    ListedColormap
        Discrete colormap with n_colors
    """
    if colormap_name == "auto":
        if n_colors <= 10:
            colormap_name = "tab10"
        elif n_colors <= 20:
            colormap_name = "tab20"
        else:
            colormap_name = "hsv"

    qualitative_maps = {
        "Set1": 9,
        "Set2": 8,
        "Set3": 12,
        "Pastel1": 9,
        "Pastel2": 8,
        "Dark2": 8,
        "Accent": 8,
        "tab10": 10,
        "tab20": 20,
        "tab20b": 20,
        "tab20c": 20,
    }

    if colormap_name in qualitative_maps:
        base_cmap = plt.get_cmap(colormap_name)
        max_colors = qualitative_maps[colormap_name]

        if n_colors <= max_colors:
            colors = [base_cmap(i) for i in range(n_colors)]
        else:
            colors = [base_cmap(i % max_colors) for i in range(n_colors)]

        return ListedColormap(colors)

    base_cmap = plt.get_cmap(colormap_name)
    if n_colors == 1:
        colors = [base_cmap(0.5)]
    else:
        colors = [base_cmap(i / (n_colors - 1)) for i in range(n_colors)]

    return ListedColormap(colors)


def _create_discrete_color_dict(
    values: np.ndarray,
    colormap: Union[str, Colormap, ListedColormap] = "auto",
    missing_color: Union[str, Tuple[float, ...]] = "gray",
) -> Dict:
    """Create a color dictionary for discrete/categorical data.

    Parameters
    ----------
    values : np.ndarray
        Array of discrete values
    colormap : str, Colormap, or ListedColormap, default 'auto'
        Colormap specification
    missing_color : str or tuple, default 'gray'
        Color to use for missing/unmapped values

    Returns
    -------
    Dict
        Dictionary mapping values to colors
    """
    values = np.asarray(values)
    unique_vals = np.unique(values[~pd.isna(values)])
    n_unique = len(unique_vals)

    if n_unique == 0:
        return {}

    if isinstance(colormap, str):
        discrete_cmap = _get_discrete_colormap(colormap, n_unique)
    elif isinstance(colormap, ListedColormap):
        discrete_cmap = colormap
    else:
        if n_unique == 1:
            colors = [colormap(0.5)]
        else:
            colors = [colormap(i / (n_unique - 1)) for i in range(n_unique)]
        discrete_cmap = ListedColormap(colors)

    color_dict = {}
    for i, val in enumerate(unique_vals):
        color_dict[val] = discrete_cmap.colors[i % len(discrete_cmap.colors)]

    if missing_color is not None:
        color_dict["__missing__"] = missing_color

    return color_dict


def _map_value_to_colors(
    values: np.ndarray,
    colormap: Union[str, Colormap, Dict] = "cmc.hawaii",
    color_norm: Optional[Tuple[float, float]] = None,
    alpha: Union[float, np.ndarray] = 1.0,
    force_discrete: Optional[bool] = None,
    missing_color: Union[str, Tuple[float, ...]] = "gray",
) -> np.ndarray:
    """Map values to colors with automatic discrete/continuous detection.

    Parameters
    ----------
    values : np.ndarray
        Array of values to map to colors
    colormap : str, Colormap, or Dict, default "cmc.hawaii"
        Colormap specification
    color_norm : tuple of float, optional
        (min, max) tuple for color normalization (continuous data only)
    alpha : float or np.ndarray, default 1.0
        Alpha value(s) for colors
    force_discrete : bool, optional
        Force discrete (True) or continuous (False) mapping. If None, auto-detect.
    missing_color : str or tuple, default 'gray'
        Color for unmapped values in discrete mode

    Returns
    -------
    np.ndarray
        RGBA color array
    """
    values = np.asarray(values)

    if isinstance(alpha, (list, np.ndarray)):
        alpha = np.asarray(alpha)
        if len(alpha) != len(values):
            raise ValueError("Alpha array must have same length as values")
    else:
        alpha = np.full(len(values), alpha)

    if values.dtype == bool:
        values = values.astype(int)

    if isinstance(colormap, dict):
        rgba_colors = np.zeros((len(values), 4))
        rgba_colors[:, 3] = alpha

        missing_rgb = (
            mcolors.to_rgb(missing_color)
            if isinstance(missing_color, str)
            else missing_color[:3]
        )

        for i, val in enumerate(values):
            if pd.isna(val):
                rgba_colors[i, :3] = missing_rgb
            elif val not in colormap:
                rgba_colors[i, :3] = missing_rgb
            else:
                color = colormap[val]
                if isinstance(color, str):
                    if color.startswith("#"):
                        rgb = tuple(
                            int(color[j : j + 2], 16) / 255.0 for j in (1, 3, 5)
                        )
                    else:
                        rgb = mcolors.to_rgb(color)
                else:
                    rgb = color[:3]
                rgba_colors[i, :3] = rgb

        if isinstance(alpha, np.ndarray):
            if not np.allclose(alpha, 1.0):
                return rgba_colors
        elif not np.isclose(alpha, 1.0):
            return rgba_colors
        return rgba_colors[:, :3]

    is_discrete = force_discrete
    if is_discrete is None:
        is_discrete = _is_discrete_data(values)

    if is_discrete:
        color_dict = _create_discrete_color_dict(values, colormap, missing_color)
        return _map_value_to_colors(
            values, color_dict, alpha=alpha, missing_color=missing_color
        )

    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap

    clean_values = values.copy().astype(float)
    nan_mask = pd.isna(clean_values)

    if color_norm is not None:
        vmin, vmax = color_norm
        norm = Normalize(vmin=vmin, vmax=vmax)
        normalized_values = norm(clean_values)
    else:
        valid_values = clean_values[~nan_mask]
        if len(valid_values) == 0:
            normalized_values = np.zeros_like(clean_values)
        else:
            vmin, vmax = np.nanmin(valid_values), np.nanmax(valid_values)
            if vmin == vmax:
                normalized_values = np.zeros_like(clean_values)
            else:
                normalized_values = (clean_values - vmin) / (vmax - vmin)

    rgba_colors = cmap(normalized_values)

    if np.any(nan_mask):
        missing_rgb = (
            mcolors.to_rgb(missing_color)
            if isinstance(missing_color, str)
            else missing_color[:3]
        )
        rgba_colors[nan_mask, :3] = missing_rgb

    rgba_colors[:, 3] = alpha

    if isinstance(alpha, np.ndarray):
        if not np.allclose(alpha, 1.0):
            return rgba_colors
    elif not np.isclose(alpha, 1.0):
        return rgba_colors
    return rgba_colors[:, :3]


def _rescale_scalar(
    value: np.ndarray,
    norm: Optional[Tuple[float, float]],
    out_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Linearly rescale a scalar array to a new range with clipping.

    Parameters
    ----------
    value : np.ndarray
        Values to rescale
    norm : tuple of float, optional
        (min, max) normalization range for the input
    out_range : tuple of float, optional
        (min, max) output range

    Returns
    -------
    np.ndarray
        Rescaled values
    """
    if norm is None:
        norm = (np.min(value), np.max(value))
    if out_range is None:
        out_range = (np.min(value), np.max(value))
    return (out_range[1] - out_range[0]) * np.asarray(
        Normalize(*norm, clip=True)(value)
    ) + out_range[0]


def _resolve_color_parameter(
    color_param: Union[str, np.ndarray, tuple, Any],
    layer: Any,
) -> Union[np.ndarray, str, tuple, None]:
    """Resolve color parameter — try feature lookup first, then pass through.

    Parameters
    ----------
    color_param : str, np.ndarray, tuple, or Any
        Color specification to resolve. If a string that matches a feature
        name, returns the feature array; otherwise returns the value as-is.
    layer : Any
        Layer object with a ``get_feature`` method (SkeletonLayer,
        PointCloudLayer, GraphLayer, etc.).

    Returns
    -------
    np.ndarray, str, tuple, or None
        Feature array if a feature name was resolved; original value otherwise.
    """
    if isinstance(color_param, str):
        try:
            return layer.get_feature(color_param)
        except (KeyError, AttributeError):
            return color_param
    return color_param


def _resolve_scalar_parameter(
    value: Union[str, np.ndarray, List, "pd.Series", float, None],
    n_vertices: int,
    norm: Optional[Tuple[float, float]] = None,
    out_range: Optional[Tuple[float, float]] = None,
    layer: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Resolve, normalize, and optionally rescale a per-vertex scalar parameter.

    Handles four input types:

    - ``None``: returns ``None`` (parameter not set).
    - ``Number`` (int/float): returns a constant array of that value.
    - ``str``: resolves as a feature name from *layer*, then normalizes.
      Feature values are always normalized (auto-infer norm when not given).
    - ``array / list / Series``: returned as-is unless *norm* or *out_range*
      is provided, in which case the values are normalized and optionally
      rescaled to *out_range*.

    Parameters
    ----------
    value : str, array-like, float, or None
        Scalar specification.
    n_vertices : int
        Number of vertices; used to fill constant arrays.
    norm : tuple of float, optional
        ``(min, max)`` normalization range for the input. When *None* and
        normalization is needed, inferred from the data range.
    out_range : tuple of float, optional
        ``(min, max)`` output range after normalization.  When *None*,
        normalized values are left in ``[0, 1]``.
    layer : Any, optional
        Layer object with a ``get_feature(name)`` method; required when
        *value* is a feature name string.

    Returns
    -------
    np.ndarray or None
        Per-vertex scalar array, or ``None`` if *value* was ``None``.

    Raises
    ------
    ValueError
        If *value* is a string but *layer* is ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "scalar must be numeric, array-like, or a feature name — not bool"
        )
    if isinstance(value, Number):
        return np.full(n_vertices, float(value))

    if isinstance(value, str):
        if layer is None:
            raise ValueError(f"layer is required to resolve feature '{value}'")
        arr = np.asarray(layer.get_feature(value), dtype=float)
        # Feature names always normalize
        n = norm if norm is not None else (float(np.nanmin(arr)), float(np.nanmax(arr)))
        normalized = np.asarray(Normalize(*n, clip=True)(arr), dtype=float)
        if out_range is not None:
            return out_range[0] + normalized * (out_range[1] - out_range[0])
        return normalized

    # Array / list / Series
    arr = np.asarray(value, dtype=float)
    if norm is not None or out_range is not None:
        n = norm if norm is not None else (float(np.nanmin(arr)), float(np.nanmax(arr)))
        normalized = np.asarray(Normalize(*n, clip=True)(arr), dtype=float)
        if out_range is not None:
            return out_range[0] + normalized * (out_range[1] - out_range[0])
        return normalized
    return arr
