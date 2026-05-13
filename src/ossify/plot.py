from numbers import Number
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .base import Cell, PointCloudLayer, SkeletonLayer
from .plot_utils import (
    _create_discrete_color_dict,
    _get_discrete_colormap,
    _is_discrete_data,
    _map_value_to_colors,
    _rescale_scalar,
    _resolve_color_to_array,
    _resolve_scalar_parameter,
    _resolve_size_with_transform,
)

__all__ = [
    "plot_cell_2d",
    "plot_morphology_2d",
    "plot_annotations_2d",
    "plot_cell_multiview",
    "plot_lineup",
    "plot_skeleton",
    "plot_points",
    "single_panel_figure",
    "multi_panel_figure",
    "add_scale_bar",
    "Rotation",
    "RotateCell",
]


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
    proj: Union[str, Callable],
) -> Callable:
    # If already a callable, return as-is
    if callable(proj):
        return proj

    # Handle string projections
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
        f"Unknown projection {proj}, expected a callable or one of 'xy', 'yx', 'yz', 'zy', 'zx', or 'xz'"
    )


# ---------------------------------------------------------------------------
# Rotation parameter resolution
# ---------------------------------------------------------------------------


def _resolve_rotation_params(
    projection: Union[str, Callable],
    rotation_angle: Optional[Union[float, int, Literal["best"]]],
    rotation_axis: Optional[Union[str, np.ndarray]],
    vertices: Optional[np.ndarray],
    center: Optional[np.ndarray],
    invert_y: bool,
) -> Union[str, Callable]:
    """Resolve inline rotation parameters into a projection callable.

    Parameters
    ----------
    projection : str or Callable
        Current projection specification.
    rotation_angle : float, int, "best", or None
        Rotation angle in degrees, or ``"best"`` for PCA-optimized.
    rotation_axis : str, np.ndarray, or None
        Rotation axis specification.
    vertices : np.ndarray or None
        Skeleton vertices for PCA modes, shape (N, 3).
    center : np.ndarray or None
        3D rotation center, shape (3,).
    invert_y : bool
        Whether to bake y-inversion into the rotation callable.

    Returns
    -------
    str or Callable
        The original projection if no rotation, or a rotation callable.

    Raises
    ------
    ValueError
        If parameters are incompatible or incomplete.
    """
    if rotation_angle is None:
        return projection

    # Cannot combine explicit projection with rotation
    if projection != "xy" and not callable(projection):
        raise ValueError(
            "Cannot combine projection with rotation_angle. "
            "Use the default projection='xy' when specifying rotation_angle."
        )
    if callable(projection):
        raise ValueError(
            "Cannot combine a callable projection with rotation_angle. "
            "Use the default projection='xy' when specifying rotation_angle."
        )

    if isinstance(rotation_angle, bool):
        raise ValueError("rotation_angle must be a number or 'best', not bool.")

    if isinstance(rotation_angle, (int, float)):
        if rotation_axis is None:
            raise ValueError(
                "rotation_axis is required when rotation_angle is numeric."
            )
        if center is None:
            raise ValueError("center is required when rotation_angle is numeric.")
        return Rotation(center, rotation_axis, rotation_angle, invert_y=invert_y)

    if rotation_angle == "best":
        if vertices is None or center is None:
            raise ValueError(
                "vertices and center are required when rotation_angle='best'."
            )
        pts_c = np.asarray(vertices, dtype=float) - center
        if rotation_axis is not None:
            k = _resolve_axis(rotation_axis)
            theta_deg = np.rad2deg(_best_angle_for_axis(pts_c, k))
            return Rotation(center, k, theta_deg, invert_y=invert_y)
        else:
            R = _pca_rotation_matrix(pts_c)
            return _build_projection_callable(center, R, None, invert_y)

    raise ValueError(
        f"rotation_angle must be a number or 'best', got {rotation_angle!r}"
    )


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

_AXIS_LABELS: Dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _resolve_axis(axis: Union[np.ndarray, Literal["x", "y", "z"]]) -> np.ndarray:
    """Return a unit vector for the given axis specification.

    Parameters
    ----------
    axis : np.ndarray or "x" | "y" | "z"
        Either a 3D vector or a string label.

    Returns
    -------
    np.ndarray
        Unit vector with shape (3,).

    Raises
    ------
    ValueError
        If a string other than "x", "y", "z" is given, or if the vector has
        zero length.
    """
    if isinstance(axis, str):
        if axis not in _AXIS_LABELS:
            raise ValueError(f"axis label must be 'x', 'y', or 'z', got {axis!r}")
        return _AXIS_LABELS[axis].copy()
    k = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(k)
    if norm == 0.0:
        raise ValueError("axis vector must have non-zero length")
    return k / norm


def _perp_basis(k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis (u, v) for the plane perpendicular to k.

    Parameters
    ----------
    k : np.ndarray
        Unit vector with shape (3,).

    Returns
    -------
    u, v : np.ndarray
        Two orthonormal vectors spanning the plane perpendicular to k.
    """
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(k, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, k) * k
    u /= np.linalg.norm(u)
    v = np.cross(k, u)
    return u, v


def _best_angle_for_axis(pts_c: np.ndarray, k: np.ndarray) -> float:
    """Find the rotation angle about k that maximizes 2D projected variance.

    Projects the centered points onto the plane perpendicular to k and runs
    2D PCA to find the angle that aligns the principal axis with the
    reference x-direction in that plane.

    Parameters
    ----------
    pts_c : np.ndarray
        Centered point cloud, shape (N, 3).
    k : np.ndarray
        Unit rotation axis, shape (3,).

    Returns
    -------
    float
        Optimal rotation angle in radians.
    """
    u, v = _perp_basis(k)
    p2 = pts_c @ np.column_stack([u, v])
    _, _, vt = np.linalg.svd(p2, full_matrices=False)
    pc1 = vt[0]
    theta = -np.arctan2(pc1[1], pc1[0])
    # Sign convention: majority of projected points should have positive x.
    R = _build_rotation_matrix(k, theta)
    projected_x = (pts_c @ R.T)[:, 0]
    if np.mean(projected_x) < 0:
        theta += np.pi
    return theta


def _build_rotation_matrix(k: np.ndarray, angle: float) -> np.ndarray:
    """Build the 3x3 Rodrigues rotation matrix for rotating by angle about k.

    Parameters
    ----------
    k : np.ndarray
        Unit rotation axis, shape (3,).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Shape (3, 3) rotation matrix.
    """
    kx, ky, kz = k
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * np.outer(k, k)
    )


def _pca_rotation_matrix(pts_c: np.ndarray) -> np.ndarray:
    """Build the globally optimal 3x3 rotation matrix from 3D PCA.

    Returns a rotation R such that R maps PC1 → x, PC2 → y, and PC3 → z,
    where PC1/PC2/PC3 are the principal components of pts_c ordered from
    highest to lowest variance. The result is guaranteed to have det = +1
    (proper rotation).

    Parameters
    ----------
    pts_c : np.ndarray
        Centered point cloud, shape (N, 3).

    Returns
    -------
    np.ndarray
        Shape (3, 3) rotation matrix.
    """
    _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
    pc1, pc2 = vt[0], vt[1]
    # Sign convention: majority of points should project to positive x then y.
    if np.mean(pts_c @ pc1) < 0:
        pc1 = -pc1
    if np.mean(pts_c @ pc2) < 0:
        pc2 = -pc2
    # Build right-handed system: PC3 = PC1 × PC2 ensures det = +1.
    pc3 = np.cross(pc1, pc2)
    return np.vstack([pc1, pc2, pc3])


def _build_projection_callable(
    center: np.ndarray,
    R: np.ndarray,
    new_center: Optional[np.ndarray],
    invert_y: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the projection closure from a precomputed rotation matrix.

    This is the single place where the (rotate → project → invert_y →
    shift) closure is constructed. Both :func:`Rotation` and the full-PCA
    branch of :func:`RotateCell` delegate here.

    Parameters
    ----------
    center : np.ndarray
        3D rotation center, shape (3,).
    R : np.ndarray
        3x3 rotation matrix.
    new_center : np.ndarray or None
        2D target position for the projected center, or None.
    invert_y : bool
        Whether to negate the y output coordinate.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function mapping (N, 3) → (N, 2).
    """
    y_sign = -1.0 if invert_y else 1.0

    if new_center is not None:
        new_center_2d = np.asarray(new_center, dtype=float)
        center_2d_after_invert = np.array([center[0], y_sign * center[1]])
        offset_2d = new_center_2d - center_2d_after_invert
    else:
        offset_2d = np.zeros(2)

    def _project(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        pts_r = (pts - center) @ R.T + center
        out = pts_r[:, :2].copy()
        out[:, 1] *= y_sign
        out += offset_2d
        return out

    return _project


def Rotation(
    center: np.ndarray,
    axis: Union[np.ndarray, Literal["x", "y", "z"]],
    angle: float,
    new_center: Optional[np.ndarray] = None,
    invert_y: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a projection callable that rotates 3D points and projects to 2D.

    The callable accepts an (N, 3) array of 3D points, applies a rotation
    about the given axis and center, projects to 2D (xy plane), optionally
    inverts y, and optionally translates the center to a new 2D location.

    Compatible with the ``projection`` parameter of all ossify plotting
    functions.

    Parameters
    ----------
    center : np.ndarray
        3D point about which the rotation is performed, shape (3,). The
        rotation center is a fixed point of the transform.
    axis : np.ndarray or "x" | "y" | "z"
        Rotation axis. String labels are converted to unit vectors.
    angle : float
        Rotation angle in degrees.
    new_center : np.ndarray, optional
        If provided, shifts the 2D output so that the projected rotation
        center appears at this 2D location. Shape (2,). Useful for
        centering a cell at the origin (``[0, 0]``) or at a specific
        layout position.
    invert_y : bool, default True
        If True, negates the y coordinate of the projected output, matching
        the image-space convention (y increases downward) used by the
        standard string projections.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function mapping (N, 3) → (N, 2).

    Examples
    --------
    Rotate 90° about the z-axis, centered at the soma:

    >>> proj = Rotation(soma_xyz, "z", 90)
    >>> plot.plot_skeleton(skel, projection=proj)

    Center the projected cell at the plot origin:

    >>> proj = Rotation(soma_xyz, "z", 90, new_center=np.array([0.0, 0.0]))
    """
    center = np.asarray(center, dtype=float)
    k = _resolve_axis(axis)
    angle_rad = np.deg2rad(angle)
    R = _build_rotation_matrix(k, angle_rad)
    return _build_projection_callable(center, R, new_center, invert_y)


def RotateCell(
    cell: "Cell",
    axis: Union[np.ndarray, Literal["x", "y", "z"], Literal["best"], None] = None,
    angle: Union[float, Literal["best"], None] = None,
    center: Optional[np.ndarray] = None,
    new_center: Optional[np.ndarray] = None,
    invert_y: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a projection callable for a cell, with automatic center and PCA modes.

    A high-level wrapper around :func:`Rotation` that extracts the rotation
    center from the cell's soma location and supports PCA-based automatic
    angle optimization.

    Parameters
    ----------
    cell : Cell
        The cell to build a projection for. Used to extract the default
        rotation center and skeleton vertices for PCA.
    axis : np.ndarray or "x" | "y" | "z" or "best" or None
        Rotation axis. String labels "x"/"y"/"z" are converted to unit
        vectors. ``"best"`` or ``None`` triggers full PCA: the minimum-
        variance axis of the skeleton is used (see Notes).
    angle : float or "best" or None
        Rotation angle in degrees, or ``"best"`` to find the optimal angle
        about the given axis via PCA. ``None`` is treated as 0.
    center : np.ndarray, optional
        3D rotation center. Defaults to ``cell.skeleton.root_location``.
    new_center : np.ndarray, optional
        2D display position for the rotation center after projection.
        Passed through to :func:`Rotation`.
    invert_y : bool, default True
        Passed through to :func:`Rotation`.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function mapping (N, 3) → (N, 2).

    Notes
    -----
    Both PCA modes share the same two-step algorithm via
    ``_best_angle_for_axis``:

    - **Axis given, angle="best"**: project skeleton vertices onto the plane
      perpendicular to the given axis, run 2D PCA, and return the angle that
      aligns the principal axis with x.
    - **axis="best" or None**: run 3D PCA first to find the minimum-variance
      direction (PC3), use that as the rotation axis, then apply the same
      constrained angle-finding step.

    Examples
    --------
    Rotate about y to the best orientation:

    >>> proj = RotateCell(cell, axis="y", angle="best")
    >>> plot.plot_cell_2d(cell, projection=proj)

    Fully automatic best view:

    >>> proj = RotateCell(cell)
    >>> plot.plot_cell_2d(cell, projection=proj)
    """
    # --- Resolve center ---
    if center is None:
        skel = cell.skeleton
        if skel is None or skel.root_location is None:
            raise ValueError(
                "Cell has no skeleton root_location; supply center explicitly."
            )
        center = np.asarray(skel.root_location, dtype=float)
    else:
        center = np.asarray(center, dtype=float)

    # --- Full-PCA branch: build R_pca directly (PC1→x, PC2→y, PC3→z) ---
    if axis is None or (isinstance(axis, str) and axis == "best"):
        skel = cell.skeleton
        if skel is None:
            raise ValueError(
                "Cell has no skeleton; cannot compute PCA. Supply axis explicitly."
            )
        pts_c = np.asarray(skel.vertices, dtype=float) - center
        R_pca = _pca_rotation_matrix(pts_c)
        return _build_projection_callable(center, R_pca, new_center, invert_y)

    # --- Resolve axis label / vector ---
    k = _resolve_axis(axis)

    # --- Resolve angle ---
    if angle == "best":
        skel = cell.skeleton
        if skel is None:
            raise ValueError(
                "Cell has no skeleton; cannot compute best angle. Supply angle explicitly."
            )
        pts_c = np.asarray(skel.vertices, dtype=float) - center
        theta = np.rad2deg(_best_angle_for_axis(pts_c, k))
    else:
        theta = float(angle) if angle is not None else 0.0

    return Rotation(center, k, theta, new_center=new_center, invert_y=invert_y)


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
    projection = projection_factory(proj=projection)
    projected = projection(vertices).astype(
        float
    )  # Ensure float type for offset operations
    projected[:, 0] += offset_h
    projected[:, 1] += offset_v
    xmin, xmax = projected[:, 0].min(), projected[:, 0].max()
    ymin, ymax = projected[:, 1].min(), projected[:, 1].max()
    return np.array([[xmin, xmax], [ymin, ymax]])


def plot_skeleton(
    skel: SkeletonLayer,
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
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
    do_autoscale_at_end = not ax.has_data() and ax.get_autoscale_on()

    # Store original projection for y-axis inversion detection
    orig_projection = projection

    # Resolve inline rotation parameters
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=skel.vertices,
        center=np.asarray(skel.root_location, dtype=float)
        if skel.root_location is not None
        else None,
        invert_y=invert_y,
    )

    projection = projection_factory(proj=projection)

    for path in skel.cover_paths_positional:
        # Convert vertex index to positional index for parent_node_array access
        path_end_vertex = path[-1]
        match skel.parent_node_array[path_end_vertex]:
            case -1:
                path_plus = path
            case parent:
                # Convert parent positional index back to vertex index
                path_plus = np.concat((path, [parent]))

        # Convert vertex indices to positional indices for vertices array access
        path_spatial = projection(skel.vertices[path_plus])
        path_spatial[:, 0] = path_spatial[:, 0] + offset_h
        path_spatial[:, 1] = path_spatial[:, 1] + offset_v
        path_segs = [
            (path_spatial[i], path_spatial[i + 1]) for i in range(len(path_spatial) - 1)
        ]

        # Extract styling for this path
        lc_kwargs = {}
        if colors is not None:
            # Convert vertex indices to positional indices for colors array access
            lc_kwargs["colors"] = colors[path_plus]
        if alpha is not None:
            # Convert vertex indices to positional indices for alpha array access
            lc_kwargs["alpha"] = alpha[path_plus]
        if linewidths is not None:
            # Convert vertex indices to positional indices for linewidths array access
            lc_kwargs["linewidths"] = linewidths[path]
        if zorder is not None:
            lc_kwargs["zorder"] = zorder

        lc = LineCollection(path_segs, capstyle="round", joinstyle="round", **lc_kwargs)
        ax.add_collection(lc)

    ax.set_aspect("equal")

    # Apply y-axis inversion if needed
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)
    if do_autoscale_at_end:
        ax.autoscale()
    return ax


def plot_points(
    points: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    palette: Optional[Union[str, Dict]] = None,
    color_norm: Optional[Tuple[float, float]] = None,
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    rotation_center: Optional[np.ndarray] = None,
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

    # Resolve inline rotation parameters
    if rotation_angle == "best":
        raise ValueError(
            "rotation_angle='best' is not supported for plot_points (no skeleton data). "
            "Use a numeric angle or pre-build a rotation callable."
        )
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=None,
        center=np.asarray(rotation_center, dtype=float)
        if rotation_center is not None
        else None,
        invert_y=invert_y,
    )

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
        colors = [palette[feature] for feature in colors]
    if colors is not None:
        if isinstance(colors, str):
            # Single color string
            scatter_kws["color"] = colors
        else:
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


def plot_annotations_2d(
    annotation: PointCloudLayer,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    alpha: float = 1,
    size: Optional[Union[str, np.ndarray, float]] = None,
    size_norm: Optional[Tuple[float, float]] = None,
    size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    sizes: Optional[np.ndarray] = (1, 30),
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    rotation_center: Optional[np.ndarray] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a 2D scatter of a :class:`PointCloudLayer` annotation.

    Parameters
    ----------
    annotation : PointCloudLayer
        Annotation layer to plot. Raw ``np.ndarray`` is also accepted, but
        feature-name resolution then requires the layer.
    color : str, np.ndarray, or tuple, optional
        Color specification. A string matching a feature name resolves to a
        per-point value array mapped through *palette*. Otherwise treated as
        a matplotlib color.
    palette : str or dict, default "coolwarm"
        Colormap or dict for scalar color mapping.
    color_norm : tuple of float, optional
        ``(min, max)`` clip range, in original (pre-transform) units.
    color_scale : {"log"} or None, optional
        Value transform applied before colormap projection. Mirrors
        :func:`plot_annotations_3d`.
    alpha : float, default 1
        Marker opacity.
    size : str, np.ndarray, or float, optional
        Marker size specification. Feature name, per-point array, or scalar.
    size_norm : tuple of float, optional
        ``(min, max)`` clip range for size mapping, in original units.
    size_scale : {"log", "sqrt", "cbrt"} or None, optional
        Value transform applied before size normalization. ``"sqrt"`` is
        useful when the feature is a cross-sectional area; ``"cbrt"`` when
        the feature is a volume. Mirrors :func:`plot_annotations_3d`.
    sizes : tuple of float, optional
        ``(min_size, max_size)`` output range for size rescaling. Default
        ``(1, 30)``.

    Returns
    -------
    plt.Axes
        Matplotlib axes with the annotation rendered.
    """
    if ax is None:
        ax = plt.gca()

    if not isinstance(annotation, PointCloudLayer):
        # Raw array — pass straight through; no feature-name resolution.
        return plot_points(
            points=np.asarray(annotation),
            sizes=size,
            colors=color,
            palette=palette,
            color_norm=color_norm,
            projection=projection,
            rotation_angle=rotation_angle,
            rotation_axis=rotation_axis,
            rotation_center=rotation_center,
            offset_h=offset_h,
            offset_v=offset_v,
            invert_y=invert_y,
            alpha=alpha,
            ax=ax,
            **kwargs,
        )

    vertices = annotation.vertices

    # Resolve color via the shared pipeline. Feature names get looked up;
    # scalar arrays get optionally log-transformed and palette-mapped.
    # Returns an (N, k) RGBA array, a scalar matplotlib color, or None.
    resolved_color = _resolve_color_to_array(
        color,
        annotation,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
    )

    # Resolve size via the shared transform pipeline. Numeric scalars pass
    # through unchanged; feature names and arrays go through the
    # log/sqrt/cbrt transform → norm → out_range pipeline.
    resolved_size = _resolve_size_with_transform(
        size,
        len(vertices),
        scale=size_scale,
        norm=size_norm,
        out_range=sizes,
        layer=annotation,
    )

    return plot_points(
        points=vertices,
        sizes=resolved_size,
        colors=resolved_color,
        # Palette intentionally not forwarded: when resolved_color is an
        # ndarray it's already pre-mapped to RGB(A), and when it's a scalar
        # color string/tuple palette is irrelevant. Forwarding palette here
        # would cause matplotlib to try to apply a colormap on top of
        # already-mapped RGB values.
        projection=projection,
        rotation_angle=rotation_angle,
        rotation_axis=rotation_axis,
        rotation_center=rotation_center,
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
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    alpha_extent: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    root_marker: bool = False,
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
        Color specification - can be feature name, array of values, or matplotlib color
    palette : str or dict, default "coolwarm"
        Colormap for mapping array values to colors
    color_norm : tuple of float, optional
        (min, max) tuple for color normalization, in the original
        (pre-transform) value space.
    color_scale : {"log"} or None, optional
        Value transform applied before colormap projection. ``"log"``
        log-transforms feature values so the colormap is distributed
        linearly in log-space. *color_norm* bounds remain in original units
        and are converted internally. Mirrors :func:`plot_morphology_3d`.
    alpha : str, np.ndarray, or float, default 1.0
        Alpha specification - can be feature name, array, or single value
    alpha_norm : tuple of float, optional
        (min, max) tuple for alpha normalization
    linewidth : str, np.ndarray, or float, default 1.0
        Linewidth specification - can be feature name, array, or single value
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

    # Resolve inline rotation parameters
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=skel.vertices,
        center=np.asarray(skel.root_location, dtype=float)
        if skel.root_location is not None
        else None,
        invert_y=invert_y,
    )

    # Resolve color through the shared pipeline. Returns an (N, k) array
    # for feature/array/pre-mapped inputs, a scalar matplotlib color for
    # uniform strings/tuples, or None.
    resolved_color = _resolve_color_to_array(
        color,
        skel,
        palette=palette,
        color_norm=color_norm,
        color_scale=color_scale,
    )

    colors_array = None
    if isinstance(resolved_color, np.ndarray):
        colors_array = resolved_color
    elif resolved_color is not None:
        import matplotlib.colors as mcolors

        single_color = mcolors.to_rgba(resolved_color)
        colors_array = np.tile(single_color, (skel.n_vertices, 1))

    # Process alpha: arrays without a norm are assumed pre-normalized to [0, 1]
    if alpha_extent is None:
        alpha_extent = (0.1, 1.0)
    rescale_alpha = isinstance(alpha, str) or alpha_norm is not None
    alpha_array = _resolve_scalar_parameter(
        alpha,
        skel.n_vertices,
        norm=alpha_norm,
        out_range=alpha_extent if rescale_alpha else None,
        layer=skel,
    )

    # Process linewidth: always normalize and rescale to widths range
    linewidth_array = _resolve_scalar_parameter(
        linewidth,
        skel.n_vertices,
        norm=linewidth_norm,
        out_range=widths,
        layer=skel,
    )

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
    if root_marker:
        if skel.root_location is not None:
            root_location = np.atleast_2d(skel.root_location)
            if root_color is None:
                if skel.base_root in skel.vertex_index:
                    root_color = (
                        colors_array[skel.root_positional]
                        if colors_array is not None
                        else None
                    )
                else:
                    raise ValueError(
                        "root_color must be provided explicitly if root is not in skeleton vertices"
                    )
            ax = plot_points(
                root_location,
                colors=root_color,
                sizes=[root_size],
                invert_y=invert_y,
                ax=ax,
                zorder=zorder + 1,
                projection=projection,
                offset_h=offset_h,
                offset_v=offset_v,
            )
    return ax


def plot_cell_2d(
    cell: Cell,
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    root_marker: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    synapses: Literal["pre", "post", "both", True, False] = False,
    pre_anno: str = "pre_syn",
    pre_color: Optional[Union[str, tuple]] = None,
    pre_palette: Union[str, dict] = "coolwarm",
    pre_color_norm: Optional[Tuple[float, float]] = None,
    syn_alpha: float = 1,
    syn_color_scale: Optional[Literal["log"]] = None,
    syn_size: Optional[Union[str, np.ndarray, float]] = None,
    syn_size_norm: Optional[Tuple[float, float]] = None,
    syn_size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    syn_sizes: Optional[np.ndarray] = (1, 30),
    post_anno: str = "post_syn",
    post_color: Optional[Union[str, tuple]] = None,
    post_palette: Union[str, dict] = "coolwarm",
    post_color_norm: Optional[Tuple[float, float]] = None,
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    units_per_inch: Optional[float] = None,
    dpi: Optional[float] = None,
    despine: bool = True,
    **syn_kwargs,
) -> plt.Axes:
    # Resolve inline rotation parameters at the top level
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=cell.skeleton.vertices,
        center=np.asarray(cell.skeleton.root_location, dtype=float)
        if cell.skeleton.root_location is not None
        else None,
        invert_y=invert_y,
    )

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
        color_scale=color_scale,
        alpha=alpha,
        alpha_norm=alpha_norm,
        linewidth=linewidth,
        linewidth_norm=linewidth_norm,
        widths=widths,
        root_marker=root_marker,
        root_size=root_size,
        root_color=root_color,
        projection=projection,
        offset_h=offset_h,
        offset_v=offset_v,
        invert_y=invert_y,
        ax=ax,
    )
    syn_common_kwargs = dict(
        alpha=syn_alpha,
        color_scale=syn_color_scale,
        size=syn_size,
        size_norm=syn_size_norm,
        size_scale=syn_size_scale,
        sizes=syn_sizes,
        ax=ax,
        offset_h=offset_h,
        offset_v=offset_v,
        invert_y=invert_y,
        projection=projection,
    )
    if synapses in ("both", "pre", True):
        if pre_anno in cell.annotations.names:
            ax = plot_annotations_2d(
                cell.annotations[pre_anno],
                color=pre_color,
                palette=pre_palette,
                color_norm=pre_color_norm,
                **syn_common_kwargs,
                **syn_kwargs,
            )
            syn_common_kwargs["ax"] = ax
    if synapses in ("both", "post", True):
        if post_anno in cell.annotations.names:
            ax = plot_annotations_2d(
                cell.annotations[post_anno],
                color=post_color,
                palette=post_palette,
                color_norm=post_color_norm,
                **syn_common_kwargs,
                **syn_kwargs,
            )
    return ax


def plot_cell_multiview(
    cell: Cell,
    layout: Literal["stacked", "side_by_side", "three_panel"] = "three_panel",
    color: Optional[Union[str, np.ndarray, tuple]] = None,
    palette: Union[str, dict] = "coolwarm",
    color_norm: Optional[Tuple[float, float]] = None,
    color_scale: Optional[Literal["log"]] = None,
    alpha: Optional[Union[str, np.ndarray, float]] = 1.0,
    alpha_norm: Optional[Tuple[float, float]] = None,
    linewidth: Optional[Union[str, np.ndarray, float]] = 1.0,
    linewidth_norm: Optional[Tuple[float, float]] = None,
    widths: Optional[tuple] = (1, 50),
    root_marker: bool = False,
    root_size: float = 100.0,
    root_color: Optional[Union[str, tuple]] = None,
    synapses: Literal["pre", "post", "both", True, False] = False,
    pre_anno: str = "pre_syn",
    pre_color: Optional[Union[str, tuple]] = None,
    pre_palette: Union[str, dict] = "coolwarm",
    pre_color_norm: Optional[Tuple[float, float]] = None,
    syn_alpha: float = 1,
    syn_color_scale: Optional[Literal["log"]] = None,
    syn_size: Optional[Union[str, np.ndarray, float]] = None,
    syn_size_norm: Optional[Tuple[float, float]] = None,
    syn_size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None,
    syn_sizes: Optional[np.ndarray] = (1, 30),
    post_anno: str = "post_syn",
    post_color: Optional[Union[str, tuple]] = None,
    post_palette: Union[str, dict] = "coolwarm",
    post_color_norm: Optional[Tuple[float, float]] = None,
    invert_y: bool = True,
    despine: bool = True,
    units_per_inch: float = 100_000,
    dpi: Optional[float] = None,
    **syn_kwargs,
) -> dict:
    fig, axes = multi_panel_figure(
        data_bounds_min=cell.skeleton.bbox[0],
        data_bounds_max=cell.skeleton.bbox[1],
        units_per_inch=units_per_inch,
        layout=layout,
        despine=despine,
        dpi=dpi,
    )
    for proj in axes:
        ax = axes[proj]
        plot_cell_2d(
            cell,
            color=color,
            palette=palette,
            color_norm=color_norm,
            color_scale=color_scale,
            alpha=alpha,
            alpha_norm=alpha_norm,
            linewidth=linewidth,
            linewidth_norm=linewidth_norm,
            widths=widths,
            root_marker=root_marker,
            root_size=root_size,
            root_color=root_color,
            projection=proj,
            invert_y=invert_y,
            synapses=synapses,
            syn_alpha=syn_alpha,
            syn_color_scale=syn_color_scale,
            syn_size=syn_size,
            syn_size_norm=syn_size_norm,
            syn_size_scale=syn_size_scale,
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
    return axes


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
    ax.set_autoscale_on(False)

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
        xy_ax.set_autoscale_on(False)

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
        zy_ax.set_autoscale_on(False)
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
        xz_ax.set_autoscale_on(False)

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
        xy_ax.set_autoscale_on(False)
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
        xy_ax.set_autoscale_on(False)

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
        xz_ax.set_autoscale_on(False)

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
        zy_ax.set_autoscale_on(False)
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


def add_scale_bar(
    ax: plt.Axes,
    length: float,
    position: Tuple[float, float] = (0.05, 0.05),
    color: str = "black",
    linewidth: float = 10.0,
    orientation: Literal["h", "v", "horizontal", "vertical"] = "h",
    feature: Optional[str] = None,
    feature_offset: float = 0.01,
    fontsize: float = 10,
) -> None:
    """Add a scale bar to an axis with precise positioning.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add scale bar to
    length : float
        Length of scale bar in data units
    position : tuple of float, default (0.05, 0.05)
        Starting position as fraction of axis dimensions (x_frac, y_frac).
        (0, 0) is bottom-left, (1, 1) is top-right.
    color : str, default "black"
        Color of the scale bar line
    linewidth : float, default 3.0
        Width of the scale bar line in points
    feature : str, optional
        Text feature for the scale bar (e.g., "100 μm")
    feature_offset : float, default 0.01
        Vertical offset for feature as fraction of axis height
    fontsize : float, default 10
        Font size for scale bar feature

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 100], [0, 50])
    >>> add_scale_bar(ax, length=20, position=(0.1, 0.1), feature="20 units")

    >>> # Add scale bar to morphology plot
    >>> fig, ax = single_panel_figure(bounds_min, bounds_max, 10)
    >>> plot_skeleton(skeleton, ax=ax)
    >>> add_scale_bar(ax, length=50, position=(0.8, 0.05), feature="50 μm")
    """
    # Get axis data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate axis data ranges
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Convert fractional position to data coordinates
    x_start = xlim[0] + position[0] * x_range
    y_start = ylim[0] + position[1] * y_range

    # Calculate end position (scale bar extends to the right)
    match orientation:
        case "h" | "horizontal":
            x_end = x_start + length
            y_end = y_start
        case "v" | "vertical":
            x_end = x_start
            if ax.yaxis_inverted():
                y_end = y_start - length
            else:
                y_end = y_start + length

    # Draw the scale bar line
    ax.plot(
        [x_start, x_end],
        [y_start, y_end],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
    )

    # Add feature if provided
    if feature is not None:
        # Position feature above the center of the scale bar
        match orientation:
            case "h" | "horizontal":
                feature_x = x_start + length / 2
                feature_y = y_start + feature_offset * y_range
            case "v" | "vertical":
                if ax.yaxis_inverted():
                    feature_x = x_start + feature_offset * x_range
                    feature_y = y_start - length / 2
                else:
                    feature_x = x_start + feature_offset * x_range
                    feature_y = y_start + length / 2

        ax.text(
            feature_x,
            feature_y,
            feature,
            ha="center",
            va="bottom",
            color=color,
            fontsize=fontsize,
        )


# ===========================================================================
# plot_lineup helpers and public API
# ===========================================================================


def _broadcast_param(val: Any, n: int) -> List[Any]:
    """Broadcast a scalar-or-list parameter to a list of length n.

    Parameters
    ----------
    val : Any
        A scalar value or a list of values.
    n : int
        Expected length.

    Returns
    -------
    List[Any]
        List of length n.

    Raises
    ------
    ValueError
        If val is a list with length != n.
    """
    if isinstance(val, list):
        if len(val) != n:
            raise ValueError(f"Expected list of length {n}, got length {len(val)}")
        return val
    return [val] * n


def _project_point_y(point_3d: np.ndarray, projection: Callable) -> float:
    """Project a single 3D point and return its projected y coordinate.

    Parameters
    ----------
    point_3d : np.ndarray
        A 3-element array representing a 3D point.
    projection : Callable
        Projection function mapping (N, 3) -> (N, 2).

    Returns
    -------
    float
        The y coordinate of the projected point.
    """
    result = projection(np.asarray(point_3d).reshape(1, 3))
    return float(result[0, 1])


def _lineup_offsets(
    cells: List[Cell],
    projection: Callable,
    gap: float,
    align: Literal["natural", "soma", "point"],
    alignment_points: Optional[List[np.ndarray]],
) -> List[Tuple[float, float]]:
    """Compute (offset_h, offset_v) for each cell in a lineup.

    Parameters
    ----------
    cells : List[Cell]
        Cells to compute offsets for.
    projection : Callable
        Projection function.
    gap : float
        Horizontal gap between cells.
    align : "natural", "soma", or "point"
        Vertical alignment mode.
    alignment_points : List[np.ndarray] or None
        One alignment point per cell; only used when align="point".

    Returns
    -------
    List[Tuple[float, float]]
        List of (offset_h, offset_v) per cell.
    """
    offsets = []
    cursor = 0.0
    for i, cell in enumerate(cells):
        skel = cell.skeleton
        bounds = _plotted_bounds(skel.vertices, projection)
        xmin, xmax = bounds[0]

        offset_h = cursor - xmin
        cursor += (xmax - xmin) + gap

        if align == "natural":
            offset_v = 0.0
        elif align == "soma":
            point = skel.root_location
            offset_v = -_project_point_y(point, projection)
        else:  # align == "point"
            point = alignment_points[i]  # type: ignore[index]
            offset_v = -_project_point_y(point, projection)

        offsets.append((offset_h, offset_v))
    return offsets


def plot_lineup(
    cells: List[Cell],
    projection: Union[str, Callable] = "xy",
    color: Optional[Union[str, np.ndarray, tuple, List]] = None,
    palette: Union[str, dict, List[Union[str, dict]]] = "coolwarm",
    color_norm: Optional[Union[Tuple[float, float], List]] = None,
    alpha: Optional[Union[str, np.ndarray, float, List]] = 1.0,
    alpha_norm: Optional[Union[Tuple[float, float], List]] = None,
    alpha_extent: Optional[Union[Tuple[float, float], List]] = None,
    linewidth: Optional[Union[str, np.ndarray, float, List]] = 1.0,
    linewidth_norm: Optional[Union[Tuple[float, float], List]] = None,
    widths: Optional[Union[tuple, List]] = (1, 50),
    root_marker: Union[bool, List[bool]] = False,
    root_size: Union[float, List[float]] = 100.0,
    root_color: Optional[Union[str, tuple, List]] = None,
    gap: float = 0.0,
    align: Literal["natural", "soma", "point"] = "natural",
    alignment_point: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
    units_per_inch: Optional[float] = None,
    dpi: Optional[float] = None,
    despine: bool = True,
) -> plt.Axes:
    """Plot multiple cells side-by-side in a single figure.

    Parameters
    ----------
    cells : List[Cell]
        Cells to plot.
    projection : str or Callable, default "xy"
        Shared projection for all cells.
    color : str, np.ndarray, tuple, or List, optional
        Per-cell or shared color specification.
    palette : str, dict, or List, default "coolwarm"
        Per-cell or shared colormap.
    color_norm : tuple or List, optional
        Per-cell or shared (min, max) color normalization.
    alpha : str, np.ndarray, float, or List, default 1.0
        Per-cell or shared alpha specification.
    alpha_norm : tuple or List, optional
        Per-cell or shared alpha normalization range.
    alpha_extent : tuple or List, optional
        Per-cell or shared alpha output range.
    linewidth : str, np.ndarray, float, or List, default 1.0
        Per-cell or shared linewidth specification.
    linewidth_norm : tuple or List, optional
        Per-cell or shared linewidth normalization.
    widths : tuple or List, optional
        Per-cell or shared (min, max) linewidth scaling.
    root_marker : bool or List[bool], default False
        Per-cell or shared root marker flag.
    root_size : float or List[float], default 100.0
        Per-cell or shared root marker size.
    root_color : str, tuple, or List, optional
        Per-cell or shared root marker color.
    gap : float, default 0.0
        Horizontal gap between cells.
    align : "natural", "soma", or "point", default "natural"
        Vertical alignment mode.
    alignment_point : np.ndarray or List[np.ndarray], optional
        Required when align="point". A single 3D point (broadcast) or one per cell.
    invert_y : bool, default True
        Whether to invert the y axis.
    ax : plt.Axes, optional
        Existing axes to plot onto. If None, a new figure is created.
    units_per_inch : float, optional
        Data units per inch for auto-sizing the figure. Used only when ax=None.
    dpi : float, optional
        Figure DPI. Used only when ax=None and units_per_inch is given.
    despine : bool, default True
        Whether to remove spines when creating a new figure.

    Returns
    -------
    plt.Axes
        Axes with all cells plotted.

    Raises
    ------
    ValueError
        If cells is empty, if a list param has the wrong length, or if
        align="point" but alignment_point is None.
    """
    if not cells:
        raise ValueError("cells must be a non-empty list")

    n = len(cells)
    proj_callable = projection_factory(projection)

    # Broadcast all per-cell styling params
    color_list = _broadcast_param(color, n)
    palette_list = _broadcast_param(palette, n)
    color_norm_list = _broadcast_param(color_norm, n)
    alpha_list = _broadcast_param(alpha, n)
    alpha_norm_list = _broadcast_param(alpha_norm, n)
    alpha_extent_list = _broadcast_param(alpha_extent, n)
    linewidth_list = _broadcast_param(linewidth, n)
    linewidth_norm_list = _broadcast_param(linewidth_norm, n)
    widths_list = _broadcast_param(widths, n)
    root_marker_list = _broadcast_param(root_marker, n)
    root_size_list = _broadcast_param(root_size, n)
    root_color_list = _broadcast_param(root_color, n)

    # Resolve alignment points
    if align == "point":
        if alignment_point is None:
            raise ValueError("alignment_point is required when align='point'")
        alignment_points: Optional[List[np.ndarray]] = _broadcast_param(
            alignment_point, n
        )
    else:
        alignment_points = None

    offsets = _lineup_offsets(cells, proj_callable, gap, align, alignment_points)

    if ax is None and units_per_inch is not None:
        # Compute combined bounds across all cells
        all_bounds = [
            _plotted_bounds(
                cells[i].skeleton.vertices,
                proj_callable,
                offsets[i][0],
                offsets[i][1],
            )
            for i in range(n)
        ]
        xmin = min(b[0, 0] for b in all_bounds)
        xmax = max(b[0, 1] for b in all_bounds)
        ymin = min(b[1, 0] for b in all_bounds)
        ymax = max(b[1, 1] for b in all_bounds)
        _, ax = single_panel_figure(
            np.array([xmin, ymin]),
            np.array([xmax, ymax]),
            units_per_inch,
            despine,
            dpi,
        )

    if ax is None:
        _, ax = plt.subplots()

    for i, cell in enumerate(cells):
        offset_h, offset_v = offsets[i]
        plot_morphology_2d(
            cell,
            projection=proj_callable,
            color=color_list[i],
            palette=palette_list[i],
            color_norm=color_norm_list[i],
            alpha=alpha_list[i],
            alpha_norm=alpha_norm_list[i],
            alpha_extent=alpha_extent_list[i],
            linewidth=linewidth_list[i],
            linewidth_norm=linewidth_norm_list[i],
            widths=widths_list[i],
            root_marker=root_marker_list[i],
            root_size=root_size_list[i],
            root_color=root_color_list[i],
            offset_h=offset_h,
            offset_v=offset_v,
            invert_y=invert_y,
            ax=ax,
        )

    return ax
