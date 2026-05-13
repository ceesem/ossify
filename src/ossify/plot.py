import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from .base import Cell, GraphLayer, MeshLayer, PointCloudLayer, SkeletonLayer
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
    "plot_lineup_grid",
    "LineupGroup",
    "plot_skeleton",
    "plot_mesh_2d",
    "plot_graph_2d",
    "plot_points",
    "single_panel_figure",
    "multi_panel_figure",
    "add_scale_bar",
    "add_layer_lines",
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
        # _should_invert_y_axis only returns True for str projections, but
        # be defensive about length — we only know how to interpret the
        # two-character axis-pair conventions ("xy", "zy", "yx", "yz").
        if not isinstance(projection, str) or len(projection) != 2:
            return ax
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

        # Extract styling for this path. Note the slicing asymmetry:
        # colors and alpha index by `path_plus` (per-vertex, length matches
        # all endpoints including the parent), but linewidths index by
        # `path` (per-segment, length matches the segment count which is
        # len(path_plus) - 1).
        lc_kwargs = {"zorder": zorder}
        if colors is not None:
            lc_kwargs["colors"] = colors[path_plus]
        if alpha is not None:
            lc_kwargs["alpha"] = alpha[path_plus]
        if linewidths is not None:
            lc_kwargs["linewidths"] = linewidths[path]

        lc = LineCollection(path_segs, capstyle="round", joinstyle="round", **lc_kwargs)
        ax.add_collection(lc)

    ax.set_aspect("equal")

    # Apply y-axis inversion if needed
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)
    if do_autoscale_at_end:
        ax.autoscale()
    return ax


def plot_mesh_2d(
    mesh: MeshLayer,
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    colors: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    edgecolors: Optional[Union[str, tuple]] = "none",
    linewidths: float = 0.0,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    zorder: int = 1,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a mesh as projected filled triangles in 2D.

    Low-level mesh renderer analogous to :func:`plot_skeleton`. Accepts
    pre-resolved per-vertex or per-face color arrays and renders projected
    triangles via :class:`matplotlib.collections.PolyCollection`.

    Parameters
    ----------
    mesh : MeshLayer
        Mesh layer to render.
    projection : str or Callable, default "xy"
        Projection function or string mapping 3D points to 2D.
    rotation_angle, rotation_axis : optional
        Inline rotation specification, same conventions as
        :func:`plot_skeleton`.
    colors : np.ndarray, optional
        Per-vertex ``(N, 3)`` / ``(N, 4)`` RGB/RGBA color array, or
        per-face ``(M, 3)`` / ``(M, 4)`` array. Per-vertex colors are
        averaged across each face's three vertices.
    alpha : float, optional
        Uniform fill opacity. Per-vertex / per-face opacity should be
        baked into the alpha channel of *colors*.
    edgecolors : str or tuple, default "none"
        Color for triangle edges. Default ``"none"`` hides edges entirely
        (clean fill). Pass e.g. ``"black"`` for a wireframe-on-surface look.
    linewidths : float, default 0.0
        Width of triangle edges in points. Only visible when
        *edgecolors* is not ``"none"``.
    offset_h, offset_v : float
        Horizontal / vertical offsets applied to the projected vertices.
    zorder : int, default 1
        Drawing order. Default ``1`` places mesh below the skeleton
        (which defaults to ``zorder=2``).
    invert_y : bool, default True
        Whether to invert the y-axis for projections containing "y".
    ax : plt.Axes, optional
        Existing matplotlib axes. A new one is created if omitted.

    Returns
    -------
    plt.Axes
        Axes with the mesh added.
    """
    if ax is None:
        ax = plt.gca()
    do_autoscale_at_end = not ax.has_data() and ax.get_autoscale_on()

    orig_projection = projection
    center = np.asarray(mesh.vertices.mean(axis=0), dtype=float)
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=mesh.vertices,
        center=center,
        invert_y=invert_y,
    )
    projection = projection_factory(proj=projection)

    verts_2d = projection(mesh.vertices).astype(float, copy=True)
    verts_2d[:, 0] += offset_h
    verts_2d[:, 1] += offset_v

    faces = mesh.faces_positional
    triangles_2d = verts_2d[faces]  # shape (M, 3, 2)

    pc_kwargs: dict = {
        "zorder": zorder,
        "edgecolors": edgecolors,
        "linewidths": linewidths,
    }
    if colors is not None:
        # Per-vertex (N, k) → average over each face's three vertices.
        # Per-face (M, k), scalar color string/tuple → pass through unchanged.
        if (
            isinstance(colors, np.ndarray)
            and colors.ndim == 2
            and colors.shape[0] == mesh.n_vertices
        ):
            pc_kwargs["facecolors"] = colors[faces].mean(axis=1)
        else:
            pc_kwargs["facecolors"] = colors
    if alpha is not None:
        pc_kwargs["alpha"] = alpha

    pc = PolyCollection(triangles_2d, **pc_kwargs)
    ax.add_collection(pc)

    ax.set_aspect("equal")
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)
    if do_autoscale_at_end:
        ax.autoscale()
    return ax


def plot_graph_2d(
    graph: GraphLayer,
    projection: Union[str, Callable] = "xy",
    rotation_angle: Optional[Union[float, int, Literal["best"]]] = None,
    rotation_axis: Optional[Union[str, np.ndarray]] = None,
    colors: Optional[np.ndarray] = None,
    alpha: Optional[Union[float, np.ndarray]] = None,
    linewidths: Optional[Union[float, np.ndarray]] = None,
    offset_h: float = 0.0,
    offset_v: float = 0.0,
    zorder: int = 2,
    invert_y: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a graph's edges as line segments in 2D.

    Low-level graph renderer analogous to :func:`plot_skeleton`. Each edge
    becomes a single segment in a :class:`matplotlib.collections.LineCollection`;
    per-vertex style arrays are averaged over the two endpoints.

    Parameters
    ----------
    graph : GraphLayer
        Graph layer to render.
    projection : str or Callable, default "xy"
        Projection function or string mapping 3D points to 2D.
    rotation_angle, rotation_axis : optional
        Inline rotation specification, same conventions as
        :func:`plot_skeleton`.
    colors : np.ndarray, optional
        Per-vertex ``(N, 3)`` / ``(N, 4)`` RGB/RGBA color array.
        Averaged over each edge's two endpoints to give per-edge colors.
    alpha : float or np.ndarray, optional
        Scalar opacity, or a per-vertex array averaged to per-edge.
    linewidths : float or np.ndarray, optional
        Scalar line width, or a per-vertex array averaged to per-edge,
        in points.
    offset_h, offset_v : float
        Horizontal / vertical offsets applied to the projected vertices.
    zorder : int, default 2
        Drawing order.
    invert_y : bool, default True
        Whether to invert the y-axis for projections containing "y".
    ax : plt.Axes, optional
        Existing matplotlib axes. A new one is created if omitted.

    Returns
    -------
    plt.Axes
        Axes with the graph added.
    """
    if ax is None:
        ax = plt.gca()
    do_autoscale_at_end = not ax.has_data() and ax.get_autoscale_on()

    orig_projection = projection
    center = np.asarray(graph.vertices.mean(axis=0), dtype=float)
    projection = _resolve_rotation_params(
        projection,
        rotation_angle,
        rotation_axis,
        vertices=graph.vertices,
        center=center,
        invert_y=invert_y,
    )
    projection = projection_factory(proj=projection)

    verts_2d = projection(graph.vertices).astype(float, copy=True)
    verts_2d[:, 0] += offset_h
    verts_2d[:, 1] += offset_v

    edges = graph.edges_positional  # shape (E, 2)
    segments_2d = verts_2d[edges]  # shape (E, 2, 2)

    lc_kwargs: dict = {"zorder": zorder, "capstyle": "round"}
    if colors is not None:
        # Per-vertex (N, k) → mean over each edge's two endpoints.
        lc_kwargs["colors"] = colors[edges].mean(axis=1)
    if alpha is not None:
        if isinstance(alpha, np.ndarray):
            lc_kwargs["alpha"] = alpha[edges].mean(axis=1)
        else:
            lc_kwargs["alpha"] = alpha
    if linewidths is not None:
        if isinstance(linewidths, np.ndarray):
            lc_kwargs["linewidths"] = linewidths[edges].mean(axis=1)
        else:
            lc_kwargs["linewidths"] = linewidths

    lc = LineCollection(segments_2d, **lc_kwargs)
    ax.add_collection(lc)

    ax.set_aspect("equal")
    ax = _apply_y_inversion_to_axes(ax, orig_projection, invert_y)
    if do_autoscale_at_end:
        ax.autoscale()
    return ax


def plot_points(
    points: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    palette: Union[str, Dict] = "coolwarm",
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
    # Markers default to borderless (matplotlib's default is a 1-pt outline,
    # which dominates small synapse markers).
    if "linewidths" not in scatter_kws:
        scatter_kws["linewidths"] = 0
    if isinstance(palette, str):
        scatter_kws["cmap"] = palette
        if color_norm is not None:
            scatter_kws["vmin"], scatter_kws["vmax"] = color_norm
    elif isinstance(palette, dict) and colors is not None:
        # Dict palette: map each feature value to its color. Works for any
        # iterable of dict keys (1-D ndarray, list, pandas Series, …).
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
        Alpha specification - can be feature name, array, or single value.
        Feature names / arrays are rescaled to ``alpha_extent``.
    alpha_norm : tuple of float, optional
        (min, max) clip range for alpha values in original feature units.
    alpha_extent : tuple of float, optional
        ``(min, max)`` output range for rescaled alpha values.  Default
        ``(0.0, 1.0)`` — i.e., the dimmest vertex is fully transparent.
        Pass e.g. ``(0.1, 1.0)`` to keep low-end vertices faintly visible.
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
        single_color = mcolors.to_rgba(resolved_color)
        colors_array = np.tile(single_color, (skel.n_vertices, 1))

    # Process alpha: arrays without a norm are assumed pre-normalized to [0, 1].
    # When alpha is a feature name or alpha_norm is given, values are rescaled
    # to alpha_extent. Default extent is [0, 1] — pass alpha_extent=(0.1, 1.0)
    # or similar to keep low-end vertices visible.
    if alpha_extent is None:
        alpha_extent = (0.0, 1.0)
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
        if ax is not None:
            raise ValueError(
                "Pass either `ax` (to paint into an existing axes) or "
                "`units_per_inch` (to size a new figure), not both."
            )
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
        elif synapses == "pre":
            # The user asked specifically for pre; missing layer is silent
            # only when "both" or True is requested (graceful degradation).
            warnings.warn(
                f"synapses='pre' requested, but no '{pre_anno}' annotation "
                f"is present on cell '{cell.name}'. Skipping.",
                stacklevel=2,
            )
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
        elif synapses == "post":
            warnings.warn(
                f"synapses='post' requested, but no '{post_anno}' annotation "
                f"is present on cell '{cell.name}'. Skipping.",
                stacklevel=2,
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
    _, axes = multi_panel_figure(
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

    # Clamp degenerate dimensions so the figure is still renderable and
    # compatible with peers in a lineup or panel. We warn — the data is
    # degenerate, and the caller likely wants to know.
    _MIN_INCHES = 0.5
    if fig_width < _MIN_INCHES or fig_height < _MIN_INCHES:
        warnings.warn(
            f"single_panel_figure received degenerate bounds "
            f"({data_width:.3g} × {data_height:.3g} units → "
            f"{fig_width:.3g} × {fig_height:.3g} inches at "
            f"units_per_inch={units_per_inch}). Clamping figure dimensions "
            f'to at least {_MIN_INCHES}" so the plot remains visible.',
            stacklevel=2,
        )
        fig_width = max(fig_width, _MIN_INCHES)
        fig_height = max(fig_height, _MIN_INCHES)

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
        case _:
            raise ValueError(
                f"orientation must be 'h', 'horizontal', 'v', or 'vertical'; "
                f"got {orientation!r}"
            )

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
            case _:
                # Unreachable — the first match would have raised, but
                # keeps the match exhaustive so the type checker is happy.
                raise ValueError(f"orientation must be h/v; got {orientation!r}")

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
            data_bounds_min=np.array([xmin, ymin]),
            data_bounds_max=np.array([xmax, ymax]),
            units_per_inch=units_per_inch,
            despine=despine,
            dpi=dpi,
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


# ===========================================================================
# plot_lineup_grid: groups, multi-row, layer guides
# ===========================================================================


# Style fields on LineupGroup that broadcast per-cell when given as a list
# (everything plot_cell_2d takes that we expect users to vary by cell).
_GROUP_PER_CELL_FIELDS = (
    "color",
    "palette",
    "color_norm",
    "color_scale",
    "alpha",
    "alpha_norm",
    "alpha_extent",
    "linewidth",
    "linewidth_norm",
    "widths",
    "root_marker",
    "root_size",
    "root_color",
)

# Style fields treated uniformly across the group (passed through as-is).
_GROUP_UNIFORM_FIELDS = (
    "synapses",
    "pre_anno",
    "pre_color",
    "pre_palette",
    "pre_color_norm",
    "post_anno",
    "post_color",
    "post_palette",
    "post_color_norm",
    "syn_alpha",
    "syn_color_scale",
    "syn_size",
    "syn_size_norm",
    "syn_size_scale",
    "syn_sizes",
)


@dataclass
class LineupGroup:
    """A labeled bag of cells with shared styling, used by :func:`plot_lineup_grid`.

    Each style field mirrors the matching keyword on :func:`plot_cell_2d`.
    Per-cell fields (``color``, ``palette``, ``alpha``, ``linewidth``, the
    root markers, etc.) accept either a scalar (broadcast across all cells
    in the group) or a list with one entry per cell. Synapse-related
    fields are uniform across the group.

    Parameters
    ----------
    cells : list of Cell
        The cells in this group.
    label : str, optional
        Title displayed above the group's cells. ``None`` suppresses it.

    Examples
    --------
    Define named styles once, then build groups:

    >>> L2A = dict(color="compartment", palette={SWC_AXON: "tab:blue",
    ...                                          SWC_DENDRITE: "navy"})
    >>> L2B = dict(color="compartment", palette={SWC_AXON: "lightblue",
    ...                                          SWC_DENDRITE: "steelblue"})
    >>> groups = [
    ...     LineupGroup(l2a_cells, label="L2a", **L2A),
    ...     LineupGroup(l2b_cells, label="L2b", **L2B),
    ... ]
    >>> plot_lineup_grid(groups=groups, row_max_width=2000,
    ...                  layer_lines={0: "L1", 250: "L2/3", 500: "L4"})

    Mix styles within a single group via per-cell broadcasts:

    >>> LineupGroup(cells, label="Comparison",
    ...             color=["compartment"] * len(cells),
    ...             palette=[L2A["palette"], L3A["palette"], L2A["palette"]],
    ...             alpha=[1.0, 1.0, 0.3])
    """

    cells: List[Cell]
    label: Optional[str] = None

    # --- Per-cell projection / rotation (handled separately from styling).
    # The rotation_* fields integrate with each cell's own PCA, so even
    # ``rotation_angle="best"`` works in a lineup: each cell rotates around
    # its own root with its own per-cell optimal angle, then the laid-out
    # bounds are recomputed from the resulting projections.
    rotation_angle: Optional[Union[float, int, Literal["best"], List]] = None
    rotation_axis: Optional[Union[str, np.ndarray, List]] = None

    # --- Skeleton styling (per-cell broadcastable) ---
    color: Optional[Union[str, np.ndarray, tuple, List]] = None
    palette: Union[str, dict, List] = "coolwarm"
    color_norm: Optional[Union[Tuple[float, float], List]] = None
    color_scale: Optional[Union[Literal["log"], List]] = None
    alpha: Union[str, np.ndarray, float, List] = 1.0
    alpha_norm: Optional[Union[Tuple[float, float], List]] = None
    alpha_extent: Optional[Union[Tuple[float, float], List]] = None
    linewidth: Union[str, np.ndarray, float, List] = 1.0
    linewidth_norm: Optional[Union[Tuple[float, float], List]] = None
    widths: Optional[Union[tuple, List]] = (1, 50)
    root_marker: Union[bool, List[bool]] = False
    root_size: Union[float, List[float]] = 100.0
    root_color: Optional[Union[str, tuple, List]] = None

    # --- Synapse styling (uniform across the group) ---
    synapses: Literal["pre", "post", "both", True, False] = False
    pre_anno: str = "pre_syn"
    pre_color: Optional[Union[str, tuple]] = None
    pre_palette: Union[str, dict] = "coolwarm"
    pre_color_norm: Optional[Tuple[float, float]] = None
    post_anno: str = "post_syn"
    post_color: Optional[Union[str, tuple]] = None
    post_palette: Union[str, dict] = "coolwarm"
    post_color_norm: Optional[Tuple[float, float]] = None
    syn_alpha: float = 1.0
    syn_color_scale: Optional[Literal["log"]] = None
    syn_size: Optional[Union[str, np.ndarray, float]] = None
    syn_size_norm: Optional[Tuple[float, float]] = None
    syn_size_scale: Optional[Literal["log", "sqrt", "cbrt"]] = None
    syn_sizes: Optional[np.ndarray] = (1, 30)

    # Escape hatch for any plot_cell_2d kwarg we don't expose explicitly.
    extra_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


def _resolve_cell_style(group: LineupGroup, cell_idx: int) -> Dict[str, Any]:
    """Resolve a single cell's plot_cell_2d kwargs from its group's fields.

    Per-cell broadcastable fields are indexed into when given as a list;
    uniform fields pass through unchanged. Rotation fields are *not*
    included — those are handled separately via :func:`_resolve_group_projections`.
    """
    n = len(group.cells)
    out: Dict[str, Any] = {}
    for name in _GROUP_PER_CELL_FIELDS:
        val = getattr(group, name)
        if isinstance(val, list):
            if len(val) != n:
                raise ValueError(
                    f"LineupGroup field {name!r} has length {len(val)} "
                    f"but group has {n} cells."
                )
            out[name] = val[cell_idx]
        else:
            out[name] = val
    for name in _GROUP_UNIFORM_FIELDS:
        out[name] = getattr(group, name)
    if group.extra_kwargs:
        out.update(group.extra_kwargs)
    return out


def _resolve_group_projections(
    group: LineupGroup,
    base_projection: Union[str, Callable],
    invert_y: bool,
) -> List[Callable]:
    """Build a per-cell projection callable for each cell in *group*.

    When the group's ``rotation_angle``/``rotation_axis`` are unset, every
    cell uses *base_projection* directly. When set (including
    ``rotation_angle="best"``), each cell's projection is its own
    rotation callable built from the cell's vertices and root location.
    """
    n = len(group.cells)
    rot_angle_list = _broadcast_param(group.rotation_angle, n)
    rot_axis_list = _broadcast_param(group.rotation_axis, n)

    projections: List[Callable] = []
    for ci, cell in enumerate(group.cells):
        ra = rot_angle_list[ci]
        rx = rot_axis_list[ci]
        if ra is None and rx is None:
            projections.append(projection_factory(base_projection))
            continue
        skel = cell.skeleton
        center = (
            np.asarray(skel.root_location, dtype=float)
            if skel is not None and skel.root_location is not None
            else None
        )
        vertices = skel.vertices if skel is not None else None
        resolved = _resolve_rotation_params(
            base_projection,
            ra,
            rx,
            vertices=vertices,
            center=center,
            invert_y=invert_y,
        )
        projections.append(projection_factory(resolved))
    return projections


def _grid_offsets(
    groups: List[LineupGroup],
    projection: Union[Callable, List[List[Callable]]],
    align: Literal["natural", "soma", "point"],
    inter_cell_gap: float,
    inter_group_gap: float,
    row_max_cells: Optional[int],
    row_max_width: Optional[float],
    row_gap: float,
    alignment_points: Optional[List[List[np.ndarray]]],
    y_axis_inverted: bool = False,
) -> Tuple[
    List[List[Tuple[float, float]]],
    List[Optional[Tuple[float, float, float]]],
]:
    """Compute per-cell offsets and per-group label anchors.

    Groups are placed left-to-right; when ``row_max_cells`` or
    ``row_max_width`` is set, groups wrap to a new row as a unit (a single
    group is never split between rows).

    Parameters
    ----------
    projection : Callable or list of list of Callable
        Either a single projection callable applied to all cells, or a
        per-cell projection (``projection[gi][ci]`` for cell ``ci`` of
        group ``gi``). The per-cell form supports per-cell rotation —
        e.g. ``rotation_angle="best"`` produces a different rotation
        callable for every cell.
    y_axis_inverted : bool, default False
        Whether the rendered y axis will be inverted on display (matplotlib's
        ``ax.invert_yaxis()``). Controls the row-stacking direction:
        subsequent rows should always appear *below* the previous one on
        screen, so when the axis is inverted we stack toward larger data y;
        when it isn't we stack toward smaller data y. Caller should compute
        this as ``invert_y and _should_invert_y_axis(projection_str)``.

    Returns
    -------
    cell_offsets : list of list of (float, float)
        ``cell_offsets[gi][ci]`` is the ``(offset_h, offset_v)`` for cell
        ``ci`` in group ``gi``, in projected data coordinates.
    group_label_anchors : list of (center_x, ymin, ymax) or None
        For each group, the horizontal center plus the min and max
        projected y of its rendered cells (after offsets applied), or
        ``None`` if the group has no label. The caller chooses whether to
        place the label above ymin or below ymax based on the axis
        inversion state.
    """
    # Normalize projection to per-cell list-of-lists so the rest of the
    # function only has one shape to handle.
    if callable(projection):
        per_cell_projections: List[List[Callable]] = [
            [projection] * len(g.cells) for g in groups
        ]
    else:
        per_cell_projections = projection

    # Step 1: gather per-cell projected bounds using each cell's own projection.
    group_data: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups):
        cells_data = []
        for ci, cell in enumerate(group.cells):
            proj = per_cell_projections[gi][ci]
            bounds = _plotted_bounds(cell.skeleton.vertices, proj)
            xmin, xmax = bounds[0]
            ymin, ymax = bounds[1]
            cells_data.append(
                {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "width": xmax - xmin,
                }
            )
        total_width = sum(c["width"] for c in cells_data) + inter_cell_gap * max(
            len(cells_data) - 1, 0
        )
        group_data.append(
            {
                "cells": cells_data,
                "total_width": total_width,
                "count": len(cells_data),
            }
        )

    # Step 2: pack groups into rows.
    rows: List[List[int]] = []
    current: List[int] = []
    current_width = 0.0
    current_count = 0
    for gi, gd in enumerate(group_data):
        if not current:
            current.append(gi)
            current_width = gd["total_width"]
            current_count = gd["count"]
            continue
        tentative_width = current_width + inter_group_gap + gd["total_width"]
        tentative_count = current_count + gd["count"]
        wrap_by_count = row_max_cells is not None and tentative_count > row_max_cells
        wrap_by_width = row_max_width is not None and tentative_width > row_max_width
        if wrap_by_count or wrap_by_width:
            rows.append(current)
            current = [gi]
            current_width = gd["total_width"]
            current_count = gd["count"]
        else:
            current.append(gi)
            current_width = tentative_width
            current_count = tentative_count
    if current:
        rows.append(current)

    # Step 3: per-row y baseline. Each row's height is the max cell ymax
    # minus min cell ymin in projected coords. Subsequent rows always
    # appear *below* on screen — we just choose the sign of the data-y
    # step based on whether the axis will be inverted on render.
    row_step_sign = +1.0 if y_axis_inverted else -1.0
    row_baselines = [0.0]
    for r in range(len(rows) - 1):
        # Height = max(ymax) - min(ymin) across all cells in this row.
        row_cells_data = [cd for gi in rows[r] for cd in group_data[gi]["cells"]]
        if row_cells_data:
            row_height = max(c["ymax"] for c in row_cells_data) - min(
                c["ymin"] for c in row_cells_data
            )
        else:
            row_height = 0.0
        row_baselines.append(row_baselines[-1] + row_step_sign * (row_height + row_gap))

    # Step 4: compute per-cell (offset_h, offset_v) and group label anchors.
    cell_offsets: List[List[Tuple[float, float]]] = [[] for _ in groups]
    group_label_anchors: List[Optional[Tuple[float, float, float]]] = [None] * len(
        groups
    )

    for row_idx, row_group_indices in enumerate(rows):
        cursor_x = 0.0
        row_y = row_baselines[row_idx]
        for gi in row_group_indices:
            gd = group_data[gi]
            group_x_start = cursor_x
            group_min_y_plotted = float("inf")
            group_max_y_plotted = float("-inf")
            for ci, cd in enumerate(gd["cells"]):
                cell = groups[gi].cells[ci]
                cell_proj = per_cell_projections[gi][ci]
                offset_h = cursor_x - cd["xmin"]
                if align == "natural":
                    offset_v = row_y
                elif align == "soma":
                    soma_y = _project_point_y(cell.skeleton.root_location, cell_proj)
                    offset_v = row_y - soma_y
                else:  # "point"
                    pt = alignment_points[gi][ci]  # type: ignore[index]
                    offset_v = row_y - _project_point_y(pt, cell_proj)
                cell_offsets[gi].append((offset_h, offset_v))
                plotted_ymin = cd["ymin"] + offset_v
                plotted_ymax = cd["ymax"] + offset_v
                group_min_y_plotted = min(group_min_y_plotted, plotted_ymin)
                group_max_y_plotted = max(group_max_y_plotted, plotted_ymax)
                cursor_x += cd["width"] + inter_cell_gap
            # Remove trailing inter_cell_gap, add inter_group_gap.
            if gd["count"] > 0:
                cursor_x -= inter_cell_gap
            cursor_x += inter_group_gap
            # Record group label anchor: horizontal center + plotted y extent.
            if groups[gi].label is not None and gd["count"] > 0:
                center_x = (group_x_start + cursor_x - inter_group_gap) / 2.0
                group_label_anchors[gi] = (
                    center_x,
                    group_min_y_plotted,
                    group_max_y_plotted,
                )

    return cell_offsets, group_label_anchors


def add_layer_lines(
    ax: plt.Axes,
    layer_lines: Dict[float, Optional[str]],
    color: str = "gray",
    linestyle: str = "--",
    linewidth: float = 0.5,
    label_fontsize: float = 9.0,
    label_pad: float = 0.01,
    label_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
) -> plt.Axes:
    """Add horizontal layer reference lines with optional left-margin labels.

    Parameters
    ----------
    ax : plt.Axes
        Axes to annotate.
    layer_lines : dict of float -> str or None
        Mapping from y-coordinate (in data units) to a label string. Pass
        ``None`` as the value to draw the line without a label.
    color : str, default "gray"
        Color for both lines and labels.
    linestyle : str, default "--"
        Line style for the reference lines.
    linewidth : float, default 0.5
        Width of the reference lines in points.
    label_fontsize : float, default 9.0
        Font size for labels.
    label_pad : float, default 0.01
        Horizontal pad between the axis edge and each label, expressed
        as a fraction of the axes width.
    label_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`Axes.text` for the
        labels. Overrides any of the defaults above.
    line_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`Axes.axhline`.
        Overrides any of the defaults above.

    Returns
    -------
    plt.Axes
        The same axes, with lines and labels added.

    Examples
    --------
    >>> add_layer_lines(ax, {0: "L1", 250: "L2/3", 500: "L4",
    ...                      800: "L5", 1100: "L6"})
    """
    line_defaults = {
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
    }
    line_defaults.update(line_kwargs or {})

    label_defaults = {
        "color": color,
        "fontsize": label_fontsize,
        "ha": "right",
        "va": "center",
    }
    label_defaults.update(label_kwargs or {})

    # Labels positioned in axes x-coords (fraction), data y-coords.
    text_transform = ax.get_yaxis_transform()
    label_x = -label_pad

    for y, label in layer_lines.items():
        ax.axhline(y, **line_defaults)
        if label is not None:
            ax.text(label_x, y, label, transform=text_transform, **label_defaults)
    return ax


def plot_lineup_grid(
    groups: List[LineupGroup],
    *,
    projection: Union[str, Callable] = "xy",
    align: Literal["natural", "soma", "point"] = "natural",
    inter_cell_gap: float = 0.0,
    inter_group_gap: float = 0.0,
    row_max_cells: Optional[int] = None,
    row_max_width: Optional[float] = None,
    row_gap: float = 0.0,
    layer_lines: Optional[Dict[float, Optional[str]]] = None,
    layer_line_kwargs: Optional[dict] = None,
    group_label_offset: float = 0.0,
    group_label_kwargs: Optional[dict] = None,
    alignment_points: Optional[List[List[np.ndarray]]] = None,
    invert_y: bool = True,
    units_per_inch: Optional[float] = None,
    dpi: Optional[float] = None,
    despine: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a grid of cell groups with per-group styling and multi-row layout.

    Each group's cells are placed contiguously, with optional gaps between
    cells and between groups. When ``row_max_cells`` or ``row_max_width``
    is set, groups wrap to new rows as units — a single group is never
    split across rows. Group labels float above each group's plotted cells
    (locally per group, not globally per row). Optional layer guide lines
    can be drawn across the full plot.

    Parameters
    ----------
    groups : list of LineupGroup
        Groups of cells with shared styling. Build these by spreading
        named style dicts into the constructor; see :class:`LineupGroup`
        for examples.
    projection : str or Callable, default "xy"
        Shared projection for all cells.
    align : {"natural", "soma", "point"}, default "natural"
        Vertical alignment mode. ``"natural"`` preserves each cell's y
        coordinate (anatomical depth); ``"soma"`` aligns each cell's
        soma to the row's reference y; ``"point"`` aligns the per-cell
        ``alignment_points`` to the row reference.
    inter_cell_gap : float, default 0.0
        Horizontal spacing between adjacent cells within a group, in
        data units.
    inter_group_gap : float, default 0.0
        Extra horizontal spacing between adjacent groups, in data units.
    row_max_cells : int, optional
        Wrap to a new row before the running cell count exceeds this.
    row_max_width : float, optional
        Wrap to a new row before the running row width (data units)
        exceeds this. When both ``row_max_cells`` and ``row_max_width``
        are given, whichever fires first triggers the wrap.
    row_gap : float, default 0.0
        Vertical spacing between rows, in data units.
    layer_lines : dict of float -> str or None, optional
        ``{y: label}`` mapping passed to :func:`add_layer_lines`. Pass
        ``None`` as a value to draw the line without a label.
    layer_line_kwargs : dict, optional
        Extra kwargs forwarded to :func:`add_layer_lines`.
    group_label_offset : float, default 0.0
        Vertical offset from each group's projected top edge to its
        label, in data units. Direction is "above on screen": for
        y-inverted plots (the default) the label sits at
        ``min_plotted_y - group_label_offset``; otherwise at
        ``max_plotted_y + group_label_offset``.
    group_label_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`Axes.text` for the
        group labels (e.g. ``fontsize``, ``color``, ``fontweight``).
    alignment_points : list of list of np.ndarray, optional
        Required when ``align="point"``. ``alignment_points[gi][ci]`` is
        the 3D anchor point for cell ``ci`` in group ``gi``.
    invert_y : bool, default True
        Invert the y-axis for string projections containing ``"y"``.
        Mirrors the behavior of other plot functions.
    units_per_inch, dpi, despine : optional
        Passed to :func:`single_panel_figure` when ``ax`` is ``None`` and
        ``units_per_inch`` is given.
    ax : plt.Axes, optional
        Existing axes to render into. A new figure is created when
        ``None``.

    Returns
    -------
    plt.Axes
        Axes with all cells, layer lines, and group labels drawn.
    """
    if not groups:
        raise ValueError("groups must be a non-empty list")
    if row_max_cells is not None and row_max_cells <= 0:
        raise ValueError("row_max_cells must be positive when given")
    if row_max_width is not None and row_max_width <= 0:
        raise ValueError("row_max_width must be positive when given")
    if align == "point" and alignment_points is None:
        raise ValueError("alignment_points is required when align='point'")

    # Resolve per-cell projection callables. When a group sets
    # rotation_angle/rotation_axis, each of its cells gets its own
    # rotation callable derived from the cell's own vertices and root.
    per_cell_projections: List[List[Callable]] = [
        _resolve_group_projections(group, projection, invert_y) for group in groups
    ]

    # Whether the rendered y axis will be inverted — drives row-stacking
    # direction so subsequent rows appear below on screen regardless of
    # projection orientation.
    y_axis_inverted = invert_y and _should_invert_y_axis(projection)

    cell_offsets, label_anchors = _grid_offsets(
        groups,
        projection=per_cell_projections,
        align=align,
        inter_cell_gap=inter_cell_gap,
        inter_group_gap=inter_group_gap,
        row_max_cells=row_max_cells,
        row_max_width=row_max_width,
        row_gap=row_gap,
        alignment_points=alignment_points,
        y_axis_inverted=y_axis_inverted,
    )

    # If no ax given and units_per_inch is set, size the figure to the
    # plotted-bounds envelope. We need to know the final bounds *after*
    # offsets, so recompute each cell's projected bounds with its offset
    # AND its own per-cell projection (which may include rotation).
    if ax is None and units_per_inch is not None:
        per_cell_bounds = [
            _plotted_bounds(
                groups[gi].cells[ci].skeleton.vertices,
                per_cell_projections[gi][ci],
                cell_offsets[gi][ci][0],
                cell_offsets[gi][ci][1],
            )
            for gi in range(len(groups))
            for ci in range(len(groups[gi].cells))
        ]
        xmin = min(b[0, 0] for b in per_cell_bounds)
        xmax = max(b[0, 1] for b in per_cell_bounds)
        ymin = min(b[1, 0] for b in per_cell_bounds)
        ymax = max(b[1, 1] for b in per_cell_bounds)
        # Pad bounds to accommodate group labels above the cells.
        if group_label_offset != 0.0:
            ymin -= group_label_offset
            ymax += group_label_offset
        _, ax = single_panel_figure(
            data_bounds_min=np.array([xmin, ymin]),
            data_bounds_max=np.array([xmax, ymax]),
            units_per_inch=units_per_inch,
            despine=despine,
            dpi=dpi,
        )

    if ax is None:
        _, ax = plt.subplots()

    # Render each cell with its group's resolved style and per-cell projection.
    # The projection has already absorbed any rotation, so plot_cell_2d
    # sees a plain callable and skips its own rotation resolution.
    for gi, group in enumerate(groups):
        for ci, cell in enumerate(group.cells):
            offset_h, offset_v = cell_offsets[gi][ci]
            style = _resolve_cell_style(group, ci)
            plot_cell_2d(
                cell,
                projection=per_cell_projections[gi][ci],
                offset_h=offset_h,
                offset_v=offset_v,
                invert_y=invert_y,
                ax=ax,
                **style,
            )

    # Layer guide lines.
    if layer_lines:
        add_layer_lines(ax, layer_lines, **(layer_line_kwargs or {}))

    # Group labels: always *above the group on screen*. With y_axis_inverted
    # the screen up direction is smaller data y, so we anchor at the
    # group's ymin and move further negative. Without inversion, we anchor
    # at ymax and move further positive.
    text_defaults = {"ha": "center", "va": "bottom", "fontsize": 10}
    if y_axis_inverted:
        text_defaults["va"] = "top"
    text_defaults.update(group_label_kwargs or {})
    for gi, anchor in enumerate(label_anchors):
        if anchor is None:
            continue
        center_x, ymin_plotted, ymax_plotted = anchor
        if y_axis_inverted:
            label_y = ymin_plotted - group_label_offset
        else:
            label_y = ymax_plotted + group_label_offset
        ax.text(center_x, label_y, groups[gi].label, **text_defaults)

    # Apply y-axis inversion at the lineup level. plot_cell_2d sees only
    # our per-cell projection *callables* (which absorb any rotation), and
    # callables can't be detected as "y"-bearing by _apply_y_inversion_to_axes.
    # So nothing inside the cells triggered the inversion — we apply it here
    # based on the original `projection` argument.
    ax = _apply_y_inversion_to_axes(ax, projection, invert_y)

    return ax
