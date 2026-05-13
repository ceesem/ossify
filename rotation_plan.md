# Rotation Plan for ossify plot.py

## Current State

`plot.py` handles 3D→2D projection through a simple coordinate-selection mechanism in `projection_factory()` (line 381). The six supported projections (`"xy"`, `"yx"`, `"xz"`, `"zx"`, `"yz"`, `"zy"`) all work by picking two of the three coordinate columns. No real 3D rotation is performed.

Every downstream plotting function (`plot_skeleton`, `plot_morphology_2d`, `plot_cell_2d`, `plot_cell_multiview`) accepts `projection: Union[str, Callable]`. When a callable is passed, `projection_factory()` returns it unchanged, and `_should_invert_y_axis()` returns `False` for callables. This means a new rotation callable slots cleanly into the existing interface without modifying callers.

The `SkeletonLayer` exposes `root_location: Optional[np.ndarray]` (a 3D coordinate) which serves as the natural center point for cell-centric rotations.

Plotting functions also expose `offset_h` and `offset_v` scalar parameters for 2D translation after projection. These are already used internally by some layout helpers.

---

## Goal

Add two public functions to `plot.py`:

1. **`Rotation(center, axis, angle, new_center=None)`** — low-level factory returning a projection callable.
2. **`RotateCell(cell, axis, angle=0, center=None, new_center=None)`** — high-level wrapper using the cell's soma/root as the default center, with a PCA-optimal "best" mode.

---

## `Rotation()` — Core Transform Factory

### Signature

```python
def Rotation(
    center: np.ndarray,
    axis: np.ndarray,
    angle: float,
    new_center: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

### Behavior

Returns a callable `f(pts: np.ndarray) -> np.ndarray` compatible with the `projection: Union[str, Callable]` parameter.

The callable:

1. Translates points to the rotation center: `pts_c = pts - center`
2. Applies a 3D rotation matrix built with Rodrigues' formula
3. Translates back: `pts_3d = pts_c_rotated + center`
4. Projects to 2D by taking the first two columns: `pts_2d = pts_3d[:, :2]`
5. If `new_center` is provided, shifts the 2D output so the rotation pivot appears at `new_center`:
   `pts_2d += (new_center - center[:2])`

### The `new_center` Parameter

`new_center` is a 2D array `[x, y]` that controls where the rotation center (soma) ends up in plot-space after projection. Three modes:

- `new_center=None` (default): no position adjustment; the cell appears at its natural 3D coordinate position projected onto the xy plane. Use this when absolute spatial relationships between cells should be preserved.
- `new_center=np.array([0.0, 0.0])`: shifts the 2D output so the rotation center sits at the plot origin. Useful for isolating a single cell in a figure without caring about absolute coordinates.
- `new_center=np.array([x, y])`: shifts the 2D output so the rotation center appears at the specified 2D location. Used by layout functions that need to place cells at prescribed positions.

The shift is always computed as `offset = new_center - center[:2]`. This is a pure 2D translation applied after projection; it does not affect the rotation or the relative geometry of the cell.

### Rodrigues' Rotation Matrix

Given unit axis vector **k** = `axis / ||axis||` and angle θ:

```text
K = [[  0, -kz,  ky],
     [ kz,   0, -kx],
     [-ky,  kx,   0]]   # skew-symmetric cross-product matrix

R = I·cos(θ) + sin(θ)·K + (1 - cos(θ))·(k ⊗ k)
```

Points transform as: `v' = R @ v`

Implemented as: `pts_rotated = pts_c @ R.T`

### Output Projection Convention

After rotation, the object is viewed "from above" along the −z axis: only xy columns are kept. This means:

- Rotating about the **z-axis** spins the object in the xy plane (same as rotating the standard xy view).
- Rotating about the **y-axis** tilts the object front-to-back, revealing xz content in the xy projection.
- Rotating about the **x-axis** tilts the object left-to-right, revealing yz content in the xy projection.
- Rotating about an **arbitrary axis** achieves any intermediate viewing angle.

### Y-axis Inversion Note

Because the returned callable is not a string, `_should_invert_y_axis()` already returns `False` for it. Callers should pass `invert_y=False` (or the default behavior is already correct) since rotation-based projections encode their own orientation.

---

## `RotateCell()` — Cell-Centric High-Level Wrapper

### Signature

```python
def RotateCell(
    cell: Cell,
    axis: Union[np.ndarray, Literal["x", "y", "z"], Literal["best"], None] = None,
    angle: Union[float, Literal["best"], None] = None,
    center: Optional[np.ndarray] = None,
    new_center: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

### Center Resolution

If `center` is `None`, the center is extracted from `cell.skeleton.root_location`. If the cell has no skeleton or `root_location` is `None`, a `ValueError` is raised asking the user to supply `center` explicitly.

The resolved 3D `center` is passed through to `Rotation()`. The `new_center` parameter is also passed through unchanged.

### Axis Label Resolution

String labels `"x"`, `"y"`, `"z"` are mapped to unit vectors:

```python
"x" → np.array([1.0, 0.0, 0.0])
"y" → np.array([0.0, 1.0, 0.0])
"z" → np.array([0.0, 0.0, 1.0])
```

Arbitrary numpy arrays are normalized to unit length before use.

### Mode Dispatch

| `axis`            | `angle`   | Behavior                                                    |
|-------------------|-----------|-------------------------------------------------------------|
| vector / label    | float     | Explicit rotation: calls `Rotation(center, axis, angle, new_center)` |
| vector / label    | `"best"`  | Optimizes rotation angle about the given axis (see §PCA)    |
| `"best"` / `None` | any       | Full PCA rotation: aligns PC1→x, PC2→y (see §PCA)          |

### PCA "Best" Mode

Both cases share the same two-step algorithm: find the rotation axis k, then find the optimal rotation angle about k via 2D PCA on the k-perpendicular projection. Case B is simply Case A with k also determined by PCA.

#### Shared implementation primitives

**`_perp_basis(k)`** — build an orthonormal basis (u, v) for the plane perpendicular to k:

- `u = normalize(x̂ - (x̂·k̂)k̂)` — Gram-Schmidt of x̂ against k̂ (fall back to ŷ if k is parallel to x̂)
- `v = k̂ × u`

**`_best_angle_for_axis(pts_c, k)`** — find the optimal rotation angle about k via 2D PCA:

1. `u, v = _perp_basis(k)`
2. Project: `p2 = pts_c @ np.column_stack([u, v])` (Nx2)
3. SVD of `p2` → PC1 = first right singular vector `[cos φ, sin φ]`
4. Return `θ* = -atan2(PC1[1], PC1[0])`

This is O(N) — one matrix multiply and one 2×2 SVD.

#### Case A: Axis given, `angle="best"`

Goal: find θ* that maximizes 2D projected variance, constrained to rotations about the given axis.

Procedure:

1. Gather and center skeleton vertices: `pts_c = pts - center`.
2. Normalize: `k̂ = axis / ||axis||`.
3. `θ* = _best_angle_for_axis(pts_c, k̂)`.
4. Return `Rotation(center, k̂, θ*, new_center)`.

#### Case B: No axis (or `axis="best"`), ignoring `angle`

Goal: find the globally optimal viewing axis and angle.

Procedure:

1. Gather and center skeleton vertices: `pts_c = pts - center`.
2. 3D SVD: `_, _, Vt = np.linalg.svd(pts_c, full_matrices=False)`.
   - `k̂ = Vt[2]` — PC3, the direction of **minimum** variance = the natural "depth" axis to look along.
3. `θ* = _best_angle_for_axis(pts_c, k̂)`.
4. Return `Rotation(center, k̂, θ*, new_center)`.

The 3D SVD costs O(N) and feeds directly into the same `_best_angle_for_axis` call used by Case A.

#### Sign ambiguity

PCA eigenvectors have a sign ambiguity. Convention: after computing θ*, check whether the majority of projected points lie in the positive-x half of the 2D output; if not, add π to θ*. Apply the same check on y. This keeps orientation consistent across cells.

---

## Future Feature: `plot_lineup`

### Concept

`plot_lineup` would display a sequence of cells arranged left-to-right on a shared set of axes: cell A plotted first, then cell B starting after a definable gap, and so on. Each cell is independently rotated/projected, and the layout engine positions the cell bodies at appropriate x-offsets so they don't overlap.

This requires two operations per cell: (1) project/rotate into 2D, and (2) translate the 2D result to the correct x-position. The question is how these two operations are coupled.

### Design Options

#### Option A: Encode position inside the transform callable (`new_center`)

`plot_lineup` computes the desired 2D soma position `(x_slot, 0)` for each cell and creates a `RotateCell(..., new_center=np.array([x_slot, 0]))` callable per cell. The callable handles both rotation and positioning.

**Pros:**
- The callable is self-contained: calling it on any skeleton produces correctly positioned output, no additional offset needed at plot time.
- `plot_lineup` can simply call `plot_cell_2d(cell, projection=callable)` for each cell without touching `offset_h`/`offset_v`.

**Cons:**
- Position determination still requires a two-pass process: bounds must be computed before `x_slot` values can be assigned, so a new callable must be created after the first pass.
- Couples two distinct concerns — rotation (a data transform) and layout (a display concern) — inside a single object. If the user later wants to reuse the rotation in a different layout, the positioning is baked in.
- `new_center` is 2D, which means the callable is no longer a pure 3D→2D transform; it is 3D→2D-with-layout. This blurs the interface.
- Does not compose with `offset_h`/`offset_v` without confusion (applying both would double-count the shift).

#### Option B: Use `offset_h`/`offset_v` in the plotting calls

`plot_lineup` computes projected bounds for each cell in a first pass, accumulates widths + gaps to determine `offset_h` per cell, then plots each cell with `plot_cell_2d(cell, projection=rotation_callable, offset_h=x_offset)`. The rotation callable is created once (or not at all, for standard projections) and reused for all cells.

**Pros:**
- Clean separation of concerns: rotation handles orientation, `offset_h` handles layout.
- The same `RotateCell(...)` callable (or even just `"xy"`) can be reused for all cells; the layout engine operates independently.
- Consistent with how `multi_panel_figure` already works — it uses `_plotted_bounds()` to compute data limits and manages offsets separately.
- `offset_h` already exists and is tested; no new mechanism needed.

**Cons:**
- Caller must explicitly compute `offset_h` for each cell, which adds boilerplate inside `plot_lineup`.
- Two-pass is inherently required either way; this option just makes that explicit in the caller.

#### Option C: `new_center` for standalone use; `offset_h` for lineup

A hybrid: `new_center` is retained in `Rotation`/`RotateCell` for the use case of "center a single cell at the origin" (common for single-cell figures), but `plot_lineup` uses `offset_h` for its layout logic.

**Pros:**
- Best of both worlds: `new_center` serves a genuine standalone-figure use case, `offset_h` serves the multi-cell layout use case.
- `plot_lineup` does not need to create per-cell callables; it creates one callable per cell for the rotation, then independently manages offsets.

**Cons:**
- Slightly more API surface (`new_center` exists even though `plot_lineup` doesn't use it), but the standalone use case is independently valuable.

### Recommendation

**Option C** (hybrid). Implement `new_center` in `Rotation`/`RotateCell` for standalone centering, but implement `plot_lineup` using `offset_h`/`offset_v`. The reasons:

1. `new_center` is genuinely useful without `plot_lineup`: a common workflow is "rotate this cell to its best orientation and center it at the origin for a clean figure." That has nothing to do with lineup.
2. Coupling layout into the projection callable (Option A) creates a leaky abstraction. The projection layer should transform coordinates; the layout layer should position things.
3. `offset_h` is already the established mechanism. `plot_lineup` can follow the same pattern as `multi_panel_figure` without inventing a new contract.

The `plot_lineup` implementation sketch would be:

```python
def plot_lineup(cells, projections, gap, ...):
    # Pass 1: compute each cell's projected width
    widths = [projected_width(cell, proj) for cell, proj in zip(cells, projections)]

    # Compute cumulative x-offsets
    offsets = cumulative_offsets(widths, gap)

    # Pass 2: plot each cell at its slot
    for cell, proj, offset in zip(cells, projections, offsets):
        plot_cell_2d(cell, projection=proj, offset_h=offset, ax=ax)
```

This does not require touching `new_center` at all, and the `projections` list can contain a mix of string projections and `RotateCell(...)` callables.

---

## Code Factorization Principle

The implementation should be as well-factorized as possible. Concretely:

- `_perp_basis(k)` and `_best_angle_for_axis(pts_c, k)` are the shared primitives; neither `Rotation` nor `RotateCell` should duplicate any of their logic.
- Case B of `RotateCell` is implemented by calling the 3D SVD to get k, then delegating entirely to the Case A path. There is no separate code branch for the two "best" modes beyond the axis selection step.
- `Rotation()` is the single source of truth for constructing the callable (rotation matrix, projection, and `new_center` shift). `RotateCell` always terminates by calling `Rotation()` — it never constructs a callable itself.
- Axis-label resolution (`"x"` → unit vector, etc.) lives in one helper and is called from one place.

## File Placement

Both `Rotation` and `RotateCell` belong in `plot.py`, placed after the existing `projection_factory()` function (around line 405). They are logical siblings of `projection_factory()` in the same "transform/projection utilities" block.

Add both to `__all__` at the top of `plot.py`.

---

## Testing

Tests go in `tests/test_plot.py` (or a new `tests/test_transforms.py`).

### `Rotation()` unit tests

- **Identity**: `Rotation(center, [0,0,1], 0)(pts)` equals `pts[:, :2]` (zero rotation = xy projection).
- **Half-turn**: Rotating 180° about z reverses the sign of both x and y in projection.
- **Quarter-turn about z**: Rotating 90° about z maps `[1,0,0]` → `[0,1,0]` in 3D, so x→0 and y→1 in projection.
- **Quarter-turn about x**: Rotating 90° about x maps `[0,1,0]` → `[0,0,1]` in 3D, which disappears in xy projection, while `[0,0,1]` → `[0,-1,0]`.
- **Center independence**: The rotation center is a fixed point — rotating about `center` leaves `center[:2]` unchanged in projection.
- **Arbitrary axis**: Rotate about `[1,1,0]` by π, verify with known analytic result.
- **Non-unit axis**: Passing `[2,0,0]` should behave identically to `[1,0,0]`.
- **`new_center` origin**: `Rotation(center, axis, angle, new_center=[0,0])(pts)` places the projected center at `[0, 0]`.
- **`new_center` translation**: The difference between output with `new_center=[a,b]` vs `new_center=None` is a uniform `[a-cx, b-cy]` shift on all points, where `[cx, cy] = center[:2]`.

### `RotateCell()` unit tests

- **Center fallback**: With `center=None`, uses `cell.skeleton.root_location`.
- **No skeleton raises**: `RotateCell(cell_no_skel, ...)` raises `ValueError`.
- **String axis labels**: `"x"`, `"y"`, `"z"` produce the same results as the corresponding unit vectors.
- **Explicit angle**: Result matches `Rotation(center, axis, angle)` applied directly.
- **`new_center` pass-through**: `RotateCell(..., new_center=[0,0])` matches `Rotation(root_loc, axis, angle, new_center=[0,0])`.
- **`angle="best"` with axis**: Returned callable produces higher 2D variance than any other sampled angle (validate on a synthetic elongated point cloud oriented at a known angle).
- **`axis="best"` PCA mode**: Applied to a synthetic point cloud whose maximum-variance direction is known, verify the projected x-axis aligns with that direction.

---

## Step-by-Step Implementation Plan

All new code goes into `src/ossify/plot.py` immediately after `projection_factory()` (line 404). Tests go into `tests/test_plot.py` as a new `TestRotation` class. Steps must be completed in order because each builds on the previous.

---

### Step 1 — `_resolve_axis(axis)` helper

Write a private function that converts an axis specification to a normalized 3D unit vector.

```python
def _resolve_axis(
    axis: Union[np.ndarray, Literal["x", "y", "z"]],
) -> np.ndarray:
```

- Map `"x"` → `[1,0,0]`, `"y"` → `[0,1,0]`, `"z"` → `[0,0,1]`.
- For ndarray input: return `axis / np.linalg.norm(axis)`. Raise `ValueError` if the norm is zero.
- Raise `ValueError` for any other string.

This is the single place axis labels are resolved; no other function should inline this logic.

---

### Step 2 — `_perp_basis(k)` helper

Write a private function that returns an orthonormal basis (u, v) for the plane perpendicular to k.

```python
def _perp_basis(k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
```

- Assumes `k` is already a unit vector (caller's responsibility after `_resolve_axis`).
- `ref = [1,0,0]`; if `|k · ref| > 0.9` (near-parallel), use `ref = [0,1,0]` instead.
- `u = normalize(ref - (ref·k)k)` — Gram-Schmidt.
- `v = np.cross(k, u)` — already unit length since k and u are orthonormal.
- Return `(u, v)`.

---

### Step 3 — `_best_angle_for_axis(pts_c, k)` helper

Write a private function that returns the optimal rotation angle about k via 2D PCA on the k-perpendicular projection.

```python
def _best_angle_for_axis(pts_c: np.ndarray, k: np.ndarray) -> float:
```

- `u, v = _perp_basis(k)`.
- `p2 = pts_c @ np.column_stack([u, v])` — Nx2 projection.
- `_, _, Vt = np.linalg.svd(p2, full_matrices=False)` — PC1 = `Vt[0]`.
- `theta = -np.arctan2(Vt[0, 1], Vt[0, 0])`.
- Apply sign convention: rotate `pts_c` by `theta` about `k`, project to 2D; if the mean x of the result is negative, add π. Same check for y.
- Return the final angle.

This is the only place the 2D PCA angle-finding logic lives.

---

### Step 4 — `_build_rotation_matrix(k, angle)` helper

Write a private function that constructs the 3×3 Rodrigues rotation matrix.

```python
def _build_rotation_matrix(k: np.ndarray, angle: float) -> np.ndarray:
```

- Assumes `k` is a unit vector.
- Build the skew-symmetric matrix K from k's components.
- Return `np.eye(3) * np.cos(angle) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(k, k)`.

Isolating this keeps `Rotation()` readable and makes the matrix independently testable.

---

### Step 5 — `Rotation()` public factory

Write the public factory function.

```python
def Rotation(
    center: np.ndarray,
    axis: np.ndarray,
    angle: float,
    new_center: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

- Convert inputs to arrays: `center = np.asarray(center, dtype=float)`, same for `axis` and `new_center`.
- `k = _resolve_axis(axis)`.
- `R = _build_rotation_matrix(k, angle)`.
- Precompute the 2D offset: `offset_2d = (np.asarray(new_center) - center[:2]) if new_center is not None else np.zeros(2)`. This is captured in the closure so the callable does no branching at call time.
- Return an inner function:
  ```python
  def _project(pts: np.ndarray) -> np.ndarray:
      pts = np.asarray(pts, dtype=float)
      pts_c = pts - center
      pts_r = pts_c @ R.T + center
      return pts_r[:, :2] + offset_2d
  ```
- Add to `__all__`.

---

### Step 6 — `RotateCell()` public wrapper

Write the public high-level wrapper.

```python
def RotateCell(
    cell: Cell,
    axis: Union[np.ndarray, Literal["x", "y", "z"], Literal["best"], None] = None,
    angle: Union[float, Literal["best"], None] = None,
    center: Optional[np.ndarray] = None,
    new_center: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
```

Body in order:

1. **Resolve center**: if `center is None`, use `cell.skeleton.root_location`; raise `ValueError` if unavailable.
2. **Full-PCA branch** (`axis in (None, "best")`): compute `pts_c = skel.vertices - center`; 3D SVD → `k = Vt[2]`; fall through to the constrained-best path below with this `k`.
3. **Axis resolution**: call `_resolve_axis(axis)` to get unit vector `k`. (For the full-PCA branch, k is already resolved above; skip this call.)
4. **Angle resolution**:
   - If `angle == "best"`: compute `pts_c` (if not already done), call `_best_angle_for_axis(pts_c, k)` to get `theta`.
   - Otherwise: `theta = float(angle)` (default 0 if `angle is None`).
5. **Return** `Rotation(center, k, theta, new_center)`.

The function never constructs a callable directly — it always terminates with a call to `Rotation()`.

Add to `__all__`.

---

### Step 7 — Update `__all__`

Add `"Rotation"` and `"RotateCell"` to the `__all__` list at the top of `plot.py`.

---

### Step 8 — Tests for private helpers

In `tests/test_plot.py`, add class `TestRotationHelpers`:

- **`test_resolve_axis_labels`**: `"x"`, `"y"`, `"z"` produce the correct unit vectors.
- **`test_resolve_axis_array`**: `[2,0,0]` normalizes to `[1,0,0]`.
- **`test_resolve_axis_zero_raises`**: `[0,0,0]` raises `ValueError`.
- **`test_resolve_axis_bad_string_raises`**: `"w"` raises `ValueError`.
- **`test_perp_basis_orthogonality`**: u·v = 0, u·k = 0, v·k = 0 for several k values including near-axis-aligned cases.
- **`test_perp_basis_unit_length`**: `||u|| == 1`, `||v|| == 1`.
- **`test_perp_basis_near_x`**: k near `[1,0,0]` falls back to y-reference without producing a zero vector.
- **`test_build_rotation_matrix_identity`**: angle=0 returns identity.
- **`test_build_rotation_matrix_quarter_z`**: 90° about z maps `[1,0,0]` → `[0,1,0]`.
- **`test_build_rotation_matrix_orthogonal`**: R·Rᵀ = I for arbitrary axis and angle.
- **`test_best_angle_for_axis_known`**: synthetic elongated point cloud aligned at a known angle in the xz plane; `_best_angle_for_axis(pts_c, [0,1,0])` returns the correct angle (within tolerance).

---

### Step 9 — Tests for `Rotation()`

Add class `TestRotation`:

- **`test_identity`**: `Rotation([0,0,0], [0,0,1], 0)(pts)` equals `pts[:, :2]`.
- **`test_half_turn_z`**: 180° about z negates both x and y.
- **`test_quarter_turn_z`**: 90° about z maps `[[1,0,0]]` to approximately `[[0,1]]`.
- **`test_quarter_turn_x`**: 90° about x maps `[[0,0,1]]` to approximately `[[0,-1]]` (disappears from y since it moves to negative y).
  - Wait, 90° about x: `[0,0,1]` → `[0,-1,0]` in 3D → projects to `[0,-1]`. Correct.
- **`test_center_is_fixed`**: the projected center point is unchanged regardless of rotation angle.
- **`test_arbitrary_axis`**: rotate `[[1,0,0]]` by π about `[1,1,0]/√2`; verify against analytic result `[[-1,0]]` (the rotation maps x→-x in this case... let me verify: rotation by π about [1,1,0]/√2 maps [1,0,0] → [0,1,0] actually no, let me not specify the exact result in the plan; just say "verify against known analytic result").
- **`test_non_unit_axis`**: `Rotation(c, [2,0,0], θ)` and `Rotation(c, [1,0,0], θ)` produce identical output.
- **`test_new_center_origin`**: projected center lies at `[0,0]` when `new_center=[0,0]`.
- **`test_new_center_arbitrary`**: projected center lies at `[a,b]` when `new_center=[a,b]`.
- **`test_new_center_uniform_shift`**: all points shift by a constant `new_center - center[:2]` relative to `new_center=None`.

---

### Step 10 — Tests for `RotateCell()`

Add class `TestRotateCell` (requires a minimal `Cell` fixture with a skeleton; reuse or extend conftest):

- **`test_center_from_root`**: with `center=None`, result matches `Rotation(root_location, axis, angle)` directly.
- **`test_explicit_center`**: explicit `center` overrides root location.
- **`test_no_skeleton_raises`**: cell without a skeleton raises `ValueError`.
- **`test_string_axis_matches_vector`**: `axis="y"` matches `axis=np.array([0,1,0])`.
- **`test_explicit_angle`**: result matches calling `Rotation()` directly with the same args.
- **`test_new_center_passthrough`**: result matches calling `Rotation(..., new_center=...)` directly.
- **`test_angle_best_constrained`**: synthetic skeleton elongated at a known angle in a plane perpendicular to y; `RotateCell(cell, axis="y", angle="best")` produces a 2D projection where the long axis aligns with x (i.e., the x-variance is larger than y-variance).
- **`test_axis_best_full_pca`**: synthetic skeleton elongated along a known 3D direction; `RotateCell(cell, axis="best")` produces a 2D projection where x-variance is the largest achievable (compare to several explicit rotations).
- **`test_full_pca_uses_pc3_as_axis`**: the callable returned by `RotateCell(cell, axis="best")` applied to a known point cloud agrees with manually computing PC3 → `_best_angle_for_axis` → `Rotation()`.

---

### Step 11 — Run tests and lint

```bash
poe test
uv run ruff check src/ tests/
```

All existing tests must continue to pass. All new tests must pass.

---

## Integration Notes

- All existing callers of `plot_skeleton`, `plot_morphology_2d`, `plot_cell_2d` are unaffected — they pass strings or `None` for `projection`.
- `plot_cell_multiview` could optionally accept `projection` per-panel; this is unchanged by this feature.
- The `invert_y=True` default in most functions is harmless for Rotation callables since `_should_invert_y_axis` returns `False` for non-strings.
- The `_plotted_bounds()` utility used in `multi_panel_figure` calls `projection_factory()` on the projection and applies it to vertices. Since `projection_factory()` passes callables through unchanged, Rotation-based projections will work correctly there too with no changes.
- No new dependencies are required: `numpy` (already a dependency) provides all needed linear algebra (`np.linalg.svd`, array operations).
