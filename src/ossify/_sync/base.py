from typing import Optional, Union

import fastremap
import numpy as np
import pandas as pd

DEFAULT_SPATIAL_COLUMNS = ["x", "y", "z"]

# Ossify's canonical identifier space. Identifiers are nominally drawn from the
# uint64 segmentation ID space, but every ID we actually support fits within the
# nonnegative range of a signed 64-bit integer. We therefore own a single
# internal representation -- ``int64`` -- so that a value never depends on
# whether an upstream source happened to hand us signed or unsigned integers.
_INT64_MAX = int(np.iinfo(np.int64).max)  # 9223372036854775807
_UINT64_INT64_MAX = np.uint64(_INT64_MAX)


def _range_error(name: str, kind: str) -> ValueError:
    if kind == "negative":
        return ValueError(
            f"{name}: identifiers must be nonnegative, but negative values were found."
        )
    return ValueError(
        f"{name}: identifiers must be <= {_INT64_MAX} (int64 max), "
        f"but larger values were found."
    )


def _validate_and_cast_integer_ndarray(
    arr: np.ndarray, null_mask: Optional[np.ndarray], name: str
) -> np.ndarray:
    """Validate a signed/unsigned integer ndarray and cast to ``int64``.

    Validation uses integer-safe comparisons only -- no value ever passes
    through a floating-point representation, which would silently lose
    precision for IDs above ``2**53``.
    """
    kind = arr.dtype.kind
    if kind == "i":
        # Signed integers already fit within int64 (int8..int64); only the
        # lower bound can be violated. Null positions carry a sentinel and are
        # excluded from the check.
        negative = arr < 0
        if null_mask is not None:
            negative = negative & ~null_mask
        if negative.any():
            raise _range_error(name, "negative")
        return arr.astype(np.int64, copy=False)
    if kind == "u":
        # Unsigned integers cannot be negative; only uint64 can exceed int64
        # max. Compare against a uint64 scalar so numpy stays in integer land
        # (mixing a uint64 array with a Python int upcasts to float64).
        too_large = arr > _UINT64_INT64_MAX
        if null_mask is not None:
            too_large = too_large & ~null_mask
        if too_large.any():
            raise _range_error(name, "too_large")
        # Every remaining value is <= int64 max, so this cast is lossless.
        return arr.astype(np.int64)
    raise TypeError(
        f"{name}: expected integer identifiers, got array of dtype {arr.dtype}."
    )


def _canonicalize_object_array(
    arr: np.ndarray, allow_null: bool, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize an object-dtype array element by element.

    Object arrays are the only place mixed Python ints, ``None``/``pd.NA`` and
    stray floats can coexist, so each element is inspected individually.
    """
    n = len(arr)
    out = np.zeros(n, dtype=np.int64)
    null_mask = np.zeros(n, dtype=bool)
    for i, v in enumerate(arr):
        if v is None or v is pd.NA:
            null_mask[i] = True
            continue
        if isinstance(v, (bool, np.bool_)):
            raise TypeError(
                f"{name}: boolean values are not valid identifiers (got {v!r})."
            )
        if isinstance(v, (float, np.floating)):
            # A float may already have lost precision before Ossify saw it, so
            # even an integral-looking float is rejected. A NaN is treated as a
            # null when nulls are permitted.
            if v != v:  # NaN
                null_mask[i] = True
                continue
            raise TypeError(
                f"{name}: float identifiers are not accepted (got {v!r}); "
                f"a float may already have lost precision above 2**53."
            )
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            if iv < 0:
                raise _range_error(name, "negative")
            if iv > _INT64_MAX:
                raise _range_error(name, "too_large")
            out[i] = iv
            continue
        raise TypeError(
            f"{name}: non-integer identifier {v!r} of type {type(v).__name__}."
        )
    if null_mask.any() and not allow_null:
        raise ValueError(f"{name}: null identifier values are not allowed here.")
    return out, null_mask


def _extract_int64(
    values, allow_null: bool, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Core: return ``(int64_ndarray, null_mask)`` for any array-like input."""
    dtype = getattr(values, "dtype", None)
    if dtype is not None and pd.api.types.is_extension_array_dtype(dtype):
        # pandas nullable integer container (Int64/UInt64/Int32/...).
        ea = values.array if isinstance(values, (pd.Series, pd.Index)) else values
        if not pd.api.types.is_integer_dtype(ea.dtype):
            raise TypeError(
                f"{name}: expected integer identifiers, got extension dtype {ea.dtype}."
            )
        null_mask = np.asarray(ea.isna())
        if null_mask.any() and not allow_null:
            raise ValueError(f"{name}: null identifier values are not allowed here.")
        # Materialize the underlying signed/unsigned native array (nulls filled
        # with a sentinel 0) then validate exactly like a plain ndarray.
        native = ea.to_numpy(dtype=ea.dtype.numpy_dtype, na_value=0)
        int64 = _validate_and_cast_integer_ndarray(native, null_mask, name)
        return int64, null_mask

    arr = values if isinstance(values, np.ndarray) else np.asarray(values)
    if arr.size == 0:
        # np.asarray([]) is float64; an empty ID container is still valid.
        return arr.astype(np.int64), np.zeros(0, dtype=bool)
    kind = arr.dtype.kind
    if kind in ("i", "u"):
        int64 = _validate_and_cast_integer_ndarray(arr, None, name)
        return int64, np.zeros(len(int64), dtype=bool)
    if kind == "f":
        raise TypeError(
            f"{name}: float identifiers are not accepted; a float may already "
            f"have lost precision above 2**53."
        )
    if kind == "b":
        raise TypeError(f"{name}: boolean values are not valid identifiers.")
    if kind == "O":
        return _canonicalize_object_array(arr, allow_null, name)
    raise TypeError(f"{name}: unsupported identifier dtype {arr.dtype}.")


def _wrap_null(int64: np.ndarray, null_mask: np.ndarray, allow_null: bool):
    """Return an ``int64`` ndarray, or a nullable ``Int64`` array if NA present."""
    if allow_null and null_mask.any():
        return pd.arrays.IntegerArray(int64, null_mask.copy())
    return int64


def canonicalize_ids(
    values,
    *,
    allow_null: bool = False,
    name: str = "id",
):
    """Convert integer identifiers to Ossify's canonical ``int64`` representation.

    Ossify treats identifiers as living in the nonnegative ``int64`` range even
    though the nominal ID space is ``uint64``. Upstream sources may hand us the
    same ID as a Python ``int``, a NumPy signed/unsigned integer, or inside a
    pandas container, and joining two differently-typed key spaces can make
    pandas coerce values through ``float64`` -- silently collapsing distinct IDs
    above ``2**53``. Canonicalizing every ID to ``int64`` before it participates
    in a lookup removes that failure mode.

    Parameters
    ----------
    values :
        The identifier(s). Accepted forms: a Python ``int``; a NumPy signed or
        unsigned integer scalar; a NumPy integer array; a list/tuple of the
        above; a pandas ``Series`` or ``Index``; or a pandas nullable integer
        (``Int64``/``UInt64``/...) container.
    allow_null :
        Whether nulls (``None``/``pd.NA``/``NaN``) are permitted. Only pass
        ``True`` where the calling mapping operation legitimately supports
        missing values; otherwise a null raises.
    name :
        Label used in error messages (typically the column or layer name).

    Returns
    -------
    :
        The canonical form, matching the input's container shape:

        - scalar in -> ``np.int64`` scalar;
        - ``Series`` in -> ``Series`` (index and name preserved);
        - ``Index`` in -> ``Index`` (name preserved);
        - array-like in -> ``np.ndarray`` of ``int64``.

        Where nulls are present (and permitted) the result uses pandas nullable
        ``Int64`` so missing values survive without a float cast; otherwise a
        plain ``int64`` container is returned.

    Raises
    ------
    TypeError
        If a value is a float (including integral-looking floats), a boolean, or
        any non-integer object.
    ValueError
        If a value is negative, exceeds ``int64`` max, or is null where nulls are
        not allowed.
    """
    # Scalars. Order matters: bool is a subclass of int, and np.bool_ must be
    # rejected before the integer check.
    if isinstance(values, (bool, np.bool_)):
        raise TypeError(f"{name}: boolean values are not valid identifiers.")
    if isinstance(values, (int, np.integer)):
        iv = int(values)
        if iv < 0:
            raise _range_error(name, "negative")
        if iv > _INT64_MAX:
            raise _range_error(name, "too_large")
        return np.int64(iv)
    if isinstance(values, (float, np.floating)):
        raise TypeError(
            f"{name}: float identifiers are not accepted; a float may already "
            f"have lost precision above 2**53."
        )

    if isinstance(values, pd.Index):
        int64, null_mask = _extract_int64(values, allow_null, name)
        return pd.Index(_wrap_null(int64, null_mask, allow_null), name=values.name)
    if isinstance(values, pd.Series):
        int64, null_mask = _extract_int64(values, allow_null, name)
        return pd.Series(
            _wrap_null(int64, null_mask, allow_null),
            index=values.index,
            name=values.name,
        )

    int64, null_mask = _extract_int64(values, allow_null, name)
    return _wrap_null(int64, null_mask, allow_null)


def _canonicalize_index(index: pd.Index) -> pd.Index:
    """Return ``index`` in canonical int64 form if it holds integer identifiers.

    Non-integer indexes (e.g. a string or float index) are returned untouched
    so canonicalization never changes behavior for data that is not an ID.
    """
    dtype = index.dtype
    if dtype.kind in ("i", "u") or (
        pd.api.types.is_extension_array_dtype(dtype)
        and pd.api.types.is_integer_dtype(dtype)
    ):
        return canonicalize_ids(index, name="node index")
    return index


def mask_and_remap(
    arr: np.ndarray,
    mask: Union[np.ndarray, list],
) -> np.ndarray:
    """Given an array in unmasked indexing and a mask,
    return the array in remapped indexing and omit rows with masked values.

    Parameters
    ----------
    arr :
        NxM array of indices
    mask :
        1D array of indices to mask, either as a boolean mask or as a list of indices
    """
    if np.array(mask).dtype == bool:
        mask = np.where(mask)[0]
    return _mask_and_remap(np.array(arr, dtype=int), mask)


def _mask_and_remap(
    arr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    mask_dict = {k: v for k, v in zip(mask, range(len(mask)))}
    mask_dict[-1] = -1

    arr_offset = arr + 1
    arr_mask_full = fastremap.remap(
        fastremap.mask_except(arr_offset, list(mask + 1)) - 1,
        mask_dict,
    )
    if len(arr_mask_full.shape) == 1:
        return arr_mask_full[~np.any(arr_mask_full == -1)]
    else:
        return arr_mask_full[~np.any(arr_mask_full == -1, axis=1)]


class Layer:
    def __init__(
        self,
        nodes: Union[pd.DataFrame, np.ndarray],
        facets: Union[pd.DataFrame, np.ndarray],
        spatial_columns: Optional[list] = None,
        relation_columns: Optional[list] = None,
        copy: bool = True,
        **kwargs,
    ):
        """A class for representing a set of nodes and their relationships (facets).

        Parameters
        ----------
        nodes :
            A DataFrame or nx3 array of the nodes/vertices/points in the layer.
            If an array is provided, it must be nx3 and the columns will be named
            "x", "y", and "z" unless `spatial_columns` is provided.
        facets :
            A DataFrame or m x k array of the facets/edges/faces in the layer.
            Some columns must correspond to indices in `nodes`, these are specified
            by `relation_columns`.
        spatial_columns :
            A list of column names in `nodes` that correspond to spatial coordinates.
            If not provided and `nodes` is a DataFrame, all columns will be used. If
            `nodes` is an array, defaults to ["x", "y", "z"].
        relation_columns :
            A list of column names in `facets` that correspond to indices in `nodes`.
            If not provided and `facets` is a DataFrame, all columns will be used.
        copy :
            Whether to copy the input DataFrames. If False, the input DataFrames
            may be modified in place.
        """
        if not isinstance(nodes, pd.DataFrame):
            if isinstance(nodes, np.ndarray):
                if nodes.shape[1] == 3:
                    nodes = pd.DataFrame(nodes, columns=DEFAULT_SPATIAL_COLUMNS)
                    if spatial_columns is None:
                        spatial_columns = DEFAULT_SPATIAL_COLUMNS
                else:
                    raise ValueError("Nodes must be an nx3 array")
            else:
                raise ValueError("Nodes must be a DataFrame or an nx3 array")

        if copy:
            nodes = nodes.copy()
        self.nodes: pd.DataFrame = nodes
        # Node indexes hold identifiers, so pin them to Ossify's canonical int64
        # representation. This keeps every downstream lookup (masking joins,
        # link joins) in a single signed key space -- mixing int64 and uint64
        # keys is what lets pandas collapse distinct IDs above 2**53 via float.
        self.nodes.index = _canonicalize_index(self.nodes.index)

        if spatial_columns is None:
            spatial_columns = []
        self.spatial_columns = spatial_columns

        if facets is None:
            facets = pd.DataFrame()
        if not isinstance(facets, pd.DataFrame):
            facets = pd.DataFrame(facets)
            if relation_columns is None:
                relation_columns = facets.columns.tolist()
        if copy:
            facets = facets.copy()
        self.facets: pd.DataFrame = facets

        if relation_columns is None:
            relation_columns = []
        self.relation_columns = relation_columns

    @property
    def vertices(self) -> np.ndarray:
        """Array of the spatial coordinates of the vertices"""
        return self.vertices_df.to_numpy(dtype=float)

    @property
    def vertices_df(self) -> pd.DataFrame:
        """DataFrame of the spatial coordinates of the vertices"""
        return (
            self.nodes[self.spatial_columns]
            if self.spatial_columns is not None
            else self.nodes
        )

    @property
    def points(self) -> np.ndarray:
        """Alias for vertices"""
        return self.vertices

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the layer."""
        return len(self.nodes)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the layer. Alias for n_nodes."""
        return self.n_nodes

    @property
    def n_points(self) -> int:
        """Number of points in the layer. Alias for n_nodes."""
        return self.n_nodes

    @property
    def n_facets(self) -> int:
        """Number of facets (edges/faces) in the layer."""
        return len(self.facets)

    @property
    def nodes_index(self) -> pd.Index:
        """Index of the nodes DataFrame."""
        return self.nodes.index

    @property
    def vertices_index(self) -> pd.Index:
        """Index of the vertices. Alias for nodes_index."""
        return self.nodes_index

    @property
    def points_index(self) -> pd.Index:
        """Index of the points. Alias for nodes_index."""
        return self.nodes_index

    @property
    def facets_index(self) -> pd.Index:
        """Index of the facets DataFrame."""
        return self.facets.index

    @property
    def edge_index(self) -> pd.Index:
        """Index of the edges. Alias for facets_index."""
        return self.facets_index

    @property
    def facets_positional(self) -> np.ndarray:
        """Array of the facets in positional indexing, such that 0 corresponds to the
        first node in its current node index ordering"""
        return mask_and_remap(self.facets[self.relation_columns], self.nodes.index)

    def query_nodes(self, query_str: str):
        """Query the nodes DataFrame and return a new layer with the
        corresponding nodes and facets.

        Parameters
        ----------
        query_str :
            A query string to pass to `pd.DataFrame.query` on the nodes DataFrame.

        Returns
        -------
        :
            A new layer with the queried nodes and corresponding facets.

        Notes
        -----
        When masking by nodes, only relationships that reference exclusively the
        remaining nodes are kept.
        """
        new_nodes = self.nodes.query(query_str)
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_nodes(self, mask: np.ndarray):
        """Mask the nodes DataFrame and return a new layer with the
        corresponding nodes and facets.

        Parameters
        ----------
        mask :
            A boolean mask array to filter the nodes DataFrame. This masking is applied
            in positional indexing (i.e. order, not key matters).

        Returns
        -------
        :
            A new layer with the masked nodes and corresponding facets.

        Notes
        -----
        When masking by nodes, only relationships that reference exclusively the
        remaining nodes are kept.
        """
        new_nodes = self.nodes.iloc[mask]
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_by_node_index(
        self,
        new_index: Union[np.ndarray, pd.Index, pd.Series],
        new_nodes: Optional[pd.DataFrame] = None,
    ):
        """Create a new layer containing only the specified nodes and their facets.

        Parameters
        ----------
        new_index :
            Index of nodes to keep in the new layer.
        new_nodes :
            Pre-filtered nodes DataFrame. If None, nodes will be filtered automatically
            based on new_index.

        Returns
        -------
        :
            A new layer instance containing only the specified nodes and facets that
            reference those nodes.

        Notes
        -----
        Only facets that reference exclusively the nodes in new_index are kept.
        """
        if new_nodes is None:
            new_nodes = self.nodes.loc[self.nodes.index.intersection(new_index)]

        new_facets = self.facets[
            self.facets[self.relation_columns].isin(new_index).all(axis=1)
        ]
        out = self.__class__((new_nodes, new_facets), **self.get_params())
        return out

    @property
    def layer_type(self) -> str:
        """String identifier of the layer type (e.g., 'mesh', 'points', 'graph')."""
        return str(self.__class__).strip(">'").split(".")[-1].lower()

    def get_params(self) -> dict:
        """Get the parameters used to initialize this layer.

        Returns
        -------
        :
            Dictionary containing layer initialization parameters.
        """
        return {
            "spatial_columns": self.spatial_columns,
            "relation_columns": self.relation_columns,
        }
