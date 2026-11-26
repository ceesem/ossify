"""Lasso skeleton vertices on a feature embedding; see where they sit on the cell.

One cell, one point per skeleton vertex. Left, the embedding of per-vertex features (e.g. ``make_skel_prop_df`` into
UMAP); right, the skeleton in 3D with its output synapses. A lasso on the left lights
up those vertices on the morphology, and the selection is live on the Python side::

    vx = VertexExplorer(cella, feature_df, x='umap0', y='umap1', hue='dist_to_root', context=cell)
    vx                      # lasso; shift adds, alt removes, click picks one vertex
    vx.selected_vertices    # skeleton vertex ids (feature_df's index)
    vx.selection_mask       # bool, aligned with feature_df's rows
    vx.selected_frame()     # their rows of feature_df

For labeling, the selection can be cut down to the longest path to the root and
marked ``primary`` or ``terminal``; unmarked vertices stay unlabeled. Labels are
kept per cell, so a widget built with :meth:`VertexExplorer.from_loader` can step
through many cells and collect them all::

    vx = VertexExplorer.from_loader(prepare, e_pf_ids, hue='dist_to_root')
    vx.labels_frame()       # root_id, vertex_id, label -- every cell visited

With ``save_dir``, every label change writes the cell's ``feature_df`` plus a
``label`` column to ``{save_dir}/{root_id}.parquet``, and a cell shown again picks
its labels back up from there. :func:`read_labels` stacks the files into one
training frame.

Copied from ``v1dd_feature_pax`` (``explorer/vertex.py``) on 2026-11-26 so it sits
beside ossify's model-training notebooks; see ``label_compartments.ipynb`` for use.

The morphology drags to rotate, shift-drags to pan, scrolls to zoom, and
double-clicks back to the XY view (y down, so pia is up in v1dd and MICrONS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import anywidget
import numpy as np
import pandas as pd
import traitlets

_HERE = Path(__file__).parent
#: Numeric hues are binned into this many colors so the morphology draws in a
#: handful of strokes rather than one per edge.
N_BINS = 64

#: Label names, in the order of the codes sent to the front end (0 is unlabeled).
LABELS = ("primary", "terminal")
_CODE = {name: i + 1 for i, name in enumerate(LABELS)}


class VertexExplorer(anywidget.AnyWidget):
    """Embedding scatter of skeleton vertices, linked to a 3D view of the skeleton.

    Parameters
    ----------
    cell
        The ossify ``Cell`` whose skeleton ``feature_df`` was computed on (e.g. the
        axon-masked cell).
    feature_df
        Per-vertex frame indexed by skeleton vertex id, as from
        ``compartments.make_skel_prop_df``. Rows with a missing ``x`` or ``y`` are
        left off the scatter but still drawn on the skeleton, in grey.
    x, y
        Embedding columns of ``feature_df``.
    hue
        Column name or array aligned with ``feature_df``. Numeric hues use ``cmap``
        between the 2nd and 98th percentiles (or ``vlim``); anything else is
        categorical. Use a column name when stepping through cells.
    context
        Optional ``Cell`` (e.g. the unmasked cell) drawn faintly beneath, with its
        root marked as the soma.
    root_id
        Key the cell's labels are kept under; defaults to ``cell.name``.
    synapses
        Annotation layer of ``cell`` drawn as red dots at the synapses' own
        positions (default ``'pre_syn'``, the outputs); ``None`` for none. With a
        selection, synapses mapped to selected vertices stay bright and the rest fade.
    save_dir
        Folder for ``{root_id}.parquet`` files of features plus labels, written on
        every label change and read back when a cell is shown. Relative paths are
        taken from the working directory. ``None`` keeps labels in memory only.
    """

    _esm = _HERE / "vertex.js"
    _css = (_HERE / "widget.css").read_text() + (_HERE / "vertex.css").read_text()

    points = traitlets.Dict().tag(
        sync=True
    )  # scatter: x, y, c (palette index), v (vertex position)
    morph = traitlets.Dict().tag(
        sync=True
    )  # skeleton: xyz (um), parent, point, synapses, context
    legend = traitlets.Dict().tag(sync=True)
    selected = traitlets.List().tag(sync=True)  # point indices
    label_codes = traitlets.List().tag(
        sync=True
    )  # per point: 0 unlabeled, else 1 + LABELS index
    nav = traitlets.Dict().tag(
        sync=True
    )  # {index, n, root_id} when stepping through cells
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        cell,
        feature_df: pd.DataFrame,
        x: str = "umap0",
        y: str = "umap1",
        hue: str | Sequence[Any] | np.ndarray | None = None,
        *,
        context=None,
        root_id: Any = None,
        cmap: str = "viridis",
        vlim: tuple[float, float] | None = None,
        palette: dict[Any, str] | None = None,
        synapses: str | None = "pre_syn",
        save_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._save_dir = _resolve_dir(save_dir)
        self._opts = dict(
            x=x, y=y, hue=hue, cmap=cmap, vlim=vlim, palette=palette, synapses=synapses
        )
        self._labels: dict[Any, dict[int, str]] = {}  # root_id -> {vertex_id: label}
        self._loader: Callable | None = None
        self._root_ids: list = []
        self._index = 0
        self.observe(self._on_select, names="selected")
        self.on_msg(self._on_msg)
        self.load(cell, feature_df, context=context, root_id=root_id)

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[Any], tuple],
        root_ids: Sequence[Any],
        start: int = 0,
        **kwargs,
    ) -> VertexExplorer:
        """A widget that steps through ``root_ids`` with its prev / next buttons.

        ``loader(root_id)`` returns ``(cell, feature_df)`` or
        ``(cell, feature_df, context)``, with the embedding columns already in
        ``feature_df``; it runs each time a cell is shown, so cache inside it if
        the UMAP is slow. Other keyword arguments go to the constructor.
        """
        root_ids = list(root_ids)
        cell, df, context = _unpack(loader(root_ids[start]))
        vx = cls(cell, df, context=context, root_id=root_ids[start], **kwargs)
        vx._loader, vx._root_ids, vx._index = loader, root_ids, start
        vx._sync_nav()
        return vx

    def load(
        self, cell, feature_df: pd.DataFrame, *, context=None, root_id: Any = None
    ) -> None:
        """Show another cell, keeping the other cells' labels and the display options."""
        o = self._opts
        skel = cell.s
        pos_of = {int(v): i for i, v in enumerate(skel.vertex_index)}
        missing = [v for v in feature_df.index if int(v) not in pos_of]
        if missing:
            raise ValueError(
                f"{len(missing)} feature_df index values are not vertices of the cell "
                f"skeleton (e.g. {missing[:3]}); pass the cell the features came from."
            )
        vpos = np.array([pos_of[int(v)] for v in feature_df.index], dtype=int)

        xy = feature_df[[o["x"], o["y"]]].to_numpy(dtype=float)
        ok = np.isfinite(xy).all(axis=1)
        rows = np.flatnonzero(ok)  # point i -> feature_df row
        cidx, palette_list, legend = _hue(
            feature_df, o["hue"], o["cmap"], o["vlim"], o["palette"]
        )

        # Shared frame: um, centred on the bounding box of everything drawn.
        xyz = skel.vertices / 1000.0
        ctx_xyz = context.s.vertices / 1000.0 if context is not None else None
        both = xyz if ctx_xyz is None else np.vstack([xyz, ctx_xyz])
        centre = (both.min(axis=0) + both.max(axis=0)) / 2

        row_of_vertex = np.full(skel.n_vertices, -1)
        row_of_vertex[vpos[rows]] = np.arange(len(rows))

        points = dict(
            x=np.round(xy[ok, 0], 4).tolist(),
            y=np.round(xy[ok, 1], 4).tolist(),
            c=cidx[ok].tolist(),
            v=vpos[ok].tolist(),
            palette=palette_list,
            names=[o["x"], o["y"]],
        )
        morph = dict(
            xyz=np.round(xyz - centre, 2).ravel().tolist(),
            parent=skel.parent_node_array.astype(int).tolist(),
            point=row_of_vertex.tolist(),  # vertex -> point, -1 if not on the scatter
            root=_root(skel),
            vindex=skel.vertex_index.astype(int).tolist(),
        )
        if ctx_xyz is not None:
            morph["context"] = dict(
                xyz=np.round(ctx_xyz - centre, 2).ravel().tolist(),
                parent=context.s.parent_node_array.astype(int).tolist(),
                root=_root(context.s),
            )
        if o["synapses"] is not None:
            morph["synapses"] = _synapses(cell, o["synapses"], centre)

        self._df, self._rows, self._n_missing = feature_df, rows, int((~ok).sum())
        self._root_id = root_id if root_id is not None else cell.name
        if not self._labels.get(self._root_id):
            self._labels[self._root_id] = self._read_saved()
        # One message, so the front end never sees points from one cell and the
        # skeleton of another.
        with self.hold_sync():
            self.selected = []
            self.points, self.morph, self.legend = points, morph, legend
            self._sync_labels()
        self._on_select(None)

    # -- python-side API --------------------------------------------------------

    @property
    def root_id(self) -> Any:
        """The cell on show."""
        return self._root_id

    @property
    def selection_mask(self) -> np.ndarray:
        """Boolean array aligned with ``feature_df``'s rows."""
        mask = np.zeros(len(self._df), bool)
        mask[self._rows[np.asarray(self.selected, dtype=int)]] = True
        return mask

    @property
    def selected_vertices(self) -> np.ndarray:
        """Skeleton vertex ids (``feature_df``'s index) of the selection."""
        return self._df.index.to_numpy()[self.selection_mask]

    def selected_frame(self) -> pd.DataFrame:
        """The selected vertices' rows of ``feature_df``."""
        return self._df[self.selection_mask]

    def select(self, vertices: Sequence[Any] | np.ndarray) -> None:
        """Set the selection from Python by vertex id, or by a boolean mask over rows."""
        vertices = np.asarray(vertices)
        if vertices.dtype == bool:
            mask = vertices
        else:
            mask = self._df.index.isin(vertices)
        self.selected = np.flatnonzero(mask[self._rows]).tolist()

    def label_selection(self, label: str | None) -> None:
        """Mark the selected vertices ``'primary'`` or ``'terminal'``; ``None`` unlabels."""
        if label is not None and label not in _CODE:
            raise ValueError(f"label must be one of {LABELS} or None, not {label!r}")
        mine = self._labels[self._root_id]
        for v in self.selected_vertices:
            if label is None:
                mine.pop(int(v), None)
            else:
                mine[int(v)] = label
        self._sync_labels()
        self._on_select(None)
        if self._save_dir is not None:
            self.save()

    def save(self) -> Path:
        """Write this cell's ``feature_df`` plus a ``label`` column to ``save_dir``."""
        if self._save_dir is None:
            raise RuntimeError(
                "no save_dir; pass save_dir=<folder> to the constructor"
            )
        mine = self._labels.get(self._root_id, {})
        out = self._df.copy()
        out.index = out.index.rename("vertex_id")
        out["label"] = pd.array([mine.get(int(v)) for v in out.index], dtype="string")
        out.insert(0, "root_id", self._root_id)
        path = self._save_dir / f"{self._root_id}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        out.reset_index().to_parquet(tmp, index=False)
        tmp.replace(path)  # never leave a half-written file
        return path

    def labels_frame(self) -> pd.DataFrame:
        """Every label so far, across cells: ``root_id, vertex_id, label``."""
        rows = [(r, v, lab) for r, d in self._labels.items() for v, lab in d.items()]
        return pd.DataFrame(rows, columns=["root_id", "vertex_id", "label"])

    def set_labels(self, frame: pd.DataFrame) -> None:
        """Merge in labels from a :meth:`labels_frame`-shaped frame (e.g. a saved one)."""
        for r, v, lab in frame[["root_id", "vertex_id", "label"]].itertuples(
            index=False
        ):
            self._labels.setdefault(r, {})[int(v)] = lab
        self._sync_labels()
        self._on_select(None)

    def goto(self, index: int) -> None:
        """Show the ``index``-th cell of a :meth:`from_loader` widget."""
        if self._loader is None:
            raise RuntimeError(
                "only a widget made with VertexExplorer.from_loader can change cells"
            )
        index = int(np.clip(index, 0, len(self._root_ids) - 1))
        if index == self._index:
            return
        root_id = self._root_ids[index]
        self.status = f"loading {root_id}..."
        try:
            cell, df, context = _unpack(self._loader(root_id))
        except Exception as e:
            self.status = f"failed to load {root_id}: {e}"
            raise
        self._index = index
        self.load(cell, df, context=context, root_id=root_id)
        self._sync_nav()

    # -- internals ----------------------------------------------------------------

    def _read_saved(self) -> dict[int, str]:
        if self._save_dir is None:
            return {}
        path = self._save_dir / f"{self._root_id}.parquet"
        if not path.exists():
            return {}
        saved = pd.read_parquet(path, columns=["vertex_id", "label"]).dropna(
            subset=["label"]
        )
        return {int(v): str(lab) for v, lab in zip(saved["vertex_id"], saved["label"])}

    def _sync_labels(self) -> None:
        mine = self._labels.get(self._root_id, {})
        ids = self._df.index.to_numpy()[self._rows]
        self.label_codes = [_CODE.get(mine.get(int(v)), 0) for v in ids]

    def _sync_nav(self) -> None:
        self.nav = dict(
            index=self._index, n=len(self._root_ids), root_id=str(self._root_id)
        )

    def _on_msg(self, widget, content, buffers) -> None:
        action = content.get("action")
        if action == "label":
            self.label_selection(content.get("label"))
        elif action == "goto":
            self.goto(self._index + int(content.get("step", 0)))

    def _on_select(self, change) -> None:
        n, total = len(self.selected), len(self._rows)
        parts = [f"{n} of {total} vertices selected" if n else f"{total} vertices"]
        if self._n_missing:
            parts.append(f"{self._n_missing} without embedding coordinates")
        codes = np.asarray(self.label_codes, dtype=int)
        parts += [f"{name} {int((codes == _CODE[name]).sum())}" for name in LABELS]
        self.status = " · ".join(parts)


def read_labels(save_dir: str | Path, labeled_only: bool = True) -> pd.DataFrame:
    """Every cell's saved features and labels as one frame (``root_id, vertex_id, ...,
    label``); with ``labeled_only``, just the labeled vertices."""
    paths = sorted(_resolve_dir(save_dir).glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=["root_id", "vertex_id", "label"])
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df[df["label"].notna()].reset_index(drop=True) if labeled_only else df


def _resolve_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    path = path.resolve()  # a relative path is taken from the working directory
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unpack(loaded) -> tuple:
    """``(cell, feature_df)`` or ``(cell, feature_df, context)`` -> all three."""
    if len(loaded) == 2:
        return (*loaded, None)
    return tuple(loaded)


def _synapses(cell, layer: str, centre: np.ndarray) -> dict[str, list]:
    """Synapse positions (um, in the shared frame) and each one's skeleton vertex
    position, -1 where it maps to none."""
    if layer not in cell.annotations.names:
        raise ValueError(
            f"{layer!r} is not an annotation layer of the cell; it has {cell.annotations.names}"
        )
    ann = cell.annotations[layer]
    xyz = ann.vertices / 1000.0 - centre
    vertex = ann.map_index_to_layer("skeleton", as_positional=True, missing=-1)
    return dict(
        xyz=np.round(xyz, 2).ravel().tolist(),
        vertex=np.asarray(vertex, dtype=int).tolist(),
    )


def _root(skel) -> int:
    """The root's position; for a masked piece (e.g. the axon alone), its one
    parentless vertex, which is the vertex nearest the soma."""
    r = skel.root_positional
    if r is not None:
        return int(r)
    tops = np.flatnonzero(skel.parent_node_array < 0)
    return int(tops[0]) if len(tops) == 1 else -1


def _hue(df, hue, cmap, vlim, palette) -> tuple[np.ndarray, list[str], dict]:
    """Per-row palette index, the palette, and a legend description."""
    import matplotlib as mpl

    n = len(df)
    if hue is None:
        return np.zeros(n, int), ["#7d7b74"], {}
    name = hue if isinstance(hue, str) else getattr(hue, "name", None) or "hue"
    values = df[hue].to_numpy() if isinstance(hue, str) else np.asarray(hue)
    if len(values) != n:
        raise ValueError(f"hue has {len(values)} values for {n} rows")

    if np.issubdtype(values.dtype, np.number) and values.dtype != bool:
        v = values.astype(float)
        finite = np.isfinite(v)
        lo, hi = (
            vlim
            if vlim is not None
            else (np.nanpercentile(v[finite], [2, 98]) if finite.any() else (0.0, 1.0))
        )
        span = (hi - lo) or 1.0
        bins = np.clip(
            ((v - lo) / span * N_BINS).astype(int, copy=False), 0, N_BINS - 1
        )
        cm = mpl.colormaps[cmap]
        colors = [mpl.colors.to_hex(cm((i + 0.5) / N_BINS)) for i in range(N_BINS)]
        # NaN gets its own grey slot at the end.
        idx = np.where(finite, bins, N_BINS)
        colors.append("#bbbbbb")
        return (
            idx,
            colors,
            dict(
                kind="numeric",
                name=name,
                lo=float(lo),
                hi=float(hi),
                ramp=colors[:N_BINS],
            ),
        )

    labels = pd.Series(values).astype("string").fillna("–")
    cats = sorted(labels.unique())
    cyc = mpl.colormaps["tab10"].colors
    colors = [
        (palette or {}).get(c) or mpl.colors.to_hex(cyc[i % len(cyc)])
        for i, c in enumerate(cats)
    ]
    lookup = {c: i for i, c in enumerate(cats)}
    return (
        labels.map(lookup).to_numpy(dtype=int),
        colors,
        dict(kind="categorical", name=name, names=cats, colors=colors),
    )
