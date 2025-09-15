import contextlib
import copy
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Self, Union

import numpy as np
import pandas as pd

from . import utils
from .data_layers import (
    GRAPH_LAYER_NAME,
    MESH_LAYER_NAME,
    SKEL_LAYER_NAME,
    GraphLayer,
    MeshLayer,
    PointCloudLayer,
    SkeletonLayer,
)
from .sync_classes import *

__all__ = [
    "Cell",
    "GraphLayer",
    "MeshLayer",
    "SkeletonLayer",
    "PointCloudLayer",
    "Link",
    "LayerManager",
    "AnnotationManager",
]


class LayerManager:
    """Unified manager for both morphological layers and annotations with flexible validation."""

    def __init__(
        self,
        managed_layers: Optional[dict] = None,
        validation: Union[str, Callable] = "any",
        context: str = "layer",
        initial_layers: Optional[list] = None,
    ):
        """Initialize the unified layer manager.

        Parameters
        ----------
        managed_layers : dict, optional
            Dictionary to store layers in. If None, creates new dict.
        validation : str or callable, default 'any'
            Type validation mode:
            - 'any': Accept any layer type
            - 'point_cloud_only': Only accept PointCloudLayer
            - callable: Custom validation function
        context : str, default 'layer'
            Context name for error messages ('layer', 'annotation', etc.)
        initial_layers : list, optional
            Initial layers to add (with validation)
        """
        self._layers = managed_layers if managed_layers is not None else {}
        self._validation = validation
        self._context = context

        # Add initial layers if provided
        if initial_layers is not None:
            for layer in initial_layers:
                self._add(layer)

    def _validate_layer(self, layer) -> None:
        """Validate layer type based on validation mode."""
        if self._validation == "any":
            # Accept any layer type
            return
        elif self._validation == "point_cloud_only":
            if not isinstance(layer, PointCloudLayer):
                raise ValueError(
                    f"{self._context.capitalize()} layers must be PointCloudLayer instances."
                )
        elif callable(self._validation):
            if not self._validation(layer):
                raise ValueError(
                    f"Layer validation failed for {self._context}: {layer}"
                )
        else:
            raise ValueError(f"Unknown validation mode: {self._validation}")

    @property
    def names(self) -> list:
        """Return a list of managed layer names."""
        return list(self._layers.keys())

    def _add(
        self, layer: Union[SkeletonLayer, GraphLayer, MeshLayer, PointCloudLayer]
    ) -> None:
        """Add a new layer to the manager. Should only be used by the Cell object."""
        self._validate_layer(layer)
        self._layers[layer.name] = layer

    def get(self, name: str, default: Any = None):
        """Get a layer by name with optional default."""
        return self._layers.get(name, default)

    def __getitem__(self, name: str):
        return self._layers[name]

    def __getattr__(self, name: str):
        if name in self._layers:
            return self._layers[name]
        else:
            raise AttributeError(
                f'{self._context.capitalize()} "{name}" does not exist.'
            )

    def __dir__(self):
        return super().__dir__() + list(self._layers.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a layer exists by name."""
        return name in self._layers

    def __iter__(self):
        """Iterate over managed layers in order."""
        return iter(self._layers.values())

    def __len__(self):
        return len(self._layers)

    def _remove(self, name: str) -> None:
        """Remove a layer from the manager. Should only be used internally."""
        if name not in self._layers:
            raise ValueError(f'{self._context.capitalize()} "{name}" does not exist.')
        del self._layers[name]

    def describe(self) -> None:
        """Generate a detailed description of all managed layers.

        Shows each layer with its metrics, labels, and links - same level of
        detail as individual layer describe() methods.

        Returns
        -------
        None
            Always returns None (prints formatted text)
        """
        print(self._describe_text())

    def _describe_text(self) -> str:
        """Generate text-based description of all managed layers."""
        lines = []

        # Header with context and count
        context_title = self._context.capitalize() + "s"
        layer_count = len(self._layers)
        lines.append(f"# {context_title} ({layer_count})")

        if layer_count == 0:
            lines.append("└── No layers present")
            return "\n".join(lines)

        # List each layer with details
        layer_items = list(self._layers.items())
        for i, (name, layer) in enumerate(layer_items):
            is_last = i == len(layer_items) - 1
            prefix = "└──" if is_last else "├──"
            continuation = "    " if is_last else "│   "

            # Layer name and type
            lines.append(f"{prefix} {name} ({layer.__class__.__name__})")

            # Get metrics
            metrics = layer._get_layer_metrics()
            lines.append(f"{continuation}├── {metrics}")

            # Labels
            if len(layer.label_names) > 0:
                label_list = ", ".join(layer.label_names)
                label_line = f"{continuation}├── Labels: [{label_list}]"
                lines.extend(self._wrap_line(label_line))
            else:
                lines.append(f"{continuation}├── Labels: []")

            # Links
            link_display = layer._get_link_display()
            if link_display:
                link_line = f"{continuation}└── Links: {link_display}"
                lines.extend(self._wrap_line(link_line))
            else:
                lines.append(f"{continuation}└── Links: []")

        return "\n".join(lines)

    def _wrap_line(self, line: str, max_width: int = 120) -> List[str]:
        """Wrap a long line to max_width characters with proper tree indentation."""
        if len(line) <= max_width:
            return [line]

        # Find where the actual content starts (after tree characters and whitespace)
        content_start = 0
        for i, char in enumerate(line):
            if char not in "├└─│ ":
                content_start = i
                break

        prefix = line[:content_start]
        content = line[content_start:]

        # Calculate available width for content
        available_width = max_width - len(prefix)

        if len(content) <= available_width:
            return [line]

        # Split the content, preserving brackets and commas
        wrapped_lines = []
        current_line = prefix + content

        while len(current_line) > max_width:
            # Find the best place to break (prefer after comma + space)
            break_pos = max_width

            # Look for comma + space within reasonable range
            search_start = max(max_width - 30, len(prefix))
            for i in range(min(max_width, len(current_line) - 1), search_start, -1):
                if current_line[i : i + 2] == ", ":
                    break_pos = i + 2
                    break

            # If no good break found, break at max_width
            if break_pos == max_width and break_pos < len(current_line):
                # Make sure we don't break in the middle of a word
                while break_pos > len(prefix) and current_line[break_pos] not in ", ":
                    break_pos -= 1
                if break_pos <= len(prefix):
                    break_pos = max_width

            wrapped_lines.append(current_line[:break_pos])
            # Use appropriate continuation indentation based on prefix
            if "└──" in prefix:
                continuation_indent = "    " + " " * max(0, (len(prefix) - 4))
            else:
                continuation_indent = "│   " + " " * max(0, (len(prefix) - 4))
            current_line = continuation_indent + current_line[break_pos:].lstrip()

        if current_line.strip():
            wrapped_lines.append(current_line)

        return wrapped_lines

    def __repr__(self) -> str:
        return f"LayerManager({self._context}s={list(self._layers.keys())})"


AnnotationManager = partial(
    LayerManager, validation="point_cloud_only", context="annotation"
)


class Cell:
    SKEL_LN = SKEL_LAYER_NAME
    GRAPH_LN = GRAPH_LAYER_NAME
    MESH_LN = MESH_LAYER_NAME

    def __init__(
        self,
        name: Optional[Union[int, str]] = None,
        morphsync: Optional[MorphSync] = None,
        meta: Optional[dict] = None,
        annotation_layers: Optional[list] = None,
    ):
        if morphsync is None:
            self._morphsync = MorphSync()
        else:
            self._morphsync = copy.deepcopy(morphsync)
        self._name = name
        if meta is None:
            self._meta = dict()
        else:
            self._meta = copy.copy(meta)
        self._managed_layers = {}
        self._layers = LayerManager(
            managed_layers=self._managed_layers, validation="any", context="layer"
        )
        self._annotations = LayerManager(
            managed_layers=None,  # Separate dict for annotations
            validation="point_cloud_only",
            context="annotation",
            initial_layers=annotation_layers,
        )

    @property
    def name(self) -> str:
        "Get the name of the cell (typically a segment id)"
        return self._name

    @property
    def meta(self) -> dict:
        "Get the metadata associated with the cell."
        return self._meta

    @property
    def layers(self) -> LayerManager:
        "Get the non-annotation layers of the cell."
        return self._layers

    def _get_layer(self, layer_name: str):
        "Get a managed layer by name."
        return self._managed_layers.get(layer_name)

    @property
    def skeleton(self) -> Optional[SkeletonLayer]:
        "Skeleton layer for the cell, if present. Otherwise, None."
        if self.SKEL_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.SKEL_LN]

    @property
    def graph(self) -> Optional[GraphLayer]:
        "Graph layer for the cell, if present. Otherwise, None."
        if self.GRAPH_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.GRAPH_LN]

    @property
    def mesh(self) -> Optional[MeshLayer]:
        "Mesh layer for the cell, if present. Otherwise, None."
        if self.MESH_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.MESH_LN]

    @property
    def annotations(self) -> LayerManager:
        "Annotation Manager for the cell, holding all annotation layers."
        return self._annotations

    @property
    def s(self) -> Optional[SkeletonLayer]:
        "Alias for skeleton."
        return self.skeleton

    @property
    def g(self) -> Optional[GraphLayer]:
        "Alias for graph."
        return self.graph

    @property
    def m(self) -> Optional[MeshLayer]:
        "Alias for mesh."
        return self.mesh

    @property
    def a(self) -> LayerManager:
        "Alias for annotations."
        return self.annotations

    @property
    def l(self) -> LayerManager:
        "Alias for layers."
        return self.layers

    @property
    def _all_objects(self) -> dict:
        "All morphological layers and annotation layers in a single dictionary."
        return {**self._managed_layers, **self._annotations._layers}

    def add_layer(
        self,
        layer: Union[PointCloudLayer, GraphLayer, SkeletonLayer, MeshLayer],
    ) -> Self:
        """Add a initialized layer to the MorphSync.

        Parameters
        ----------
        layer : Union[PointCloudLayer, GraphLayer, SkeletonLayer, MeshLayer]
            The layer to add.

        Raises
        ------
        ValueError
            If the layer already exists or if the layer type is incorrect.
        """
        name = layer.name
        if name in self._managed_layers:
            raise ValueError(f'Layer "{name}" already exists!')
        if name == self.MESH_LN and not issubclass(type(layer), MeshLayer):
            raise ValueError(f'Layer "{name}" must be a MeshLayer!')
        if name == self.SKEL_LN and not issubclass(type(layer), SkeletonLayer):
            raise ValueError(f'Layer "{name}" must be a SkeletonLayer!')
        if name == self.GRAPH_LN and not issubclass(type(layer), GraphLayer):
            raise ValueError(f'Layer "{name}" must be a GraphLayer!')
        if self._morphsync != layer._morphsync:
            raise ValueError("Incompatible MorphSync objects.")
        self._layers._add(layer)
        return self

    def add_mesh(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, MeshLayer],
        faces: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        linkage: Optional[Link] = None,
        spatial_columns: Optional[list] = None,
    ) -> Self:
        """Add a mesh layer to the MorphSync.

        Parameters
        ----------
        vertices : Union[np.ndarray, pd.DataFrame, MeshLayer]
            The vertices of the mesh, or a MeshLayer object.
        faces : Union[np.ndarray, pd.DataFrame]
            The faces of the mesh. If faces are provided as a dataframe, faces should be in dataframe indices.
        labels : Optional[Union[dict, pd.DataFrame]]
            Additional labels for the mesh. If passed as dictionary, the key is the label name and the values are an array of label values.
        vertex_index : Optional[Union[str, np.ndarray]]
            The column to use as a vertex index for the mesh, if vertices are a dataframe.
        linkage : Optional[Link]
            The linkage information for the mesh.
        spatial_columns: Optional[list] = None
            The spatial columns for the mesh, if vertices are a dataframe.

        Returns
        -------
        Self
        """
        if self.mesh is not None:
            raise ValueError('"Mesh already exists!')

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)
        if isinstance(vertices, MeshLayer):
            self.add_layer(vertices)
        else:
            self._layers._add(
                MeshLayer(
                    name=self.MESH_LN,
                    vertices=vertices,
                    faces=faces,
                    labels=labels,
                    morphsync=self._morphsync,
                    spatial_columns=spatial_columns,
                    linkage=linkage,
                    vertex_index=vertex_index,
                )
            )
            self._managed_layers[self.MESH_LN]._register_cell(self)
        return self

    def add_skeleton(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonLayer],
        edges: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        root: Optional[int] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        linkage: Optional[Link] = None,
        spatial_columns: Optional[list] = None,
        inherited_properties: Optional[dict] = None,
    ) -> Self:
        """
        Add a skeleton layer to the MorphSync.

        Parameters
        ----------
        vertices : Union[np.ndarray, pd.DataFrame, SkeletonLayer]
            The vertices of the skeleton, or a SkeletonLayer object.
        edges : Union[np.ndarray, pd.DataFrame]
            The edges of the skeleton.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the skeleton.
        root : Optional[int]
            The root vertex for the skeleton, required of the edges are not already consistent with a single root.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the skeleton.
        linkage : Optional[Link]
            The linkage information for the skeleton. Typically, you will define the source vertices for the skeleton if using a graph-to-skeleton mapping.
        spatial_columns: Optional[list] = None
            The spatial columns for the skeleton, if vertices are a dataframe.

        Returns
        -------
        Self
        """
        if self.skeleton is not None:
            raise ValueError('"Skeleton already exists!')

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if isinstance(vertices, SkeletonLayer):
            self.add_layer(vertices)
        else:
            self._layers._add(
                SkeletonLayer(
                    name=self.SKEL_LN,
                    vertices=vertices,
                    edges=edges,
                    labels=labels,
                    root=root,
                    morphsync=self._morphsync,
                    spatial_columns=spatial_columns,
                    linkage=linkage,
                    vertex_index=vertex_index,
                    inherited_properties=inherited_properties,
                )
            )
            self._managed_layers[self.SKEL_LN]._register_cell(self)
        return self

    def add_graph(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonLayer],
        edges: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        spatial_columns: Optional[list] = None,
        linkage: Optional[Link] = None,
    ) -> Self:
        """
        Add the core graph layer to a MeshWork object.
        Additional graph layers can be used, but they must be added separately and with unique names.

        Parameters
        ----------
        vertices : Union[np.ndarray, pd.DataFrame, SkeletonLayer]
            The vertices of the graph.
        edges : Union[np.ndarray, pd.DataFrame]
            The edges of the graph.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the graph.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the graph.
        spatial_columns: Optional[list] = None
            The spatial columns for the graph, if vertices are a dataframe.

        Returns
        -------
        Self
        """
        if self.graph is not None:
            raise ValueError('"Graph already exists!')
        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)
        if isinstance(vertices, GraphLayer):
            self.add_layer(
                vertices,
            )
        else:
            self._layers._add(
                GraphLayer(
                    name=self.GRAPH_LN,
                    vertices=vertices,
                    edges=edges,
                    labels=labels,
                    morphsync=self._morphsync,
                    spatial_columns=spatial_columns,
                    linkage=linkage,
                    vertex_index=vertex_index,
                )
            )
            self._managed_layers[self.GRAPH_LN]._register_cell(self)
        return self

    def add_point_annotations(
        self,
        name: str,
        vertices: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[Link] = None,
        vertices_from_linkage: bool = False,
    ) -> Self:
        """
        Add point annotations to the MeshWork object.  This is intended for annotations which are typically sparse and represent specific features, unlike general point clouds that represent the morphology of the cell.

        Parameters
        ----------
        name : str
            The name of the annotation layer.
        vertices : Union[np.ndarray, pd.DataFrame]
            The vertices of the annotation layer.
        spatial_columns : Optional[list]
            The spatial columns for the annotation layer.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the annotation layer.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the annotation layer.
        linkage : Optional[Link]
            The linkage information for the annotation layer. Typically, you will define the target vertices for annotations.
        vertices_from_linkage : bool
            If True, the vertices will be inferred from the linkage mapping rather than the provided vertices. This is useful if you want to create an annotation layer that directly maps to another layer without providing separate vertex coordinates.

        Returns
        -------
        Self
        """

        if name in self._managed_layers:
            raise ValueError(f"Layer '{name}' already exists.")

        if isinstance(spatial_columns, str) or spatial_columns is None:
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if vertices is None and not vertices_from_linkage:
            raise ValueError(
                "Either vertices or vertices_from_linkage must be provided."
            )

        if isinstance(vertices, PointCloudLayer):
            anno = PointCloudLayer(
                name=name,
                vertices=vertices.vertices,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            if vertices_from_linkage:
                if not isinstance(vertices, pd.DataFrame):
                    raise ValueError(
                        "Vertices must be a DataFrame when using vertices_from_linkage."
                    )
                if linkage.map_value_is_index:
                    sp_verts = (
                        self._all_objects[linkage.target]
                        .vertex_df.loc[linkage.mapping]
                        .values
                    )
                else:
                    sp_verts = self._all_objects[linkage.target].vertices[
                        linkage.mapping
                    ]
                for ii, col in enumerate(spatial_columns):
                    vertices[col] = sp_verts[:, ii]
            anno = PointCloudLayer(
                name=name,
                vertices=vertices,
                spatial_columns=spatial_columns,
                vertex_index=vertex_index,
                labels=labels,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        anno._register_cell(self)
        self._annotations._add(anno)
        return self

    def add_point_layer(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[Link] = None,
    ) -> Self:
        """
        Add point layer to the MeshWork object. This is intended for general point clouds that represent the morphology of the cell, unlike annotations which are typically sparse and represent specific features.

        Parameters
        ----------
        name : str
            The name of the point layer.
        vertices : Union[np.ndarray, pd.DataFrame]
            The vertices of the point layer.
        spatial_columns : Optional[list]
            The spatial columns for the point layer.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the annotation layer.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the annotation layer.
        linkage : Optional[Link]
            The linkage information for the annotation layer. Typically, you will define the target vertices for annotations.

        Returns
        -------
        Self
        """

        if name in self._managed_layers:
            raise ValueError(f"Layer '{name}' already exists.")

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if isinstance(vertices, PointCloudLayer):
            layer = PointCloudLayer(
                name=name,
                vertices=vertices.vertices,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            layer = PointCloudLayer(
                name=name,
                vertices=vertices,
                spatial_columns=spatial_columns,
                vertex_index=vertex_index,
                labels=labels,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        layer._register_cell(self)
        self._layers._add(layer)
        return self

    def apply_mask(
        self, layer: str, mask: np.ndarray, as_positional: bool = False
    ) -> Self:
        """Create a new Cell with vertices masked out.

        Parameters
        ----------
        layer : str
            The layer name that the mask is based on.
        mask : np.ndarray
            The mask to apply. Values that are True are preserved, while values that are False are discarded.
            Can be a boolean array or an array of vertices.
        as_positional : bool
            If mask is an array of vertices, this sets whether indices are in dataframe indices or as_positional indices.

        Returns
        -------
        Self
        """
        new_morphsync = self.layers[layer]._mask_morphsync(
            mask, as_positional=as_positional
        )
        return self.__class__._from_existing(new_morphsync, self)

    def __repr__(self) -> str:
        layers = self.layers.names
        annos = self.annotations.names
        return f"Cell(name={self.name}, layers={sorted(layers)}, annotations={annos})"

    @classmethod
    def _from_existing(cls, new_morphsync: MorphSync, old_obj: Self) -> Self:
        """Build a new Cell from existing data and a new morphsync."""
        new_obj = cls(
            name=old_obj.name,
            morphsync=new_morphsync,
            meta=old_obj.meta,
        )
        for old_layer in old_obj.layers:
            new_layer = old_layer.__class__._from_existing(new_morphsync, old_layer)
            new_layer._register_cell(new_obj)
            new_obj._layers._add(new_layer)
        for old_anno in old_obj.annotations:
            new_anno = old_anno.__class__._from_existing(new_morphsync, old_anno)
            new_anno._register_cell(new_obj)
            new_obj._annotations._add(new_anno)
        return new_obj

    def copy(self) -> Self:
        """Create a deep copy of the Cell."""
        return self.__class__._from_existing(copy.deepcopy(self._morphsync), self)

    def transform(
        self, transform: Union[np.ndarray, callable], inplace: bool = False
    ) -> Self:
        """Apply a spatial transformation to all spatial layers in the Cell.

        Parameters
        ----------
        transform : Union[np.ndarray, callable]
            If an array, must be the same shape as the vertices of the layer(s).
            If a callable, must take in a (N, 3) array and return a (N, 3) array.
        inplace : bool
            If True, modify the current Cell. If False, return a new Cell.

        Returns
        -------
        Self
            The transformed Cell.
        """
        if not inplace:
            target = self.copy()
        else:
            target = self
        for layer in target._all_objects.values():
            layer.transform(transform, inplace=True)
        return target

    @property
    def labels(self) -> pd.DataFrame:
        """Return a DataFrame listing all label columns across all layers. Each label is a row, with the layer name and label name as columns."""
        all_labels = []
        for layer in self.layers:
            all_labels.append(
                pd.DataFrame({"layer": layer.name, "labels": layer.labels.columns})
            )
        return pd.concat(all_labels)

    def get_labels(
        self,
        labels: Union[str, list],
        target_layer: str,
        source_layers: Optional[Union[str, list]] = None,
        agg: Union[str, dict] = "median",
    ) -> pd.DataFrame:
        """Map label columns from various sources to a target layer.

        Parameters
        ----------
        labels : Union[str, list]
            The labels to map from the source layer.
        target_layer : str
            The target layer to map all labels to.
        source_layers : Optional[Union[str, list]]
            The source layers to map the labels from. Unnecessary if labels are unique.
        agg : Union[str, dict]
            The aggregation method to use when mapping the labels.
            Anything pandas `groupby.agg` takes, as well as "majority" which will is a majority vote across the mapped indices via the stats.mode function.

        Returns
        -------
        pd.DataFrame
            The mapped labels for the target layer.
        """
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(source_layers, str):
            source_layers = [source_layers]
        elif source_layers is None:
            source_layers = [None] * len(labels)
        remap_labels = []
        for label, source_layer in zip(labels, source_layers):
            label_row = self.labels.query("labels == @label")
            if label_row.shape[0] == 0:
                raise ValueError(f'Label "{label}" not found in any layer.')
            if label_row.shape[0] > 1 and source_layer is None:
                raise ValueError(
                    f'Label "{label}" found in multiple layers, please specify a source layer.'
                )
            if source_layer is None:
                source_layer = label_row.iloc[0]["layer"]
            remap_labels.append(
                self.layers[source_layer].map_labels_to_layer(
                    labels=label,
                    layer=target_layer,
                    agg=agg,
                )
            )
        return pd.concat(remap_labels, axis=1)

    @contextlib.contextmanager
    def mask_context(
        self,
        layer: str,
        mask: np.ndarray,
    ) -> Generator[Self, None, None]:
        """Create a masked version of the MeshWork object in a context state.

        Parameters
        ----------
        layer: str
            The name of the layer to which the mask applies.
        mask : array or None
            A boolean array with the same number of elements as mesh vertices. True elements are kept, False are masked out.

        Example
        -------
        >>> with mesh.mask_context("layer_name", mask) as masked_mesh:
        >>>     result = my_favorite_function(masked_mesh)
        """
        nrn_out = self.apply_mask(layer, mask)
        try:
            yield nrn_out
        finally:
            pass

    def _cleanup_links(self, layer_name: str) -> None:
        """Remove all links involving the specified layer from MorphSync."""
        links_to_remove = []
        for link_key in self._morphsync.links.keys():
            if layer_name in link_key:
                links_to_remove.append(link_key)

        for link_key in links_to_remove:
            del self._morphsync.links[link_key]

    def remove_layer(self, name: str) -> Self:
        """Remove a morphological layer from the Cell.

        Parameters
        ----------
        name : str
            The name of the layer to remove

        Returns
        -------
        Self
            The Cell object for method chaining

        Raises
        ------
        ValueError
            If the layer does not exist or is a core layer that cannot be removed
        """
        # Check if layer exists
        if name not in self._managed_layers:
            raise ValueError(f'Layer "{name}" does not exist.')

        # Prevent removal of core layers if they have specific restrictions
        # (This could be extended based on business logic requirements)

        # Remove from layer manager
        self._layers._remove(name)

        # Remove from MorphSync layers
        if name in self._morphsync.layers:
            del self._morphsync.layers[name]

        # Clean up all related links
        self._cleanup_links(name)

        return self

    def remove_annotation(self, name: str) -> Self:
        """Remove an annotation layer from the Cell.

        Parameters
        ----------
        name : str
            The name of the annotation to remove

        Returns
        -------
        Self
            The Cell object for method chaining

        Raises
        ------
        ValueError
            If the annotation does not exist
        """
        # Check if annotation exists
        if name not in self._annotations:
            raise ValueError(f'Annotation "{name}" does not exist.')

        # Remove from annotation manager
        self._annotations._remove(name)

        # Remove from MorphSync layers
        if name in self._morphsync.layers:
            del self._morphsync.layers[name]

        # Clean up all related links
        self._cleanup_links(name)

        return self

    def describe(self, html: bool = False) -> None:
        """Generate a hierarchical summary description of the cell and its layers.

        Provides a tree-like overview including:
        - Cell name and basic info
        - Layers section with expandable details
        - Annotations section with expandable details

        Parameters
        ----------
        html : bool, optional
            If True, create expandable HTML widgets in Jupyter.
            If False, print formatted string (default).

        Returns
        -------
        None
            Always returns None (prints text or displays HTML widgets)

        Examples
        --------
        >>> cell.describe()  # Prints formatted string
        >>> cell.describe(html=True)  # Shows HTML widgets in Jupyter
        """
        if html:
            self._describe_html()
        else:
            print(self._describe_text())

    def _describe_text(self) -> str:
        """Generate text-based hierarchical description."""
        lines = []

        # Cell header
        cell_name = str(self.name) if self.name is not None else "Unnamed"

        # Count layers, annotations, and potential linkages
        num_layers = self._count_total_layers()
        num_annotations = len(self.annotations)
        num_linkages = (
            len(self._morphsync.links) if hasattr(self._morphsync, "links") else 0
        )

        lines.append(f"# Cell: {cell_name}")

        # Layers section with count
        lines.append(f"├── Layers ({num_layers})")

        # Always show skeleton, mesh, graph in that order
        core_layers = [
            ("skeleton", self.skeleton),
            ("mesh", self.mesh),
            ("graph", self.graph),
        ]

        # Additional layers (non-core layers)
        core_layer_names = {self.MESH_LN, self.SKEL_LN, self.GRAPH_LN}
        additional_layers = [
            layer for layer in self.layers if layer.name not in core_layer_names
        ]

        all_layers = core_layers + [(layer.name, layer) for layer in additional_layers]

        for i, (layer_name, layer_obj) in enumerate(all_layers):
            is_last_layer = (
                i == len(all_layers) - 1 and num_annotations == 0 and num_linkages == 0
            )

            if is_last_layer:
                prefix = "└──"
            else:
                prefix = "├──"

            if layer_obj is not None:
                vertex_count = len(layer_obj.vertices)

                if hasattr(layer_obj, "faces"):  # Mesh layers
                    face_count = (
                        len(layer_obj.faces) if len(layer_obj.faces.shape) > 1 else 0
                    )
                    lines.append(
                        f"│   {prefix} {layer_name}: {vertex_count} vertices, {face_count} faces"
                    )
                elif hasattr(layer_obj, "edges"):  # Graph/Skeleton layers
                    edge_count = (
                        len(layer_obj.edges) if len(layer_obj.edges.shape) > 1 else 0
                    )
                    lines.append(
                        f"│   {prefix} {layer_name}: {vertex_count} vertices, {edge_count} edges"
                    )
                else:
                    lines.append(f"│   {prefix} {layer_name}: {vertex_count} vertices")
            else:
                lines.append(f"│   {prefix} {layer_name}: not present")

        # Annotations section
        if num_annotations > 0:
            if num_linkages > 0:
                lines.append(f"├── Annotations ({num_annotations})")
            else:
                lines.append(f"└── Annotations ({num_annotations})")

            for i, annotation in enumerate(self.annotations):
                is_last_annotation = i == len(self.annotations) - 1

                if is_last_annotation:
                    prefix = "└──"
                else:
                    prefix = "├──"

                if num_linkages == 0:
                    continuation = "    " if is_last_annotation else "│   "
                else:
                    continuation = "│   "

                vertex_count = len(annotation.vertices)
                lines.append(
                    f"{continuation}{prefix} {annotation.name}: {vertex_count} points"
                )

        # Linkage section (if there are any links)
        if num_linkages > 0:
            # Process links to detect bidirectional connections
            processed_pairs = set()
            link_display_items = []

            for link_key in self._morphsync.links.keys():
                source, target = link_key
                reverse_key = (target, source)

                # Skip if we've already processed this pair in reverse
                if reverse_key in processed_pairs:
                    continue

                # Check if bidirectional link exists
                if reverse_key in self._morphsync.links:
                    # Bidirectional link
                    link_display_items.append(f"{source} <-> {target}")
                    processed_pairs.add(link_key)
                    processed_pairs.add(reverse_key)
                else:
                    # Unidirectional link
                    link_display_items.append(f"{source} → {target}")
                    processed_pairs.add(link_key)

            # Show the actual number of connection pairs being displayed
            num_connections = len(link_display_items)
            lines.append(f"└── Linkage ({num_connections} connections)")

            for i, link_display in enumerate(link_display_items):
                is_last_link = i == len(link_display_items) - 1
                prefix = "└──" if is_last_link else "├──"
                lines.append(f"    {prefix} {link_display}")

        return "\n".join(lines)

    def _wrap_line(
        self, line: str, max_width: int, continuation_indent: str
    ) -> List[str]:
        """Wrap a long line to max_width characters with proper indentation."""
        if len(line) <= max_width:
            return [line]

        # Find where the actual content starts (after tree characters and whitespace)
        content_start = 0
        for i, char in enumerate(line):
            if char not in "├└─│ ":
                content_start = i
                break

        prefix = line[:content_start]
        content = line[content_start:]

        # Calculate available width for content
        available_width = max_width - len(prefix)

        if len(content) <= available_width:
            return [line]

        # Split the content, preserving brackets and commas
        wrapped_lines = []
        current_line = prefix + content

        while len(current_line) > max_width:
            # Find the best place to break (prefer after comma + space)
            break_pos = max_width

            # Look for comma + space within reasonable range
            search_start = max(max_width - 30, len(prefix))
            for i in range(min(max_width, len(current_line) - 1), search_start, -1):
                if current_line[i : i + 2] == ", ":
                    break_pos = i + 2
                    break

            # If no good break found, break at max_width
            if break_pos == max_width and break_pos < len(current_line):
                # Make sure we don't break in the middle of a word
                while break_pos > len(prefix) and current_line[break_pos] not in ", ":
                    break_pos -= 1
                if break_pos <= len(prefix):
                    break_pos = max_width

            wrapped_lines.append(current_line[:break_pos])
            current_line = continuation_indent + current_line[break_pos:].lstrip()

        if current_line.strip():
            wrapped_lines.append(current_line)

        return wrapped_lines

    def _describe_html(self) -> None:
        """Generate HTML widgets with expandable sections for Jupyter."""
        try:
            import uuid

            from IPython.display import HTML, display

            # Generate unique IDs for this cell description
            cell_id = str(uuid.uuid4())[:8]

            # Cell name
            cell_name = str(self.name) if self.name is not None else "Unnamed"

            # Build layers section
            layers_html = self._build_layers_html()

            # Build annotations section
            annotations_html = self._build_annotations_html()

            # Complete HTML with styling
            html_content = f"""
            <style>
            .cell-describe-{{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 10px 0;
                border-left: 3px solid #007acc;
                padding-left: 10px;
            }}
            .cell-title {{
                font-size: 18px;
                font-weight: bold;
                color: var(--jp-content-font-color1, var(--vscode-foreground, #000000));
                margin-bottom: 10px;
                background: var(--jp-layout-color0, var(--vscode-editor-background, transparent));
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid var(--jp-border-color1, var(--vscode-panel-border, #007acc));
            }}
            .section-header {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 8px 12px;
                cursor: pointer;
                user-select: none;
                margin: 5px 0;
                border-radius: 4px;
                transition: background-color 0.2s;
                color: #000000;
                font-weight: 500;
            }}
            .section-header:hover {{
                background: #e9ecef;
            }}
            .section-header::before {{
                content: '▶ ';
                transition: transform 0.2s;
                display: inline-block;
            }}
            .section-header.expanded::before {{
                transform: rotate(90deg);
            }}
            .section-content {{
                display: none;
                padding: 10px 15px;
                margin-left: 20px;
                border-left: 2px solid #dee2e6;
                background: #fafbfc;
            }}
            .section-content.expanded {{
                display: block;
            }}
            .layer-item {{
                margin: 8px 0;
                padding: 6px;
                background: white;
                border-radius: 3px;
                border-left: 3px solid #28a745;
            }}
            .layer-item.not-present {{
                border-left-color: #dc3545;
                opacity: 0.7;
            }}
            .layer-name {{
                font-weight: bold;
                color: #495057;
            }}
            .layer-details {{
                font-size: 0.9em;
                color: #6c757d;
                margin-top: 4px;
            }}
            .annotation-item {{
                margin: 8px 0;
                padding: 6px;
                background: white;
                border-radius: 3px;
                border-left: 3px solid #17a2b8;
            }}
            </style>
            
            <div class="cell-describe">
                <div class="cell-title">Cell: {cell_name}</div>
                
                <div class="section-header" onclick="toggleSection('{cell_id}_layers')">
                    Layers ({self._count_present_layers()}/{self._count_total_layers()})
                </div>
                <div class="section-content" id="{cell_id}_layers">
                    {layers_html}
                </div>
                
                <div class="section-header" onclick="toggleSection('{cell_id}_annotations')">
                    Annotations ({len(self.annotations)})
                </div>
                <div class="section-content" id="{cell_id}_annotations">
                    {annotations_html}
                </div>
            </div>
            
            <script>
            function toggleSection(sectionId) {{
                const content = document.getElementById(sectionId);
                const header = content.previousElementSibling;
                
                if (content.classList.contains('expanded')) {{
                    content.classList.remove('expanded');
                    header.classList.remove('expanded');
                }} else {{
                    content.classList.add('expanded');
                    header.classList.add('expanded');
                }}
            }}
            </script>
            """

            display(HTML(html_content))
            return None

        except ImportError:
            # Fallback to text version if not in Jupyter
            return self._describe_text()

    def _count_present_layers(self) -> int:
        """Count how many core layers are present."""
        count = 0
        if self.mesh is not None:
            count += 1
        if self.skeleton is not None:
            count += 1
        if self.graph is not None:
            count += 1
        # Add additional layers
        core_layer_names = {self.MESH_LN, self.SKEL_LN, self.GRAPH_LN}
        count += len([l for l in self.layers if l.name not in core_layer_names])
        return count

    def _count_total_layers(self) -> int:
        """Count total possible layers (3 core + additional)."""
        core_layer_names = {self.MESH_LN, self.SKEL_LN, self.GRAPH_LN}
        additional_count = len(
            [l for l in self.layers if l.name not in core_layer_names]
        )
        return 3 + additional_count

    def _build_layers_html(self) -> str:
        """Build HTML for layers section."""
        html_parts = []

        # Core layers
        core_layers = [
            (self.MESH_LN, self.mesh, "mesh"),
            (self.SKEL_LN, self.skeleton, "skeleton"),
            (self.GRAPH_LN, self.graph, "graph"),
        ]

        for layer_name, layer_obj, layer_type in core_layers:
            if layer_obj is not None:
                vertex_count = len(layer_obj.vertices)

                if hasattr(layer_obj, "faces"):
                    face_count = (
                        len(layer_obj.faces) if len(layer_obj.faces.shape) > 1 else 0
                    )
                    details = f"vertices: {vertex_count:,} | faces: {face_count:,}"
                elif hasattr(layer_obj, "edges"):
                    edge_count = (
                        len(layer_obj.edges) if len(layer_obj.edges.shape) > 1 else 0
                    )
                    details = f"vertices: {vertex_count:,} | edges: {edge_count:,}"
                else:
                    details = f"vertices: {vertex_count:,}"

                # Add labels info
                if hasattr(layer_obj, "label_names") and len(layer_obj.label_names) > 0:
                    labels_str = ", ".join(layer_obj.label_names)
                    details += f" | labels: [{labels_str}]"
                else:
                    details += " | labels: []"

                html_parts.append(f"""
                <div class="layer-item">
                    <div class="layer-name">{layer_name} ({layer_type})</div>
                    <div class="layer-details">{details}</div>
                </div>
                """)
            else:
                html_parts.append(f"""
                <div class="layer-item not-present">
                    <div class="layer-name">{layer_name} ({layer_type})</div>
                    <div class="layer-details">not present</div>
                </div>
                """)

        # Additional layers
        core_layer_names = {self.MESH_LN, self.SKEL_LN, self.GRAPH_LN}
        for layer in self.layers:
            if layer.name not in core_layer_names:
                vertex_count = len(layer.vertices)

                if hasattr(layer, "faces"):
                    face_count = len(layer.faces) if len(layer.faces.shape) > 1 else 0
                    details = f"vertices: {vertex_count:,} | faces: {face_count:,}"
                elif hasattr(layer, "edges"):
                    edge_count = len(layer.edges) if len(layer.edges.shape) > 1 else 0
                    details = f"vertices: {vertex_count:,} | edges: {edge_count:,}"
                else:
                    details = f"vertices: {vertex_count:,}"

                # Add labels info
                if hasattr(layer, "label_names") and len(layer.label_names) > 0:
                    labels_str = ", ".join(layer.label_names)
                    details += f" | labels: [{labels_str}]"
                else:
                    details += " | labels: []"

                html_parts.append(f"""
                <div class="layer-item">
                    <div class="layer-name">{layer.name} ({layer.layer_type})</div>
                    <div class="layer-details">{details}</div>
                </div>
                """)

        return (
            "\n".join(html_parts)
            if html_parts
            else '<div style="color: #6c757d; font-style: italic;">No layers present</div>'
        )

    def _build_annotations_html(self) -> str:
        """Build HTML for annotations section."""
        if len(self.annotations) == 0:
            return '<div style="color: #6c757d; font-style: italic;">No annotations present</div>'

        html_parts = []
        for annotation in self.annotations:
            vertex_count = len(annotation.vertices)
            details = f"vertices: {vertex_count:,}"

            # Add labels info
            if hasattr(annotation, "label_names") and len(annotation.label_names) > 0:
                labels_str = ", ".join(annotation.label_names)
                details += f" | labels: [{labels_str}]"
            else:
                details += " | labels: []"

            html_parts.append(f"""
            <div class="annotation-item">
                <div class="layer-name">{annotation.name}</div>
                <div class="layer-details">{details}</div>
            </div>
            """)

        return "\n".join(html_parts)
