import contextlib
import copy
from functools import partial
from typing import Any, Callable, Generator, Optional, Self, Union

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
        self._labels = {}
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
