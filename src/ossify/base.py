import contextlib
import copy
from typing import Any, Optional, Self, Union

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
]


class LayerManager:
    def __init__(
        self,
        managed_layers: dict,
    ):
        self._managed_layers = managed_layers

    @property
    def names(self) -> list:
        """Return a list of managed layer names."""
        return list(self._managed_layers.keys())

    def add(self, layer: Union[SkeletonLayer, GraphLayer]) -> None:
        self._managed_layers[layer.name] = layer

    def get(self, name: str) -> Optional[Union[SkeletonLayer, GraphLayer]]:
        return self._managed_layers.get(name)

    def __getitem__(self, name: str) -> Optional[Union[SkeletonLayer, GraphLayer]]:
        return self._managed_layers.get(name)

    def __getattr__(self, name: str) -> Optional[Union[SkeletonLayer, GraphLayer]]:
        if name in self._managed_layers:
            return self._managed_layers[name]
        else:
            raise AttributeError(f'Layer "{name}" does not exist.')

    def __dir__(self):
        return super().__dir__() + list(self._managed_layers.keys())

    def __iter__(self):
        """Iterate over managed layers in order."""
        return iter(self._managed_layers.values())

    def __len__(self):
        return len(self._managed_layers)

    def __repr__(self) -> str:
        return str(self.names)


class AnnotationManager:
    def __init__(
        self,
        managed_layers: list,
        annotation_layers: Optional[list] = None,
    ):
        self._annotations = {}
        self._managed_layers = managed_layers
        if annotation_layers is not None:
            for layer in annotation_layers:
                if issubclass(type(layer), PointCloudLayer):
                    self.add(layer)
                else:
                    raise ValueError("Annotation layers must be point clouds.")

    def add(self, layer: PointCloudLayer) -> None:
        self._annotations[layer.name] = layer

    def get(self, name: str, default: Any = None) -> PointCloudLayer:
        return self._annotations.get(name, default)

    def __getitem__(self, name: str) -> PointCloudLayer:
        return self._annotations[name]

    def __getattr__(self, name: str) -> PointCloudLayer:
        if name in self._annotations:
            return self._annotations[name]
        else:
            raise AttributeError(f'Annotation "{name}" does not exist.')

    def __len__(self):
        return len(self._managed_layers)

    def __dir__(self):
        return super().__dir__() + list(self._annotations.keys())

    @property
    def names(self) -> list:
        """Return a list of annotation names."""
        return list(self._annotations.keys())

    def __contains__(self, name: str) -> bool:
        """Check if an annotation exists by name."""
        return name in self._annotations

    def __len__(self) -> int:
        """Return the number of annotations."""
        return len(self._annotations)

    def __iter__(self):
        """Iterate over annotations in order."""
        return iter(self._annotations.values())

    def __repr__(self) -> str:
        return f"AnnotationManager(annotations={list(self._annotations.keys())})"


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
        self._layers = LayerManager(self._managed_layers)
        self._annotations = AnnotationManager(
            managed_layers=self._managed_layers,
            annotation_layers=annotation_layers,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def layers(self) -> dict:
        return self._layers

    @property
    def layer_df(self) -> pd.DataFrame:
        return self._morphsync.layers

    def _get_layer(self, layer_name: str):
        return self._managed_layers.get(layer_name)

    @property
    def skeleton(self) -> SkeletonLayer:
        if self.SKEL_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.SKEL_LN]

    @property
    def graph(self) -> GraphLayer:
        if self.GRAPH_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.GRAPH_LN]

    @property
    def mesh(self) -> MeshLayer:
        if self.MESH_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.MESH_LN]

    @property
    def annotations(self) -> AnnotationManager:
        return self._annotations

    @property
    def _all_objects(self) -> dict:
        return {**self._managed_layers, **self._annotations._annotations}

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
        self._managed_layers[name] = layer
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
        if self.mesh is not None:
            raise ValueError('"Mesh already exists!')

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)
        if isinstance(vertices, MeshLayer):
            self.add_layer(vertices, self.MESH_LN)
        else:
            self._managed_layers[self.MESH_LN] = MeshLayer(
                name=self.MESH_LN,
                vertices=vertices,
                faces=faces,
                labels=labels,
                morphsync=self._morphsync,
                spatial_columns=spatial_columns,
                linkage=linkage,
                vertex_index=vertex_index,
            )
            self._managed_layers[self.MESH_LN]._register_cell(self)
        return self

        pass

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
            self.add_layer(vertices, self.SKEL_LN)
        else:
            self._managed_layers[self.SKEL_LN] = SkeletonLayer(
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
            self.add_layer(vertices, self.GRAPH_LN)
        else:
            self._managed_layers[self.GRAPH_LN] = GraphLayer(
                name=self.GRAPH_LN,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphsync=self._morphsync,
                spatial_columns=spatial_columns,
                linkage=linkage,
                vertex_index=vertex_index,
            )
            self._managed_layers[self.GRAPH_LN]._register_cell(self)
        return self

    def add_point_annotations(
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
        Add point annotations to the MeshWork object.

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

        Returns
        -------
        Self
        """

        if name in self._managed_layers:
            raise ValueError(f"Layer '{name}' already exists.")

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if isinstance(vertices, PointCloudLayer):
            anno = PointCloudLayer(
                name=name,
                vertices=vertices.vertices,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
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
        self._annotations.add(anno)
        return self

    def apply_mask(
        self, layer: str, mask: np.ndarray, positional: bool = False
    ) -> Self:
        """Mask the"""
        new_morphsync = self.layers[layer]._mask_morphsync(mask, positional=positional)
        return self.__class__._from_existing(new_morphsync, self)

    def __repr__(self) -> str:
        layers = self.layers.names
        annos = self.annotations.names
        return f"Cell(name={self.name}, layers={sorted(layers)}, annotations={annos})"

    @classmethod
    def _from_existing(cls, new_morphsync, old_obj) -> Self:
        new_obj = cls(
            name=old_obj.name,
            morphsync=new_morphsync,
            meta=old_obj.meta,
        )
        for old_layer in old_obj.layers:
            new_layer = old_layer.__class__._from_existing(new_morphsync, old_layer)
            new_layer._register_cell(new_obj)
            new_obj._layers.add(new_layer)
        for old_anno in old_obj.annotations:
            new_anno = old_anno.__class__._from_existing(new_morphsync, old_anno)
            new_anno._register_cell(new_obj)
            new_obj._annotations.add(new_anno)
        return new_obj

    @property
    def labels(self) -> pd.DataFrame:
        """Return a DataFrame of all label columns across all layers."""
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
    ):
        """Create a masked version of the MeshWork object in a context state.

        Parameters
        ----------
        mask : array or None,
            A boolean array with the same number of elements as mesh vertices. True elements are kept, False are masked out. If None, resets the mask entirely.
        """
        nrn_out = self.apply_mask(layer, mask)
        try:
            yield nrn_out
        finally:
            pass
