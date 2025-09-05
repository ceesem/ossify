import contextlib
import copy
import uuid
from abc import ABC, abstractmethod
from numbers import Number
from typing import TYPE_CHECKING, Generator, List, Literal, Optional, Self, Tuple, Union

import fastremap
import numpy as np
import pandas as pd
import trimesh
from scipy import sparse, spatial

from . import graph_functions as gf
from . import utils
from .sync_classes import *

if TYPE_CHECKING:
    from .base import Cell

SKEL_LAYER_NAME = "skeleton"
GRAPH_LAYER_NAME = "graph"
MESH_LAYER_NAME = "mesh"


class EdgeMixin(ABC):
    _csgraph = None
    _csgraph_binary = None

    @property
    def edges(self) -> np.ndarray:
        """
        Get the edges of the layer in dataframe indices.
        """
        return self.layer.edges

    @property
    def edge_df(self) -> pd.DataFrame:
        """
        Get the edges of the layer as a DataFrame.
        """
        return self.layer.edges_df

    @property
    def edges_positional(self) -> np.ndarray:
        """
        Get the edges of the layer in positional indices.
        """
        return self.layer.edges_positional

    def _map_edges_to_index(
        self, edges: np.ndarray, vertex_indices: np.ndarray
    ) -> np.ndarray:
        """Remap positional edges to vertex indices.

        Parameters
        ----------
        edges : np.ndarray
            Edge array with positional indices.
        vertex_indices : np.ndarray
            Array of vertex indices to map to.

        Returns
        -------
        np.ndarray
            Edge array with vertex indices instead of positional indices.
        """
        index_map = {ii: v for ii, v in enumerate(vertex_indices)}
        return fastremap.remap(edges, index_map)

    @property
    def csgraph(self) -> sparse.csr_matrix:
        """
        Get the compressed sparse graph representation of the layer with Euclidean edge weights.
        """
        if self._csgraph is None:
            self._csgraph = utils.build_csgraph(
                self.vertices,
                self.edges_positional,
                euclidean_weight=True,
                directed=True,
            )
        return self._csgraph

    @property
    def csgraph_binary(self) -> sparse.csr_matrix:
        """
        Get the unweighted compressed sparse graph representation of the layer.
        """
        if self._csgraph_binary is None:
            self._csgraph_binary = utils.build_csgraph(
                self.vertices,
                self.edges_positional,
                euclidean_weight=False,
                directed=True,
            )
        return self._csgraph_binary

    @property
    def csgraph_undirected(self) -> sparse.csr_matrix:
        """
        Get the undirected compressed sparse graph representation of the layer with Euclidean edge weights.
        """
        return self.csgraph + self.csgraph.T

    @property
    def csgraph_binary_undirected(self) -> sparse.csr_matrix:
        """
        Get the unweighted and undirected compressed sparse graph representation of the layer.
        """
        return self.csgraph_binary + self.csgraph_binary.T

    def _reset_derived_properties(self) -> None:
        """Reset cached derived properties to force recomputation.

        This method clears cached sparse graph representations to ensure
        they are recomputed when next accessed.
        """
        self._csgraph = None
        self._csgraph_binary = None

    def distance_between(
        self,
        sources: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        as_positional=False,
        limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get the distance between two sets of vertices in the skeleton.

        Parameters
        ----------
        sources : Optional[np.ndarray]
            The source vertices. If None, all vertices are used.
        targets : Optional[np.ndarray]
            The target vertices. If None, all vertices are used.
        as_positional: bool
            Whether the input vertices are positional (i.e., masks or indices).
            Must be the same for sources and targets.
        limit: Optional[float]
            The maximum distance to consider in the graph distance lookup. If None, no limit is applied.
            Distances above this will be set to infinity.

        Returns
        -------
        np.ndarray
            The distance between each source and target vertex, of dimensions len(sources) x len(targets).
        """
        # Sources must be positional for the dijkstra
        sources, as_positional_sources = self._vertices_to_positional(
            sources, as_positional
        )
        targets, as_positional_targets = self._vertices_to_positional(
            targets, as_positional
        )
        if as_positional_sources != as_positional_targets:
            raise ValueError(
                "sources and targets must both be positional or both be indices. Masks are implicitly positional."
            )
        if limit is None:
            limit = np.inf
        return gf.source_target_distances(
            sources=sources,
            targets=targets,
            csgraph=self.csgraph_undirected,
            limit=limit,
        )

    def path_between(
        self,
        source: int,
        target: int,
        as_positional=False,
        as_vertices=False,
    ) -> np.ndarray:
        """
        Get the shortest path between two vertices in the skeleton.

        Parameters
        ----------
        source : int
            The source vertex.
        target : int
            The target vertex.
        as_positional: bool
            Whether the input vertices are positional (i.e., masks or indices).
            Must be the same for sources and targets.
        as_vertices: bool
            Whether to return the path as vertex IDs or 3d positions.

        Returns
        -------
        np.ndarray
            The shortest path between each source and target vertex, indices if as_positional is False, or nx3 array if `as_vertices` is True.
        """
        # Sources must be positional for the dijkstra
        st, _ = self._vertices_to_positional([source, target], as_positional)
        source = st[0]
        target = st[1]
        path_positional = gf.shortest_path(
            source=source,
            target=target,
            csgraph=self.csgraph_binary_undirected,
        )
        if as_positional and not as_vertices:
            return self.vertex_index[path_positional]
        else:
            if as_vertices:
                return self.vertices[path_positional]
            else:
                return path_positional


class FaceMixin(ABC):
    _csgraph = None
    _trimesh = None

    @property
    def faces_positional(self) -> np.ndarray:
        """Return the triangle face indices of the mesh in positional indices."""
        return self.layer.faces

    @property
    def faces(self) -> np.ndarray:
        """Return the triangle face indices of the mesh in raw indices."""
        return self.vertex_index[self.faces_positional]

    @property
    def as_trimesh(self) -> trimesh.Trimesh:
        """Return the mesh as a trimesh.Trimesh object. Note that Trimesh works entirely in positional vertices."""
        if self._trimesh is None:
            self._trimesh = trimesh.Trimesh(
                vertices=self.vertices, faces=self.faces_positional, process=False
            )
        return self._trimesh

    @property
    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple of (vertices, faces_positional) expected by many mesh-processing algorithms."""
        return self.vertices, self.faces_positional

    @property
    def edges(self) -> np.ndarray:
        """Return the edge indices of the mesh in raw indices. Note that for each connected vertex pair u,v, there are both edges (u,v) and (v,u)."""
        return self.vertex_index[self.as_trimesh.edges]

    @property
    def edges_positional(self) -> np.ndarray:
        """Return the edge indices of the mesh in positional indices. Note that for each connected vertex pair u,v, there are both edges (u,v) and (v,u)."""
        return self.as_trimesh.edges

    @property
    def csgraph(self) -> sparse.csr_matrix:
        """Generate a compressed sparse graph representation of the mesh as a network."""
        if self._csgraph is None:
            self._csgraph = utils.build_csgraph(
                self.vertices,
                self.edges_positional,
                euclidean_weight=True,
                directed=False,
            )
        return self._csgraph

    def _map_faces_to_index(
        self, faces: np.ndarray, vertex_indices: np.ndarray
    ) -> np.ndarray:
        """Remap positional faces to vertex indices.

        Parameters
        ----------
        faces : np.ndarray
            Face array with positional indices.
        vertex_indices : np.ndarray
            Array of vertex indices to map to.

        Returns
        -------
        np.ndarray
            Face array with vertex indices instead of positional indices.
        """
        index_map = {ii: v for ii, v in enumerate(vertex_indices)}
        return fastremap.remap(faces, index_map)

    def surface_area(
        self,
        vertices: Optional[np.ndarray] = None,
        as_positional: bool = True,
        inclusive: bool = False,
    ) -> float:
        """Calculate the surface area of the mesh, or a subset of vertices.

        Parameters
        ----------
        vertices : Optional[np.ndarray], optional
            Vertex indices to calculate surface area for. If None, uses entire mesh.
        as_positional : bool, optional
            Whether the input vertices are positional indices or vertex indices. Default True.
        inclusive : bool, optional
            Whether to include faces that are covered by any vertex (True) or only those fully covered (False). Default False.

        Returns
        -------
        float
            The surface area of the mesh or the subset of vertices.
        """
        if vertices is None:
            return self.as_trimesh.area
        else:
            vertices, _ = self._vertices_to_positional(vertices, as_positional)
            mask = np.full(self.n_vertices, False)
            mask[vertices] = True
            if inclusive:
                face_mask = np.any(mask[self.as_trimesh.faces], axis=1)
            else:
                face_mask = np.all(mask[self.as_trimesh.faces], axis=1)
            return self.as_trimesh.area_faces[face_mask].sum()


# General properties for layers with points
class PointMixin(ABC):
    def _setup_properties(
        self,
        name: str,
        morphsync: Optional[PointSync] = None,
        vertices: Union[np.ndarray, pd.DataFrame] = None,
        spatial_columns: Optional[list] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Process basic label data when building the layer.

        Parameters
        ----------
        name : str
            Name of the layer.
        morphsync : Optional[PointSync], optional
            MorphSync object to use. If None, creates a new one.
        vertices : Union[np.ndarray, pd.DataFrame], optional
            Vertex data as array or DataFrame.
        spatial_columns : Optional[list], optional
            List of column names for spatial coordinates.
        labels : Optional[Union[dict, pd.DataFrame]], optional
            Additional label data to attach to vertices.
        vertex_index : Optional[Union[str, np.ndarray]], optional
            Vertex index column name or array.

        Returns
        -------
        Tuple[pd.DataFrame, List[str], List[str]]
            Processed vertices DataFrame, spatial column names, and label column names.
        """
        self._name = name
        self._kdtree = None
        if morphsync is None:
            self._morphsync = MorphSync()
        else:
            self._morphsync = morphsync
        vertices, spatial_columns, label_columns = utils.process_vertices(
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        self._spatial_columns = spatial_columns
        self._label_columns = label_columns
        return vertices, spatial_columns, label_columns

    def _setup_linkage(
        self,
        linkage: Optional[Link] = None,
    ) -> None:
        """Add linkage information to the layer.

        Parameters
        ----------
        linkage : Optional[Link], optional
            Link object containing mapping information between layers.
        """
        if linkage is not None:
            if linkage.source is None:
                linkage.source = self.layer_name
            elif linkage.target is None:
                linkage.target = self.layer_name
            if isinstance(linkage.mapping, str):
                linkage.mapping = (
                    self._morphsync._layers[linkage.source]
                    .nodes[linkage.mapping]
                    .values
                )
            self._process_linkage(linkage)

    def _vertices_to_positional(
        self,
        vertices: Optional[np.ndarray],
        as_positional: bool,
        vertex_index: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Map vertex index to positional indices whether inputs are positional, masks, indices.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            Array of vertex indices, positional indices, or boolean mask. If None, all vertices used.
        as_positional : bool
            Whether input vertices are positional indices (True) or vertex indices (False).
        vertex_index : Optional[np.ndarray], optional
            Custom vertex index array to use for mapping. If None, uses self.vertex_index.

        Returns
        -------
        Tuple[np.ndarray, bool]
            Tuple of (positional_indices, is_positional_flag).
        """
        if vertices is None:
            vertices = np.arange(self.n_vertices)
            as_positional = True
        else:
            vertices = np.array(vertices)
            if np.issubdtype(vertices.dtype, np.bool_):
                if len(vertices) != self.n_vertices:
                    raise ValueError(
                        "If vertices is a boolean array, it must have the same length as the number of vertices."
                    )
                vertices = np.flatnonzero(vertices)
                as_positional = True
            if not as_positional:
                if vertex_index is None:
                    vertex_index_map = self.vertex_index_map
                else:
                    vertex_index_map = {v: i for i, v in enumerate(vertex_index)}
                vertices = fastremap.remap(vertices, vertex_index_map)
            vertices = np.array(vertices)
        return vertices, as_positional

    @property
    def name(self) -> str:
        """Layer name."""
        return self._name

    @property
    def layer(self) -> Facet:
        """Get the morphsync layer associated with the data layer."""
        return self._get_layer(self.layer_name)

    def _get_layer(self, layer_name: str) -> Facet:  # type: ignore
        return self._morphsync._layers[layer_name]

    @property
    def vertices(self) -> np.ndarray:
        """Get the Nx3 vertex positions of the data layer."""
        return self.layer.vertices

    @property
    def vertex_df(self) -> pd.DataFrame:
        """Get the Nx3 vertex positions of the data layer as an indexed DataFrame."""
        return self.layer.vertices_df

    @property
    def vertex_index(self) -> np.ndarray:
        """Get vertex indices as a numpy array."""
        return np.array(self.layer.vertices_index)

    @property
    def vertex_index_map(self) -> dict:
        """Get a dictionary mapping vertex indices to their positional indices."""
        map = {int(v): ii for ii, v in enumerate(self.vertex_index)}
        map[-1] = -1
        return map

    @property
    def nodes(self) -> pd.DataFrame:
        """Get the complete DataFrame of vertices and all associated data, including both spatial columns and labels."""
        return self.layer.nodes

    @property
    def spatial_columns(self) -> list:
        """Get the list of column names associated with the x, y, and z positions."""
        return self._spatial_columns

    @property
    def label_names(self) -> list:
        """Get the list of column names associated with labels (non-spatial columns)."""
        return self._label_columns

    @property
    def labels(self) -> pd.DataFrame:
        """Get the labels DataFrame."""
        return self.nodes[self.label_names]

    @property
    def n_vertices(self) -> int:
        """Get the number of vertices in the data layer."""
        return self.layer.n_vertices

    @property
    def bbox(self) -> np.array:
        """Get the axis-aligned bounding box (min, max) of the data layer's vertices."""
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def kdtree(self) -> spatial.KDTree:
        """Get the KDTree for the data layer's vertices for efficient spatial queries. See scipy.spatial.KDTree for documentation."""
        if self._kdtree is None:
            self._kdtree = spatial.KDTree(self.vertices)
        return self._kdtree

    def get_label(self, key: str) -> np.ndarray:
        """Get a label array from the labels DataFrame.

        Parameters
        ----------
        key : str
            Column name of the label to retrieve.

        Returns
        -------
        np.ndarray
            Array of label values for all vertices.
        """
        return self.labels[key].values

    def add_label(
        self,
        label: Union[list, np.ndarray, dict, pd.DataFrame],
        name: Optional[str] = None,
    ) -> Self:
        """Add a new layer to the DataFrame.

        Parameters
        ----------
        label: Union[list, np.ndarray, dict, pd.Series, pd.DataFrame]
            The label data to add. If an array or list, it should follow the vertex order.
        name: Optional[str]
            The name of the label column (required if label is a list or np.ndarray).

        Returns
        -------
        Self
            The updated DataLayer instance.
        """
        if isinstance(label, list) or isinstance(label, np.ndarray):
            label = pd.DataFrame(label, index=self.vertex_index, columns=[name])
        elif isinstance(label, dict):
            label = pd.DataFrame(label, index=self.vertex_index)
        elif isinstance(label, pd.DataFrame) or isinstance(label, pd.Series):
            label = label.loc[self.vertex_index]
        else:
            raise ValueError("Label must be a list, np.ndarray, dict or pd.DataFrame.")

        if label.shape[0] != self.n_vertices:
            raise ValueError("Label must have the same number of rows as vertices.")
        if np.any(label.columns.isin(self.nodes.columns)):
            raise ValueError('"Label name already exists in the nodes DataFrame.")')

        self._morphsync._layers[self.layer_name].nodes = self.nodes.merge(
            label,
            left_index=True,
            right_index=True,
            how="left",
            validate="1:1",
        )
        self._label_columns += list(label.columns)
        return self

    def _map_range_to_range(self, layer: str, source_index: np.ndarray) -> np.ndarray:
        """Map source indices to all corresponding target indices without 1:1 constraint.

        Takes the dataframe from get_mapping and returns all values in the target layer,
        without respecting a 1:1 mapping between indices.

        Parameters
        ----------
        layer : str
            Target layer name to map to.
        source_index : np.ndarray
            Source indices to map from.

        Returns
        -------
        np.ndarray
            All corresponding target indices for the source indices.
        """
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        return mapping.values

    def _map_index_one_to_one(
        self, layer: str, source_index: np.ndarray, validate: bool = False
    ) -> np.ndarray:
        """Map source indices to single target indices in a one-to-one manner.

        Takes the list of source indices and returns one target index for each source.
        Maintains the order of the source indices.

        Parameters
        ----------
        layer : str
            Target layer name to map to.
        source_index : np.ndarray
            Source indices to map from.
        validate : bool, optional
            Whether to validate for ambiguous mappings, i.e. multiple targets are possible. Default False.

        Returns
        -------
        np.ndarray
            One target index for each source index, maintaining order.
        """
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        if validate:
            if np.any(mapping.index.duplicated()):
                raise ValueError(
                    f"Ambiguous index mapping from {self.layer_name} to {layer}."
                )
        return mapping[~mapping.index.duplicated(keep="first")].loc[source_index].values

    def _map_index_to_list_of_lists(self, layer: str, source_index: np.ndarray) -> dict:
        """Map source indices to lists of all corresponding target indices.

        Takes the list of source indices and returns a list of all target indices for each source.

        Parameters
        ----------
        layer : str
            Target layer name to map to.
        source_index : np.ndarray
            Source indices to map from.

        Returns
        -------
        dict
            Dictionary mapping each source index to a list of all corresponding target indices.
        """
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        mapping_dict = mapping.groupby(by=mapping.index).agg(list).to_dict()
        return {k: np.array(v) for k, v in mapping_dict.items()}

    def map_labels_to_layer(
        self, labels: Union[str, list], layer: str, agg: Union[str, dict]
    ) -> pd.DataFrame:
        """Map labels from one layer to another.

        Parameters
        ----------
        labels: Union[str, list]
            The labels to map from the source layer.
        layer: str
            The target layer to map the labels to.
        agg: Union[str, dict]
            The aggregation method to use when mapping the labels.
            This can take anything pandas `groupby.agg` takes, as well as
            "majority" which will is a majority vote across the mapped indices
            via the stats.mode function.

        Returns
        -------
        pd.DataFrame
            The mapped labels for the target layer.
        """
        if layer == self.layer_name:
            return self.nodes[labels]
        if isinstance(labels, str):
            labels = [labels]
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=self.vertex_index
        )
        mapping_merged = mapping.to_frame().merge(
            self.nodes[labels],
            left_index=True,
            right_index=True,
            how="left",
        )
        if agg == "majority":
            agg = utils.majority_agg()
        return (
            mapping_merged.groupby(layer)
            .agg(agg)
            .loc[self._morphsync._layers[layer].nodes.index]
        )

    def _map_index_to_layer(
        self,
        layer: str,
        source_index: np.ndarray,
        as_positional: bool,
        how: str,
        validate: bool = False,
    ) -> Union[np.ndarray, dict]:
        """Master function for mapping indices from source to target layer with various strategies.

        Parameters
        ----------
        layer : str
            Target layer name to map to.
        source_index : np.ndarray
            Source indices to map from.
        as_positional : bool
            Whether indices are positional (True) or vertex indices (False).
        how : str
            Mapping strategy: 'one_to_one', 'range_to_range', or 'one_to_list'.
        validate : bool, optional
            Whether to validate mapping consistency. Default False.

        Returns
        -------
        Union[np.ndarray, dict]
            Mapped indices. Array for 'one_to_one' and 'range_to_range', dict for 'one_to_list'.
        """
        if layer == self.layer_name:
            return source_index
        if source_index is None:
            if as_positional:
                source_index = np.arange(len(self.vertex_index))
            else:
                source_index = self.vertex_index
        source_index = np.asarray(source_index)
        if as_positional or np.issubdtype(source_index.dtype, np.bool):
            source_index = self.vertex_index[source_index]
        if layer in self._morphsync._layers:
            match how:
                case "one_to_one":
                    mapping = self._map_index_one_to_one(
                        layer=layer, source_index=source_index, validate=validate
                    )
                case "range_to_range":
                    mapping = self._map_range_to_range(
                        layer=layer, source_index=source_index
                    )
                case "one_to_list":
                    mapping = self._map_index_to_list_of_lists(
                        layer=layer, source_index=source_index
                    )
            if as_positional:
                if isinstance(mapping, np.ndarray):
                    mapping = fastremap.remap(
                        mapping,
                        {
                            int(k): ii
                            for ii, k in enumerate(
                                np.array(
                                    self._morphsync._layers[layer].vertices_index.values
                                )
                            )
                        },
                    )
                elif isinstance(mapping, dict):
                    mapping = {
                        self.vertex_index_map[k]: fastremap.remap(
                            v,
                            {
                                int(kk): ii
                                for ii, kk in enumerate(
                                    np.array(
                                        self._morphsync._layers[
                                            layer
                                        ].vertices_index.values
                                    )
                                )
                            },
                        )
                        for k, v in mapping.items()
                    }
            return mapping
        else:
            raise ValueError(f"Layer '{layer}' does not exist.")

    def map_index_to_layer(
        self,
        layer: str,
        source_index: Optional[np.ndarray] = None,
        as_positional: bool = False,
        validate: bool = False,
    ) -> np.ndarray:
        """Map each vertex index from the current layer to a single index in the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source index to map from. If None, all vertices are used. Can also be a boolean array.
        as_positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.
        validate : bool
            Whether to raise an error is the mapping is ambiguous, i.e. it is not clear which target index to use.

        Returns
        -------
        np.ndarray
            The mapped indices in the target layer.
            There will be exactly one target index for each source index, no matter how many viable target indices there are.
            If `as_positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            as_positional=as_positional,
            how="one_to_one",
            validate=validate,
        )

    def map_region_to_layer(
        self,
        layer: str,
        source_index: Optional[np.ndarray] = None,
        as_positional: bool = False,
    ) -> np.ndarray:
        """Map each vertex index from the current layer to the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source indices to map from. If None, all vertices are used. Can also be a boolean array.
        as_positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.

        Returns
        -------
        np.ndarray
            All mapped indices in the target layer.
            Not necessarily the same length as the source indices, because it maps a region to another region.
            If `as_positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            as_positional=as_positional,
            how="range_to_range",
        )

    def map_index_to_layer_region(
        self,
        layer: str,
        source_index: Optional[np.ndarray] = None,
        as_positional: bool = False,
    ) -> dict:
        """Map each vertex index from the current layer to a list of all appropriate vertices in the target layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source indices to map from. If None, all vertices are used. Can also be a boolean array.
        as_positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.

        Returns
        -------
        dict
            A dictionary mapping each source index to a list of all mapped target indices.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            as_positional=as_positional,
            how="one_to_list",
        )

    def map_mask_to_layer(
        self, layer: str, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Map a boolean mask from the current layer to the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        mask : Optional[np.ndarray]
            The boolean mask to map from. If None, all vertices are used.

        Returns
        -------
        np.ndarray
            The mapped boolean mask in the target layer.
            There may be multiple target indices for each source index, depending on the region mapping.
            If `as_positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        mask = np.array(mask)
        if len(mask) != self.n_vertices and np.issubdtype(mask.dtype, np.bool_):
            raise ValueError(
                "Mask must be a boolean array with the same length as the number of vertices."
            )
        mapping = self._map_index_to_layer(
            layer=layer,
            source_index=mask,
            as_positional=True,
            how="range_to_range",
        )
        mask_out = np.full(self._cell._all_objects[layer].n_vertices, False)
        mask_out[mapping] = True
        return mask_out

    def _mask_morphsync(
        self, mask: np.ndarray, as_positional: bool = False
    ) -> "MorphSync":
        """Apply a mask to the underlying morphsync indexing infrastructure.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask or indices to apply.
        as_positional : bool, optional
            Whether mask contains positional indices. Default False.

        Returns
        -------
        MorphSync
            New MorphSync object with mask applied.
        """
        if as_positional:
            bool_mask = np.full(self.n_vertices, False)
            bool_mask[mask] = True
            mask = bool_mask
        else:
            mask = np.array(mask)
            if len(mask) == self.n_vertices and np.issubdtype(mask.dtype, np.bool_):
                mask = mask.astype(bool)
            else:
                mask = self.vertex_index.isin(mask)
        return self._morphsync.apply_mask(
            layer_name=self.layer_name,
            mask=mask,
        )

    def _process_linkage(
        self,
        full_link: Link,
    ) -> None:
        """Process a Link object and add it to the morphsync.

        Parameters
        ----------
        full_link : Link
            Link object containing source, target, and mapping information.
        """
        source_layer = self._get_layer(full_link.source)
        target_layer = self._get_layer(full_link.target)

        if len(full_link.mapping) == source_layer.n_vertices:
            self._morphsync.add_link(
                source=full_link.source,
                target=full_link.target,
                mapping=full_link.mapping_to_index(target_layer.nodes),
            )
        else:
            raise ValueError("Mapping must have the same number of rows as vertices.")

    @classmethod
    @abstractmethod
    def _from_existing(cls, new_morphsync: MorphSync, old_obj: Self) -> Self:
        pass

    def apply_mask(
        self,
        mask: np.ndarray,
        as_positional: bool = False,
        self_only: bool = False,
    ) -> Union[Self, "Cell"]:
        """Apply a mask on the current layer. Returns a new object with the masked morphsync.
        If the object is associated with a CellSync, a new CellSync will be created, otherwise
        a new object of the same class will be returned.

        Properties
        ----------
        mask: np.ndarray
            The mask to apply, either in boolean, vertex index, or positional index form.
        as_positional: bool
            If providing indices, specify if they are positional indices (True) or vertex indices (False).
        self_only: bool
            If True, only apply the mask to the current object and not to any associated CellSync.

        Returns
        -------
        masked_object Union[Self, "CellSync"]
            Either a new object of the same class or a new CellSync will be returned.
        """
        new_morphsync = self._mask_morphsync(mask=mask, as_positional=as_positional)
        if self._cell is None or self_only:
            return self.__class__._from_existing(
                new_morphsync=new_morphsync, old_obj=self
            )
        else:
            return self._cell.__class__._from_existing(
                new_morphsync=new_morphsync, old_obj=self._cell
            )

    @contextlib.contextmanager
    def mask_context(self, mask: np.ndarray) -> Generator[Self, None, None]:
        """Context manager to temporarily apply a mask via the current layer.

        Parameters
        ----------
        mask: np.ndarray
            The mask to apply, either in boolean, vertex index, or positional index form.

        Yields
        ------
        Self
            A new object of the same class with the mask applied.

        Example
        -------
        >>> with cell.skeleton.mask_context(mask) as masked_cell:
        >>>     masked_path_length = masked_cell.mesh.surface_area()
        """
        new_self = self.apply_mask(mask=mask)
        try:
            yield new_self
        finally:
            pass

    def _register_cell(self, mws: "Cell") -> None:
        """Register a Cell object with this layer.

        Parameters
        ----------
        mws : Cell
            Cell object to register with this layer.
        """
        self._cell = mws


class GraphLayer(PointMixin, EdgeMixin):
    layer_name = GRAPH_LAYER_NAME
    layer_type = "graph"

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[Link] = None,
        existing: bool = False,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )

        self._cell = None

        if not existing:
            if vertex_index:
                edges = self._map_edges_to_index(edges, vertices.index)
            self._morphsync.add_graph(
                graph=(vertices, edges),
                name=self.layer_name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)

    @classmethod
    def _from_existing(
        cls,
        new_morphsync: MorphSync,
        old_obj: Self,
    ) -> Self:
        "Generate a new SkeletonSync derived from an existing morphsync and skeleton metadata, no need for new vertices or edges."
        new_obj = cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            edges=old_obj.edges,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            existing=True,
        )
        return new_obj

    def map_annotations_to_label(
        self,
        annotation: str,
        distance_threshold: float,
        agg: Union[str, dict] = "count",
        chunk_size: int = 1000,
        validate: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Aggregate a point annotation to a label on the layer.

        Parameters
        ----------
        annotation : str
            The name of the annotation layer to aggregate.
        distance_threshold : float
            Maximum distance to consider for aggregation.
        agg : Union[str, dict], optional
            Aggregation method. Can be 'count' or dict of aggregation functions. Default 'count'.
        chunk_size : int, optional
            Size of processing chunks for memory efficiency. Default 1000.
        validate : bool, optional
            Whether to validate mapping consistency. Default False.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            Aggregated annotation values. Series for 'count', DataFrame for dict aggregations.
        """
        idx_list, prox_list = gf.build_proximity_lists_chunked(
            self.vertices,
            self.csgraph,
            distance_threshold=distance_threshold,
            chunk_size=chunk_size,
        )
        prox_df = pd.DataFrame(
            {
                "idx": self.vertex_index[idx_list],
                "prox_idx": self.vertex_index[prox_list],
            }
        )
        anno_df = self._morphsync._layers[annotation].nodes
        local_vertex = self._cell.annotations[annotation].map_index_to_layer(
            self.layer_name, validate=validate
        )
        local_vertex_col = f"temp_{uuid.uuid4().hex}"
        anno_df[local_vertex_col] = local_vertex
        prox_df = prox_df.merge(
            anno_df,
            left_on="prox_idx",
            right_on=local_vertex_col,
            how="left",
        )
        anno_df.drop(columns=local_vertex_col, inplace=True)
        if agg == "count":
            count_ser = prox_df.groupby("idx")[local_vertex_col].count()
            count_ser.name = f"{annotation}_count"
            return count_ser
        elif isinstance(agg, dict):
            agg_df = prox_df.groupby("idx").agg(**agg)
            return agg_df
        else:
            raise ValueError(
                f"Unknown aggregation type: {agg}. Must be 'count' or a dict."
            )

    def __repr__(self) -> str:
        return f"GraphLayer(name={self.name}, vertices={self.vertices.shape[0]}, edges={self.edges.shape[0]})"


class SkeletonLayer(GraphLayer):
    layer_name = SKEL_LAYER_NAME
    layer_type = "skeleton"

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        root: Optional[int] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[dict] = None,
        inherited_properties: Optional[dict] = None,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )

        if inherited_properties is None:
            # Add as a morphsync layer
            if vertex_index:
                edges = self._map_edges_to_index(edges, vertices.index)
            self._morphsync.add_graph(
                graph=(vertices, edges),
                name=self.layer_name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)

            # Establish the root and then build the base properties
            self._root = self._infer_root(root)
            self._dag_cache = gf.DAGCache(
                root=self.root_positional
            )  # Cache of properties associated with rooted skeletons
            self._dag_cache.parent_node_array = self._apply_root_to_edges(root)

            self._set_base_properties(
                base_properties={
                    "base_root": self.root_positional,
                    "base_vertex_index": self.vertex_index,
                    "base_parent_array": self.parent_node_array,
                    "base_csgraph": self.csgraph,
                    "base_csgraph_binary": self.csgraph_binary,
                }
            )
        else:
            # Infer all values from inherited properties and/or the existing morphsync.
            self._set_base_properties(base_properties=inherited_properties)

            old_root_pos = inherited_properties.get("base_root")
            old_root_idx = inherited_properties.get("base_vertex_index")[old_root_pos]

            if old_root_idx in self.vertex_index:
                self._root = old_root_idx
            else:
                self._root = None

            self._dag_cache = gf.DAGCache(root=self.root_positional)
            self._dag_cache.parent_node_array = gf.build_parent_node_array(
                self.vertices, self.edges_positional
            )

        self._cell = None

    @classmethod
    def _from_existing(
        cls,
        new_morphsync: MorphSync,
        old_obj: Self,
    ) -> Self:
        "Generate a new SkeletonSync derived from an existing morphsync and skeleton metadata, no need for new vertices or edges."
        new_obj = cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            edges=old_obj.edges,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            inherited_properties=old_obj._base_properties,
        )
        return new_obj

    @property
    def root(self) -> Optional[int]:
        """Get the root node index.

        Returns
        -------
        Optional[int]
            Root node index, or None if no root is set.
        """
        return self._root

    @property
    def root_positional(self) -> Optional[int]:
        """Get the root node positional index.

        Returns
        -------
        Optional[int]
            Root node positional index, or None if no root is set.
        """
        if self._root is None:
            return None
        return np.flatnonzero(self.vertex_index == self.root)[0]

    @property
    def root_location(self) -> Optional[np.ndarray]:
        """Get the spatial coordinates of the root node.

        Returns
        -------
        Optional[np.ndarray]
            3D coordinates of the root node, or None if no root is set.
        """
        if self.root == -1:
            return None
        return self.vertex_df.loc[self.root, self.spatial_columns].values

    @property
    def parent_node_array(self) -> np.ndarray:
        """Get the parent node array for the skeleton, or -1 for a missing parent."""
        if self._dag_cache.parent_node_array is None:
            self._dag_cache.parent_node_array = gf.build_parent_node_array(
                self.vertices, self.edges_positional
            )
        return self._dag_cache.parent_node_array

    @property
    def branch_points(self) -> np.ndarray:
        "List of branch points of the skeleton based on vertex index"
        return self.vertex_index[self.branch_points_positional]

    @property
    def branch_points_positional(self) -> np.ndarray:
        "List of branch points of the skeleton based on positional index"
        if self._dag_cache.branch_points is None:
            self._dag_cache.branch_points = gf.find_branch_points(self.csgraph_binary)
        return self._dag_cache.branch_points

    @property
    def end_points(self) -> np.ndarray:
        "List of end points of the skeleton based on vertex index"
        return self.vertex_index[self.end_points_positional]

    @property
    def end_points_positional(self) -> np.ndarray:
        "List of end points of the skeleton based on positional index"
        if self._dag_cache.end_points is None:
            self._dag_cache.end_points = gf.find_end_points(self.csgraph_binary)
        return self._dag_cache.end_points

    @property
    def end_points_undirected(self) -> np.ndarray:
        "List of end points of the skeleton based on vertex index potentially including root if a leaf node"
        return self.vertex_index[self.end_points_undirected_positional]

    @property
    def end_points_undirected_positional(self) -> np.ndarray:
        "List of end points of the skeleton based on positional index potentially including root if a leaf node"
        return np.flatnonzero(
            self.csgraph_binary_undirected.sum(axis=1) == 1
        )  # Only one neighbor

    @property
    def branch_points_undirected(self) -> np.ndarray:
        "List of branch points of the skeleton based on vertex index potentially including root if a leaf node"
        return self.vertex_index[self.branch_points_undirected_positional]

    @property
    def branch_points_undirected_positional(self) -> np.ndarray:
        "List of branch points of the skeleton based on positional index potentially including root if a leaf node"
        return np.flatnonzero(
            self.csgraph_binary_undirected.sum(axis=1) > 2
        )  # More than 2 neighbors

    @property
    def n_end_points(self) -> int:
        "Number of end points in the skeleton"
        return self.end_points.shape[0]

    @property
    def n_branch_points(self) -> int:
        "Number of branch points in the skeleton"
        return self.branch_points.shape[0]

    @property
    def topo_points(self) -> np.ndarray:
        "All vertices not along a segment: branch points, end points, and root node"

        return self.vertex_index[self.topo_points_positional]

    @property
    def topo_points_positional(self) -> np.ndarray:
        "All vertices not along a segment: branch points, end points, and root node"
        bp_ep_rp = np.concatenate(
            (self.branch_points_positional, self.end_points_positional)
        )
        if self.root is not None:
            bp_ep_rp = np.concatenate((bp_ep_rp, [self.root_positional]))
        return np.unique(bp_ep_rp)

    @property
    def n_topo_points(self) -> int:
        "Number of topological points in the skeleton"
        return self.topo_points.shape[0]

    @property
    def parentless_nodes(self) -> np.ndarray:
        "List of nodes by vertex index that do not have any parents, including any root node."
        return self.vertex_index[self.parentless_nodes_positional]

    @property
    def parentless_nodes_positional(self) -> np.ndarray:
        "List of nodes by positional index that do not have any parents, including any root node."
        return np.flatnonzero(self.parent_node_array == -1)

    @property
    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the vertices and (positional) edges of the graph as a tuple, which is a common input to many functions.
        """
        return self.vertices, self.edges_positional

    def _set_base_properties(self, base_properties: Optional[dict] = None) -> None:
        """Set or update base properties for the skeleton.

        Parameters
        ----------
        base_properties : Optional[dict], optional
            Dictionary of base properties. If None, uses current skeleton properties.
        """
        if not base_properties:
            self._base_properties["base_root"] = self.root
            self._base_properties["base_vertex_index"] = self.vertex_index
            self._base_properties["base_parent_array"] = self.parent_node_array
            self._base_properties["base_csgraph"] = self.csgraph
            self._base_properties["base_csgraph_binary"] = self.csgraph_binary
        else:
            self._base_properties = copy.deepcopy(base_properties)

    @property
    def base_root(self) -> int:
        """Get the base root from original unmasked skeleton.

        Returns
        -------
        int
            Base root index from original skeleton.
        """
        return self._base_properties["base_root"]

    @property
    def base_csgraph(self) -> sparse.csr_matrix:
        """Get the base sparse graph from original unmasked skeleton.

        Returns
        -------
        sparse.csr_matrix
            Base compressed sparse graph with Euclidean edge weights.
        """
        return self._base_properties["base_csgraph"]

    @property
    def base_csgraph_binary(self) -> sparse.csr_matrix:
        """Get the base binary sparse graph from original unmasked skeleton.

        Returns
        -------
        sparse.csr_matrix
            Base compressed sparse graph with binary edge weights.
        """
        return self._base_properties["base_csgraph_binary"]

    @property
    def base_vertex_index(self) -> Union[str, np.ndarray]:
        """Get the base vertex index from original unmasked skeleton.

        Returns
        -------
        Union[str, np.ndarray]
            Base vertex indices from original skeleton.
        """
        return self._base_properties["base_vertex_index"]

    @property
    def base_parent_array(self) -> np.ndarray:
        """Get the base parent array from original unmasked skeleton.

        Returns
        -------
        np.ndarray
            Base parent node array from original skeleton.
        """
        return self._base_properties["base_parent_array"]

    def _reset_derived_properties(self) -> None:
        super()._reset_derived_properties()
        self._dag_cache = gf.DAGCache()

    def _infer_root(self, root: Optional[int]) -> int:
        """Infer the root node from the graph structure or validate provided root.

        Parameters
        ----------
        root : Optional[int]
            Proposed root node index. If None, attempts to infer from graph structure.

        Returns
        -------
        int
            The root node index.

        Raises
        ------
        ValueError
            If no root specified and multiple potential roots found.
        """
        if root is not None:
            return int(root)
        else:
            potential_roots = np.flatnonzero(self.csgraph_binary.sum(axis=1) == 0)
            if len(potential_roots) == 1:
                return int(potential_roots[0])
            else:
                raise ValueError(
                    "No root specified and edges are not consistent with a single root. Please set a valid root."
                )

    def _apply_root_to_edges(
        self, root: Optional[int], apply_to_all_components: bool = False
    ) -> np.ndarray:
        """Reorient edges so that children are always first in the edge list.

        Parameters
        ----------
        root : Optional[int]
            Root node to orient edges from. If None, uses self._root.
        apply_to_all_components : bool, optional
            Whether to reorient all connected components. Default False.

        Returns
        -------
        np.ndarray
            Parent node array after edge reorientation.
        """
        if root is None:
            root = self._root

        _, lbls = sparse.csgraph.connected_components(self.csgraph_binary)

        root_comp = lbls[root]
        if apply_to_all_components:
            comps_to_reroot = np.unique(lbls)
        else:
            comps_to_reroot = np.array([root_comp])

        edges_positional_new = self.edges_positional

        for comp in comps_to_reroot:
            if comp == root_comp:
                comp_root = int(root)
            else:
                comp_root = utils.find_far_points_graph(
                    self.csgraph_binary,
                    start_ind=np.flatnonzero(lbls == comp)[0],
                    multicomponent=True,
                )[0]

            d = sparse.csgraph.dijkstra(
                self.csgraph_binary, directed=False, indices=comp_root
            )

            # Make edges in edge list orient as [child, parent]
            # Where each child only has one parent
            # And the root has no parent. (Thus parent is closer than child)
            edge_slice = np.any(
                np.isin(edges_positional_new, np.flatnonzero(lbls == comp)), axis=1
            )

            edge_subset = edges_positional_new[edge_slice]
            is_ordered = d[edge_subset[:, 1]] < d[edge_subset[:, 0]]
            e1 = np.where(is_ordered, edge_subset[:, 0], edge_subset[:, 1])
            e2 = np.where(is_ordered, edge_subset[:, 1], edge_subset[:, 0])
            edges_positional_new[edge_slice] = np.stack((e1, e2)).T

        # Update facets/edges
        for ii in [0, 1]:
            self._morphsync._layers[self.layer_name].facets[ii] = self.vertex_index[
                edges_positional_new[:, ii]
            ]

    def reroot(self, new_root: int, as_positional=False) -> Self:
        """Reroot to a new index. Important: that this will reset any inherited properties from an unmasked skeleton!

        Parameters
        ----------
        new_root : int
            The new root index to set.
        as_positional: bool, optional
            Whether the new root is a positional index. If False, the new root is treated as a vertex label.

        Returns
        -------
        Self
        """
        self._reset_derived_properties()
        if not as_positional:
            new_root = np.flatnonzero(self.vertex_index == new_root)[0]
        self._root = new_root
        self._dag_cache.root = self._root
        self._apply_root_to_edges(new_root)
        self._set_base_properties(
            base_properties={
                "base_root": new_root,
                "base_vertex_index": self.vertex_index,
                "base_parent_array": self.parent_node_array,
                "base_csgraph": self.csgraph,
            }
        )
        return self

    def distance_to_root(
        self, vertices: Optional[np.ndarray] = None, as_positional=False
    ) -> np.ndarray:
        """
        Get the distance to the root for each vertex in the skeleton, or for a subset of vertices.
        Always uses the original skeleton topology, so that root is inherited from the original root even if it is
        currently masked out. E.g. if you mask an axon only, you can still get distance to the root soma even if
        the soma is not in your current object.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance from the root for. If None, all vertices are used.
        as_positional : bool
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        np.ndarray
            The distance from the root for each vertex.
        """
        # Vertices must be positional for the dijkstra
        vertices, as_positional = self._vertices_to_positional(
            vertices, as_positional, vertex_index=self.base_vertex_index
        )
        if self._dag_cache.distance_to_root is not None:
            dtr = self._dag_cache.distance_to_root
        else:
            dtr = sparse.csgraph.dijkstra(
                self.base_csgraph,
                directed=False,
                indices=self.base_root,
            )
            self._dag_cache.distance_to_root = dtr
        return dtr[vertices]

    def hops_to_root(
        self,
        vertices: Optional[np.ndarray] = None,
        as_positional=False,
    ) -> np.ndarray:
        """Distance to root in number of hops between vertices. Always works on the base graph, whether the root is masked out or not.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance from the root for. If None, all vertices are used.
        as_positional : bool
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        np.ndarray
            The distance from the root for each vertex.
        """
        vertices, _ = self._vertices_to_positional(vertices, as_positional)
        if self._dag_cache.hops_to_root is not None:
            htr = self._dag_cache.hops_to_root
        else:
            htr = sparse.csgraph.dijkstra(
                self.base_csgraph_binary,
                directed=False,
                indices=self.base_root,
            )
            self._dag_cache.hops_to_root = htr
        return htr[vertices]

    def child_vertices(self, vertices=None, as_positional=False) -> dict:
        """Get mapping from vertices to their child nodes.

        Parameters
        ----------
        vertices : Union[np.ndarray, List[int]]
            The vertices to get the child nodes for.
        as_positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        dict
            A dictionary mapping each vertex to its child nodes.
        """
        if isinstance(vertices, Number):
            vertices = [vertices]
        vertices, as_positional = self._vertices_to_positional(vertices, as_positional)
        cinds = gf.build_child_node_dictionary(vertices, self.csgraph_binary)
        if as_positional:
            return cinds
        else:
            new_cinds = {}
            for k, v in cinds.items():
                new_cinds[self.vertex_index[k]] = self.vertex_index[v]
            return new_cinds

    def downstream_vertices(
        self, vertex, inclusive=False, as_positional=False
    ) -> np.ndarray:
        """Get all vertices downstream of a specified vertex

        Parameters
        ----------
        vertex : Union[int, np.ndarray]
            The vertex to get the downstream vertices for.
        inclusive: bool, optional
            Whether to include the specified vertex in the downstream vertices.
        as_positional : bool, optional
            Whether the vertex is a positional index. If False, it is treated as a vertex label.

        Returns
        -------
        np.ndarray
            The downstream vertices, following the same mode as the as_positional parameter.
        """
        vertex, as_positional = self._vertices_to_positional([vertex], as_positional)
        ds_inds = gf.get_subtree_nodes(
            subtree_root=vertex[0], edges=self.edges_positional
        )
        if inclusive:
            ds_inds = np.concatenate(([vertex[0]], ds_inds))
        if as_positional:
            return ds_inds
        else:
            return self.vertex_index[ds_inds]

    def cable_length(
        self, vertices: Optional[Union[list, np.ndarray]] = None, as_positional=False
    ) -> float:
        """The net cable length of the subgraph formed by given vertices. If no vertices are provided, the entire graph is used.

        Parameters
        ----------
        vertices : Optional[Union[list, np.ndarray]]
            The vertices to include in the subgraph. If None, the entire graph is used.
        as_positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        float
            The net cable length of the subgraph.
        """

        vertices, _ = self._vertices_to_positional(vertices, as_positional)
        return float(self.csgraph[np.ix_(vertices, vertices)].sum())

    def lowest_common_ancestor(
        self, u: int, v: int, as_positional=False
    ) -> Optional[int]:
        """Get the lowest common ancestor of two vertices in the skeleton.

        Parameters
        ----------
        u : int
            The first vertex.
        v : int
            The second vertex.
        as_positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        Optional[int]
            The lowest common ancestor of the two vertices, or None if not found.
        """
        uv, as_positional = self._vertices_to_positional([u, v], as_positional)
        u = uv[0]
        v = uv[1]
        return gf.lca(
            u,
            v,
            self.vertices,
            self.edges_positional,
            self._dag_cache,
        )

    @property
    def cover_paths(self) -> List[np.ndarray]:
        """A collection of unbranched paths from each end point toward the root.
        Each path ends when it hits a vertex that's already been visited in a previous path.
        Paths are represented in dataframe indices.
        """
        return [self.vertex_index[path] for path in self.cover_paths_positional]

    @property
    def cover_paths_positional(self) -> List[np.ndarray]:
        """A collection of unbranched paths from each end point toward the root.
        Each path ends when it hits a vertex that's already been visited in a previous path.
        Paths are represented in positional indices.
        """
        if self._dag_cache.cover_paths is None:
            self._dag_cache.cover_paths = gf.build_cover_paths(
                self.end_points_positional,
                self.parent_node_array,
                self.distance_to_root(as_positional=True),
                self._dag_cache,
            )
        return self._dag_cache.cover_paths

    def cover_paths_specific(
        self, sources: Union[np.ndarray, list], as_positional: bool = False
    ) -> List[np.ndarray]:
        """Get cover paths starting from specific source vertices.

        Parameters
        ----------
        sources : Union[np.ndarray, list]
            The source vertices to start the cover paths from.
        as_positional : bool, optional
            Whether the sources are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        list
            A list of cover paths, each path is a list of vertex indices, ordered as the typical `cover_paths` method.
        """
        sources, as_positional = self._vertices_to_positional(sources, as_positional)
        cps = gf.compute_cover_paths(
            sources,
            self.parent_node_array,
            self.distance_to_root(as_positional=True),
        )
        if not as_positional:
            return [self.vertex_index[path] for path in cps]
        return cps

    @property
    def segments_positional(self) -> List[np.ndarray]:
        """
        Get the segments of the layer, a list of arrays where each array represents an unbranched span from end point or branch point to the upstring branch point or root (non-inclusive).
        Segments are presented in positional indices.

        Returns
        -------
        List[np.ndarray]
            List of segment arrays in positional indices.
        """
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_vertices(as_positional=True),
                self.hops_to_root(as_positional=True),
            )
        return self._dag_cache.segments

    @property
    def segments(self) -> List[np.ndarray]:
        """
        Get the segments of the layer, a list of arrays where each array represents an unbranched span from end point or branch point to the upstring branch point or root (non-inclusive).
        Segments are presented in dataframe indices.
        """
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_vertices(as_positional=True),
                self.hops_to_root(as_positional=True),
            )
        return [self.vertex_index[seg] for seg in self._dag_cache.segments]

    @property
    def segments_plus_positional(self) -> List[np.ndarray]:
        """Segments plus their parent node in positional indices.

        Returns
        -------
        List[np.ndarray]
            List of segment arrays including parent nodes in positional indices.
        """
        segs = self.segments_positional
        return [
            np.concatenate((seg, [self.parent_node_array[seg[-1]]])) for seg in segs
        ]

    @property
    def segments_plus(self) -> List[np.ndarray]:
        """Segments plus their parent node in dataframe indices.

        Returns
        -------
        List[np.ndarray]
            List of segment arrays including parent nodes in dataframe indices.
        """
        return [self.vertices[seg] for seg in self.segments_plus_positional]

    @property
    def segment_map(self) -> np.ndarray:
        """Get the mapping from each vertex to its segment index"""
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_vertices(as_positional=True),
                self.hops_to_root(as_positional=True),
            )
        return self._dag_cache.segment_map

    def expand_to_segment(
        self, vertices: Union[np.ndarray, List[int]], as_positional: bool = False
    ) -> List[np.ndarray]:
        """For each vertex in vertices, get the corresponding segment.

        Parameters
        ----------
        vertices : Union[np.ndarray, List[int]]
            Vertices to expand to their segments.
        as_positional : bool, optional
            Whether vertices are positional indices. Default False.

        Returns
        -------
        List[np.ndarray]
            List of segments corresponding to input vertices.
        """
        vertices, as_positional = self._vertices_to_positional(vertices, as_positional)
        segment_ids = self.segment_map[vertices]

        if as_positional:
            return [self.segments_positional[ii] for ii in segment_ids]
        else:
            return [self.segments[ii] for ii in segment_ids]

    @property
    def half_edge_length(self) -> np.ndarray:
        """Get the sum length of half-edges from a vertices to all parents and children.

        Returns
        -------
        np.ndarray
            Array of half-edge lengths for each vertex.
        """
        return np.array(self.csgraph_undirected.sum(axis=0)).flatten() / 2

    def map_annotations_to_label(
        self,
        annotation: str,
        distance_threshold: float,
        agg: Union[Literal["count", "density"], dict] = "count",
        chunk_size: int = 1000,
        validate: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Aggregates a point annotation to a label on the layer based on a maximum proximity.

        Parameters
        ----------
        annotation : str
            The name of the annotation to project.
        distance_threshold : float
            The maximum distance to consider for projecting annotations.
        agg : Union[Literal["count", "density"], dict], optional
            The aggregation method to use. Can be "count", or a dict specifying custom aggregations
            on the annotations properties as per the `groupby.agg` method.
            * "count" returns how many annotations are within the given radius.
            * "density" returns the count of annotations divided by the subgraph path length measured in half-edge-lengths per vertex.
            * To make a new label called "aggregate_label" that is the median "size" of a point annotation,
            it would be {"aggregate_label": ('size', 'median')}. Multiple labels can be specified at the same time in this manner.
        chunk_size : int, optional
            The size of chunks to process at a time, which limits memory consumption. Defaults to 1000.

        Returns
        -------
        pd.Series or pd.DataFrame
            A series (with 'count' or 'density') or dataframe (with dictionary agg) containing the projected annotation values for each vertex.
        """
        if agg == "density":
            agg_temp = "count"
        else:
            agg_temp = agg
        result = super().map_annotations_to_label(
            annotation,
            distance_threshold,
            agg=agg_temp,
            chunk_size=chunk_size,
            validate=validate,
        )
        if agg == "density":
            count_len_df = pd.concat(
                (
                    pd.Series(
                        data=self.half_edge_length,
                        index=self.vertex_index,
                        name="net_length",
                    ),
                    result,
                ),
                axis=1,
            )
            return pd.Series(
                data=count_len_df[result.name] / count_len_df["net_length"],
                index=count_len_df.index,
                name=f"{annotation}_density",
            )
        else:
            return result

    def __repr__(self) -> str:
        return f"SkeletonLayer(name={self.name}, vertices={self.vertices.shape[0]}, edges={self.edges.shape[0]})"


class PointCloudLayer(PointMixin):
    layer_type = "points"

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[dict] = None,
        existing: bool = False,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        if not existing:
            self._morphsync.add_points(
                points=vertices,
                name=self._name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)
        self._cell = None

    @classmethod
    def _from_existing(cls, new_morphsync, old_obj) -> Self:
        return cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            existing=True,
        )

    @property
    def layer_name(self) -> str:
        """
        Get the name of the layer.
        """
        return self._name

    def __repr__(self) -> str:
        return f"PointCloudLayer(name={self.name}, vertices={self.vertices.shape[0]})"

    def distance_to_root(
        self, vertices: Optional[np.ndarray] = None, as_positional: bool = False
    ) -> np.ndarray:
        """
        Get the distance to the root for each vertex in the point cloud along the skeleton, or for a subset of vertices.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance to the root for. If None, all vertices are used.
        as_positional : bool, optional
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.
            By default False.

        Returns
        -------
        np.ndarray
            The distance to the root for each vertex.
        """
        if self._cell is None:
            raise ValueError("PointCloud is not attached to a Cell object.")
        if self._cell.skeleton is None:
            raise ValueError("Cell does not have a Skeleton object.")

        if vertices is None:
            vertices = self.vertex_index
        skel_idx = self.map_index_to_layer(
            layer=SKEL_LAYER_NAME, source_index=vertices, as_positional=as_positional
        )
        return self._cell.skeleton.distance_to_root(
            vertices=skel_idx, as_positional=as_positional
        )

    def distance_between(
        self,
        vertices: Optional[np.ndarray] = None,
        as_positional: bool = False,
        via: Literal["skeleton", "graph", "mesh"] = "skeleton",
        limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get the distance between each pair of vertices in the point cloud along the skeleton.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance between. If None, all vertices are used.
        as_positional : bool, optional
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.
            By default False.
        via: Literal["skeleton", "graph", "mesh"], optional
            The method to use for calculating distances. Can be "skeleton", "graph", or "mesh". Default is "skeleton".
        limit: Optional[float], optional
            The maximum distance to consider when calculating distances. If None, no limit is applied.

        Returns
        -------
        np.ndarray
            The distance between each pair of vertices.
        """
        if self._cell is None:
            raise ValueError("PointCloud is not attached to a Cell object.")
        if via not in self._morphsync._layers:
            raise ValueError(f"Cell does not have a {via.capitalize()} object.")

        vertices, as_positional = self._vertices_to_positional(vertices, as_positional)

        target_idx = self.map_index_to_layer(
            layer=via, source_index=vertices, as_positional=as_positional
        )
        return self._cell.layers[via].distance_between(
            sources=target_idx,
            targets=target_idx,
            as_positional=as_positional,
            limit=limit,
        )

    def filter(
        self,
        mask: np.ndarray,
        layer: str,
    ) -> pd.DataFrame:
        """Filter point cloud by a mask on a specific layer.

        Parameters
        ----------
        mask: np.ndarray
            The mask to filter by. Either an explicit mask array or a boolean mask.
        layer: str
            The layer that the mask is associated with.

        Returns
        -------
        pd.DataFrame
            The dataframe filtered by the mask.
        """
        if self._cell is None:
            raise ValueError("PointCloud is not attached to a Cell object.")
        source_layer = self._cell.layers[layer]
        target_mask = source_layer.map_mask_to_layer(mask=mask, layer=self.layer_name)
        return self.nodes[target_mask]


class MeshLayer(FaceMixin, PointMixin):
    layer_name = MESH_LAYER_NAME
    layer_type = "mesh"

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        faces: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[Link] = None,
        existing: bool = False,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )

        self._cell = None

        if not existing:
            if vertex_index:
                faces = self._map_faces_to_index(faces, vertices.index)
            self._morphsync.add_mesh(
                mesh=(vertices, faces),
                name=self.layer_name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)

    @classmethod
    def _from_existing(
        cls,
        new_morphsync: MorphSync,
        old_obj: Self,
    ) -> Self:
        "Generate a new SkeletonSync derived from an existing morphsync and skeleton metadata, no need for new vertices or edges."
        new_obj = cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            faces=old_obj.faces,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            existing=True,
        )
        return new_obj

    def __repr__(self) -> str:
        return f"MeshLayer(name={self.name}, vertices={self.vertices.shape[0]}, faces={self.faces.shape[0]})"
