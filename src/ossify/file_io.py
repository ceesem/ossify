import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Optional, Union
from urllib import parse

import cloudfiles
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
from numpy import savez_compressed
from scipy.sparse import load_npz, save_npz

from ossify import utils

from .base import Cell
from .data_layers import GraphLayer, Link, MeshLayer, PointCloudLayer, SkeletonLayer

if TYPE_CHECKING:
    import scipy

__all__ = ["load_cell", "save_cell", "CellFiles", "import_legacy_meshwork"]

PUTABLE_SCHEMES = ["s3", "gs", "file", "mem"]
METADATA_FILENAME = "metadata.json"
OSSIFY_EXTENSION = "osy"


def load_cell(source: Union[str, BinaryIO]) -> Cell:
    """Load a neuronal object from a file path or file object.

    Parameters
    ----------
    source : Union[str, BinaryIO]
        The path to the file/cloudpath or an open binary file object.

    Returns
    -------
    Cell
        The loaded Cell object.

    """
    if isinstance(source, str):
        return _load_from_path(source)
    else:
        return _load_from_file_object(source)


def _load_from_path(path: str) -> Cell:
    """Load from a file path using cloudfiles."""
    filename = path.split("/")[-1]
    basepath = "/".join(path.split("/")[:-1])
    return CellFiles(basepath).load(filename)


def _load_from_file_object(file_obj: BinaryIO) -> Cell:
    """Load from an open binary file object."""
    file_bytes = file_obj.read()
    return _import_neuron_from_bytes(file_bytes)


def save_cell(
    cell: Cell,
    file: Union[str, BinaryIO, None] = None,
    allow_overwrite: bool = False,
) -> None:
    """Save Cell to a file path or file object.

    Parameters
    ----------
    cell : Cell
        The Cell object to save.
    file: Union[str, BinaryIO, None]
        File path, open binary file object, or None (uses "<cell.name>.osy" as a string).
    allow_overwrite : bool
        Whether to allow overwriting existing files.
    """
    if isinstance(file, str) or file is None:
        _save_to_path(cell, file, allow_overwrite)
    else:
        _save_to_file_object(cell, file)


def _save_to_path(cell: Cell, path: Optional[str], allow_overwrite: bool) -> None:
    """Save to a file path using cloudfiles."""
    if path:
        basepath = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
    else:
        basepath = "."
        name = f"{cell.name}.{OSSIFY_EXTENSION}"
    CellFiles(basepath).save(cell, name, allow_overwrite=allow_overwrite)


def _save_to_file_object(cell: Cell, file_obj: BinaryIO) -> None:
    """Save to an open binary file object."""
    tar_bytes = _export_neuron_to_bytes(cell)
    file_obj.write(tar_bytes)


def _export_neuron_to_bytes(cell: Cell) -> bytes:
    """Export Cell to bytes data."""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as tf:
        _export_neuron_to_tar(cell, tf)
    return b.getvalue()


def _export_neuron_to_tar(cell: Cell, tf: tarfile.TarFile) -> None:
    """Export Cell to tar file."""
    # export metadata
    add_file_to_tar(
        name=METADATA_FILENAME,
        data=extract_metadata(cell),
        tf=tf,
    )

    # Export all layers
    for l in cell.layers:
        match l.layer_type:
            case "skeleton":
                export_skeleton_layer(l, tf)
            case "graph":
                export_graph_layer(l, tf)
            case "mesh":
                export_mesh_layer(l, tf)
            case "points":
                export_point_cloud_layer(l, tf, as_annotation=False)

    # Export all annotations
    for anno in cell.annotations:
        export_point_cloud_layer(anno, tf, as_annotation=True)

    # Export linkage
    export_linkage(cell, tf)


def _import_neuron_from_bytes(file: bytes) -> Cell:
    """Import Cell from bytes data."""
    with tarfile.open(fileobj=io.BytesIO(file), mode="r") as tf:
        files = {f.name: f for f in tf.getmembers()}
        metadata = load_dict(files[METADATA_FILENAME], tf)
        _validate_archive_structure(metadata, files)

        cell = Cell(name=metadata["name"], meta=metadata["meta"])
        cell = _process_layers(metadata, files, tf, cell)
        cell = _process_annotations(metadata, files, tf, cell)
        cell = _process_linkage(metadata, tf, cell)
    return cell


def _validate_archive_structure(metadata: dict, files: dict):
    """Validate that all expected files exist in the archive."""
    missing_files = []

    # Check required files exist
    for layer_name, layer_info in metadata["structure"]["layers"].items():
        required_files = [f"layers/{layer_name}/meta.json"]

        # Add type-specific required files
        if layer_info["type"] in ["skeleton", "graph"]:
            required_files.extend(
                [
                    f"layers/{layer_name}/nodes.feather",
                    f"layers/{layer_name}/edges.npz",
                ]
            )
        elif layer_info["type"] == "points":
            required_files.append(f"layers/{layer_name}/nodes.feather")

        missing_files.extend([f for f in required_files if f not in files])

    # Check annotations
    for anno_name in metadata["structure"]["annotations"]:
        required_files = [
            f"annotations/{anno_name}/meta.json",
            f"annotations/{anno_name}/nodes.feather",
        ]
        missing_files.extend([f for f in required_files if f not in files])

    # Check linkage references valid layers
    if "linkage" in metadata:
        all_layer_names = set(metadata["structure"]["layers"].keys()) | set(
            metadata["structure"]["annotations"].keys()
        )
        for linkage_pair in metadata["linkage"]:
            for layer_name in linkage_pair:
                if layer_name not in all_layer_names:
                    raise ValueError(f"Linkage references unknown layer: {layer_name}")

    if missing_files:
        raise ValueError(f"Missing required files: {missing_files}")


def _process_layers(metadata, files, tf, cell) -> Cell:
    """Process layer data from tar file."""
    for layer_name, layer_info in metadata["structure"]["layers"].items():
        match layer_info["type"]:
            case "skeleton":
                cell = build_skeleton(
                    parse_skeleton_files(layer_name, files, tf), cell=cell
                )
            case "graph":
                cell = build_graph(parse_graph_files(layer_name, files, tf), cell=cell)
            case "mesh":
                cell = build_mesh(parse_mesh_files(layer_name, files, tf), cell=cell)
            case "points":
                cell = build_point_cloud(
                    parse_point_cloud_files(layer_name, files, tf, as_annotation=False),
                    cell=cell,
                    as_annotation=False,
                )
    return cell


def _process_annotations(metadata, files, tf, cell) -> Cell:
    """Process annotation data from tar file."""
    for anno_name, anno_info in metadata["structure"]["annotations"].items():
        cell = build_point_cloud(
            parse_point_cloud_files(anno_name, files, tf, as_annotation=True),
            cell=cell,
            as_annotation=True,
        )
    return cell


def _process_linkage(metadata, tf, cell) -> Cell:
    """Process linkage data from tar file."""
    linkages = metadata["linkage"]
    for linkage_pair in linkages:
        cell = build_linkage(linkage_pair, tf, cell=cell)
    return cell


class CellFiles:
    def __init__(self, path, use_https: bool = True):
        self.path = path
        if parse.urlparse(path).scheme in ["s3", "gs", "https", "http"]:
            self.cf = cloudfiles.CloudFiles(path, use_https=use_https)
            self._remote = True
        elif parse.urlparse(path).scheme == "mem":
            self.cf = cloudfiles.CloudFiles(path)
            self._remote = False
        else:
            self.cf = cloudfiles.CloudFiles(
                "file://" + str((Path(path).expanduser().absolute()))
            )
            self._remote = False
        self._saveable = parse.urlparse(self.cf.cloudpath).scheme in PUTABLE_SCHEMES

    @property
    def saveable(self):
        return self._saveable

    @property
    def remote(self):
        return self._remote

    def save(
        self,
        cell: Cell,
        filename: Optional[str] = None,
        allow_overwrite: bool = False,
    ):
        if not self._saveable:
            raise ValueError(
                f"Path {self.path} is not writable. Please provide a path with one of the following schemes: {PUTABLE_SCHEMES}"
            )
        if filename is None:
            filename = f"{cell.name}.{OSSIFY_EXTENSION}"
        if not allow_overwrite:
            # Split up to avoid network
            if self.cf.exists(filename):
                raise FileExistsError(
                    f"{filename} already exists in path {self.cf.cloudpath}."
                )
        tar_bytes = _export_neuron_to_bytes(cell)
        self.cf.put(filename, tar_bytes)

    def load(self, filename: str) -> Cell:
        f = self.cf.get(filename, raw=True)
        if not f:
            raise FileNotFoundError(
                f"{filename} not found in path {self.cf.cloudpath}."
            )
        return self._import_neuron(f)

    def _import_neuron(self, file: bytes) -> Cell:
        return _import_neuron_from_bytes(file)


def load_dict(tinfo, tf) -> dict:
    return orjson.loads(tf.extractfile(tinfo).read())


def load_dataframe(tinfo, tf) -> pd.DataFrame:
    df_buf = pa.BufferReader(tf.extractfile(tinfo).read())
    return pd.read_feather(df_buf)


def load_array(tinfo, tf) -> np.ndarray:
    buf = io.BytesIO(tf.extractfile(tinfo).read())
    return np.load(buf)["data"]


def load_sparse_matrix(tinfo, tf) -> "scipy.sparse.csgraph.csr_matrix":
    buf = io.BytesIO(tf.extractfile(tinfo).read())
    return load_npz(buf)


def export_linkage(cell, tf) -> None:
    datapath = f"linkage"
    unique_linkage = np.unique(
        [sorted(b) for b in list(cell._morphsync.links.keys())], axis=0
    ).tolist()
    for linkage_pair in unique_linkage:
        add_file_to_tar(
            name=f"{datapath}/{linkage_pair[0]}/{linkage_pair[1]}/linkage.feather",
            data=bytesio_feather(cell._morphsync.links[tuple(linkage_pair)]),
            tf=tf,
        )


def export_skeleton_layer(
    layer,
    tf,
) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {
                "spatial_columns": layer.spatial_columns,
                "name": layer.name,
                "root": layer.root,
            }
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/edges.npz",
        data=bytesio_array(layer.edges),
        tf=tf,
    )
    save_skeleton_base_properties(
        datapath,
        layer._base_properties,
        tf,
    )


def export_graph_layer(
    layer,
    tf,
) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/edges.npz",
        data=bytesio_array(layer.edges),
        tf=tf,
    )


def export_point_cloud_layer(
    layer,
    tf,
    as_annotation: bool = True,
) -> None:
    if as_annotation:
        datapath = f"annotations/{layer.name}"
    else:
        datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )


def export_mesh_layer(layer, tf) -> None:
    datapath = f"layers/{layer.name}"
    add_file_to_tar(
        name=f"{datapath}/meta.json",
        data=dict_to_bytesio_json(
            {"spatial_columns": layer.spatial_columns, "name": layer.name}
        ),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/nodes.feather",
        data=bytesio_feather(layer.nodes),
        tf=tf,
    )
    add_file_to_tar(
        name=f"{datapath}/faces.npz",
        data=bytesio_array(layer.faces),
        tf=tf,
    )


def extract_metadata(cell) -> bytes:
    layer_structure = {l.name: {"type": l.layer_type} for l in cell.layers}
    annotation_structure = {a.name: {"type": a.layer_type} for a in cell.annotations}
    all_links = [sorted(b) for b in list(cell._morphsync.links.keys())]
    linkages = np.unique([sorted(b) for b in all_links], axis=0).tolist()
    metadata = {
        "name": cell.name,
        "meta": cell.meta,
        "file_version": 1.0,
        "structure": {"layers": layer_structure, "annotations": annotation_structure},
        "linkage": linkages,
    }
    metadata_bytes = orjson.dumps(
        metadata,
        option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
    )
    return metadata_bytes


def bytesio_feather(df, compression="zstd") -> bytes:
    buf = io.BytesIO()
    df.to_feather(buf, compression=compression)
    return buf.getvalue()


def bytesio_sparse_matrix(sparse_matrix) -> bytes:
    buf = io.BytesIO()
    save_npz(buf, sparse_matrix)
    return buf.getvalue()


def bytesio_array(data) -> bytes:
    buf = io.BytesIO()
    savez_compressed(buf, data=data)
    return buf.getvalue()


def dict_to_bytesio_json(d) -> bytes:
    return orjson.dumps(
        d,
        option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
    )


def add_file_to_tar(name, data, tf) -> bytes:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def save_skeleton_base_properties(
    path, base_properties, tf, matrix_properties=["base_csgraph", "base_csgraph_binary"]
) -> None:
    datapath = f"{path}/base_properties"

    add_file_to_tar(
        name=f"{datapath}/properties.json",
        data=dict_to_bytesio_json(
            {k: v for k, v in base_properties.items() if k not in matrix_properties}
        ),
        tf=tf,
    )
    for v in matrix_properties:
        add_file_to_tar(
            name=f"{datapath}/properties_{v}.npz",
            data=bytesio_sparse_matrix(base_properties[v]),
            tf=tf,
        )


def parse_skeleton_files(layer_name, files, tf) -> dict:
    prefix = f"layers/{layer_name}"
    skeleton_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    skeleton_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    skeleton_parts["nodes"] = load_dataframe(files[fn], tf)
                case ("edges.npz", layer_name):
                    skeleton_parts["edges"] = load_array(files[fn], tf)
                case ("properties.json", "base_properties"):
                    skeleton_parts["base_properties"] = load_dict(files[fn], tf)
                case ("properties_base_csgraph.npz", "base_properties"):
                    skeleton_parts["base_csgraph"] = load_sparse_matrix(files[fn], tf)
                case ("properties_base_csgraph_binary.npz", "base_properties"):
                    skeleton_parts["base_csgraph_binary"] = load_sparse_matrix(
                        files[fn], tf
                    )
    return skeleton_parts


def build_skeleton(
    skeleton_parts: dict, cell: "Cell" = None
) -> Union[SkeletonLayer, Cell]:
    inherited_properties = {}
    inherited_properties.update(skeleton_parts["base_properties"])
    inherited_properties["base_csgraph"] = skeleton_parts.get("base_csgraph")
    inherited_properties["base_csgraph_binary"] = skeleton_parts.get(
        "base_csgraph_binary"
    )
    if cell is None:
        skel_sync = SkeletonLayer(
            name=skeleton_parts["meta"]["name"],
            vertices=skeleton_parts["nodes"],
            edges=skeleton_parts["edges"],
            root=skeleton_parts["meta"]["root"],
            spatial_columns=skeleton_parts["meta"]["spatial_columns"],
        )
        skel_sync._set_base_properties(inherited_properties)
        return skel_sync
    else:
        cell.add_skeleton(
            vertices=skeleton_parts["nodes"],
            edges=skeleton_parts["edges"],
            root=skeleton_parts["meta"]["root"],
            spatial_columns=skeleton_parts["meta"]["spatial_columns"],
        )
        cell.layers[skeleton_parts["meta"]["name"]]._set_base_properties(
            inherited_properties
        )
        return cell


def parse_mesh_files(layer_name, files, tf):
    prefix = f"layers/{layer_name}"
    mesh_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    mesh_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    mesh_parts["nodes"] = load_dataframe(files[fn], tf)
                case ("faces.npz", layer_name):
                    mesh_parts["faces"] = load_array(files[fn], tf)
    return mesh_parts


def build_mesh(mesh_parts: dict, cell: Cell = None) -> Union[MeshLayer, Cell]:
    if cell is None:
        mesh_sync = MeshLayer(
            name=mesh_parts["meta"]["name"],
            vertices=mesh_parts["nodes"],
            faces=mesh_parts["faces"],
            spatial_columns=mesh_parts["meta"]["spatial_columns"],
        )
        return mesh_sync
    else:
        cell.add_mesh(
            vertices=mesh_parts["nodes"],
            faces=mesh_parts["faces"],
            spatial_columns=mesh_parts["meta"]["spatial_columns"],
        )
        return cell


def parse_graph_files(layer_name, files, tf) -> dict:
    prefix = f"layers/{layer_name}"
    graph_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    graph_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    graph_parts["nodes"] = load_dataframe(files[fn], tf)
                case ("edges.npz", layer_name):
                    graph_parts["edges"] = load_array(files[fn], tf)
    return graph_parts


def build_graph(graph_parts: dict, cell: Cell = None) -> Union[GraphLayer, Cell]:
    if cell is None:
        graph_sync = GraphLayer(
            name=graph_parts["meta"]["name"],
            vertices=graph_parts["nodes"],
            edges=graph_parts["edges"],
            spatial_columns=graph_parts["meta"]["spatial_columns"],
        )
        return graph_sync
    else:
        cell.add_graph(
            vertices=graph_parts["nodes"],
            edges=graph_parts["edges"],
            spatial_columns=graph_parts["meta"]["spatial_columns"],
        )
        return cell


def parse_point_cloud_files(layer_name, files, tf, as_annotation: bool = True) -> dict:
    if as_annotation:
        prefix = f"annotations/{layer_name}"
    else:
        prefix = f"layers/{layer_name}"
    pcd_parts = {}
    for fn in files.keys():
        if fn.startswith(prefix):
            path_parts = Path(fn).parts
            match (path_parts[-1], path_parts[-2]):
                case ("meta.json", layer_name):
                    pcd_parts["meta"] = load_dict(files[fn], tf)
                case ("nodes.feather", layer_name):
                    pcd_parts["nodes"] = load_dataframe(files[fn], tf)
    return pcd_parts


def build_point_cloud(
    pcd_parts, cell: "Cell" = None, as_annotation: bool = True
) -> Union[PointCloudLayer, Cell]:
    if cell is None:
        point_cloud_sync = PointCloudLayer(
            name=pcd_parts["meta"]["name"],
            vertices=pcd_parts["nodes"],
            spatial_columns=pcd_parts["meta"]["spatial_columns"],
        )
        return point_cloud_sync
    else:
        if as_annotation:
            cell.add_point_annotations(
                name=pcd_parts["meta"]["name"],
                vertices=pcd_parts["nodes"],
                spatial_columns=pcd_parts["meta"]["spatial_columns"],
            )
        return cell


def build_linkage(
    linkage_pair,
    tf,
    cell,
) -> Cell:
    prefix = f"linkage/{linkage_pair[0]}/{linkage_pair[1]}"
    link_df = load_dataframe(f"{prefix}/linkage.feather", tf)
    # Determine source based on the length of the vertices in the mapping and in the skeleton layer
    if len(link_df) == len(cell._all_objects[linkage_pair[0]].nodes):
        source_layer = linkage_pair[0]
        target_layer = linkage_pair[1]
    elif len(link_df) == len(cell._all_objects[linkage_pair[1]].nodes):
        source_layer = linkage_pair[1]
        target_layer = linkage_pair[0]
    else:
        raise ValueError("Linkage DataFrame does not match any layer.")

    layer = cell._all_objects[source_layer]
    cell._all_objects[source_layer]._process_linkage(
        Link(
            link_df.set_index(source_layer).loc[layer.vertex_index][target_layer],
            source=source_layer,
            target=target_layer,
        )
    )
    return cell


def import_legacy_meshwork(
    filename: Union[str, BinaryIO], l2_skeleton: bool = True, as_pcg_skel: bool = False
) -> tuple[Cell, np.ndarray]:
    """Import a legacy MeshWork file as a Cell object. Requires `h5py`, which can be installed with `pip install ossify[legacy]`.

    Parameters
    ----------
    filename : Union[str, BinaryIO]
        The path to the MeshWork file or an open binary file object.
    l2_skeleton : bool
        Whether to import the skeleton as a level 2 skeleton (True, puts "mesh" data into cell.graph) or as a mesh (False, puts "mesh" data into cell.mesh).
    as_pcg_skel : bool
        Whether to process the skeleton to remap typical pcg-skel built skeleton annotations like segment properties and volume properties into features.
        Volume properties are mapped to graph features (since they are associated with L2 graph vertices) while segment properties are mapped to skeleton features.

    Returns
    -------
    tuple[Cell, np.ndarray]
        The imported Cell object and a boolean mask indicating which mesh vertices correspond to skeleton nodes.
        Note the mask is not pre-applied, since masking removes data in ossify.
        You can apply the mask with `cell.graph.apply_mask(mask)` (if l2_skeleton is True) or `cell.mesh.apply_mask(mask)` (if l2_skeleton is False).
    """
    mwi = MeshworkImporter(filename, l2_skeleton=l2_skeleton)
    cell, node_mask = mwi.import_cell(process_pcg_skel=as_pcg_skel)
    return cell, node_mask


def _process_pcg_skel_import(cell: Cell) -> Cell:
    if "compartment" in cell.annotations:
        cell.skeleton.add_feature(
            cell.skeleton.map_annotations_to_feature(
                "compartment",
                distance_threshold=0,
                agg={"compartment": ("compartment", "mean")},
            ),
            "compartment",
        )
    if "segment_properties" in cell.annotations:
        features = cell.skeleton.map_annotations_to_feature(
            "segment_properties",
            distance_threshold=0,
            agg={
                "area": ("area", "mean"),
                "area_factor": ("area", "mean"),
                "len": ("len", "mean"),
                "r_eff": ("r_eff", "mean"),
                "strahler": ("strahler", "mean"),
                "vol": ("vol", "mean"),
            },
        )
        cell.skeleton.add_feature(features)
    if "lvl2_ids" in cell.annotations:
        cell.graph.add_feature(cell.annotations.lvl2_ids.features[["lvl2_id"]])
    if "is_axon" in cell.annotations:
        is_axon = cell.skeleton.nodes.index.isin(cell.annotations.is_axon.vertex_index)
        cell.skeleton.add_feature(is_axon, "is_axon")
    if "vol_prop" in cell.annotations:
        cell.graph.add_feature(cell.annotations.vol_prop.features)
    for anno in [
        "compartment",
        "segment_properties",
        "lvl2_ids",
        "is_axon",
        "vol_prop",
    ]:
        if anno in cell.annotations:
            cell.remove_annotation(anno)
    return cell


class MeshworkImporter:
    NULL_VERSION = 1

    def __init__(self, filename: Union[str, BinaryIO], l2_skeleton: bool = True):
        try:
            import h5py

            self.h5py = h5py
        except ImportError:
            self.h5py = None
            raise ImportError(
                "h5py is required to import legacy MeshWork files. Install ossify with `pip install ossify[legacy]` to enable importing legacy meshwork files."
            )
        self.file = filename
        self.l2_skeleton = l2_skeleton

        self.meta = None
        self.cell = None
        self.mesh_mask = None
        self.node_mask = None
        self.anno_load_function = {
            1: self._load_dataframe_pandas,
            2: self._load_dataframe_generic,
        }
        self.annotation_dfs = None

    def import_cell(self, process_pcg_skel: bool = False) -> tuple[Cell, np.ndarray]:
        self.import_metadata()
        self.cell = Cell(name=self.meta.get("seg_id", "unknown"), meta=self.meta)

        self.import_mesh(self.l2_skeleton)
        self.import_skeleton()
        self.import_annotations()
        if process_pcg_skel:
            cell_out = _process_pcg_skel_import(self.cell)
            return cell_out, self.node_mask
        else:
            return self.cell, self.node_mask

    @property
    def version(self):
        return self.meta.get("version", self.NULL_VERSION)

    def _load_dataframe_generic(self, table_name):
        key = f"annotations/{table_name}/data"
        with self.h5py.File(self.file, "r") as f:
            dat = f[key][()].tobytes()
            df = pd.DataFrame.from_records(orjson.loads(dat))
            try:
                df.index = np.array(df.index, dtype="int")
            except:
                pass
        return df

    def _load_dataframe_pandas(self, table_name):
        return pd.read_hdf(self.file, f"annotations/{table_name}/data")

    def import_metadata(self):
        meta = {}
        with self.h5py.File(self.file, "r") as f:
            meta["seg_id"] = f.attrs.get("seg_id", None)
            meta["voxel_resolution"] = f.attrs.get("voxel_resolution", None)
            meta["version"] = f.attrs.get("version", self.NULL_VERSION)
        self.meta = meta

    def import_mesh(self, l2_skeleton: bool = True):
        with self.h5py.File(self.file, "r") as f:
            verts = f["mesh/vertices"][()]
            faces = f["mesh/faces"][()]
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)

            if "link_edges" in f["mesh"].keys():
                link_edges = f["mesh/link_edges"][()]
            else:
                link_edges = None

            node_mask = f["mesh/node_mask"][()]
            voxel_scaling = f["mesh"].attrs.get("voxel_scaling", None)
            mesh_mask = f["mesh/mesh_mask"][()]

        if voxel_scaling is None:
            voxel_scaling = np.atleast_2d(np.asarray([1.0, 1.0, 1.0]))
        verts = verts * voxel_scaling

        if l2_skeleton:
            self.cell.add_graph(
                vertices=verts,
                edges=link_edges,
            )
        else:
            self.cell.add_mesh(
                vertices=verts,
                faces=faces,
            )
        self.mesh_mask = mesh_mask
        self.node_mask = node_mask

    def import_skeleton(self):
        with self.h5py.File(self.file, "r") as f:
            if "skeleton" not in f:
                return

            if "meta" in f["skeleton"].keys():
                meta = orjson.loads(f["skeleton/meta"][()].tobytes())
            else:
                meta = {}

            verts = f["skeleton/vertices"][()]
            edges = f["skeleton/edges"][()]
            root = f["skeleton/root"][()]
            mesh_to_skel_map = f["skeleton/mesh_to_skel_map"][()]

            if "radius" in f["skeleton"].keys():
                radius = f["skeleton/radius"][()]
            else:
                radius = None
            voxel_scaling = f["skeleton"].attrs.get("voxel_scaling", None)

            if voxel_scaling is None:
                voxel_scaling = np.atleast_2d(np.asarray([1.0, 1.0, 1.0]))
            verts = verts * voxel_scaling

        if self.l2_skeleton:
            source_layer = "graph"
        else:
            source_layer = "mesh"
        self.cell.add_skeleton(
            vertices=verts,
            edges=edges,
            root=root,
            features={"radius": radius} if radius is not None else None,
            linkage=Link(
                mesh_to_skel_map, source=source_layer, map_value_is_index=False
            ),
        )

    def import_annotations(self):
        with self.h5py.File(self.file, "r") as f:
            if "annotations" not in f:
                return {}
            table_names = list(f["annotations"].keys())

            annotation_dfs = {}
            for table_name in table_names:
                annotation_dfs[table_name] = {}
                annotation_dfs[table_name]["data"] = self.anno_load_function[
                    self.version
                ](table_name)
                dset = f[f"annotations/{table_name}"]
                annotation_dfs[table_name]["anchor_to_mesh"] = bool(
                    dset.attrs.get("anchor_to_mesh")
                )
                annotation_dfs[table_name]["point_column"] = dset.attrs.get(
                    "point_column", None
                )
                annotation_dfs[table_name]["max_distance"] = dset.attrs.get(
                    "max_distance"
                )
                if bool(dset.attrs.get("defined_index", False)):
                    annotation_dfs[table_name]["index_column"] = dset.attrs.get(
                        "index_column", None
                    )
        for name, df_info in annotation_dfs.items():
            df = df_info["data"]

            pt_col = df.get("point_column", None)
            if pt_col is None:
                vertices_from_linkage = True
                spatial_cols = None
            else:
                vertices_from_linkage = False
                spatial_cols = utils.process_spatial_columns(pt_col)
                for ii, c in enumerate(spatial_cols):
                    df[c] = df[pt_col].values[:, ii]
                df.drop(columns=[pt_col], inplace=True)

            linkage_col = df_info.get("index_column")
            mapping = df[linkage_col].values
            df.drop(columns=[linkage_col], inplace=True)

            if self.l2_skeleton:
                target = "graph"
            else:
                target = "mesh"

            self.cell.add_point_annotations(
                name,
                vertices=df,
                spatial_columns=None,
                linkage=Link(
                    mapping=mapping,
                    target=target,
                    map_value_is_index=False,
                ),
                vertices_from_linkage=vertices_from_linkage,
            )
