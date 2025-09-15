import numpy as np
import orjson
import pandas as pd
import pytest

from ossify import file_io


@pytest.fixture
def root_id():
    with open("tests/data/root_id.json", "r") as f:
        root_id = int(f.read())
    return root_id


@pytest.fixture
def pre_syn_df():
    return pd.read_feather("tests/data/pre_l2.feather")


@pytest.fixture
def post_syn_df():
    return pd.read_feather("tests/data/post_l2.feather")


@pytest.fixture
def skel_dict():
    with open("tests/data/skel.json", "r") as f:
        skel_dict = orjson.loads(f.read())
    return skel_dict


@pytest.fixture
def l2_graph():
    with open("tests/data/l2graph.json") as f:
        l2_graph = orjson.loads(f.read())
    return l2_graph


@pytest.fixture
def l2_df():
    l2_df = pd.read_feather("tests/data/l2properties.feather")
    l2_df.reset_index(inplace=True)
    return l2_df


@pytest.fixture
def synapse_spatial_columns():
    return ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]


@pytest.fixture
def l2_spatial_columns():
    return ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]


@pytest.fixture
def nrn():
    with open("tests/data/test_cell_no_mesh.osy", "rb") as f:
        nrn = file_io.load_cell(f)
    return nrn


@pytest.fixture
def cell_with_mesh():
    with open("tests/data/test_cell_with_mesh.osy", "rb") as f:
        nrn = file_io.load_cell(f)
    return nrn


# ============================================================================
# Mock Data Fixtures for Unit Tests
# ============================================================================


@pytest.fixture
def simple_skeleton_data():
    """Linear skeleton with 5 vertices and 4 edges."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # root
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],  # tip
        ]
    )
    edges = np.array(
        [
            [1, 0],  # child -> parent relationship
            [2, 1],
            [3, 2],
            [4, 3],
        ]
    )
    vertex_indices = np.array([100, 101, 102, 103, 104])
    return vertices, edges, vertex_indices


@pytest.fixture
def branched_skeleton_data():
    """Y-shaped skeleton with branch point."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # root (100)
            [1.0, 0.0, 0.0],  # branch point (101)
            [2.0, 1.0, 0.0],  # tip 1 (102)
            [2.0, -1.0, 0.0],  # tip 2 (103)
            [3.0, 1.0, 0.0],  # extended tip 1 (104)
        ]
    )
    edges = np.array(
        [
            [1, 0],  # branch point -> root
            [2, 1],  # tip 1 -> branch point
            [3, 1],  # tip 2 -> branch point
            [4, 2],  # extended tip 1 -> tip 1
        ]
    )
    vertex_indices = np.array([100, 101, 102, 103, 104])
    return vertices, edges, vertex_indices


@pytest.fixture
def simple_mesh_data():
    """Tetrahedron mesh."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_indices = np.array([200, 201, 202, 203])
    return vertices, faces, vertex_indices


@pytest.fixture
def simple_graph_data():
    """Small graph network."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]])
    vertex_indices = np.array([300, 301, 302, 303, 304])
    return vertices, edges, vertex_indices


@pytest.fixture
def mock_point_annotations():
    """Small point cloud with features."""
    vertices = np.array(
        [[0.5, 0.2, 0.1], [1.5, 0.1, 0.2], [2.5, 0.3, 0.0], [3.2, 0.1, 0.1]]
    )

    df = pd.DataFrame(
        {
            "x": vertices[:, 0],
            "y": vertices[:, 1],
            "z": vertices[:, 2],
            "annotation_type": ["synapse", "synapse", "bouton", "synapse"],
            "confidence": [0.9, 0.85, 0.95, 0.8],
            "size": [1.2, 0.8, 2.1, 1.0],
        }
    )
    df.index = [400, 401, 402, 403]
    return df


@pytest.fixture
def spatial_columns():
    """Standard spatial column names."""
    return ["x", "y", "z"]


@pytest.fixture
def mock_features():
    """Sample feature data for testing."""
    return {
        "radius": [0.5, 0.6, 0.4, 0.7, 0.3],
        "compartment": ["soma", "dendrite", "dendrite", "dendrite", "axon"],
        "distance_to_root": [0.0, 1.0, 2.0, 3.0, 4.0],
    }
