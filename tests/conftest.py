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
    with open("tests/data/test_meshwork.osy", "rb") as f:
        nrn = file_io.load_morphology(f)
    return nrn
