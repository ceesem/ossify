"""Tests for the structured compartment-labeling composition.

``ProbaVertexModel`` (encode) feeds ``StructuredLabeler`` (decode + clean). The
decode/validation tests run with a tiny synthetic skeleton and a fake estimator
(no ML framework or sample data needed). The end-to-end tests -- including the
"any sklearn estimator" path -- are gated on the v1dd sample meshwork and the
optional xgboost/sklearn extras.
"""

import pathlib

import numpy as np
import pytest

from ossify import Cell
from ossify.compartments import ProbaVertexModel, StructuredLabeler
from ossify.structured_prediction import TransitionSchema

DATA = pathlib.Path(__file__).parent / "data"
MESHWORK = DATA / "v1dd_864691132533489754.h5"
MODEL = "v1dd_ds15_us0_bd0.json"

FEATURE_COLUMNS = [
    "area_um2",
    "vol_um3",
    "max_dt_um",
    "vol_to_area",
    "syn_in",
    "syn_out",
    "down_area_um2",
    "down_vol_um3",
    "down_max_dt_um",
    "down_vol_to_area",
    "down_syn_in",
    "down_syn_out",
]


class _FakeProba:
    """Minimal sklearn-style classifier: fixed classes_, identity predict_proba."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)


class _FixedEncoder:
    """Encoder stub: returns pre-set per-vertex probabilities as unaries.

    Duck-types the bits ``StructuredLabeler`` uses (``predict_unaries`` and
    ``classes_``), so the decode tail can be exercised with synthetic scores
    without any feature extraction or fitted model.
    """

    def __init__(self, scores, classes):
        self._scores = np.asarray(scores, dtype=float)
        self.classes_ = list(classes)

    def predict_unaries(self, cell):
        return np.log(np.clip(self._scores, 1e-9, None))


def _encoder(estimator, feature_columns=("f",), **kwargs):
    return ProbaVertexModel(estimator, list(feature_columns), **kwargs)


def _schema(revert=50.0, default_cost=0.0):
    return TransitionSchema(
        classes=["dendrite", "axon"],
        transitions={("axon", "dendrite"): revert},
        root_classes=["dendrite"],
        default_cost=default_cost,
    )


def _chain_cell(n=6):
    verts = np.array([[i, 0, 0] for i in range(n)], dtype=float)
    edges = np.column_stack([np.arange(1, n), np.arange(0, n - 1)])
    cell = Cell()
    cell.add_skeleton(verts, edges, root=0)
    return cell


class TestClassValidation:
    def test_integer_placeholder_classes_ok(self):
        m = StructuredLabeler(_encoder(_FakeProba([0, 1])), _schema())
        assert m.classes == ("dendrite", "axon")

    def test_bool_placeholder_classes_ok(self):
        # [False, True] == [0, 1] -> treated as positional placeholders.
        StructuredLabeler(_encoder(_FakeProba([False, True])), _schema())

    def test_string_classes_must_match_order(self):
        StructuredLabeler(_encoder(_FakeProba(["dendrite", "axon"])), _schema())
        with pytest.raises(ValueError, match="must match schema.classes"):
            StructuredLabeler(_encoder(_FakeProba(["axon", "dendrite"])), _schema())

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="predicts 3"):
            StructuredLabeler(_encoder(_FakeProba([0, 1, 2])), _schema())

    def test_missing_classes_attr_is_allowed(self):
        class NoClasses:
            def predict_proba(self, X):
                return np.asarray(X, dtype=float)

        # No classes_ on the estimator -> validated against unary width at decode.
        StructuredLabeler(_encoder(NoClasses()), _schema())


class TestDecodeTail:
    def test_absorb_collapses_noisy_tip(self):
        cell = _chain_cell(6)
        scores = np.array([[0.99, 0.01]] * 4 + [[0.4, 0.6]] * 2)
        on = StructuredLabeler(
            _FixedEncoder(scores, [0, 1]), _schema(), absorb_min_size=2
        )
        off = StructuredLabeler(_FixedEncoder(scores, [0, 1]), _schema())
        np.testing.assert_array_equal(
            on.predict(cell, return_labels_as="index"), [0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            off.predict(cell, return_labels_as="index"), [0, 0, 0, 0, 1, 1]
        )

    def test_default_cost_resists_trivial_flip(self):
        cell = _chain_cell(4)
        # a single weakly-axon leaf; with no switching penalty it flips...
        scores = np.array([[0.99, 0.01]] * 3 + [[0.45, 0.55]])
        weak = StructuredLabeler(
            _FixedEncoder(scores, [0, 1]), _schema(default_cost=0.0)
        )
        np.testing.assert_array_equal(
            weak.predict(cell, return_labels_as="index"), [0, 0, 0, 1]
        )
        # ...but a switching penalty larger than its log-odds pins it to dendrite.
        firm = StructuredLabeler(
            _FixedEncoder(scores, [0, 1]), _schema(default_cost=1.0)
        )
        np.testing.assert_array_equal(
            firm.predict(cell, return_labels_as="index"), [0, 0, 0, 0]
        )

    def test_label_index_mapping(self):
        cell = _chain_cell(4)
        scores = np.array([[0.99, 0.01]] * 2 + [[0.1, 0.9]] * 2)
        m = StructuredLabeler(_FixedEncoder(scores, [0, 1]), _schema())
        idx = m.predict(cell, return_labels_as="index")
        np.testing.assert_array_equal(
            m.predict(cell, return_labels_as="label"),
            ["dendrite", "dendrite", "axon", "axon"],
        )
        np.testing.assert_array_equal(m._as_labels(idx, "index"), idx)
        with pytest.raises(ValueError):
            m._as_labels(idx, "nonsense")


# --- End-to-end on the real sample, incl. the "any sklearn model" path -------

pytestmark_data = pytest.mark.skipif(
    not MESHWORK.exists(), reason="v1dd sample meshwork not present"
)


@pytestmark_data
def test_from_config_end_to_end():
    pytest.importorskip("xgboost")
    pytest.importorskip("h5py")
    import ossify

    cell, _ = ossify.import_legacy_meshwork(str(MESHWORK), as_pcg_skel=True)
    schema = _schema(revert=10.0)
    m = StructuredLabeler.from_config(schema, MODEL, absorb_min_size=3)
    n = cell.skeleton.n_vertices

    labels = m.predict(cell)
    assert labels.shape == (n,)
    assert set(np.unique(labels)).issubset({"dendrite", "axon"})

    lab_idx, edge_cost = m.predict_with_violations(cell, return_labels_as="index")
    assert lab_idx.shape == (n,) and edge_cost.shape == (n,)
    assert lab_idx.dtype.kind in "iu"


@pytestmark_data
def test_accepts_plain_sklearn_estimator():
    """Nothing is xgboost-specific: a sklearn RandomForest drops in unchanged."""
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    pytest.importorskip("h5py")
    from sklearn.ensemble import RandomForestClassifier

    import ossify
    from ossify.compartments import ProbaVertexModel, make_skel_prop_df

    cell, _ = ossify.import_legacy_meshwork(str(MESHWORK), as_pcg_skel=True)
    feats = make_skel_prop_df(cell, downstream_hops=15)
    X = feats[FEATURE_COLUMNS].values
    # supervise on the bundled model's structured labels just to fit a sklearn model
    y = (
        StructuredLabeler.from_config(_schema(revert=10.0), MODEL)
        .predict(cell, return_labels_as="index")
        .astype(int)
    )
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert list(rf.classes_) == [0, 1]  # positional placeholders; schema names them

    m = StructuredLabeler(
        ProbaVertexModel(rf, FEATURE_COLUMNS, downstream_hops=15),
        _schema(revert=10.0),
        absorb_min_size=3,
    )
    labels = m.predict(cell)
    assert labels.shape == (cell.skeleton.n_vertices,)
    assert set(np.unique(labels)).issubset({"dendrite", "axon"})
