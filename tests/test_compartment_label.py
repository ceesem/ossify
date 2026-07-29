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
from ossify.compartments import (
    DEFAULT_UNARY_CLIP,
    ProbaVertexModel,
    StructuredLabeler,
)
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

    def __init__(self, scores, classes, unary_clip=None):
        self._scores = np.asarray(scores, dtype=float)
        self.classes_ = list(classes)
        self._unary_clip = unary_clip

    def predict_unaries(self, cell):
        floor = 1e-9 if self._unary_clip is None else np.exp(-self._unary_clip)
        return np.clip(
            np.log(np.clip(self._scores, floor, None)), None, self._unary_clip
        )


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


class TestUnaryClipping:
    def test_unary_clip_validation(self):
        for bad in (0.0, -0.1, -50.0, None):
            with pytest.raises(ValueError, match="unary_clip"):
                ProbaVertexModel(_FakeProba([0, 1]), ["f"], unary_clip=bad)

    def test_guards_log_of_zero(self):
        # The point of the clip: log(0) -> -inf would poison the decode; the clip
        # bounds the zero column to exactly -unary_clip and keeps everything finite.
        u = _encoder(_FakeProba([0, 1]), unary_clip=50.0).to_unaries(
            np.array([[1.0, 0.0]])
        )
        assert np.all(np.isfinite(u))
        np.testing.assert_allclose(u[0, 1], -50.0, rtol=1e-12)
        # the certain class sits at log(1) = 0, so the log-odds gap is exactly 50.
        np.testing.assert_allclose(u[0, 0] - u[0, 1], 50.0, rtol=1e-12)

    def test_large_clip_does_not_change_a_decode(self):
        # A clip set comfortably above the costs never binds -> identical labels.
        cell = _chain_cell(6)
        scores = np.array([[0.99, 0.01]] * 4 + [[0.4, 0.6]] * 2)
        schema = _schema(revert=10.0, default_cost=1.0)
        guarded = StructuredLabeler(
            _FixedEncoder(scores, [0, 1], unary_clip=50.0), schema
        )
        plain = StructuredLabeler(_FixedEncoder(scores, [0, 1]), schema)
        np.testing.assert_array_equal(
            guarded.predict(cell, return_labels_as="index"),
            plain.predict(cell, return_labels_as="index"),
        )

    def test_default_clip_is_a_finite_rail(self):
        enc = _encoder(_FakeProba([0, 1]))  # uses DEFAULT_UNARY_CLIP
        u = enc.to_unaries(np.array([[1.0, 0.0]]))
        # the zero column lands at -DEFAULT_UNARY_CLIP rather than blowing up to -inf
        assert np.all(np.isfinite(u))
        np.testing.assert_allclose(u[0, 1], -DEFAULT_UNARY_CLIP, rtol=1e-12)

    def test_small_clip_can_change_results(self):
        # Opt-in regime: set *below* a cost and the guard becomes a confidence cap.
        # A leaf strongly labeled axon against a default_cost=2.0 switch penalty:
        # at full strength its log-odds (~6.9) flips it; clipping below 2.0 holds it.
        cell = _chain_cell(4)
        scores = np.array([[0.999, 0.001]] * 3 + [[0.001, 0.999]])
        schema = _schema(default_cost=2.0)
        loud = StructuredLabeler(_FixedEncoder(scores, [0, 1]), schema)
        np.testing.assert_array_equal(
            loud.predict(cell, return_labels_as="index"), [0, 0, 0, 1]
        )
        quiet = StructuredLabeler(_FixedEncoder(scores, [0, 1], unary_clip=1.5), schema)
        np.testing.assert_array_equal(
            quiet.predict(cell, return_labels_as="index"), [0, 0, 0, 0]
        )


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


class TestSerialization:
    def test_encoder_without_config_ref_raises(self):
        # Built directly from an estimator (not via from_config) -> no known
        # source file to point a reloaded config at.
        labeler = StructuredLabeler(
            _encoder(_FakeProba([0, 1])), _schema(revert=10.0), absorb_min_size=5
        )
        with pytest.raises(ValueError, match="not built via from_config"):
            labeler.to_dict()

    def test_estimator_without_save_model_raises(self):
        labeler = StructuredLabeler(_encoder(_FakeProba([0, 1])), _schema())
        with pytest.raises(TypeError, match="save_model"):
            labeler.to_dict(model_file="whatever.json")

    def test_custom_feature_spec_blocks_serialization(self):
        from ossify.compartments import MappedFeature

        enc = _encoder(_FakeProba([0, 1]), feature_spec=[MappedFeature("a", "b")])
        labeler = StructuredLabeler(enc, _schema())
        with pytest.raises(ValueError, match="feature_spec"):
            labeler.to_dict()

    def test_unsupported_encoder_type_raises(self):
        class _OtherEncoder:
            classes_ = [0, 1]

            def predict_unaries(self, cell):
                raise NotImplementedError

        labeler = StructuredLabeler(_OtherEncoder(), _schema())
        with pytest.raises(TypeError, match="not supported"):
            labeler.to_dict()

    def test_from_dict_rejects_unknown_encoder_type(self):
        d = {
            "schema": _schema().to_dict(),
            "encoder": {"type": "SomethingElse"},
            "absorb_min_size": None,
            "absorb_min_weight": None,
        }
        with pytest.raises(ValueError, match="Unknown or unsupported encoder type"):
            StructuredLabeler.from_dict(d)

    def test_from_dict_rejects_newer_schema_version(self):
        d = {
            "version": StructuredLabeler._SCHEMA_VERSION + 1,
            "schema": _schema().to_dict(),
            "encoder": {"type": "SomethingElse"},
            "absorb_min_size": None,
            "absorb_min_weight": None,
        }
        with pytest.raises(ValueError, match="schema version"):
            StructuredLabeler.from_dict(d)


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
    m = StructuredLabeler(
        ProbaVertexModel.from_config(MODEL), schema, absorb_min_size=3
    )
    n = cell.skeleton.n_vertices

    labels = m.predict(cell)
    assert labels.shape == (n,)
    assert set(np.unique(labels)).issubset({"dendrite", "axon"})

    lab_idx, edge_cost = m.predict_with_violations(cell, return_labels_as="index")
    assert lab_idx.shape == (n,) and edge_cost.shape == (n,)
    assert lab_idx.dtype.kind in "iu"


@pytestmark_data
def test_save_config_load_config_roundtrip(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("h5py")
    import ossify

    cell, _ = ossify.import_legacy_meshwork(str(MESHWORK), as_pcg_skel=True)
    schema = _schema(revert=10.0)
    labeler = StructuredLabeler(
        ProbaVertexModel.from_config(MODEL), schema, absorb_min_size=3
    )

    import ossify as ossify_pkg

    d = labeler.to_dict()
    assert d["type"] == "StructuredLabeler"
    assert d["version"] == StructuredLabeler._SCHEMA_VERSION
    assert d["ossify_version"] == ossify_pkg.__version__
    assert d["encoder"]["type"] == "ProbaVertexModel"
    assert d["encoder"]["version"] == ProbaVertexModel._SCHEMA_VERSION
    assert d["encoder"]["config"] == MODEL
    assert d["absorb_min_size"] == 3
    assert d["absorb_min_weight"] is None

    path = tmp_path / "labeler.json"
    labeler.save_config(path)
    reloaded = StructuredLabeler.load_config(path)

    np.testing.assert_array_equal(reloaded.predict(cell), labeler.predict(cell))


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
        StructuredLabeler(ProbaVertexModel.from_config(MODEL), _schema(revert=10.0))
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
