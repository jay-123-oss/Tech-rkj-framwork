"""
Phase 2 Gate Verification Test Suite: Native Pickle / Joblib / Multiprocessing Protocol.

Covers:
1. Bit-exact pickle serialization for SubspaceRegressor (predictions, P matrix, weights, stats).
2. Bit-exact pickle serialization for SubspaceClassifier (predictions, probas, P matrix, weights, classes).
3. Joblib disk dump/load roundtrip verification.
4. Scikit-learn clone() compliance.
5. Multiprocessing execution via cross_val_score(n_jobs=2).
6. Corrupt and truncated payload resilience (clean ValueError).
7. Continual learning persistence post-unpickling.
"""

import pickle
import tempfile
import numpy as np
import pytest
import joblib
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import NotFittedError

from tech_rkj import SubspaceRegressor, SubspaceClassifier


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 12
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    true_w = np.random.randn(n_features)
    y = (X @ true_w + 0.1 * np.random.randn(n_samples)).astype(np.float64)
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(1337)
    n_samples = 240
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 3-class target
    logits = X[:, :3]
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


def test_01_regressor_pickle_bit_exact(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(
        num_heads=4,
        head_circuits=128,
        alpha=5.0,
        frontend_mode="zscore",
    )
    reg.fit(X, y)

    orig_preds = reg.predict(X)
    orig_p_inv = reg.p_inv_
    orig_weights = reg.weights_
    orig_samples_seen = reg.n_samples_seen_

    # Serialize and deserialize via pickle
    pickled = pickle.dumps(reg, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(pickled)

    # 1. Prediction parity: exact bit-for-bit equality
    restored_preds = restored.predict(X)
    max_pred_diff = np.max(np.abs(orig_preds - restored_preds))
    assert max_pred_diff == 0.0, f"Regression predictions drift: {max_pred_diff}"
    assert np.array_equal(orig_preds, restored_preds)

    # 2. P_inv matrix preservation across all heads
    restored_p_inv = restored.p_inv_
    assert len(orig_p_inv) == len(restored_p_inv)
    for h in range(len(orig_p_inv)):
        diff_p = np.max(np.abs(orig_p_inv[h] - restored_p_inv[h]))
        assert diff_p == 0.0, f"Head {h} P_inv drift: {diff_p}"
        assert np.array_equal(orig_p_inv[h], restored_p_inv[h])

    # 3. Weights preservation across all heads
    restored_weights = restored.weights_
    assert len(orig_weights) == len(restored_weights)
    for h in range(len(orig_weights)):
        diff_w = np.max(np.abs(orig_weights[h] - restored_weights[h]))
        assert diff_w == 0.0, f"Head {h} weights drift: {diff_w}"
        assert np.array_equal(orig_weights[h], restored_weights[h])

    # 4. Metadata and sample counter preservation
    assert restored.n_samples_seen_ == orig_samples_seen
    assert restored.n_features_in_ == reg.n_features_in_
    assert restored.is_fitted_ is True


def test_02_classifier_pickle_bit_exact(classification_data):
    X, y = classification_data
    clf = SubspaceClassifier(
        num_heads=4,
        head_circuits=128,
        alpha=8.0,
        frontend_mode="zscore",
    )
    clf.fit(X, y)

    orig_preds = clf.predict(X)
    orig_probas = clf.predict_proba(X)
    orig_p_inv = clf.p_inv_
    orig_weights = clf.weights_
    orig_classes = clf.classes_
    orig_samples_seen = clf.n_samples_seen_

    # Serialize and deserialize via pickle
    pickled = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(pickled)

    # 1. Prediction and probability bit-exact parity
    restored_preds = restored.predict(X)
    assert np.array_equal(orig_preds, restored_preds)

    restored_probas = restored.predict_proba(X)
    max_proba_diff = np.max(np.abs(orig_probas - restored_probas))
    assert max_proba_diff == 0.0, f"Classification proba drift: {max_proba_diff}"
    assert np.array_equal(orig_probas, restored_probas)

    # 2. P_inv matrix preservation across all heads
    restored_p_inv = restored.p_inv_
    assert len(orig_p_inv) == len(restored_p_inv)
    for h in range(len(orig_p_inv)):
        diff_p = np.max(np.abs(orig_p_inv[h] - restored_p_inv[h]))
        assert diff_p == 0.0, f"Head {h} P_inv drift: {diff_p}"

    # 3. Weights preservation across all heads
    restored_weights = restored.weights_
    assert len(orig_weights) == len(restored_weights)
    for h in range(len(orig_weights)):
        diff_w = np.max(np.abs(orig_weights[h] - restored_weights[h]))
        assert diff_w == 0.0, f"Head {h} weights drift: {diff_w}"

    # 4. Classes and metadata preservation
    assert np.array_equal(orig_classes, restored.classes_)
    assert restored.n_classes_ == clf.n_classes_
    assert restored.n_samples_seen_ == orig_samples_seen
    assert restored.is_fitted_ is True


def test_03_joblib_file_roundtrip(regression_data, classification_data):
    X_reg, y_reg = regression_data
    X_cls, y_cls = classification_data

    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X_reg, y_reg)
    clf = SubspaceClassifier(num_heads=2, head_circuits=64).fit(X_cls, y_cls)

    with tempfile.NamedTemporaryFile(suffix=".joblib") as f_reg:
        joblib.dump(reg, f_reg.name)
        loaded_reg = joblib.load(f_reg.name)
        assert np.array_equal(reg.predict(X_reg), loaded_reg.predict(X_reg))
        assert loaded_reg.is_fitted_ is True

    with tempfile.NamedTemporaryFile(suffix=".joblib") as f_cls:
        joblib.dump(clf, f_cls.name)
        loaded_cls = joblib.load(f_cls.name)
        assert np.array_equal(clf.predict(X_cls), loaded_cls.predict(X_cls))
        assert loaded_cls.is_fitted_ is True


def test_04_sklearn_clone_compliance():
    # 1. Regressor clone
    reg = SubspaceRegressor(
        num_heads=6,
        head_circuits=192,
        alpha=15.0,
        frontend_mode="identity",
    )
    cloned_reg = clone(reg)
    assert cloned_reg.num_heads == 6
    assert cloned_reg.head_circuits == 192
    assert cloned_reg.alpha == 15.0
    assert cloned_reg.frontend_mode == "identity"
    assert cloned_reg.is_fitted_ is False
    assert cloned_reg._core is None

    # Verify cloned model can be fitted independently
    X = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randn(50).astype(np.float64)
    cloned_reg.fit(X, y)
    assert cloned_reg.is_fitted_ is True
    assert len(cloned_reg.predict(X)) == 50

    # 2. Classifier clone
    clf = SubspaceClassifier(
        num_heads=3,
        head_circuits=96,
        alpha=12.0,
        frontend_mode="zscore",
    )
    cloned_clf = clone(clf)
    assert cloned_clf.num_heads == 3
    assert cloned_clf.head_circuits == 96
    assert cloned_clf.alpha == 12.0
    assert cloned_clf.frontend_mode == "zscore"
    assert cloned_clf.is_fitted_ is False
    assert cloned_clf._core is None

    y_cls = np.random.choice([0, 1], size=50)
    cloned_clf.fit(X, y_cls)
    assert cloned_clf.is_fitted_ is True
    assert len(cloned_clf.predict(X)) == 50


def test_05_multiprocessing_cross_val_score(regression_data, classification_data):
    X_reg, y_reg = regression_data
    X_cls, y_cls = classification_data

    # Multi-process cross-validation (n_jobs=2) on Regressor
    reg = SubspaceRegressor(num_heads=2, head_circuits=64, alpha=5.0)
    scores_reg = cross_val_score(reg, X_reg, y_reg, cv=3, n_jobs=2)
    assert len(scores_reg) == 3
    # Check that scores are computed and finite
    assert not np.isnan(scores_reg).any()
    assert not np.isinf(scores_reg).any()

    # Multi-process cross-validation (n_jobs=2) on Classifier
    clf = SubspaceClassifier(num_heads=2, head_circuits=64, alpha=5.0)
    scores_cls = cross_val_score(clf, X_cls, y_cls, cv=3, n_jobs=2)
    assert len(scores_cls) == 3
    assert not np.isnan(scores_cls).any()
    assert (scores_cls >= 0.0).all() and (scores_cls <= 1.0).all()


def test_06_corrupt_payload_resilience(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X, y)

    # 1. Corrupted internal bincode byte buffer
    state = reg.__getstate__()
    state["_core_bytes"] = b"corrupted_garbage_bytes_0xDEADBEEF"

    corrupted_reg = SubspaceRegressor()
    with pytest.raises(ValueError, match="Corrupted or invalid state payload"):
        corrupted_reg.__setstate__(state)

    # 2. Truncated internal bincode byte buffer
    valid_bytes = state["_core_bytes"]
    state["_core_bytes"] = valid_bytes[: len(valid_bytes) // 4]
    with pytest.raises(ValueError, match="Corrupted or invalid state payload"):
        corrupted_reg.__setstate__(state)

    # 3. Unpickling corrupted pickle stream raises exception cleanly
    pickled = pickle.dumps(reg)
    corrupted_pickled = pickled[: len(pickled) // 2]
    with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
        pickle.loads(corrupted_pickled)


def test_07_continual_learning_post_unpickle(regression_data):
    X, y = regression_data
    n_initial = 150
    X_init, y_init = X[:n_initial], y[:n_initial]
    X_stream, y_stream = X[n_initial:], y[n_initial:]

    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X_init, y_init)
    assert reg.n_samples_seen_ == n_initial

    # Pickle roundtrip
    pickled = pickle.dumps(reg)
    restored = pickle.loads(pickled)
    assert restored.n_samples_seen_ == n_initial

    # Resume continual online streaming on restored model
    for i in range(len(X_stream)):
        restored.update_online(X_stream[i], y_stream[i])

    assert restored.n_samples_seen_ == len(X)
    preds = restored.predict(X_stream)
    assert len(preds) == len(X_stream)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
