"""
Phase 4 Gate Verification Test Suite: Unified scikit-learn + Deep Learning API.

Covers:
1. scikit-learn Pipeline with StandardScaler + SubspaceRegressor.
2. scikit-learn Pipeline with StandardScaler + SubspaceClassifier.
3. GridSearchCV hyperparameter tuning on SubspaceRegressor.
4. GridSearchCV hyperparameter tuning on SubspaceClassifier.
5. Sample weight weighting support in fit() and partial_fit().
6. Input validation (check_X_y, check_array, NaN/Inf rejection).
7. PyTorch Hybrid Head (TorchSubspaceHead) in an nn.Sequential backbone.
8. PyTorch Hybrid Head classification with probability distribution.
9. PyTorch Hybrid Head continual streaming online learning.
10. PyTorch Hybrid Head .rkj save/load roundtrip persistence.
"""

import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

from tech_rkj import SubspaceRegressor, SubspaceClassifier, TorchSubspaceHead


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n_samples = 150
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + 0.1 * np.random.randn(n_samples)).astype(
        np.float64
    )
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(99)
    n_samples = 180
    n_features = 8
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)
    return X, y


def test_01_sklearn_pipeline_regressor(regression_data):
    X, y = regression_data
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", SubspaceRegressor(num_heads=2, head_circuits=64, alpha=5.0)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()

    score = pipe.score(X, y)
    assert score > 0.80, f"Pipeline score too low: {score}"


def test_02_sklearn_pipeline_classifier(classification_data):
    X, y = classification_data
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SubspaceClassifier(num_heads=2, head_circuits=64, alpha=5.0)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)

    probas = pipe.predict_proba(X)
    assert probas.shape == (len(y), 2)
    # Check probabilities sum to 1.0
    assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    score = pipe.score(X, y)
    assert score > 0.85, f"Pipeline accuracy too low: {score}"


def test_03_gridsearch_cv_regressor(regression_data):
    X, y = regression_data
    param_grid = {
        "alpha": [1.0, 10.0],
        "num_heads": [2, 4],
    }
    grid = GridSearchCV(
        SubspaceRegressor(head_circuits=64),
        param_grid=param_grid,
        cv=3,
        n_jobs=2,
    )
    grid.fit(X, y)

    assert grid.best_estimator_ is not None
    assert grid.best_score_ is not None
    preds = grid.predict(X)
    assert len(preds) == len(y)


def test_04_gridsearch_cv_classifier(classification_data):
    X, y = classification_data
    param_grid = {
        "alpha": [2.0, 8.0],
        "num_heads": [2, 4],
    }
    grid = GridSearchCV(
        SubspaceClassifier(head_circuits=64),
        param_grid=param_grid,
        cv=3,
        n_jobs=2,
    )
    grid.fit(X, y)

    assert grid.best_estimator_ is not None
    assert grid.best_score_ is not None
    preds = grid.predict(X)
    assert len(preds) == len(y)


def test_05_sample_weight_support(regression_data):
    X, y = regression_data
    weights = np.ones(len(y), dtype=np.float64)
    # Heavily weight the first 10 samples
    weights[:10] = 50.0

    reg = SubspaceRegressor(num_heads=2, head_circuits=64)
    reg.fit(X, y, sample_weight=weights)

    assert reg.is_fitted_ is True
    preds = reg.predict(X)
    assert len(preds) == len(y)

    # Partial fit with weights
    reg.partial_fit(X[:10], y[:10], sample_weight=weights[:10])
    assert reg.n_samples_seen_ == len(y) + 10


def test_06_input_validation_and_nan_rejection(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor()

    # 1. NaN in features rejected
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        reg.fit(X_nan, y)

    # 2. Inf in features rejected
    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="contains infinity"):
        reg.fit(X_inf, y)

    # 3. 1D X rejected
    with pytest.raises(ValueError, match="Expected 2D array"):
        reg.fit(X[:, 0], y)

    # 4. Dimension mismatch during predict rejected
    reg.fit(X, y)
    with pytest.raises(ValueError, match="Feature mismatch"):
        reg.predict(X[:, : X.shape[1] - 1])


def test_07_torch_subspace_head_hybrid_backbone(regression_data):
    X, y = regression_data
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    # Construct deep PyTorch backbone + closed-form subspace head
    in_dim = X.shape[1]
    hidden_dim = 16
    backbone = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
    )
    head = TorchSubspaceHead(
        in_features=hidden_dim,
        out_features=1,
        task="regression",
        num_heads=4,
        head_circuits=64,
    )

    # Extract embeddings through PyTorch backbone
    with torch.no_grad():
        embeddings = backbone(X_t)

    # Instantaneous closed-form fitting on deep embeddings (0 backprop, 0 epochs)
    head.fit(embeddings, y_t)

    # PyTorch forward pass returns torch.Tensor
    preds_t = head(embeddings)
    assert isinstance(preds_t, torch.Tensor)
    assert preds_t.shape == (len(y), 1)

    # Predict tensor method
    preds_direct = head.predict(embeddings)
    assert isinstance(preds_direct, torch.Tensor)
    assert np.allclose(preds_t.squeeze().numpy(), preds_direct.numpy(), atol=1e-5)


def test_08_torch_subspace_head_classification(classification_data):
    X, y = classification_data
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    head = TorchSubspaceHead(
        in_features=X.shape[1],
        out_features=2,
        task="classification",
        num_heads=3,
        head_circuits=64,
    )

    head.fit(X_t, y_t)

    # Forward pass returns probability distribution tensor
    probas = head(X_t)
    assert isinstance(probas, torch.Tensor)
    assert probas.shape == (len(y), 2)
    assert torch.allclose(torch.sum(probas, dim=1), torch.ones(len(y)), atol=1e-5)

    # Predict returns discrete class labels tensor
    preds = head.predict(X_t)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (len(y),)
    # Predicted discrete class should match argmax of probabilities
    assert torch.equal(preds, torch.argmax(probas, dim=1))


def test_09_torch_subspace_head_continual_learning(regression_data):
    X, y = regression_data
    n_init = 100
    X_init, y_init = torch.from_numpy(X[:n_init]), torch.from_numpy(y[:n_init])
    X_stream, y_stream = torch.from_numpy(X[n_init:]), torch.from_numpy(y[n_init:])

    head = TorchSubspaceHead(
        in_features=X.shape[1],
        out_features=1,
        task="regression",
        num_heads=2,
        head_circuits=64,
    )
    head.fit(X_init, y_init)

    # Stream continual updates on new samples
    for i in range(len(X_stream)):
        head.update_online(X_stream[i], y_stream[i])

    assert head.model.n_samples_seen_ == len(X)
    preds = head(X_stream)
    assert len(preds) == len(X_stream)
    assert not torch.isnan(preds).any()


def test_10_torch_subspace_head_rkj_persistence(regression_data):
    X, y = regression_data
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    head = TorchSubspaceHead(
        in_features=X.shape[1],
        out_features=1,
        task="regression",
        num_heads=2,
        head_circuits=64,
    )
    head.fit(X_t, y_t)
    orig_preds = head(X_t)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        head.save(tmp.name)
        restored_head = TorchSubspaceHead.load(tmp.name)

        restored_preds = restored_head(X_t)
        assert torch.equal(orig_preds, restored_preds)
