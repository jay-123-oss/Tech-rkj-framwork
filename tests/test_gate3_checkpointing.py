"""
Phase 3 Gate Verification Test Suite: Native Checkpointer & State Persistence Engine (.rkj format).

Covers:
1. SubspaceRegressor .rkj persistence with CRC32 checksum.
2. SubspaceClassifier .rkj persistence with CRC32 checksum.
3. SHA-256 cryptographic checksum persistence and verification.
4. Memory-mapped (mmap) zero-copy vs direct file read equivalence.
5. Invalid magic header rejection (b"BAD\x00").
6. Model type mismatch rejection (Regressor vs Classifier).
7. Payload bit-flip corruption detection via checksum.
8. Truncated checkpoint file rejection.
9. Atomic write rollback safety (existing file untouched on write failure).
10. Continual online learning resumption post-.rkj restoration.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

import tech_rkj
from tech_rkj import SubspaceRegressor, SubspaceClassifier
from tech_rkj.persistence import (
    MAGIC_V2,
    HEADER_SIZE,
    MODEL_TYPE_REGRESSOR,
    MODEL_TYPE_CLASSIFIER,
)


@pytest.fixture
def regression_data():
    np.random.seed(101)
    n_samples = 150
    n_features = 8
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    w = np.random.randn(n_features)
    y = (X @ w + 0.05 * np.random.randn(n_samples)).astype(np.float64)
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(202)
    n_samples = 180
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.sum(X[:, :3], axis=1) > 0.0).astype(np.int64)
    return X, y


def test_01_regressor_save_load_crc32(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(
        num_heads=4,
        head_circuits=128,
        alpha=6.0,
        frontend_mode="zscore",
    )
    reg.fit(X, y)

    orig_preds = reg.predict(X)
    orig_p_inv = reg.p_inv_
    orig_weights = reg.weights_
    orig_samples = reg.n_samples_seen_

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        save_path = reg.save(tmp.name, checksum="crc32")
        assert Path(save_path).exists()

        # Load back via classmethod
        restored = SubspaceRegressor.load(save_path)

        # 1. Prediction bit-exact parity
        restored_preds = restored.predict(X)
        diff_preds = np.max(np.abs(orig_preds - restored_preds))
        assert diff_preds == 0.0, f"Predictions drift: {diff_preds}"
        assert np.array_equal(orig_preds, restored_preds)

        # 2. P_inv parity
        for h in range(len(orig_p_inv)):
            assert np.array_equal(orig_p_inv[h], restored.p_inv_[h])

        # 3. Weights parity
        for h in range(len(orig_weights)):
            assert np.array_equal(orig_weights[h], restored.weights_[h])

        # 4. Metadata parity
        assert restored.n_samples_seen_ == orig_samples
        assert restored.n_features_in_ == reg.n_features_in_
        assert restored.is_fitted_ is True


def test_02_classifier_save_load_crc32(classification_data):
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
    orig_classes = clf.classes_

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        save_path = clf.save(tmp.name, checksum="crc32")
        restored = SubspaceClassifier.load(save_path)

        assert np.array_equal(orig_preds, restored.predict(X))
        assert np.array_equal(orig_probas, restored.predict_proba(X))
        assert np.array_equal(orig_classes, restored.classes_)
        assert restored.n_classes_ == clf.n_classes_
        assert restored.is_fitted_ is True


def test_03_sha256_cryptographic_checksum(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X, y)
    orig_preds = reg.predict(X)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        save_path = reg.save(tmp.name, checksum="sha256")

        # Top-level tech_rkj.load
        loaded = tech_rkj.load(save_path)
        assert isinstance(loaded, SubspaceRegressor)
        assert np.array_equal(orig_preds, loaded.predict(X))


def test_04_mmap_vs_direct_read_parity(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=3, head_circuits=64).fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        reg.save(tmp.name)

        loaded_mmap = SubspaceRegressor.load(tmp.name, use_mmap=True)
        loaded_direct = SubspaceRegressor.load(tmp.name, use_mmap=False)

        preds_mmap = loaded_mmap.predict(X)
        preds_direct = loaded_direct.predict(X)
        assert np.array_equal(preds_mmap, preds_direct)


def test_05_invalid_magic_header_rejection():
    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        # Write corrupted header with bad magic bytes
        bad_header = b"BAD\x00" + (b"\x00" * 60)
        tmp.write(bad_header)
        tmp.flush()

        with pytest.raises(ValueError, match="Invalid magic header"):
            tech_rkj.load(tmp.name)


def test_06_model_type_mismatch_rejection(classification_data):
    X, y = classification_data
    clf = SubspaceClassifier(num_heads=2, head_circuits=64).fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        clf.save(tmp.name)

        # Attempt to load classifier checkpoint with SubspaceRegressor.load
        with pytest.raises(ValueError, match="Model type mismatch"):
            SubspaceRegressor.load(tmp.name)


def test_07_corrupted_payload_checksum_rejection(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        reg.save(tmp.name, checksum="crc32")

        with open(tmp.name, "rb") as f:
            content = bytearray(f.read())

        # Flip a bit in the payload (near the end of the file)
        content[-10] ^= 0xFF

        with open(tmp.name, "wb") as f:
            f.write(content)
            f.flush()

        with pytest.raises(ValueError, match="integrity check failed"):
            SubspaceRegressor.load(tmp.name)


def test_08_truncated_file_rejection(regression_data):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        reg.save(tmp.name)

        with open(tmp.name, "rb") as f:
            content = f.read()

        # 1. Truncate below HEADER_SIZE
        with open(tmp.name, "wb") as f:
            f.write(content[:32])
            f.flush()

        with pytest.raises(ValueError, match="smaller than header"):
            SubspaceRegressor.load(tmp.name)

        # 2. Truncate inside payload
        with open(tmp.name, "wb") as f:
            f.write(content[: len(content) - 100])
            f.flush()

        with pytest.raises(ValueError, match="truncated"):
            SubspaceRegressor.load(tmp.name)


def test_09_atomic_save_rollback_safety(regression_data, monkeypatch):
    X, y = regression_data
    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "checkpoint.rkj"

        # 1. Save valid initial model
        reg.save(target_file)
        orig_bytes = target_file.read_bytes()
        orig_mtime = target_file.stat().st_mtime_ns

        # 2. Create another model to save
        reg_new = SubspaceRegressor(num_heads=4, head_circuits=128).fit(X, y)

        # Simulate unexpected I/O error during os.fsync
        def mock_fsync(fd):
            raise OSError("Simulated disk write fault")

        monkeypatch.setattr("os.fsync", mock_fsync)

        with pytest.raises(OSError, match="Simulated disk write fault"):
            reg_new.save(target_file)

        # 3. Assert original file is 100% intact and untouched
        assert target_file.exists()
        assert target_file.read_bytes() == orig_bytes
        assert target_file.stat().st_mtime_ns == orig_mtime

        # Verify no orphan temporary files left behind
        temp_files = list(Path(tmpdir).glob(".*.tmp.*"))
        assert len(temp_files) == 0


def test_10_continual_learning_post_rkj_load(regression_data):
    X, y = regression_data
    n_split = 100
    X_train, y_train = X[:n_split], y[:n_split]
    X_stream, y_stream = X[n_split:], y[n_split:]

    reg = SubspaceRegressor(num_heads=2, head_circuits=64).fit(X_train, y_train)

    with tempfile.NamedTemporaryFile(suffix=".rkj") as tmp:
        reg.save(tmp.name)
        restored = SubspaceRegressor.load(tmp.name)

        # Perform streaming online updates on restored model
        for i in range(len(X_stream)):
            restored.update_online(X_stream[i], y_stream[i])

        assert restored.n_samples_seen_ == len(X)
        preds = restored.predict(X_stream)
        assert len(preds) == len(X_stream)
        assert not np.isnan(preds).any()
