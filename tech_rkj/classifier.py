from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

from tech_rkj._core import SubspaceClassifierCore


class SubspaceClassifier(BaseEstimator, ClassifierMixin):
    """
    SubspaceClassifier: High-performance AVX2/NEON closed-form orthogonal subspace
    classifier with continual online learning and bit-exact state persistence.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_circuits: int = 256,
        alpha: float = 10.0,
        forgetting_factor: float = 1.0,
        low_freq_scale: float = 0.12,
        high_freq_scale: float = 0.32,
        frontend_mode: str = "zscore",
        num_classes: Optional[int] = None,
        input_dim: Optional[int] = None,
    ):
        self.num_heads = num_heads
        self.head_circuits = head_circuits
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.low_freq_scale = low_freq_scale
        self.high_freq_scale = high_freq_scale
        self.frontend_mode = frontend_mode
        self.num_classes = num_classes
        self.input_dim = input_dim

        self._core = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self.is_fitted_ = False

        if input_dim is not None and num_classes is not None:
            self._init_core(input_dim, num_classes)

    def _init_core(self, input_dim: int, num_classes: int):
        self._core = SubspaceClassifierCore(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=self.num_heads,
            head_circuits=self.head_circuits,
            alpha=self.alpha,
            forgetting_factor=self.forgetting_factor,
            low_freq_scale=self.low_freq_scale,
            high_freq_scale=self.high_freq_scale,
            frontend_mode=self.frontend_mode,
        )
        self.n_features_in_ = input_dim
        self.n_classes_ = num_classes

    def fit(self, X, y, sample_weight=None):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=np.float32)

        n_samples, n_features = X_arr.shape
        classes, y_idx = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)
        if self.num_classes is not None and self.num_classes > n_classes:
            n_classes = self.num_classes

        self.n_classes_ = n_classes
        self._init_core(n_features, self.n_classes_)

        y_idx_arr = np.ascontiguousarray(y_idx, dtype=np.int64)
        if sample_weight is not None:
            w_arr = np.ascontiguousarray(sample_weight, dtype=np.float64)
            self._core.partial_fit(X_arr, y_idx_arr, w_arr)
        else:
            self._core.fit(X_arr, y_idx_arr)

        self.n_samples_seen_ = self._core.n_samples_seen
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self._core is None:
            X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
            if classes is None:
                if self.classes_ is not None:
                    classes = self.classes_
                elif self.num_classes is not None:
                    classes = np.arange(self.num_classes)
                else:
                    raise ValueError(
                        "classes must be passed on the first call to partial_fit."
                    )
            self.classes_ = np.asarray(classes)
            self.n_classes_ = len(self.classes_)
            self._init_core(X_arr.shape[1], self.n_classes_)
        else:
            X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
                )

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[val] for val in y_arr], dtype=np.int64)
        w_arr = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )

        self._core.partial_fit(X_arr, y_idx, w_arr)
        self.n_samples_seen_ = self._core.n_samples_seen
        self.is_fitted_ = True
        return self

    def update_online(self, x, label, sample_weight: float = 1.0):
        x_arr = np.ascontiguousarray(x, dtype=np.float32).ravel()
        if self._core is None:
            raise ValueError("Model is not fitted. Call fit or partial_fit first.")
        if len(x_arr) != self.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_features_in_}, got {len(x_arr)}"
            )

        if self.classes_ is not None:
            idx_matches = np.where(self.classes_ == label)[0]
            if len(idx_matches) == 0:
                raise ValueError(f"Label {label} not found in classes_ {self.classes_}")
            label_idx = int(idx_matches[0])
        else:
            label_idx = int(label)

        self._core.update_online(x_arr, label_idx, float(sample_weight))
        self.n_samples_seen_ = self._core.n_samples_seen
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X_arr = check_array(X, accept_sparse=False, dtype=np.float32)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
            )

        raw_preds = self._core.predict(X_arr)
        if self.classes_ is not None and len(self.classes_) > 0:
            return self.classes_[raw_preds]
        return raw_preds

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X_arr = check_array(X, accept_sparse=False, dtype=np.float32)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
            )
        return self._core.predict_proba(X_arr)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self, "is_fitted_")
        preds = self.predict(X)
        return accuracy_score(y, preds, sample_weight=sample_weight)

    def memory_mb(self) -> float:
        if self._core is not None:
            return self._core.memory_mb()
        return 0.0

    def save(self, path: Union[str, Path], checksum: str = "crc32") -> str:
        """Save this model to an .rkj binary checkpoint."""
        from tech_rkj.persistence import save as rkj_save

        return rkj_save(self, path, checksum=checksum)

    @classmethod
    def load(cls, path: Union[str, Path], use_mmap: bool = True) -> "SubspaceClassifier":
        """Load a SubspaceClassifier from an .rkj binary checkpoint."""
        from tech_rkj.persistence import load as rkj_load, MODEL_TYPE_CLASSIFIER

        return rkj_load(path, expected_type=MODEL_TYPE_CLASSIFIER, use_mmap=use_mmap)

    @property
    def p_inv_(self):
        """Return the P_inv covariance matrices across heads."""
        if self._core is None:
            return None
        return [self._core.get_p_inv(h) for h in range(self.num_heads)]

    @property
    def weights_(self):
        """Return the weight vectors across heads."""
        if self._core is None:
            return None
        return [self._core.get_weights(h) for h in range(self.num_heads)]

    # =========================================================================
    # SERIALIZATION / PICKLE / JOBLIB PROTOCOL
    # =========================================================================

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "num_heads": self.num_heads,
            "head_circuits": self.head_circuits,
            "alpha": self.alpha,
            "forgetting_factor": self.forgetting_factor,
            "low_freq_scale": self.low_freq_scale,
            "high_freq_scale": self.high_freq_scale,
            "frontend_mode": self.frontend_mode,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "n_features_in_": self.n_features_in_,
            "n_samples_seen_": self.n_samples_seen_,
            "is_fitted_": self.is_fitted_,
            "_core_bytes": (
                self._core.__getstate__() if self._core is not None else None
            ),
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.num_heads = state.get("num_heads", 4)
        self.head_circuits = state.get("head_circuits", 256)
        self.alpha = state.get("alpha", 10.0)
        self.forgetting_factor = state.get("forgetting_factor", 1.0)
        self.low_freq_scale = state.get("low_freq_scale", 0.12)
        self.high_freq_scale = state.get("high_freq_scale", 0.32)
        self.frontend_mode = state.get("frontend_mode", "zscore")
        self.num_classes = state.get("num_classes")
        self.input_dim = state.get("input_dim")
        self.classes_ = state.get("classes_")
        self.n_classes_ = state.get("n_classes_")
        self.n_features_in_ = state.get("n_features_in_")
        self.n_samples_seen_ = state.get("n_samples_seen_", 0)
        self.is_fitted_ = state.get("is_fitted_", False)

        core_bytes = state.get("_core_bytes")
        if (
            core_bytes is not None
            and self.n_features_in_ is not None
            and self.n_classes_ is not None
        ):
            self._init_core(self.n_features_in_, self.n_classes_)
            try:
                self._core.__setstate__(core_bytes)
            except Exception as e:
                raise ValueError(f"Corrupted or invalid state payload: {e}") from e
        else:
            self._core = None


AICoreClassifier = SubspaceClassifier
