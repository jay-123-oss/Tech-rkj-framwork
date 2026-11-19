from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

from tech_rkj._core import SubspaceRegressorCore


class SubspaceRegressor(BaseEstimator, RegressorMixin):
    """
    SubspaceRegressor: High-performance AVX2/NEON closed-form orthogonal subspace
    regressor with continual online learning and bit-exact state persistence.
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
        input_dim: Optional[int] = None,
    ):
        self.num_heads = num_heads
        self.head_circuits = head_circuits
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.low_freq_scale = low_freq_scale
        self.high_freq_scale = high_freq_scale
        self.frontend_mode = frontend_mode
        self.input_dim = input_dim

        self._core = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self.is_fitted_ = False

        if input_dim is not None:
            self._init_core(input_dim)

    def _init_core(self, input_dim: int):
        self._core = SubspaceRegressorCore(
            input_dim=input_dim,
            num_heads=self.num_heads,
            head_circuits=self.head_circuits,
            alpha=self.alpha,
            forgetting_factor=self.forgetting_factor,
            low_freq_scale=self.low_freq_scale,
            high_freq_scale=self.high_freq_scale,
            frontend_mode=self.frontend_mode,
        )
        self.n_features_in_ = input_dim

    def fit(self, X, y, sample_weight=None):
        X_arr, y_arr = check_X_y(
            X, y, accept_sparse=False, dtype=np.float32, y_numeric=True
        )
        y_arr = np.ascontiguousarray(y_arr, dtype=np.float64)

        n_samples, n_features = X_arr.shape
        self._init_core(n_features)

        if sample_weight is not None:
            w_arr = np.ascontiguousarray(sample_weight, dtype=np.float64)
            self._core.partial_fit(X_arr, y_arr, w_arr)
        else:
            self._core.fit(X_arr, y_arr)

        self.n_samples_seen_ = self._core.n_samples_seen
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, sample_weight=None):
        if self._core is None:
            X_arr, y_arr = check_X_y(
                X, y, accept_sparse=False, dtype=np.float32, y_numeric=True
            )
            self._init_core(X_arr.shape[1])
        else:
            X_arr, y_arr = check_X_y(
                X, y, accept_sparse=False, dtype=np.float32, y_numeric=True
            )
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
                )

        y_arr = np.ascontiguousarray(y_arr, dtype=np.float64)
        w_arr = (
            np.ascontiguousarray(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )
        self._core.partial_fit(X_arr, y_arr, w_arr)

        self.n_samples_seen_ = self._core.n_samples_seen
        self.is_fitted_ = True
        return self

    def update_online(self, x, y, sample_weight: float = 1.0):
        x_arr = np.ascontiguousarray(x, dtype=np.float32).ravel()
        if self._core is None:
            self._init_core(len(x_arr))
        elif len(x_arr) != self.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_features_in_}, got {len(x_arr)}"
            )

        self._core.update_online(x_arr, float(y), float(sample_weight))
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
        return self._core.predict(X_arr)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self, "is_fitted_")
        preds = self.predict(X)
        return r2_score(y, preds, sample_weight=sample_weight)

    def memory_mb(self) -> float:
        if self._core is not None:
            return self._core.memory_mb()
        return 0.0

    def save(self, path: Union[str, Path], checksum: str = "crc32") -> str:
        """Save this model to an .rkj binary checkpoint."""
        from tech_rkj.persistence import save as rkj_save

        return rkj_save(self, path, checksum=checksum)

    @classmethod
    def load(cls, path: Union[str, Path], use_mmap: bool = True) -> "SubspaceRegressor":
        """Load a SubspaceRegressor from an .rkj binary checkpoint."""
        from tech_rkj.persistence import load as rkj_load, MODEL_TYPE_REGRESSOR

        return rkj_load(path, expected_type=MODEL_TYPE_REGRESSOR, use_mmap=use_mmap)

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
        state = {
            "num_heads": self.num_heads,
            "head_circuits": self.head_circuits,
            "alpha": self.alpha,
            "forgetting_factor": self.forgetting_factor,
            "low_freq_scale": self.low_freq_scale,
            "high_freq_scale": self.high_freq_scale,
            "frontend_mode": self.frontend_mode,
            "input_dim": self.input_dim,
            "n_features_in_": self.n_features_in_,
            "n_samples_seen_": self.n_samples_seen_,
            "is_fitted_": self.is_fitted_,
            "_core_bytes": (
                self._core.__getstate__() if self._core is not None else None
            ),
        }
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.num_heads = state.get("num_heads", 4)
        self.head_circuits = state.get("head_circuits", 256)
        self.alpha = state.get("alpha", 10.0)
        self.forgetting_factor = state.get("forgetting_factor", 1.0)
        self.low_freq_scale = state.get("low_freq_scale", 0.12)
        self.high_freq_scale = state.get("high_freq_scale", 0.32)
        self.frontend_mode = state.get("frontend_mode", "zscore")
        self.input_dim = state.get("input_dim")
        self.n_features_in_ = state.get("n_features_in_")
        self.n_samples_seen_ = state.get("n_samples_seen_", 0)
        self.is_fitted_ = state.get("is_fitted_", False)

        core_bytes = state.get("_core_bytes")
        if core_bytes is not None and self.n_features_in_ is not None:
            self._init_core(self.n_features_in_)
            try:
                self._core.__setstate__(core_bytes)
            except Exception as e:
                raise ValueError(f"Corrupted or invalid state payload: {e}") from e
        else:
            self._core = None


AICoreRegressor = SubspaceRegressor
