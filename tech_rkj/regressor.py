import ctypes
import os
from typing import Optional
import numpy as np

_MODE_MAP = {"identity": 0, "zscore": 1, "vision_gabor": 2}


class SubspaceRegressor:
    """
    SubspaceRegressor: High-performance AVX2 closed-form orthogonal subspace regressor in Rust.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        num_heads: int = 4,
        head_circuits: int = 384,
        alpha: float = 5.0,
        forgetting_factor: float = 1.0,
        low_freq_scale: float = 0.12,
        high_freq_scale: float = 0.32,
        frontend_mode: str = "zscore",
        lib_path: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_circuits = head_circuits
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.low_freq_scale = low_freq_scale
        self.high_freq_scale = high_freq_scale
        self.frontend_mode = frontend_mode

        if lib_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            default_so = os.path.join(here, "bin", "libaicore_fw.so")
            fallback_so = os.path.join(here, "libaicore_fw.so")
            if os.path.exists(default_so):
                lib_path = default_so
            elif os.path.exists(fallback_so):
                lib_path = fallback_so
            else:
                lib_path = default_so

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Native binary '{lib_path}' not found. Please verify package installation."
            )

        self._lib_path = lib_path
        self._lib = ctypes.CDLL(lib_path)
        self._bind_signatures()
        self._handle = None
        self.handle = None

        if self.input_dim is not None:
            self._init_engine()

    def _bind_signatures(self):
        c_f32_p = ctypes.POINTER(ctypes.c_float)
        c_f64_p = ctypes.POINTER(ctypes.c_double)

        self._lib.aicore_create.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ]
        self._lib.aicore_create.restype = ctypes.c_void_p
        self._lib.aicore_free.argtypes = [ctypes.c_void_p]

        self._lib.aicore_fit_regression.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            c_f64_p,
            ctypes.c_size_t,
        ]
        self._lib.aicore_fit_regression.restype = ctypes.c_int

        self._lib.aicore_predict_regression_batch.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            c_f64_p,
            ctypes.c_size_t,
        ]

        self._lib.aicore_update_regression_online.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            ctypes.c_double,
            ctypes.c_double,
        ]

        self._lib.aicore_memory_mb.argtypes = [ctypes.c_void_p]
        self._lib.aicore_memory_mb.restype = ctypes.c_double

    def _init_engine(self):
        mode_id = _MODE_MAP.get(self.frontend_mode, 1)
        # num_classes = 1 for continuous regression
        self._handle = self._lib.aicore_create(
            self.input_dim,
            1,
            self.num_heads,
            self.head_circuits,
            self.alpha,
            self.forgetting_factor,
            self.low_freq_scale,
            self.high_freq_scale,
            mode_id,
        )
        if not self._handle:
            raise RuntimeError("Failed to initialize Rust AICoreEngine for regression")
        self.handle = self._handle

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        y_c = np.ascontiguousarray(y, dtype=np.float64)

        if X_c.ndim == 1:
            X_c = X_c.reshape(-1, 1)

        N, D = X_c.shape
        if self._handle is None or self.input_dim != D:
            if self._handle is not None:
                self._lib.aicore_free(self._handle)
                self._handle = None
            self.input_dim = D
            self._init_engine()

        if len(X_c) != len(y_c):
            raise ValueError(
                f"Length mismatch: X has {len(X_c)} samples, y has {len(y_c)} samples"
            )

        rc = self._lib.aicore_fit_regression(
            self._handle,
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            N,
        )
        if rc != 0:
            raise RuntimeError(f"aicore_fit_regression failed with code {rc}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._handle is None:
            raise RuntimeError("SubspaceRegressor is not fitted yet. Call .fit() first.")
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        if X_c.ndim == 1:
            X_c = X_c.reshape(-1, 1)
        N = len(X_c)
        preds = np.empty(N, dtype=np.float64)
        self._lib.aicore_predict_regression_batch(
            self._handle,
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            N,
        )
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R^2 (coefficient of determination) regression score.
        """
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot > 1e-12:
            return float(1.0 - (ss_res / ss_tot))
        return 0.0

    def update_online(self, x: np.ndarray, y: float, sample_weight: float = 1.0):
        if self._handle is None:
            raise RuntimeError("SubspaceRegressor is not fitted yet. Call .fit() first.")
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        self._lib.aicore_update_regression_online(
            self._handle,
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            float(y),
            float(sample_weight),
        )

    def memory_mb(self) -> float:
        if self._handle:
            return float(self._lib.aicore_memory_mb(self._handle))
        return 0.0

    def __del__(self):
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            self._lib.aicore_free(self._handle)
            self._handle = None
            self.handle = None


# Alias for backward compatibility
AICoreRegressor = SubspaceRegressor

