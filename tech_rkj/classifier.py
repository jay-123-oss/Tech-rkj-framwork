import ctypes
import os
from typing import Optional
import numpy as np

_MODE_MAP = {"identity": 0, "zscore": 1, "vision_gabor": 2}


class SubspaceClassifier:
    """
    SubspaceClassifier: High-performance AVX2 closed-form and continual learning classifier in Rust.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 4,
        head_circuits: int = 384,
        alpha: float = 8.0,
        forgetting_factor: float = 1.0,
        low_freq_scale: float = 0.055,
        high_freq_scale: float = 0.115,
        frontend_mode: str = "identity",
        lib_path: Optional[str] = None,
    ):
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

        self._lib = ctypes.CDLL(lib_path)
        self._bind_signatures()
        self.input_dim = input_dim
        self.num_classes = num_classes

        mode_id = _MODE_MAP.get(frontend_mode, 0)
        self._handle = self._lib.aicore_create(
            input_dim,
            num_classes,
            num_heads,
            head_circuits,
            alpha,
            forgetting_factor,
            low_freq_scale,
            high_freq_scale,
            mode_id,
        )
        if not self._handle:
            raise RuntimeError("Failed to initialize Rust AICoreEngine")
        self.handle = self._handle

    def _bind_signatures(self):
        c_f32_p = ctypes.POINTER(ctypes.c_float)
        c_i64_p = ctypes.POINTER(ctypes.c_int64)

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
        self._lib.aicore_fit.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            c_i64_p,
            ctypes.c_size_t,
        ]
        self._lib.aicore_fit.restype = ctypes.c_int
        self._lib.aicore_update_online.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self._lib.aicore_predict_batch.argtypes = [
            ctypes.c_void_p,
            c_f32_p,
            c_i64_p,
            ctypes.c_size_t,
        ]
        self._lib.aicore_memory_mb.argtypes = [ctypes.c_void_p]
        self._lib.aicore_memory_mb.restype = ctypes.c_double

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        y_c = np.ascontiguousarray(y, dtype=np.int64)
        if len(X_c) != len(y_c):
            raise ValueError(
                f"Length mismatch: X has {len(X_c)} samples, y has {len(y_c)} samples"
            )
        rc = self._lib.aicore_fit(
            self._handle,
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(y_c),
        )
        if rc != 0:
            raise RuntimeError(f"aicore_fit failed with code {rc}")
        return self

    def update_online(
        self, x: np.ndarray, label: int, sample_weight: float = 1.0
    ):
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        self._lib.aicore_update_online(
            self._handle,
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(label),
            float(sample_weight),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        preds = np.empty(len(X_c), dtype=np.int64)
        self._lib.aicore_predict_batch(
            self._handle,
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(X_c),
        )
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(preds == y_arr))

    def memory_mb(self) -> float:
        if self._handle:
            return float(self._lib.aicore_memory_mb(self._handle))
        return 0.0

    def __del__(self):
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            self._lib.aicore_free(self._handle)
            self._handle = None
            self.handle = None


# Backward compatibility alias
AICoreClassifier = SubspaceClassifier

