"""
PyTorch Deep Learning Hybrid Subspace Head for tech-rkj.

Attaches directly to any PyTorch neural network backbone (CNN, ViT, MLP,
Transformer) to replace traditional gradient-descent linear classification /
regression heads with an AVX2/NEON closed-form, sub-10us continual learning head.
"""

from typing import Optional, Union, Any
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    nn = object  # type: ignore

from tech_rkj.classifier import SubspaceClassifier
from tech_rkj.regressor import SubspaceRegressor
from tech_rkj.persistence import save as rkj_save, load as rkj_load


class TorchSubspaceHead(nn.Module if _TORCH_AVAILABLE else object):
    """
    TorchSubspaceHead: Deep Learning hybrid subspace continual learning head.

    Features:
    - Zero gradient backprop requirement: instantaneous closed-form fitting.
    - Zero-copy tensor interop: directly handles torch.Tensor on CPU.
    - Continual online learning: streaming updates without catastrophic forgetting.
    - Full PyTorch nn.Module hierarchy compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        task: str = "regression",
        num_heads: int = 4,
        head_circuits: int = 256,
        alpha: float = 10.0,
        forgetting_factor: float = 1.0,
        low_freq_scale: float = 0.12,
        high_freq_scale: float = 0.32,
        frontend_mode: str = "identity",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use TorchSubspaceHead. Install it via 'pip install torch'."
            )

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task.lower().strip()
        self.num_heads = num_heads
        self.head_circuits = head_circuits
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.low_freq_scale = low_freq_scale
        self.high_freq_scale = high_freq_scale
        self.frontend_mode = frontend_mode

        if self.task == "regression":
            self.model = SubspaceRegressor(
                num_heads=num_heads,
                head_circuits=head_circuits,
                alpha=alpha,
                forgetting_factor=forgetting_factor,
                low_freq_scale=low_freq_scale,
                high_freq_scale=high_freq_scale,
                frontend_mode=frontend_mode,
                input_dim=in_features,
            )
        elif self.task in ("classification", "classifier"):
            self.model = SubspaceClassifier(
                num_classes=out_features,
                num_heads=num_heads,
                head_circuits=head_circuits,
                alpha=alpha,
                forgetting_factor=forgetting_factor,
                low_freq_scale=low_freq_scale,
                high_freq_scale=high_freq_scale,
                frontend_mode=frontend_mode,
                input_dim=in_features,
            )
        else:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: 'regression', 'classification'."
            )

        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def _to_numpy(
        self, tensor: Any, dtype: np.dtype = np.float32
    ) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().astype(dtype, copy=False)
        return np.ascontiguousarray(tensor, dtype=dtype)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
    ) -> "TorchSubspaceHead":
        """Fit closed-form weights instantaneously on deep embeddings."""
        X_np = self._to_numpy(X, dtype=np.float32)
        y_dtype = np.float64 if self.task == "regression" else np.int64
        y_np = self._to_numpy(y, dtype=y_dtype)
        w_np = (
            self._to_numpy(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )

        self.model.fit(X_np, y_np, sample_weight=w_np)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        classes: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> "TorchSubspaceHead":
        """Stream new batch of embeddings into online subspace head."""
        X_np = self._to_numpy(X, dtype=np.float32)
        y_dtype = np.float64 if self.task == "regression" else np.int64
        y_np = self._to_numpy(y, dtype=y_dtype)
        w_np = (
            self._to_numpy(sample_weight, dtype=np.float64)
            if sample_weight is not None
            else None
        )

        if self.task == "regression":
            self.model.partial_fit(X_np, y_np, sample_weight=w_np)
        else:
            self.model.partial_fit(X_np, y_np, classes=classes, sample_weight=w_np)
        return self

    def update_online(
        self,
        x: Any,
        y: Union[float, int, Any],
        sample_weight: float = 1.0,
    ) -> "TorchSubspaceHead":
        """Single-sample continual update."""
        x_np = self._to_numpy(x, dtype=np.float32).ravel()
        if isinstance(y, torch.Tensor):
            y_val = y.item()
        else:
            y_val = y
        self.model.update_online(x_np, y_val, sample_weight=sample_weight)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch model pipelines.
        Returns:
            torch.Tensor: predicted continuous values for regression (B, 1) or
                          class probability distribution for classification (B, C).
        """
        device = x.device
        X_np = self._to_numpy(x, dtype=np.float32)

        if self.task == "regression":
            preds_np = self.model.predict(X_np)
            out_tensor = (
                torch.from_numpy(preds_np)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(-1)
            )
        else:
            probas_np = self.model.predict_proba(X_np)
            out_tensor = torch.from_numpy(probas_np).to(
                device=device, dtype=torch.float32
            )

        return out_tensor

    def predict(self, x: Any) -> torch.Tensor:
        """Return discrete class predictions or scalar regressions as a Torch Tensor."""
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        X_np = self._to_numpy(x, dtype=np.float32)
        preds_np = self.model.predict(X_np)
        return torch.from_numpy(preds_np).to(device=device)

    def predict_proba(self, x: Any) -> torch.Tensor:
        """Return class probability distributions as a Torch Tensor."""
        if self.task != "classification":
            raise AttributeError(
                "predict_proba is only available for classification task."
            )
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        X_np = self._to_numpy(x, dtype=np.float32)
        probas_np = self.model.predict_proba(X_np)
        return torch.from_numpy(probas_np).to(device=device, dtype=torch.float32)

    def save(self, path: Union[str, Path], checksum: str = "crc32") -> str:
        """Save underlying model to .rkj checkpoint."""
        return self.model.save(path, checksum=checksum)

    @classmethod
    def load(
        cls, path: Union[str, Path], use_mmap: bool = True
    ) -> "TorchSubspaceHead":
        """Load from .rkj checkpoint into a TorchSubspaceHead."""
        loaded_model = rkj_load(path, use_mmap=use_mmap)
        task = (
            "regression"
            if isinstance(loaded_model, SubspaceRegressor)
            else "classification"
        )
        head = cls(
            in_features=loaded_model.n_features_in_,
            out_features=getattr(loaded_model, "n_classes_", 1),
            task=task,
            num_heads=loaded_model.num_heads,
            head_circuits=loaded_model.head_circuits,
            alpha=loaded_model.alpha,
            forgetting_factor=loaded_model.forgetting_factor,
            low_freq_scale=loaded_model.low_freq_scale,
            high_freq_scale=loaded_model.high_freq_scale,
            frontend_mode=loaded_model.frontend_mode,
        )
        head.model = loaded_model
        return head
