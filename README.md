# tech-rkj

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/tech-rkj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-brightgreen.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/SIMD-AVX2%20%7C%20FMA%20%7C%20AVX--512%20%7C%20NEON-blueviolet.svg)]()

**Production-Grade, Cross-Platform Deterministic ML + Continual Learning Framework in Rust & Python.**

`tech-rkj` delivers instant, single-pass closed-form training with zero backpropagation loops, alongside sub-10µs zero-allocation continual streaming learning without catastrophic forgetting. Built with PyO3 0.22, runtime multi-architecture SIMD detection (AVX-512, AVX2+FMA, ARM NEON), full scikit-learn estimator compliance, bit-exact `.rkj` container persistence, and seamless PyTorch deep learning hybrid integration.

---

## Key Features

- **Single-Pass Closed-Form Solve**: Solves $(Z^T Z + \alpha I)^{-1} Z^T Y$ across multi-head orthogonal random projections in parallel using Rayon and zero-allocation BLAS kernels.
- **Multi-Architecture SIMD Runtime**: Automatically detects CPU capabilities at runtime to select between AVX-512, AVX2+FMA (x86_64), ARM NEON (Apple Silicon / aarch64), or portable auto-vectorized scalar kernels.
- **Continual Streaming Learning (< 10µs / sample)**: Zero-allocation hot path recursive least squares (RLS) rank-1 updates with Cauchy-Schwarz bounded covariance stabilization.
- **Full Scikit-Learn Ecosystem Compliance**: Drop-in compatible with `Pipeline`, `GridSearchCV`, `cross_val_score`, `clone()`, `check_is_fitted`, and multi-process concurrency (`joblib`, `multiprocessing`).
- **Native Checkpointer Engine (`.rkj` v2.0)**: Fixed 64-byte aligned header container with magic bytes `b"RKJ\x02"`, CRC32 / SHA-256 payload integrity validation, memory-mapped (`mmap`) zero-copy deserialization, and atomic write-rename-replace rollback safety.
- **PyTorch Deep Learning Hybrid Head (`TorchSubspaceHead`)**: Attach directly to any PyTorch neural network backbone (CNN, ViT, MLP, Transformer) to replace gradient-descent output heads with closed-form, continual subspace heads.
- **Native Serialization**: Fully compatible with Python's standard `pickle`, `joblib`, and Python 3.9 through 3.13 via universal ABI3 wheels.

---

## Installation

```bash
pip install tech-rkj
```

To install optional PyTorch hybrid integration:
```bash
pip install "tech-rkj[torch]"
```

*Requirements*: Linux, macOS (Intel & Apple Silicon), or Windows; Python $\ge 3.9$; NumPy $\ge 1.21$.

---

## Quickstart

### 1. High-Performance Regression (`SubspaceRegressor`)
```python
import numpy as np
from tech_rkj import SubspaceRegressor

# Generate dataset
X = np.random.randn(1000, 20).astype(np.float32)
y = (X[:, 0] * 3.5 - X[:, 1] * 2.1 + 0.1 * np.random.randn(1000)).astype(np.float64)

# Initialize and fit in a single closed-form pass (0 backprop, 0 epochs)
reg = SubspaceRegressor(
    num_heads=4,
    head_circuits=256,
    alpha=10.0,
    frontend_mode="zscore",
)
reg.fit(X, y)

print(f"R^2 Score: {reg.score(X, y):.4f}")
print(f"Memory Footprint: {reg.memory_mb():.2f} MB")

# Sub-10µs online streaming update without retraining
reg.update_online(X[0], y[0], sample_weight=1.0)
```

### 2. Multi-Class Classification (`SubspaceClassifier`)
```python
from sklearn.datasets import load_digits
from tech_rkj import SubspaceClassifier

X, y = load_digits(return_X_y=True)

clf = SubspaceClassifier(
    num_heads=4,
    head_circuits=384,
    alpha=1.0,
    frontend_mode="zscore",
)
clf.fit(X, y)

print(f"Accuracy: {clf.score(X, y) * 100:.2f}%")
print(f"Predicted Probabilities:\n{clf.predict_proba(X[:2])}")
```

### 3. Scikit-Learn Pipelines & GridSearchCV
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tech_rkj import SubspaceRegressor

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SubspaceRegressor()),
])

param_grid = {
    "model__alpha": [1.0, 5.0, 10.0],
    "model__num_heads": [2, 4],
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=2)
grid.fit(X, y)
print(f"Best Parameters: {grid.best_params_}")
```

### 4. PyTorch Hybrid Deep Learning Head (`TorchSubspaceHead`)
```python
import torch
import torch.nn as nn
from tech_rkj import TorchSubspaceHead

# 1. Define neural backbone + closed-form subspace head
backbone = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
)
subspace_head = TorchSubspaceHead(
    in_features=32,
    out_features=10,
    task="classification",
    num_heads=4,
    head_circuits=256,
)

# 2. Extract deep embeddings through backbone
inputs = torch.randn(256, 64)
targets = torch.randint(0, 10, (256,))

with torch.no_grad():
    embeddings = backbone(inputs)

# 3. Instantaneous closed-form fitting on deep representations (0 backprop loops!)
subspace_head.fit(embeddings, targets)

# 4. Standard PyTorch forward pass returns torch.Tensor
logits = subspace_head(embeddings)
predictions = subspace_head.predict(embeddings)
```

### 5. Bit-Exact Checkpointing (`.rkj` Container Format)
```python
import tech_rkj
from tech_rkj import SubspaceRegressor

# Save model with hardware-accelerated CRC32 or cryptographic SHA-256 integrity check
reg.save("model.rkj", checksum="crc32")

# Fast zero-copy memory-mapped load
restored = SubspaceRegressor.load("model.rkj", use_mmap=True)

# Alternatively, load directly via top-level API
model = tech_rkj.load("model.rkj")
```

---

## Checkpoint Format Specification (`.rkj` v2.0)

| Offset | Length | Type | Description |
|---|---|---|---|
| `0..4` | 4 bytes | `[u8; 4]` | Magic header: `b"RKJ\x02"` |
| `4` | 1 byte | `u8` | Model Type: `0x01` = Regressor, `0x02` = Classifier |
| `5` | 1 byte | `u8` | Checksum Type: `0x01` = CRC32, `0x02` = SHA-256 |
| `6..8` | 2 bytes | `u16` | Reserved flags (`0x0000`) |
| `8..12` | 4 bytes | `u32` (LE) | Metadata JSON string byte length $M$ |
| `12..20` | 8 bytes | `u64` (LE) | Binary Bincode payload byte length $P$ |
| `20..52` | 32 bytes | `[u8; 32]` | Integrity checksum (CRC32 lower 4 bytes or SHA-256 32 bytes) |
| `52..64` | 12 bytes | `[u8; 12]` | Reserved padding (Header aligned to 64 bytes) |
| `64..64+M` | $M$ bytes | `UTF-8` | JSON Metadata Block (hyperparameters, schema, timestamps) |
| `64+M..` | $P$ bytes | Binary | Native Bincode serialized engine state |

---

## License

MIT License. Copyright (c) 2026 Jaydeep Prajapati.
