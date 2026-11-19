# tech-rkj

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/tech-rkj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/jay-123-oss/Tech-rkj-framwork/actions/workflows/wheels.yml/badge.svg)](https://github.com/jay-123-oss/Tech-rkj-framwork/actions/workflows/wheels.yml)
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
- **Cross-Platform ABI3 Wheels**: Pre-compiled for Linux (x86_64, aarch64), macOS Universal2 (Apple Silicon M1-M4 & Intel x86_64), and Windows x64.

---

## Installation & Wheel Support Matrix

```bash
pip install tech-rkj
```

To install optional PyTorch hybrid integration:
```bash
pip install "tech-rkj[torch]"
```

### Verified Multi-Platform Wheel Distribution

| OS Platform | Architecture | Target Triple | Wheel Tag | CI Status |
|---|---|---|---|---|
| **Linux** | x86_64 (Intel / AMD) | `x86_64-unknown-linux-gnu` | `cp39-abi3-manylinux_2_34_x86_64` | `Passing` |
| **Linux** | aarch64 (ARM64) | `aarch64-unknown-linux-gnu` | `cp39-abi3-manylinux_2_34_aarch64` | `Passing` |
| **macOS** | Universal2 (Apple Silicon M1-M4 + Intel) | `universal2-apple-darwin` | `cp39-abi3-macosx_universal2` | `Passing` |
| **Windows** | x64 (MSVC) | `x86_64-pc-windows-msvc` | `cp39-abi3-win_amd64` | `Passing` |
| **Any (Source)** | Sdist (Crates.io + PyO3) | Source Tree | `tech_rkj-0.2.0.tar.gz` | `Passing` |

*Compatibility*: Python 3.9, 3.10, 3.11, 3.12, and 3.13 via Python Stable ABI (`abi3`).

---

## Empirical Benchmarks & Numerical Guarantees

### 1. Collinear Feature Stability ($r = 0.90$ to $0.9999$)
In severe ill-conditioned collinear settings where feature correlation reaches $r = 0.9999$ (Gram matrix condition number $\kappa(X^T X) \approx 7.58 \times 10^4$), `SubspaceRegressor` preserves full correlation structure without eigenvalue collapse:

| Correlation Level | Gram Condition $\kappa(X^T X)$ | Scipy OLS $R^2$ | Sklearn Ridge $R^2$ | SubspaceRegressor $R^2$ | Prediction Corr vs Ridge | Min Eigenvalue $\lambda_{\min}(P)$ |
|---|---|---|---|---|---|---|
| $r = 0.9000$ | $7.99 \times 10^1$ | 0.999720 | 0.999719 | **0.996537** | **0.998414** | $4.15 \times 10^{-5}$ |
| $r = 0.9900$ | $7.60 \times 10^2$ | 0.999695 | 0.999655 | **0.998179** | **0.999239** | $3.83 \times 10^{-5}$ |
| $r = 0.9990$ | $7.58 \times 10^3$ | 0.999695 | 0.999505 | **0.998485** | **0.999429** | $3.80 \times 10^{-5}$ |
| $r = 0.9999$ | $7.58 \times 10^4$ | 0.999696 | 0.999620 | **0.998727** | **0.999480** | $3.79 \times 10^{-5}$ |

### 2. Long-Horizon Continual Streaming Stability (1,000 Collinear Updates, $r = 0.999$)
Under continuous sample-by-sample Sherman-Morrison rank-1 updates, Cauchy-Schwarz bounding ($|P_{ij}| \le \sqrt{P_{ii} P_{jj}}$) prevents finite-precision floating point divergence, keeping condition numbers strictly bounded:

| Stream Step | Test Set $R^2$ | Min Eigenvalue $\lambda_{\min}(P)$ | Max Eigenvalue $\lambda_{\max}(P)$ | Condition Number $\kappa(P)$ | Numerical Drift |
|---|---|---|---|---|---|
| **Step 200** | 0.3039 | $1.40 \times 10^{-4}$ | $1.000000$ | $7.14 \times 10^3$ | $0.00$ |
| **Step 400** | 0.3698 | $6.91 \times 10^{-5}$ | $0.999999$ | $1.45 \times 10^4$ | $0.00$ |
| **Step 600** | 0.4038 | $4.64 \times 10^{-5}$ | $0.999999$ | $2.16 \times 10^4$ | $0.00$ |
| **Step 800** | 0.4369 | $3.48 \times 10^{-5}$ | $0.999998$ | $2.87 \times 10^4$ | $0.00$ |
| **Step 1000** | 0.4324 | $2.78 \times 10^{-5}$ | $0.999998$ | $3.59 \times 10^4$ | $0.00$ |

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
