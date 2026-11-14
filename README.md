# tech-rkj

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/tech-rkj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-linux__x86__64-brightgreen.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/AVX2%20%2B%20FMA-Vectorized-blueviolet.svg)]()

**Deterministic AVX2 Closed-Form & Online Continual Learning Subspace Engine in Rust.**

`tech-rkj` delivers instant, single-pass closed-form training with zero backprop loops, alongside sub-30µs streaming continual learning without catastrophic forgetting.

---

## Key Features

- **Single-Pass Closed-Form Solve**: Direct Gauss-Jordan orthogonal subspace solver $(Z^T Z + \alpha I)^{-1} Z^T Y$ eliminates iterative backpropagation loops entirely.
- **Hardware-Vectorized AVX2 + FMA**: Rank-1 updates ($P^{-1} \leftarrow P^{-1} - \gamma k k^T$ and $W \leftarrow W + \gamma k e^T$) unrolled with 256-bit SIMD intrinsics.
- **Continual Streaming Learning (< 30µs)**: Real-time sample-by-sample recursive least squares (RLS) updates with zero heap allocation per stream step.
- **Zero-Day & Few-Shot Detection**: Novelty-aware learning gain floor avoids $P^{-1}$ saturation on rare classes after large batch fits.
- **Supervised Fisher / ANOVA Feature Weighting**: Automatic between-class to within-class variance ratio scaling for tabular datasets to surpass gradient boosted trees (e.g. XGBoost).
- **Scikit-Learn Compatible API**: Drop-in `.fit(X, y)`, `.predict(X)`, `.score(X, y)`, `.update_online(x, label, sample_weight)`, and `.memory_mb()`.

---

## Installation

```bash
pip install tech-rkj
```

*Requirements*: Linux (x86_64 with AVX2 support), Python >= 3.9, NumPy >= 1.21.

---

## Quickstart

### Batch Fitting & Evaluation
```python
import numpy as np
from sklearn.datasets import load_digits
from tech_rkj import SubspaceClassifier

# Load dataset
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
y = y.astype(np.int64)

# Split 80/20
X_train, X_test = X[:1400], X[1400:]
y_train, y_test = y[:1400], y[1400:]

# Initialize classifier
clf = SubspaceClassifier(
    input_dim=64,
    num_classes=10,
    num_heads=4,
    head_circuits=256,
    alpha=1.0,
    low_freq_scale=0.18,
    high_freq_scale=0.38,
)

# Single-pass fit (no backpropagation)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Memory Footprint: {clf.memory_mb():.2f} MB")
```

### Continual Online Streaming (< 30µs / sample)
```python
# Stream new incoming sample dynamically without retraining
clf.update_online(X_test[0], int(y_test[0]), sample_weight=1.0)

# High-priority / zero-day novelty sample (e.g. few-shot rare event)
clf.update_online(X_test[1], int(y_test[1]), sample_weight=5.0)
```

### Tabular Datasets with Fisher / ANOVA Feature Weighting
```python
# For tabular datasets with noisy or uninformative features
clf_tab = SubspaceClassifier(
    input_dim=32,
    num_classes=4,
    num_heads=4,
    head_circuits=384,
    alpha=0.5,
    low_freq_scale=0.15,
    high_freq_scale=0.35,
    frontend_mode="zscore",  # Enables ANOVA F-statistic feature weighting
)
clf_tab.fit(X_tabular_train, y_tabular_train)
```

---

## API Reference

### `SubspaceClassifier`

```python
SubspaceClassifier(
    input_dim: int,
    num_classes: int,
    num_heads: int = 4,
    head_circuits: int = 384,
    alpha: float = 8.0,
    forgetting_factor: float = 1.0,
    low_freq_scale: float = 0.055,
    high_freq_scale: float = 0.115,
    frontend_mode: str = "identity",  # "identity" | "zscore" | "vision_gabor"
    lib_path: Optional[str] = None,
)
```

- `fit(X, y)`: Fits the subspace model using single-pass closed-form Gauss-Jordan elimination.
- `predict(X)`: Predicts class labels for sample array `X`.
- `score(X, y)`: Computes mean classification accuracy on `(X, y)`.
- `update_online(x, label, sample_weight=1.0)`: Performs sub-30µs streaming RLS rank-1 update.
- `memory_mb()`: Returns the total internal memory footprint of the engine in megabytes.

*(Note: `AICoreClassifier` is retained as an alias for backward compatibility.)*

---

## Benchmark Highlights

- **Full MNIST 28x28 (784-D)**: Trained single-pass in ~2.1 seconds (**2.9x faster** than 10-epoch MLP backpropagation).
- **Continual Learning Retention**: **95.25%** base class retention after streaming completely new unseen classes.
- **Zero-Day Few-Shot Detection**: **82.14%** accuracy on 40 few-shot zero-day samples injected after 6,000 normal samples (with 99.00% normal class retention).
- **Tabular 32D Benchmark**: **84.90%** accuracy (exceeds XGBoost baseline of 83.83%).

---

## License

MIT License. Copyright (c) 2026 Jaydeep Prajapati.

