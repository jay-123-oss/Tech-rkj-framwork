# Tech-RKJ (`tech-rkj`)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/tech-rkj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Rust Engine](https://img.shields.io/badge/Core-Rust%20AVX2%20%2B%20FMA-orange.svg)](#architecture)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](#installation)

**Deterministic AVX2/FMA Closed-Form & Online Continual Learning Framework in Rust**

`tech-rkj` is a zero-backpropagation machine learning and continual-learning runtime written in **Rust** (`x86-64-v3` AVX2 + FMA SIMD kernels) with a clean, scikit-learn-compatible Python API (`SubspaceClassifier` & `SubspaceRegressor`).

Instead of iterative gradient descent or static decision trees that require full dataset retraining, `tech-rkj` solves optimal weights analytically in a **single closed-form pass** and adapts to live data streams in **~105 µs/sample** using full-matrix **Sherman-Morrison Recursive Least Squares (RLS)** — preventing catastrophic forgetting without replay buffers.

---

## Key Capabilities

- **Single-Pass Closed-Form Solver:** Computes exact ridge-regularized subspace projections $(Z^T Z + \alpha I)^{-1} Z^T Y$ via partial-pivoting Gauss-Jordan elimination in one step — zero epochs, zero learning-rate tuning.
- **True Online Continual Learning (`update_online`):** Absorbs new samples and brand-new unseen classes one sample at a time in **~105 µs** with near-zero forgetting (`< 1%` drop on base classes).
- **Novelty-Aware Few-Shot Adaptation:** Automatically prevents precision matrix ($P^{-1}$) starvation when rare or zero-day patterns arrive after large batch fits.
- **Built-in Feature Frontends:**
  - `identity`: Direct pass-through for normalized signals and frozen Deep Learning backbone embeddings (`ResNet`, `CLIP`, `MiniLM`).
  - `zscore`: Supervised **ANOVA F-statistic / Correlation Feature Weighting** + standardization that suppresses noisy tabular columns.
  - `vision_gabor`: Multi-orientation spatial Gabor filter bank for 2D image patches.
- **Ultra-Low Footprint:** Deterministic `< 5 MB` RAM footprint and `0.23 J` energy consumption, engineered for edge CPUs, robotics telemetry, and real-time streaming pipelines.

---

## Installation

Install directly from PyPI (pre-compiled universal `manylinux` wheel with bundled AVX2/FMA runtime — no Rust compiler required):

```bash
pip install tech-rkj
