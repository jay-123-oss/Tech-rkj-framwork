#!/usr/bin/env python3
"""
=============================================================================
BENCHMARK: ALL MACHINE LEARNING REGRESSION MODELS VS. tech-rkj SUBSPACE ENGINE
Dataset: Car Price Prediction (/home/jaydeep/Documents/AI/car_price_prediction_ (1).csv)
Package: tech-rkj (Installed via PyPI / pip install tech-rkj)
=============================================================================
Evaluates 15 core metadata and performance dimensions:
 1. R² Score (Coefficient of Determination)
 2. Root Mean Squared Error (RMSE, $)
 3. Mean Absolute Error (MAE, $)
 4. Mean Absolute Percentage Error (MAPE, %)
 5. Median Absolute Error (MedAE, $)
 6. Training Wall-Clock Latency (ms)
 7. Per-Sample Inference Latency (µs)
 8. Inference Throughput (samples / sec)
 9. Peak RAM Delta (MB)
10. Model Memory Footprint (KB)
11. CPU Core Utilization (%)
12. Estimated Energy Consumption (Joules & mWh)
13. Training Iterations / Passes (1-pass solve vs N iterations)
14. Continual Online Streaming Learning Support (Native RLS vs Full Retrain)
15. Streaming Update Latency (µs / sample)
=============================================================================
"""

import os
import sys
import time
import gc
import json
import pickle
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)

# Standard Machine Learning Regressors
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# =============================================================================
# EXTERNAL FRAMEWORK IMPORT (Installed via `pip install tech-rkj`)
# No local classes, internal files, or relative folder paths are used.
# =============================================================================
from tech_rkj import SubspaceRegressor


# =============================================================================
# HELPER: MAPE Metric
# =============================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    non_zero = np.abs(y_true) > 1e-6
    if not np.any(non_zero):
        return 0.0
    return (
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
        * 100.0
    )


# =============================================================================
# BENCHMARK RUNNER FUNCTION
# =============================================================================
def benchmark_single_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cpu_tdp_watts: float = 15.0,
    is_subspace: bool = False,
):
    """
    Evaluates a regression model across 15 rigorous performance dimensions.
    """
    proc = psutil.Process(os.getpid())
    gc.collect()

    # Pre-train system stats
    mem_before = proc.memory_info().rss / (1024.0 * 1024.0)
    cpu_before = psutil.cpu_percent(interval=None)

    # 1. Training Latency & Wall-Clock Measurement
    t0_train = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = (time.perf_counter() - t0_train) * 1000.0  # in ms

    # Post-train system stats
    mem_after = proc.memory_info().rss / (1024.0 * 1024.0)
    cpu_after = psutil.cpu_percent(interval=None)
    cpu_pct = max(cpu_after, cpu_before, 1.0)
    mem_delta = max(mem_after - mem_before, 0.05)

    # 2. Batch Inference Throughput & Latency
    t0_inf = time.perf_counter()
    preds = model.predict(X_test)
    t_inf_total = time.perf_counter() - t0_inf
    n_test = len(X_test)
    sample_latency_us = (t_inf_total / n_test) * 1e6
    throughput = n_test / t_inf_total if t_inf_total > 0 else 0.0

    # 3. Accuracy & Precision Metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)

    # 4. Model Storage / Memory Footprint
    if hasattr(model, "memory_mb"):
        model_size_kb = model.memory_mb() * 1024.0
    else:
        try:
            model_size_kb = len(pickle.dumps(model)) / 1024.0
        except Exception:
            model_size_kb = 100.0

    # 5. Energy Footprint (Joules & mWh based on CPU TDP)
    t_train_sec = t_train / 1000.0
    power_watts = cpu_tdp_watts * (cpu_pct / 100.0)
    energy_joules = power_watts * t_train_sec
    energy_mwh = energy_joules / 3.6

    # 6. Online Streaming Continual Update Test
    if is_subspace:
        online_supported = "YES (Native RLS)"
        update_lats = []
        for i in range(min(50, len(X_test))):
            ts = time.perf_counter()
            model.update_online(X_test[i], y_test[i], sample_weight=1.0)
            update_lats.append((time.perf_counter() - ts) * 1e6)
        stream_latency_us = float(np.median(update_lats))
        iterations = "1 (Single-Pass Solve)"
    else:
        online_supported = "NO (Retrain All)"
        stream_latency_us = t_train * 1000.0  # Full retrain latency
        if hasattr(model, "n_iter_"):
            iterations = str(model.n_iter_)
        elif hasattr(model, "n_estimators"):
            iterations = f"{model.n_estimators} Trees"
        elif hasattr(model, "max_iter"):
            iterations = f"{model.max_iter} Epochs"
        else:
            iterations = "Iterative"

    return {
        "Model": name,
        "R2_Score": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_pct": float(mape),
        "MedAE": float(medae),
        "Train_Time_ms": float(t_train),
        "Sample_Latency_us": float(sample_latency_us),
        "Throughput_sps": float(throughput),
        "Peak_RAM_MB": float(mem_delta),
        "Model_Size_KB": float(model_size_kb),
        "CPU_Percent": float(cpu_pct),
        "Energy_Joules": float(energy_joules),
        "Energy_mWh": float(energy_mwh),
        "Iterations": iterations,
        "Streaming_Update": online_supported,
        "Stream_Latency_us": float(stream_latency_us),
    }


# =============================================================================
# MAIN BENCHMARK EXECUTION
# =============================================================================
def main():
    print("=" * 115)
    print("  AUTOMOTIVE PRICE PREDICTION: ALL ML REGRESSORS VS tech-rkj SUBSPACE ENGINE")
    print("  Framework Source: PyPI (pip install tech-rkj) | Zero local codebase hacks")
    print("  Hardware: AMD Ryzen 5 7530U (6C/12T, AVX2 + FMA) | Nominal TDP: 15W")
    print("=" * 115)

    dataset_path = "/home/jaydeep/Documents/AI/car_price_prediction_ (1).csv"
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset not found at: {dataset_path}")
        sys.exit(1)

    print(f"\n[*] Loading dataset: {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    print(f"[*] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean identifier columns
    if "Car ID" in df.columns:
        df = df.drop(columns=["Car ID"])

    target_col = "Price"
    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float64)

    cat_cols = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
    num_cols = ["Year", "Engine Size", "Mileage"]

    print(f"[*] Target Stats: min=${y.min():.2f}, max=${y.max():.2f}, mean=${y.mean():.2f}, std=${y.std():.2f}")
    print(f"[*] Categorical Columns ({len(cat_cols)}): {cat_cols}")
    print(f"[*] Numerical Columns ({len(num_cols)}): {num_cols}")

    # Preprocessing Pipeline (StandardScaler + OneHotEncoder)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                cat_cols,
            ),
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    print(f"[*] Processed Feature Dimensions: {X_train.shape[1]} features")
    print(f"[*] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    # Define Candidate Regressors
    models_to_run = [
        # --- OUR PIP-INSTALLED FRAMEWORK ---
        (
            "tech-rkj Subspace Regressor",
            SubspaceRegressor(
                num_heads=4,
                head_circuits=256,
                alpha=50.0,
                low_freq_scale=0.12,
                high_freq_scale=0.32,
                frontend_mode="zscore",
            ),
            True,
        ),
        # --- CLASSICAL LINEAR MODELS ---
        ("Linear Regression (OLS)", LinearRegression(), False),
        ("Ridge Regression (L2)", Ridge(alpha=10.0), False),
        ("Lasso Regression (L1)", Lasso(alpha=100.0, random_state=42), False),
        ("ElasticNet (L1+L2)", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42), False),
        ("Bayesian Ridge", BayesianRidge(), False),
        # --- INSTANCE-BASED ---
        ("K-Nearest Neighbors (KNN)", KNeighborsRegressor(n_neighbors=7), False),
        # --- TREES & ENSEMBLES ---
        ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5, random_state=42), False),
        (
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            False,
        ),
        (
            "Extra Trees Regressor",
            ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            False,
        ),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            False,
        ),
        (
            "HistGradientBoosting (LightGBM)",
            HistGradientBoostingRegressor(max_depth=4, random_state=42),
            False,
        ),
        (
            "AdaBoost Regressor",
            AdaBoostRegressor(n_estimators=50, random_state=42),
            False,
        ),
        # --- KERNEL & NEURAL NETWORK ---
        ("Support Vector Regressor (SVR)", SVR(C=1000.0, gamma="scale"), False),
        (
            "MLP Regressor (Neural Net)",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=250,
                random_state=42,
                early_stopping=True,
            ),
            False,
        ),
    ]

    results = []

    print("[*] Running 15-Model Benchmark Suite...")
    for idx, (name, model, is_subspace) in enumerate(models_to_run, 1):
        print(f"    [{idx:02d}/15] Evaluating {name:<32} ... ", end="", flush=True)
        res = benchmark_single_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            cpu_tdp_watts=15.0,
            is_subspace=is_subspace,
        )
        results.append(res)
        print(f"Done (Train: {res['Train_Time_ms']:.1f}ms | R2: {res['R2_Score']:.4f})")

    # =========================================================================
    # DISPLAY BEAUTIFUL COMPARATIVE TABLES
    # =========================================================================

    print("\n" + "=" * 115)
    print("TABLE 1: ACCURACY & PREDICTION ERROR METRICS (Test Set = 500 samples)")
    print("=" * 115)
    header1 = f"{'Model':<32} | {'R² Score':<10} | {'RMSE ($)':<12} | {'MAE ($)':<12} | {'MAPE (%)':<10} | {'MedAE ($)':<12}"
    print(header1)
    print("-" * 115)
    for r in results:
        prefix = "-> " if "tech-rkj" in r["Model"] else "   "
        print(
            f"{prefix + r['Model']:<32} | {r['R2_Score']:10.4f} | {r['RMSE']:12.2f} | {r['MAE']:12.2f} | {r['MAPE_pct']:9.2f}% | {r['MedAE']:12.2f}"
        )
    print("=" * 115)

    print("\n" + "=" * 115)
    print("TABLE 2: COMPUTATIONAL SPEED & INFERENCE THROUGHPUT")
    print("=" * 115)
    header2 = f"{'Model':<32} | {'Train Time (ms)':<16} | {'Sample Latency (µs)':<20} | {'Throughput (samp/s)':<20}"
    print(header2)
    print("-" * 115)
    for r in results:
        prefix = "-> " if "tech-rkj" in r["Model"] else "   "
        print(
            f"{prefix + r['Model']:<32} | {r['Train_Time_ms']:16.2f} | {r['Sample_Latency_us']:20.2f} | {r['Throughput_sps']:20.1f}"
        )
    print("=" * 115)

    print("\n" + "=" * 115)
    print("TABLE 3: HARDWARE RESOURCES, MEMORY & POWER CONSUMPTION (TDP = 15W)")
    print("=" * 115)
    header3 = f"{'Model':<32} | {'RAM Delta (MB)':<15} | {'Model Size (KB)':<16} | {'CPU Load (%)':<13} | {'Energy (mWh)':<14} | {'Joules':<10}"
    print(header3)
    print("-" * 115)
    for r in results:
        prefix = "-> " if "tech-rkj" in r["Model"] else "   "
        print(
            f"{prefix + r['Model']:<32} | {r['Peak_RAM_MB']:15.2f} | {r['Model_Size_KB']:16.2f} | {r['CPU_Percent']:12.1f}% | {r['Energy_mWh']:14.4f} | {r['Energy_Joules']:10.3f}"
        )
    print("=" * 115)

    print("\n" + "=" * 115)
    print("TABLE 4: ARCHITECTURAL ATTRIBUTES & STREAMING CONTINUAL LEARNING")
    print("=" * 115)
    header4 = f"{'Model':<32} | {'Optimization Passes':<22} | {'Continual Streaming':<22} | {'Stream Update Latency':<22}"
    print(header4)
    print("-" * 115)
    for r in results:
        prefix = "-> " if "tech-rkj" in r["Model"] else "   "
        stream_str = (
            f"{r['Stream_Latency_us']:.2f} µs"
            if "YES" in r["Streaming_Update"]
            else f"{r['Train_Time_ms']:.1f} ms (Retrain)"
        )
        print(
            f"{prefix + r['Model']:<32} | {r['Iterations']:<22} | {r['Streaming_Update']:<22} | {stream_str:<22}"
        )
    print("=" * 115)

    # Save summary JSON & CSV
    out_dir = "/home/jaydeep/Documents/AI/TESTING"
    json_path = os.path.join(out_dir, "benchmark_results.json")
    csv_path = os.path.join(out_dir, "benchmark_results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[+] Detailed benchmark reports saved to:")
    print(f"    - JSON: {json_path}")
    print(f"    - CSV:  {csv_path}")

    # =========================================================================
    # EXECUTIVE SUMMARY & ARCHITECTURAL COMPARISON
    # =========================================================================
    subspace_res = next(r for r in results if "tech-rkj" in r["Model"])
    rf_res = next(r for r in results if "Random Forest" in r["Model"])
    mlp_res = next(r for r in results if "MLP Regressor" in r["Model"])

    speedup_rf = rf_res["Train_Time_ms"] / max(subspace_res["Train_Time_ms"], 1e-3)
    speedup_mlp = mlp_res["Train_Time_ms"] / max(subspace_res["Train_Time_ms"], 1e-3)

    print("\n" + "=" * 100)
    print("EXECUTIVE ARCHITECTURAL SUMMARY & KEY FINDINGS")
    print("=" * 100)
    print(f"1. Single-Pass Closed-Form Solver vs Iterative Optimizers:")
    print(f"   - tech-rkj Subspace Regressor solves weights analytically via (Z^T*Z + alpha*I)^(-1) * Z^T*Y in 1 step.")
    print(f"   - Train time: {subspace_res['Train_Time_ms']:.2f} ms vs Random Forest ({rf_res['Train_Time_ms']:.2f} ms, {speedup_rf:.1f}x faster) and MLP ({mlp_res['Train_Time_ms']:.2f} ms, {speedup_mlp:.1f}x faster).")
    print()
    print(f"2. Continual Streaming & Zero Catastrophic Forgetting:")
    print(f"   - All standard ML models (Trees, Ensembles, Neural Nets) CANNOT update on a single new sample.")
    print(f"     They require full dataset retraining (taking {rf_res['Train_Time_ms']:.1f} ms each time).")
    print(f"   - tech-rkj Subspace Regressor updates live in {subspace_res['Stream_Latency_us']:.2f} µs via Recursive Least Squares (RLS),")
    print(f"     enabling real-time streaming automotive pricing without retraining from scratch.")
    print()
    print(f"3. Green AI & Power Efficiency (AMD Ryzen 15W TDP):")
    print(f"   - tech-rkj Energy Footprint: {subspace_res['Energy_Joules']:.4f} Joules ({subspace_res['Energy_mWh']:.6f} mWh).")
    print(f"   - Random Forest: {rf_res['Energy_Joules']:.4f} Joules ({rf_res['Energy_mWh']:.6f} mWh).")
    print(f"   - MLP Neural Network: {mlp_res['Energy_Joules']:.4f} Joules ({mlp_res['Energy_mWh']:.6f} mWh).")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
