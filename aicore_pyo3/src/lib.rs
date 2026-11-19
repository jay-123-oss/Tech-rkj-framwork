//! PyO3 Rust extension module for tech-rkj.
//!
//! Provides high-performance AVX2/NEON subspace regression and classification
//! with zero-copy NumPy array interop, zero-allocation hot paths, and bit-exact
//! binary serialization (bincode / Serde) for pickle and joblib support.

#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::useless_conversion
)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use serde::{Deserialize, Serialize};

use aicore_fw::engine::{AICoreConfig, MultiHeadSubspaceEngine};
use aicore_fw::frontend::{IdentityFrontend, OnlineZScoreFrontend, ZScoreStats};

// =============================================================================
// STATE STRUCTS FOR BIT-EXACT SERIALIZATION
// =============================================================================

#[derive(Clone, Serialize, Deserialize)]
struct RegressorState {
    engine: MultiHeadSubspaceEngine,
    frontend_mode: String,
    zscore_stats: Option<ZScoreStats>,
    input_dim: usize,
    n_samples_seen: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct ClassifierState {
    engine: MultiHeadSubspaceEngine,
    frontend_mode: String,
    zscore_stats: Option<ZScoreStats>,
    input_dim: usize,
    num_classes: usize,
    n_samples_seen: usize,
}

// =============================================================================
// SUBSPACE REGRESSOR CORE
// =============================================================================

#[pyclass(module = "tech_rkj._core")]
pub struct SubspaceRegressorCore {
    engine: MultiHeadSubspaceEngine,
    zscore: OnlineZScoreFrontend,
    identity: IdentityFrontend,
    frontend_mode: String,
    input_dim: usize,
    n_samples_seen: usize,
}

#[pymethods]
impl SubspaceRegressorCore {
    #[new]
    #[pyo3(signature = (
        input_dim,
        num_heads = 4,
        head_circuits = 256,
        alpha = 10.0,
        forgetting_factor = 1.0,
        low_freq_scale = 0.12,
        high_freq_scale = 0.32,
        frontend_mode = "zscore"
    ))]
    pub fn new(
        input_dim: usize,
        num_heads: usize,
        head_circuits: usize,
        alpha: f64,
        forgetting_factor: f64,
        low_freq_scale: f32,
        high_freq_scale: f32,
        frontend_mode: &str,
    ) -> PyResult<Self> {
        let config = AICoreConfig {
            input_dim,
            num_heads,
            head_circuits,
            alpha,
            forgetting_factor,
            target_accuracy: 0.95,
            num_classes: 1, // continuous regression
            low_freq_scale,
            high_freq_scale,
        };

        let engine = MultiHeadSubspaceEngine::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize engine: {}", e)))?;

        Ok(Self {
            engine,
            zscore: OnlineZScoreFrontend::new(input_dim),
            identity: IdentityFrontend::new(input_dim),
            frontend_mode: frontend_mode.to_string(),
            input_dim,
            n_samples_seen: 0,
        })
    }

    pub fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Targets array must be contiguous"))?;
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        if self.frontend_mode == "zscore" {
            self.engine.fit_regression_batch(&self.zscore, x_slice, y_slice);
        } else {
            self.engine.fit_regression_batch(&self.identity, x_slice, y_slice);
        }

        self.n_samples_seen += num_samples;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn partial_fit(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Targets array must be contiguous"))?;
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let weights_slice = match &sample_weight {
            Some(w) => Some(w.as_slice().map_err(|_| PyValueError::new_err("Weights must be contiguous"))?),
            None => None,
        };

        for s in 0..num_samples {
            let sample_x = &x_slice[s * feat_dim..(s + 1) * feat_dim];
            let target_y = y_slice[s];
            let w = weights_slice.map(|ws| ws[s]).unwrap_or(1.0);

            if self.frontend_mode == "zscore" {
                self.engine.update_regression_online(&self.zscore, sample_x, target_y, w);
            } else {
                self.engine.update_regression_online(&self.identity, sample_x, target_y, w);
            }
        }

        self.n_samples_seen += num_samples;
        Ok(())
    }

    pub fn update_online(&mut self, x: PyReadonlyArray1<f32>, y: f64, sample_weight: f64) -> PyResult<()> {
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Sample array must be contiguous"))?;

        if x_slice.len() != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.input_dim, x_slice.len()
            )));
        }

        if self.frontend_mode == "zscore" {
            self.engine.update_regression_online(&self.zscore, x_slice, y, sample_weight);
        } else {
            self.engine.update_regression_online(&self.identity, x_slice, y, sample_weight);
        }

        self.n_samples_seen += 1;
        Ok(())
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let mut out_preds = vec![0.0f64; num_samples];

        if self.frontend_mode == "zscore" {
            self.engine.predict_regression_batch(&self.zscore, x_slice, &mut out_preds);
        } else {
            self.engine.predict_regression_batch(&self.identity, x_slice, &mut out_preds);
        }

        Ok(PyArray1::from_vec_bound(py, out_preds))
    }

    pub fn score(&self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Targets array must be contiguous"))?;
        let num_samples = y_slice.len();
        if num_samples == 0 {
            return Ok(0.0);
        }

        let py = unsafe { Python::assume_gil_acquired() };
        let preds_bound = self.predict(py, x)?;
        let preds = unsafe { preds_bound.as_slice()? };

        let mut sum_y = 0.0f64;
        for &val in y_slice {
            sum_y += val;
        }
        let y_mean = sum_y / (num_samples as f64);

        let mut ss_tot = 0.0f64;
        let mut ss_res = 0.0f64;
        for i in 0..num_samples {
            let diff_tot = y_slice[i] - y_mean;
            ss_tot += diff_tot * diff_tot;
            let diff_res = y_slice[i] - preds[i];
            ss_res += diff_res * diff_res;
        }

        if ss_tot < 1e-12 {
            Ok(0.0)
        } else {
            Ok(1.0 - (ss_res / ss_tot))
        }
    }

    #[getter]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    #[getter]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[getter]
    pub fn y_mean(&self) -> f64 {
        self.engine.y_mean
    }

    #[getter]
    pub fn y_std(&self) -> f64 {
        self.engine.y_std
    }

    pub fn memory_mb(&self) -> f64 {
        let mut total_bytes = 0usize;
        for head in &self.engine.heads {
            total_bytes += head.proj_f32.len() * 4;
            total_bytes += head.bias_f32.len() * 4;
            total_bytes += head.p_inv_f64.len() * 8;
            total_bytes += head.w_f64.len() * 8;
            total_bytes += head.z_f64.len() * 8;
            total_bytes += head.k_f64.len() * 8;
        }
        (total_bytes as f64) / (1024.0 * 1024.0)
    }

    pub fn get_weights<'py>(&self, py: Python<'py>, head_idx: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if head_idx >= self.engine.heads.len() {
            return Err(PyValueError::new_err("Head index out of range"));
        }
        Ok(PyArray1::from_slice_bound(py, &self.engine.heads[head_idx].w_f64))
    }

    pub fn get_p_inv<'py>(&self, py: Python<'py>, head_idx: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if head_idx >= self.engine.heads.len() {
            return Err(PyValueError::new_err("Head index out of range"));
        }
        Ok(PyArray1::from_slice_bound(py, &self.engine.heads[head_idx].p_inv_f64))
    }

    #[getter]
    pub fn zscore_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        if self.frontend_mode == "zscore" {
            let stats = self.zscore.stats.read().unwrap();
            dict.set_item("fitted", stats.fitted)?;
            dict.set_item("mean", PyArray1::from_slice_bound(py, &stats.mean))?;
            dict.set_item("std", PyArray1::from_slice_bound(py, &stats.std))?;
            dict.set_item("feature_weights", PyArray1::from_slice_bound(py, &stats.feature_weights))?;
        }
        Ok(dict)
    }

    // =========================================================================
    // SERDE / PICKLE HOOKS
    // =========================================================================

    pub fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyBytes>)> {
        let cls = py.get_type_bound::<Self>();
        let args = PyTuple::new_bound(py, [self.input_dim]);
        let state = self.__getstate__(py)?;
        Ok((cls.into_any(), args, state))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let zstats = if self.frontend_mode == "zscore" {
            Some(self.zscore.stats.read().unwrap().clone())
        } else {
            None
        };

        let state = RegressorState {
            engine: self.engine.clone(),
            frontend_mode: self.frontend_mode.clone(),
            zscore_stats: zstats,
            input_dim: self.input_dim,
            n_samples_seen: self.n_samples_seen,
        };

        let encoded = bincode::serialize(&state).map_err(|e| {
            PyValueError::new_err(format!("Serialization failure: {}", e))
        })?;

        Ok(PyBytes::new_bound(py, &encoded))
    }

    pub fn __setstate__(&mut self, state_bytes: Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state_bytes.as_bytes();
        let state: RegressorState = bincode::deserialize(bytes).map_err(|e| {
            PyValueError::new_err(format!("Invalid/corrupt state payload: {}", e))
        })?;

        self.engine = state.engine;
        self.frontend_mode = state.frontend_mode;
        self.input_dim = state.input_dim;
        self.n_samples_seen = state.n_samples_seen;

        if let Some(zstats) = state.zscore_stats {
            self.zscore = OnlineZScoreFrontend::new(self.input_dim);
            *self.zscore.stats.write().unwrap() = zstats;
        } else {
            self.zscore = OnlineZScoreFrontend::new(self.input_dim);
        }
        self.identity = IdentityFrontend::new(self.input_dim);

        Ok(())
    }
}

// =============================================================================
// SUBSPACE CLASSIFIER CORE
// =============================================================================

#[pyclass(module = "tech_rkj._core")]
pub struct SubspaceClassifierCore {
    engine: MultiHeadSubspaceEngine,
    zscore: OnlineZScoreFrontend,
    identity: IdentityFrontend,
    frontend_mode: String,
    input_dim: usize,
    num_classes: usize,
    n_samples_seen: usize,
}

#[pymethods]
impl SubspaceClassifierCore {
    #[new]
    #[pyo3(signature = (
        input_dim,
        num_classes,
        num_heads = 4,
        head_circuits = 256,
        alpha = 10.0,
        forgetting_factor = 1.0,
        low_freq_scale = 0.12,
        high_freq_scale = 0.32,
        frontend_mode = "zscore"
    ))]
    pub fn new(
        input_dim: usize,
        num_classes: usize,
        num_heads: usize,
        head_circuits: usize,
        alpha: f64,
        forgetting_factor: f64,
        low_freq_scale: f32,
        high_freq_scale: f32,
        frontend_mode: &str,
    ) -> PyResult<Self> {
        let config = AICoreConfig {
            input_dim,
            num_heads,
            head_circuits,
            alpha,
            forgetting_factor,
            target_accuracy: 0.95,
            num_classes,
            low_freq_scale,
            high_freq_scale,
        };

        let engine = MultiHeadSubspaceEngine::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize engine: {}", e)))?;

        Ok(Self {
            engine,
            zscore: OnlineZScoreFrontend::new(input_dim),
            identity: IdentityFrontend::new(input_dim),
            frontend_mode: frontend_mode.to_string(),
            input_dim,
            num_classes,
            n_samples_seen: 0,
        })
    }

    pub fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Labels array must be contiguous"))?;
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let y_usize: Vec<usize> = y_slice.iter().map(|&lbl| lbl as usize).collect();

        if self.frontend_mode == "zscore" {
            self.engine.fit_batch(&self.zscore, x_slice, &y_usize);
        } else {
            self.engine.fit_batch(&self.identity, x_slice, &y_usize);
        }

        self.n_samples_seen += num_samples;
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    pub fn partial_fit(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Labels array must be contiguous"))?;
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let weights_slice = match &sample_weight {
            Some(w) => Some(w.as_slice().map_err(|_| PyValueError::new_err("Weights must be contiguous"))?),
            None => None,
        };

        for s in 0..num_samples {
            let sample_x = &x_slice[s * feat_dim..(s + 1) * feat_dim];
            let label = y_slice[s] as usize;
            let w = weights_slice.map(|ws| ws[s]).unwrap_or(1.0);

            if self.frontend_mode == "zscore" {
                self.engine.update_online(&self.zscore, sample_x, label, w);
            } else {
                self.engine.update_online(&self.identity, sample_x, label, w);
            }
        }

        self.n_samples_seen += num_samples;
        Ok(())
    }

    pub fn update_online(&mut self, x: PyReadonlyArray1<f32>, label: usize, sample_weight: f64) -> PyResult<()> {
        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Sample array must be contiguous"))?;

        if x_slice.len() != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.input_dim, x_slice.len()
            )));
        }

        if self.frontend_mode == "zscore" {
            self.engine.update_online(&self.zscore, x_slice, label, sample_weight);
        } else {
            self.engine.update_online(&self.identity, x_slice, label, sample_weight);
        }

        self.n_samples_seen += 1;
        Ok(())
    }

    pub fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let mut preds = vec![0i64; num_samples];
        for s in 0..num_samples {
            let row = &x_slice[s * feat_dim..(s + 1) * feat_dim];
            let pred = if self.frontend_mode == "zscore" {
                self.engine.predict(&self.zscore, row)
            } else {
                self.engine.predict(&self.identity, row)
            };
            preds[s] = pred as i64;
        }

        Ok(PyArray1::from_vec_bound(py, preds))
    }

    pub fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_shape = x.shape();
        let num_samples = x_shape[0];
        let feat_dim = x_shape[1];
        let c = self.num_classes;

        if feat_dim != self.input_dim {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim, feat_dim
            )));
        }

        let x_slice = x.as_slice()
            .map_err(|_| PyValueError::new_err("Feature matrix must be contiguous"))?;

        let mut probas = vec![0.0f64; num_samples * c];
        for s in 0..num_samples {
            let row = &x_slice[s * feat_dim..(s + 1) * feat_dim];
            let out_p = &mut probas[s * c..(s + 1) * c];
            if self.frontend_mode == "zscore" {
                self.engine.predict_proba(&self.zscore, row, out_p);
            } else {
                self.engine.predict_proba(&self.identity, row, out_p);
            }
        }

        let py_array = PyArray2::from_vec2_bound(
            py,
            &probas.chunks(c).map(|ch| ch.to_vec()).collect::<Vec<_>>(),
        )?;
        Ok(py_array)
    }

    pub fn score(&self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<i64>) -> PyResult<f64> {
        let y_slice = y.as_slice()
            .map_err(|_| PyValueError::new_err("Labels array must be contiguous"))?;
        let num_samples = y_slice.len();
        if num_samples == 0 {
            return Ok(0.0);
        }

        let py = unsafe { Python::assume_gil_acquired() };
        let preds_bound = self.predict(py, x)?;
        let preds = unsafe { preds_bound.as_slice()? };

        let mut correct = 0usize;
        for i in 0..num_samples {
            if preds[i] == y_slice[i] {
                correct += 1;
            }
        }

        Ok((correct as f64) / (num_samples as f64))
    }

    #[getter]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    #[getter]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[getter]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn memory_mb(&self) -> f64 {
        let mut total_bytes = 0usize;
        for head in &self.engine.heads {
            total_bytes += head.proj_f32.len() * 4;
            total_bytes += head.bias_f32.len() * 4;
            total_bytes += head.p_inv_f64.len() * 8;
            total_bytes += head.w_f64.len() * 8;
            total_bytes += head.z_f64.len() * 8;
            total_bytes += head.k_f64.len() * 8;
        }
        (total_bytes as f64) / (1024.0 * 1024.0)
    }

    pub fn get_weights<'py>(&self, py: Python<'py>, head_idx: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if head_idx >= self.engine.heads.len() {
            return Err(PyValueError::new_err("Head index out of range"));
        }
        Ok(PyArray1::from_slice_bound(py, &self.engine.heads[head_idx].w_f64))
    }

    pub fn get_p_inv<'py>(&self, py: Python<'py>, head_idx: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if head_idx >= self.engine.heads.len() {
            return Err(PyValueError::new_err("Head index out of range"));
        }
        Ok(PyArray1::from_slice_bound(py, &self.engine.heads[head_idx].p_inv_f64))
    }

    #[getter]
    pub fn zscore_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        if self.frontend_mode == "zscore" {
            let stats = self.zscore.stats.read().unwrap();
            dict.set_item("fitted", stats.fitted)?;
            dict.set_item("mean", PyArray1::from_slice_bound(py, &stats.mean))?;
            dict.set_item("std", PyArray1::from_slice_bound(py, &stats.std))?;
            dict.set_item("feature_weights", PyArray1::from_slice_bound(py, &stats.feature_weights))?;
        }
        Ok(dict)
    }

    // =========================================================================
    // SERDE / PICKLE HOOKS
    // =========================================================================

    pub fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyBytes>)> {
        let cls = py.get_type_bound::<Self>();
        let args = PyTuple::new_bound(py, [self.input_dim, self.num_classes]);
        let state = self.__getstate__(py)?;
        Ok((cls.into_any(), args, state))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let zstats = if self.frontend_mode == "zscore" {
            Some(self.zscore.stats.read().unwrap().clone())
        } else {
            None
        };

        let state = ClassifierState {
            engine: self.engine.clone(),
            frontend_mode: self.frontend_mode.clone(),
            zscore_stats: zstats,
            input_dim: self.input_dim,
            num_classes: self.num_classes,
            n_samples_seen: self.n_samples_seen,
        };

        let encoded = bincode::serialize(&state).map_err(|e| {
            PyValueError::new_err(format!("Serialization failure: {}", e))
        })?;

        Ok(PyBytes::new_bound(py, &encoded))
    }

    pub fn __setstate__(&mut self, state_bytes: Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state_bytes.as_bytes();
        let state: ClassifierState = bincode::deserialize(bytes).map_err(|e| {
            PyValueError::new_err(format!("Invalid/corrupt state payload: {}", e))
        })?;

        self.engine = state.engine;
        self.frontend_mode = state.frontend_mode;
        self.input_dim = state.input_dim;
        self.num_classes = state.num_classes;
        self.n_samples_seen = state.n_samples_seen;

        if let Some(zstats) = state.zscore_stats {
            self.zscore = OnlineZScoreFrontend::new(self.input_dim);
            *self.zscore.stats.write().unwrap() = zstats;
        } else {
            self.zscore = OnlineZScoreFrontend::new(self.input_dim);
        }
        self.identity = IdentityFrontend::new(self.input_dim);

        Ok(())
    }
}

// =============================================================================
// PYMODULE REGISTRATION
// =============================================================================

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SubspaceRegressorCore>()?;
    m.add_class::<SubspaceClassifierCore>()?;
    m.add("__version__", "0.2.0")?;
    Ok(())
}
