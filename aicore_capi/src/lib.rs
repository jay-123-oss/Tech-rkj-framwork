use aicore_fw::frontend::FeatureFrontend;
use aicore_fw::frontend::IdentityFrontend;
use aicore_fw::frontend::OnlineZScoreFrontend;
use aicore_fw::frontend::VisionGaborFrontend;
use aicore_fw::engine::{MultiHeadSubspaceEngine, AICoreConfig};
use std::slice;

enum FrontendKind {
    Identity(IdentityFrontend),
    ZScore(OnlineZScoreFrontend),
    VisionGabor(VisionGaborFrontend),
}

impl FeatureFrontend for FrontendKind {
    fn extract(&self, raw: &[f32], out: &mut [f32]) {
        match self {
            FrontendKind::Identity(f) => f.extract(raw, out),
            FrontendKind::ZScore(f) => f.extract(raw, out),
            FrontendKind::VisionGabor(f) => f.extract(raw, out),
        }
    }
    fn output_dim(&self) -> usize {
        match self {
            FrontendKind::Identity(f) => f.output_dim(),
            FrontendKind::ZScore(f) => f.output_dim(),
            FrontendKind::VisionGabor(f) => f.output_dim(),
        }
    }
    fn fit_stats(&self, x_batch: &[f32], y_labels: &[usize], num_classes: usize) {
        match self {
            FrontendKind::Identity(f) => f.fit_stats(x_batch, y_labels, num_classes),
            FrontendKind::ZScore(f) => f.fit_stats(x_batch, y_labels, num_classes),
            FrontendKind::VisionGabor(f) => f.fit_stats(x_batch, y_labels, num_classes),
        }
    }
    fn fit_stats_regression(&self, x_batch: &[f32], y_targets: &[f64]) {
        match self {
            FrontendKind::Identity(f) => f.fit_stats_regression(x_batch, y_targets),
            FrontendKind::ZScore(f) => f.fit_stats_regression(x_batch, y_targets),
            FrontendKind::VisionGabor(f) => f.fit_stats_regression(x_batch, y_targets),
        }
    }
}

pub struct CEngineHandle {
    engine: MultiHeadSubspaceEngine,
    frontend: FrontendKind,
    raw_dim: usize,
}

/// frontend_mode: 0 = identity, 1 = zscore, 2 = vision_gabor
#[no_mangle]
pub extern "C" fn aicore_create(
    input_dim: usize,
    num_classes: usize,
    num_heads: usize,
    head_circuits: usize,
    alpha: f64,
    forgetting_factor: f64,
    low_freq_scale: f32,
    high_freq_scale: f32,
    frontend_mode: i32,
) -> *mut CEngineHandle {
    let frontend = match frontend_mode {
        1 => FrontendKind::ZScore(OnlineZScoreFrontend::new(input_dim)),
        2 => FrontendKind::VisionGabor(VisionGaborFrontend::new(256)),
        _ => FrontendKind::Identity(IdentityFrontend::new(input_dim)),
    };

    let cfg = AICoreConfig {
        input_dim,
        num_heads,
        head_circuits,
        alpha,
        forgetting_factor,
        target_accuracy: 0.90,
        num_classes,
        low_freq_scale,
        high_freq_scale,
    };

    match MultiHeadSubspaceEngine::new(cfg) {
        Ok(engine) => Box::into_raw(Box::new(CEngineHandle {
            engine,
            frontend,
            raw_dim: input_dim,
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn aicore_free(handle: *mut CEngineHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn aicore_fit(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    y_ptr: *const i64,
    num_samples: usize,
) -> i32 {
    if handle.is_null() || x_ptr.is_null() || y_ptr.is_null() {
        eprintln!("aicore_fit: null handle or pointer");
        return -2;
    }
    let h = &mut *handle;
    let x_slice = slice::from_raw_parts(x_ptr, num_samples * h.raw_dim);
    let y_raw = slice::from_raw_parts(y_ptr, num_samples);
    let y_usize: Vec<usize> = y_raw.iter().map(|&v| v as usize).collect();

    h.engine.fit_batch(&h.frontend, &x_slice[..num_samples * h.raw_dim], &y_usize[..num_samples]);
    0
}

#[no_mangle]
pub unsafe extern "C" fn aicore_update_online(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    label: usize,
    sample_weight: f64,
) {
    if handle.is_null() || x_ptr.is_null() {
        return;
    }
    let h = &mut *handle;
    let x_slice = slice::from_raw_parts(x_ptr, h.raw_dim);
    h.engine.update_online(&h.frontend, x_slice, label, sample_weight);
}

#[no_mangle]
pub unsafe extern "C" fn aicore_predict_batch(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    out_preds: *mut i64,
    num_samples: usize,
) {
    if handle.is_null() || x_ptr.is_null() || out_preds.is_null() || num_samples == 0 {
        return;
    }
    let h = &mut *handle;
    let dim = h.raw_dim;
    let x_slice = slice::from_raw_parts(x_ptr, num_samples * dim);
    let preds_slice = slice::from_raw_parts_mut(out_preds, num_samples);
    for i in 0..num_samples {
        let sample = &x_slice[i * dim..(i + 1) * dim];
        preds_slice[i] = h.engine.predict(&h.frontend, sample) as i64;
    }
}

#[no_mangle]
pub unsafe extern "C" fn aicore_fit_regression(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    y_ptr: *const f64,
    num_samples: usize,
) -> i32 {
    if handle.is_null() || x_ptr.is_null() || y_ptr.is_null() || num_samples == 0 {
        eprintln!("aicore_fit_regression: null handle or pointer or 0 samples");
        return -2;
    }
    let h = &mut *handle;
    let x_slice = slice::from_raw_parts(x_ptr, num_samples * h.raw_dim);
    let y_slice = slice::from_raw_parts(y_ptr, num_samples);

    h.engine.fit_regression_batch(&h.frontend, x_slice, y_slice);
    0
}

#[no_mangle]
pub unsafe extern "C" fn aicore_predict_regression_batch(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    out_preds: *mut f64,
    num_samples: usize,
) {
    if handle.is_null() || x_ptr.is_null() || out_preds.is_null() || num_samples == 0 {
        return;
    }
    let h = &mut *handle;
    let dim = h.raw_dim;
    let x_slice = slice::from_raw_parts(x_ptr, num_samples * dim);
    let preds_slice = slice::from_raw_parts_mut(out_preds, num_samples);
    h.engine.predict_regression_batch(&h.frontend, x_slice, preds_slice);
}

#[no_mangle]
pub unsafe extern "C" fn aicore_update_regression_online(
    handle: *mut CEngineHandle,
    x_ptr: *const f32,
    y_target: f64,
    sample_weight: f64,
) {
    if handle.is_null() || x_ptr.is_null() {
        return;
    }
    let h = &mut *handle;
    let x_slice = slice::from_raw_parts(x_ptr, h.raw_dim);
    h.engine.update_regression_online(&h.frontend, x_slice, y_target, sample_weight);
}

#[no_mangle]
pub unsafe extern "C" fn aicore_memory_mb(handle: *const CEngineHandle) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    let h = &*handle;
    (h.engine.memory_footprint_bytes() as f64) / (1024.0 * 1024.0)
}
