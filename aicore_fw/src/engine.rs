//! Multi-head subspace engine implementation.

use crate::frontend::FeatureFrontend;
use crate::head::SubspaceHead;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct AICoreConfig {
    pub input_dim: usize,
    pub num_heads: usize,
    pub head_circuits: usize,
    pub alpha: f64,
    pub forgetting_factor: f64,
    pub target_accuracy: f64,
    pub num_classes: usize,
    pub low_freq_scale: f32,
    pub high_freq_scale: f32,
}

#[derive(Debug)]
pub enum AICoreError {
    InvalidConfig(&'static str),
}

impl std::fmt::Display for AICoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AICoreError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for AICoreError {}

#[derive(Debug)]
pub struct MultiHeadSubspaceEngine {
    pub heads: Vec<SubspaceHead>,
    pub input_dim: usize,
    pub num_classes: usize,
    pub num_heads: usize,
    pub head_circuits: usize,
    pub alpha: f64,
    pub forgetting_factor: f64,
    pub low_freq_scale: f32,
    pub high_freq_scale: f32,
    pub y_mean: f64,
    pub y_std: f64,
    feat_buf: Vec<f32>,
    scores_buf: Vec<f64>,
    residual_buf: Vec<f64>,
}

impl MultiHeadSubspaceEngine {
    pub fn new(config: AICoreConfig) -> Result<Self, AICoreError> {
        if config.input_dim == 0 || config.num_classes == 0 || config.num_heads == 0 || config.head_circuits == 0 {
            return Err(AICoreError::InvalidConfig("Dimensions and counts must be non-zero"));
        }
        let mut heads = Vec::with_capacity(config.num_heads);
        for h in 0..config.num_heads {
            let seed = 42u64.wrapping_add((h as u64) * 10007);
            heads.push(SubspaceHead::new(
                config.input_dim,
                config.head_circuits,
                config.num_classes,
                config.alpha,
                config.low_freq_scale,
                config.high_freq_scale,
                seed,
            ));
        }
        let max_dim = config.input_dim.max(512);
        Ok(Self {
            heads,
            input_dim: config.input_dim,
            num_classes: config.num_classes,
            num_heads: config.num_heads,
            head_circuits: config.head_circuits,
            alpha: config.alpha,
            forgetting_factor: config.forgetting_factor,
            low_freq_scale: config.low_freq_scale,
            high_freq_scale: config.high_freq_scale,
            y_mean: 0.0,
            y_std: 1.0,
            feat_buf: vec![0.0f32; max_dim],
            scores_buf: vec![0.0f64; config.num_classes],
            residual_buf: vec![0.0f64; config.num_classes],
        })
    }

    pub fn fit_batch<F: FeatureFrontend>(
        &mut self,
        frontend: &F,
        x_batch: &[f32],
        y_labels: &[usize],
    ) {
        let num_samples = y_labels.len();
        if num_samples == 0 {
            return;
        }
        frontend.fit_stats(x_batch, y_labels, self.num_classes);

        let feat_dim = frontend.output_dim();
        let num_heads = self.heads.len();
        let d_circuits = self.head_circuits;
        let alpha = self.alpha;

        let mut feat_all = vec![0.0f32; num_samples * feat_dim];
        for s in 0..num_samples {
            let raw_x = &x_batch[s * self.input_dim..(s + 1) * self.input_dim];
            let out_feat = &mut feat_all[s * feat_dim..(s + 1) * feat_dim];
            frontend.extract(raw_x, out_feat);
        }

        self.heads.par_iter_mut().for_each(|head| {
            let mut z_batch = vec![0.0f64; num_samples * d_circuits];
            for s in 0..num_samples {
                let feat_s = &feat_all[s * feat_dim..(s + 1) * feat_dim];
                let out_z = &mut z_batch[s * d_circuits..(s + 1) * d_circuits];
                head.project_one(feat_s, out_z);
            }
            head.fit_closed_form(&z_batch, y_labels, alpha, num_heads);
        });
    }

    pub fn fit_regression_batch<F: FeatureFrontend>(
        &mut self,
        frontend: &F,
        x_batch: &[f32],
        y_targets: &[f64],
    ) {
        let num_samples = y_targets.len();
        if num_samples == 0 {
            return;
        }

        let mut sum_y = 0.0f64;
        for &y in y_targets {
            sum_y += y;
        }
        let y_mean = sum_y / (num_samples as f64);

        let mut sum_var_y = 0.0f64;
        for &y in y_targets {
            let diff = y - y_mean;
            sum_var_y += diff * diff;
        }
        let y_std = (sum_var_y / (num_samples as f64)).sqrt().max(1e-6);

        self.y_mean = y_mean;
        self.y_std = y_std;

        let mut y_norm = vec![0.0f64; num_samples];
        for s in 0..num_samples {
            y_norm[s] = (y_targets[s] - y_mean) / y_std;
        }

        frontend.fit_stats_regression(x_batch, y_targets);

        let feat_dim = frontend.output_dim();
        let num_heads = self.heads.len();
        let d_circuits = self.head_circuits;
        let alpha = self.alpha;

        let mut feat_all = vec![0.0f32; num_samples * feat_dim];
        for s in 0..num_samples {
            let raw_x = &x_batch[s * self.input_dim..(s + 1) * self.input_dim];
            let out_feat = &mut feat_all[s * feat_dim..(s + 1) * feat_dim];
            frontend.extract(raw_x, out_feat);
        }

        self.heads.par_iter_mut().for_each(|head| {
            let mut z_batch = vec![0.0f64; num_samples * d_circuits];
            for s in 0..num_samples {
                let feat_s = &feat_all[s * feat_dim..(s + 1) * feat_dim];
                let out_z = &mut z_batch[s * d_circuits..(s + 1) * d_circuits];
                head.project_one(feat_s, out_z);
            }
            head.fit_closed_form_regression(&z_batch, &y_norm, alpha, num_heads);
        });
    }

    #[inline(always)]
    pub fn update_online<F: FeatureFrontend>(
        &mut self,
        frontend: &F,
        x: &[f32],
        label: usize,
        sample_weight: f64,
    ) {
        let feat_dim = frontend.output_dim();
        if self.feat_buf.len() < feat_dim {
            self.feat_buf.resize(feat_dim, 0.0);
        }
        frontend.extract(x, &mut self.feat_buf[..feat_dim]);

        let num_heads = self.heads.len();
        let num_classes = self.num_classes;

        self.scores_buf.fill(0.0);

        for head in &mut self.heads {
            head.project_self(&self.feat_buf[..feat_dim]);
            head.compute_logits_self(&mut self.scores_buf);
        }

        let inv_heads = 1.0 / (num_heads as f64);
        for cl in 0..num_classes {
            let target = if cl == label { 1.0 } else { 0.0 };
            self.residual_buf[cl] = (target - self.scores_buf[cl]) * inv_heads;
        }

        // Sequential per-head RLS update with zero allocations (sub-30us)
        for head in &mut self.heads {
            head.update_rls_self(&self.residual_buf, sample_weight);
        }
    }

    pub fn predict<F: FeatureFrontend>(&self, frontend: &F, x: &[f32]) -> usize {
        let feat_dim = frontend.output_dim();
        let mut feat_buf = vec![0.0f32; feat_dim];
        frontend.extract(x, &mut feat_buf);

        let mut scores = vec![0.0f64; self.num_classes];
        for head in &self.heads {
            head.forward(&feat_buf, &mut scores);
        }

        let mut best_class = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (cl, &score) in scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_class = cl;
            }
        }
        best_class
    }

    pub fn predict_proba<F: FeatureFrontend>(&self, frontend: &F, x: &[f32], out_proba: &mut [f64]) {
        let feat_dim = frontend.output_dim();
        let mut feat_buf = vec![0.0f32; feat_dim];
        frontend.extract(x, &mut feat_buf);

        let mut scores = vec![0.0f64; self.num_classes];
        for head in &self.heads {
            head.forward(&feat_buf, &mut scores);
        }

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0f64;
        for (i, &s) in scores.iter().enumerate() {
            let exp_s = (s - max_score).exp();
            out_proba[i] = exp_s;
            sum_exp += exp_s;
        }
        if sum_exp > 0.0 {
            for p in out_proba.iter_mut() {
                *p /= sum_exp;
            }
        }
    }

    pub fn predict_regression<F: FeatureFrontend>(&self, frontend: &F, x: &[f32]) -> f64 {
        let feat_dim = frontend.output_dim();
        let mut feat_buf = vec![0.0f32; feat_dim];
        frontend.extract(x, &mut feat_buf);

        let mut pred_norm = 0.0f64;
        let mut z = vec![0.0f64; self.head_circuits];
        for head in &self.heads {
            head.project_one(&feat_buf, &mut z);
            pred_norm += head.compute_scalar(&z);
        }
        pred_norm * self.y_std + self.y_mean
    }

    pub fn predict_regression_batch<F: FeatureFrontend>(
        &self,
        frontend: &F,
        x_batch: &[f32],
        out_preds: &mut [f64],
    ) {
        let num_samples = out_preds.len();
        let feat_dim = frontend.output_dim();
        let mut feat_buf = vec![0.0f32; feat_dim];
        let mut z = vec![0.0f64; self.head_circuits];

        for s in 0..num_samples {
            let raw_x = &x_batch[s * self.input_dim..(s + 1) * self.input_dim];
            frontend.extract(raw_x, &mut feat_buf);

            let mut pred_norm = 0.0f64;
            for head in &self.heads {
                head.project_one(&feat_buf, &mut z);
                pred_norm += head.compute_scalar(&z);
            }
            out_preds[s] = pred_norm * self.y_std + self.y_mean;
        }
    }

    #[inline(always)]
    pub fn update_regression_online<F: FeatureFrontend>(
        &mut self,
        frontend: &F,
        x: &[f32],
        y_target: f64,
        sample_weight: f64,
    ) {
        let feat_dim = frontend.output_dim();
        if self.feat_buf.len() < feat_dim {
            self.feat_buf.resize(feat_dim, 0.0);
        }
        frontend.extract(x, &mut self.feat_buf[..feat_dim]);

        let num_heads = self.heads.len();
        let y_norm = (y_target - self.y_mean) / self.y_std;

        let mut pred_norm = 0.0f64;
        for head in &mut self.heads {
            head.project_self(&self.feat_buf[..feat_dim]);
            pred_norm += head.compute_scalar_self();
        }

        let res = (y_norm - pred_norm) / (num_heads as f64);
        let res_slice = [res];

        for head in &mut self.heads {
            head.update_rls_self(&res_slice, sample_weight);
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for head in &self.heads {
            total += std::mem::size_of::<SubspaceHead>();
            total += head.proj_f32.capacity() * 4;
            total += head.bias_f32.capacity() * 4;
            total += head.p_inv_f64.capacity() * 8;
            total += head.w_f64.capacity() * 8;
            total += head.z_f64.capacity() * 8;
            total += head.k_f64.capacity() * 8;
        }
        total += self.feat_buf.capacity() * 4;
        total += self.scores_buf.capacity() * 8;
        total += self.residual_buf.capacity() * 8;
        total
    }
}
