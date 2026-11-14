//! Feature frontends for AICore classifier with supervised ANOVA/Fisher feature weighting.

use std::sync::RwLock;

pub trait FeatureFrontend: Send + Sync {
    fn extract(&self, raw: &[f32], out: &mut [f32]);
    fn output_dim(&self) -> usize;
    fn fit_stats(&self, _x_batch: &[f32], _y_labels: &[usize], _num_classes: usize) {}
    fn fit_stats_regression(&self, _x_batch: &[f32], _y_targets: &[f64]) {}
}

#[derive(Debug, Clone)]
pub struct IdentityFrontend {
    dim: usize,
}

impl IdentityFrontend {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl FeatureFrontend for IdentityFrontend {
    fn extract(&self, raw: &[f32], out: &mut [f32]) {
        let n = raw.len().min(out.len());
        out[..n].copy_from_slice(&raw[..n]);
    }

    fn output_dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug)]
struct ZScoreStats {
    mean: Vec<f32>,
    std: Vec<f32>,
    feature_weights: Vec<f32>,
    fitted: bool,
}

#[derive(Debug)]
pub struct OnlineZScoreFrontend {
    dim: usize,
    stats: RwLock<ZScoreStats>,
}

impl OnlineZScoreFrontend {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            stats: RwLock::new(ZScoreStats {
                mean: vec![0.0; dim],
                std: vec![1.0; dim],
                feature_weights: vec![1.0; dim],
                fitted: false,
            }),
        }
    }

    pub fn fit_stats(&self, x_batch: &[f32], y_labels: &[usize], num_classes: usize) {
        let num_samples = y_labels.len();
        if num_samples == 0 {
            return;
        }
        let dim = self.dim;
        let n_f = num_samples as f32;

        // 1. Global per-feature mean
        let mut mean = vec![0.0f32; dim];
        for s in 0..num_samples {
            let row = &x_batch[s * dim..(s + 1) * dim];
            for j in 0..dim {
                mean[j] += row[j];
            }
        }
        for j in 0..dim {
            mean[j] /= n_f;
        }

        // 2. Global per-feature std: sigma = (var + 1e-6).sqrt()
        let mut std = vec![1.0f32; dim];
        for s in 0..num_samples {
            let row = &x_batch[s * dim..(s + 1) * dim];
            for j in 0..dim {
                let diff = row[j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for j in 0..dim {
            let var = std[j] / n_f;
            std[j] = (var + 1e-6).sqrt();
        }

        // 3. Supervised ANOVA F-statistic feature weighting
        let mut feature_weights = vec![1.0f32; dim];
        if num_samples > num_classes && num_classes > 1 {
            let c = num_classes;
            let mut n_c = vec![0usize; c];
            let mut sum_c = vec![0.0f64; c * dim];

            // Compute standardized features z_nj = (x_nj - mu[j]) / sigma[j] and accumulate class sums
            for s in 0..num_samples {
                let label = y_labels[s];
                if label < c {
                    n_c[label] += 1;
                    let row = &x_batch[s * dim..(s + 1) * dim];
                    let c_off = label * dim;
                    for j in 0..dim {
                        let z_j = ((row[j] - mean[j]) / std[j]) as f64;
                        sum_c[c_off + j] += z_j;
                    }
                }
            }

            // Per-class means mu_c[c][j]
            let mut mu_c = vec![0.0f64; c * dim];
            for cl in 0..c {
                let cnt = n_c[cl];
                if cnt > 0 {
                    let inv_cnt = 1.0 / (cnt as f64);
                    let c_off = cl * dim;
                    for j in 0..dim {
                        mu_c[c_off + j] = sum_c[c_off + j] * inv_cnt;
                    }
                }
            }

            // Between-class MSB = sum_c n_c * (mu_c,j)^2 / (C - 1)
            let df_b = (c - 1) as f64;
            let mut msb = vec![0.0f64; dim];
            for cl in 0..c {
                let cnt = n_c[cl] as f64;
                if cnt > 0.0 {
                    let c_off = cl * dim;
                    for j in 0..dim {
                        let m_cj = mu_c[c_off + j];
                        msb[j] += cnt * m_cj * m_cj;
                    }
                }
            }
            for j in 0..dim {
                msb[j] /= df_b;
            }

            // Within-class SSW = sum_n (z_nj - mu_{y_n, j})^2
            let mut ssw = vec![0.0f64; dim];
            for s in 0..num_samples {
                let label = y_labels[s];
                if label < c {
                    let row = &x_batch[s * dim..(s + 1) * dim];
                    let c_off = label * dim;
                    for j in 0..dim {
                        let z_j = ((row[j] - mean[j]) / std[j]) as f64;
                        let diff = z_j - mu_c[c_off + j];
                        ssw[j] += diff * diff;
                    }
                }
            }
            let df_w = (num_samples - c) as f64;

            let mut f_stat = vec![0.0f64; dim];
            let mut f_sum = 0.0f64;
            for j in 0..dim {
                let msw = (ssw[j] / df_w).max(1e-4);
                let f_val = msb[j] / msw;
                f_stat[j] = f_val;
                f_sum += f_val;
            }

            let f_mean = f_sum / (dim as f64);
            let f_mean_clamped = f_mean.max(1e-6);

            let mut raw_w = vec![0.0f64; dim];
            let mut sum_sq = 0.0f64;
            for j in 0..dim {
                let ratio = f_stat[j] / f_mean_clamped;
                let w = ratio.powf(0.65).clamp(0.15, 2.5);
                raw_w[j] = w;
                sum_sq += w * w;
            }

            // Normalize feature_weights so (1/D) * sum(w_j^2) == 1.0
            let s2 = (sum_sq / (dim as f64)).max(1e-12);
            let norm = s2.sqrt();
            for j in 0..dim {
                feature_weights[j] = (raw_w[j] / norm) as f32;
            }
        }

        let mut st = self.stats.write().unwrap();
        st.mean = mean;
        st.std = std;
        st.feature_weights = feature_weights;
        st.fitted = true;
    }

    pub fn fit_stats_regression(&self, x_batch: &[f32], y_targets: &[f64]) {
        let num_samples = y_targets.len();
        if num_samples == 0 {
            return;
        }
        let dim = self.dim;
        let n_f = num_samples as f32;
        let n_f64 = num_samples as f64;

        // 1. Global per-feature mean
        let mut mean = vec![0.0f32; dim];
        for s in 0..num_samples {
            let row = &x_batch[s * dim..(s + 1) * dim];
            for j in 0..dim {
                mean[j] += row[j];
            }
        }
        for j in 0..dim {
            mean[j] /= n_f;
        }

        // 2. Global per-feature std: sigma = (var + 1e-6).sqrt()
        let mut std = vec![1.0f32; dim];
        for s in 0..num_samples {
            let row = &x_batch[s * dim..(s + 1) * dim];
            for j in 0..dim {
                let diff = row[j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for j in 0..dim {
            let var = std[j] / n_f;
            std[j] = (var + 1e-6).sqrt();
        }

        // 3. Target stats: y_mean and y_std
        let mut sum_y = 0.0f64;
        for &y in y_targets {
            sum_y += y;
        }
        let y_mean = sum_y / n_f64;

        let mut sum_var_y = 0.0f64;
        for &y in y_targets {
            let diff = y - y_mean;
            sum_var_y += diff * diff;
        }
        let y_std = (sum_var_y / n_f64).sqrt().max(1e-6);

        // 4. Supervised correlation feature weighting
        // |cov(x_j, y)| / (std_x_j * y_std) raised to 0.5, clamped to [0.35, 2.5], RMS-normalized to 1.0
        let mut corrs = vec![0.0f64; dim];
        let mut sum_corr = 0.0f64;

        for s in 0..num_samples {
            let y_diff = y_targets[s] - y_mean;
            let row = &x_batch[s * dim..(s + 1) * dim];
            for j in 0..dim {
                let x_diff = (row[j] - mean[j]) as f64;
                corrs[j] += x_diff * y_diff;
            }
        }

        for j in 0..dim {
            let cov = (corrs[j] / n_f64).abs();
            let c_val = cov / ((std[j] as f64) * y_std).max(1e-6);
            corrs[j] = c_val;
            sum_corr += c_val;
        }

        let mean_corr = (sum_corr / (dim as f64)).max(1e-6);

        let mut feature_weights = vec![1.0f32; dim];
        let mut sum_sq = 0.0f64;
        let mut raw_w = vec![0.0f64; dim];

        for j in 0..dim {
            let ratio = corrs[j] / mean_corr;
            let w = ratio.powf(0.5).clamp(0.35, 2.5);
            raw_w[j] = w;
            sum_sq += w * w;
        }

        let rms = ((sum_sq / (dim as f64)).max(1e-12)).sqrt();
        for j in 0..dim {
            feature_weights[j] = (raw_w[j] / rms) as f32;
        }

        let mut st = self.stats.write().unwrap();
        st.mean = mean;
        st.std = std;
        st.feature_weights = feature_weights;
        st.fitted = true;
    }
}

impl FeatureFrontend for OnlineZScoreFrontend {
    fn extract(&self, raw: &[f32], out: &mut [f32]) {
        let dim = self.dim;
        let n = raw.len().min(dim).min(out.len());
        let st = self.stats.read().unwrap();
        if st.fitted {
            for j in 0..n {
                out[j] = ((raw[j] - st.mean[j]) / st.std[j]) * st.feature_weights[j];
            }
        } else {
            out[..n].copy_from_slice(&raw[..n]);
        }
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn fit_stats(&self, x_batch: &[f32], y_labels: &[usize], num_classes: usize) {
        OnlineZScoreFrontend::fit_stats(self, x_batch, y_labels, num_classes);
    }

    fn fit_stats_regression(&self, x_batch: &[f32], y_targets: &[f64]) {
        OnlineZScoreFrontend::fit_stats_regression(self, x_batch, y_targets);
    }
}

#[derive(Debug, Clone)]
pub struct VisionGaborFrontend {
    output_dim: usize,
}

impl VisionGaborFrontend {
    pub fn new(output_dim: usize) -> Self {
        Self { output_dim }
    }
}

impl FeatureFrontend for VisionGaborFrontend {
    fn extract(&self, raw: &[f32], out: &mut [f32]) {
        let n = raw.len().min(out.len());
        out[..n].copy_from_slice(&raw[..n]);
        if out.len() > n {
            out[n..].fill(0.0);
        }
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}
