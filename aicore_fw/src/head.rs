//! Subspace head implementation for AICore classifier.

use crate::math_avx2::simd_dot_f32;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_uniform().clamp(1e-15, 1.0 - 1e-15);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct SubspaceHead {
    pub proj_f32: Vec<f32>,
    pub bias_f32: Vec<f32>,
    pub p_inv_f64: Vec<f64>,
    pub w_f64: Vec<f64>,
    pub d_circuits: usize,
    pub d_in: usize,
    pub num_classes: usize,
    pub z_f64: Vec<f64>,
    pub k_f64: Vec<f64>,
}

impl SubspaceHead {
    pub fn new(
        d_in: usize,
        d_circuits: usize,
        num_classes: usize,
        alpha: f64,
        low_freq_scale: f32,
        high_freq_scale: f32,
        seed: u64,
    ) -> Self {
        let d = d_circuits;
        let proj_size = d * d_in;
        let p_inv_size = d * d;
        let w_size = d * num_classes;

        let mut rng = SplitMix64::new(seed);

        let mut proj_f32 = vec![0.0f32; proj_size];
        let half = d / 2;
        for i in 0..d {
            let scale = if i < half { low_freq_scale } else { high_freq_scale } as f64;
            for j in 0..d_in {
                let g = rng.next_gaussian();
                proj_f32[i * d_in + j] = (g * scale) as f32;
            }
        }

        let mut bias_f32 = vec![0.0f32; d];
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..d {
            bias_f32[i] = (rng.next_uniform() * two_pi) as f32;
        }

        let mut p_inv_f64 = vec![0.0f64; p_inv_size];
        let inv_alpha = 1.0 / alpha.max(1e-6);
        for i in 0..d {
            p_inv_f64[i * d + i] = inv_alpha;
        }

        let w_f64 = vec![0.0f64; w_size];
        let z_f64 = vec![0.0f64; d];
        let k_f64 = vec![0.0f64; d];

        Self {
            proj_f32,
            bias_f32,
            p_inv_f64,
            w_f64,
            d_circuits: d,
            d_in,
            num_classes,
            z_f64,
            k_f64,
        }
    }

    #[inline(always)]
    pub fn project_one(&self, x: &[f32], out_z: &mut [f64]) {
        for i in 0..self.d_circuits {
            let row = &self.proj_f32[i * self.d_in..(i + 1) * self.d_in];
            let dot = simd_dot_f32(row, x);
            out_z[i] = ((dot + self.bias_f32[i]) as f64).cos();
        }
    }

    #[inline(always)]
    pub fn project_self(&mut self, x: &[f32]) {
        for i in 0..self.d_circuits {
            let row = &self.proj_f32[i * self.d_in..(i + 1) * self.d_in];
            let dot = simd_dot_f32(row, x);
            self.z_f64[i] = ((dot + self.bias_f32[i]) as f64).cos();
        }
    }

    #[inline(always)]
    pub fn compute_logits(&self, z: &[f64], scores: &mut [f64]) {
        let d = self.d_circuits;
        let c = self.num_classes;
        for cl in 0..c {
            let mut sum = 0.0f64;
            for i in 0..d {
                sum += self.w_f64[i * c + cl] * z[i];
            }
            scores[cl] += sum;
        }
    }

    #[inline(always)]
    pub fn compute_logits_self(&self, scores: &mut [f64]) {
        self.compute_logits(&self.z_f64, scores);
    }

    #[inline(always)]
    pub fn compute_scalar(&self, z: &[f64]) -> f64 {
        let d = self.d_circuits;
        let mut sum = 0.0f64;
        for i in 0..d {
            sum += self.w_f64[i] * z[i];
        }
        sum
    }

    #[inline(always)]
    pub fn compute_scalar_self(&self) -> f64 {
        self.compute_scalar(&self.z_f64)
    }

    #[inline(always)]
    pub fn forward(&self, x: &[f32], scores: &mut [f64]) {
        let d = self.d_circuits;
        let mut z = vec![0.0f64; d];
        self.project_one(x, &mut z);
        self.compute_logits(&z, scores);
    }

    pub fn fit_closed_form(
        &mut self,
        z_batch: &[f64],
        y_labels: &[usize],
        alpha: f64,
        num_heads: usize,
    ) {
        let n = y_labels.len();
        if n == 0 {
            return;
        }
        let d = self.d_circuits;
        let c = self.num_classes;

        // Build Gram matrix G = Z^T Z + alpha * I_d [d x d]
        let mut g = vec![0.0f64; d * d];
        for s in 0..n {
            let z_s = &z_batch[s * d..(s + 1) * d];
            for i in 0..d {
                let zi = z_s[i];
                let row_off = i * d;
                for j in i..d {
                    g[row_off + j] += zi * z_s[j];
                }
            }
        }
        for i in 0..d {
            g[i * d + i] += alpha;
            for j in 0..i {
                g[i * d + j] = g[j * d + i];
            }
        }

        // Build RHS B = Z^T Y [d x c], where Y[s, y_labels[s]] = 1.0 / num_heads
        let inv_h = 1.0 / (num_heads as f64);
        let mut b = vec![0.0f64; d * c];
        for s in 0..n {
            let cl = y_labels[s];
            if cl < c {
                let z_s = &z_batch[s * d..(s + 1) * d];
                for i in 0..d {
                    b[i * c + cl] += z_s[i] * inv_h;
                }
            }
        }

        // Augmented matrix [d x (2d + c)] for G * [P_inv | W] = [I_d | B]
        let m = 2 * d + c;
        let mut aug = vec![0.0f64; d * m];
        for i in 0..d {
            let row_off = i * m;
            for j in 0..d {
                aug[row_off + j] = g[i * d + j];
            }
            aug[row_off + d + i] = 1.0;
            for cl in 0..c {
                aug[row_off + 2 * d + cl] = b[i * c + cl];
            }
        }

        // Gauss-Jordan elimination with partial pivoting
        for col in 0..d {
            let mut max_val = aug[col * m + col].abs();
            let mut pivot_row = col;
            for r in (col + 1)..d {
                let val = aug[r * m + col].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = r;
                }
            }

            if pivot_row != col {
                for k in 0..m {
                    let tmp = aug[col * m + k];
                    aug[col * m + k] = aug[pivot_row * m + k];
                    aug[pivot_row * m + k] = tmp;
                }
            }

            let pivot = aug[col * m + col];
            let inv_pivot = if pivot.abs() > 1e-15 { 1.0 / pivot } else { 1.0 };
            aug[col * m + col] = 1.0;
            for k in (col + 1)..m {
                aug[col * m + k] *= inv_pivot;
            }

            for r in 0..d {
                if r != col {
                    let factor = aug[r * m + col];
                    if factor.abs() > 1e-15 {
                        aug[r * m + col] = 0.0;
                        for k in (col + 1)..m {
                            let sub = factor * aug[col * m + k];
                            aug[r * m + k] -= sub;
                        }
                    }
                }
            }
        }

        // Extract P_inv and W
        for i in 0..d {
            let row_off = i * m;
            for j in 0..d {
                self.p_inv_f64[i * d + j] = aug[row_off + d + j];
            }
            for cl in 0..c {
                self.w_f64[i * c + cl] = aug[row_off + 2 * d + cl];
            }
        }
    }

    pub fn fit_closed_form_regression(
        &mut self,
        z_batch: &[f64],
        y_norm: &[f64],
        alpha: f64,
        num_heads: usize,
    ) {
        let n = y_norm.len();
        if n == 0 {
            return;
        }
        let d = self.d_circuits;
        let c = self.num_classes; // c is 1 for regression

        // Build Gram matrix G = Z^T Z + alpha * I_d [d x d]
        let mut g = vec![0.0f64; d * d];
        for s in 0..n {
            let z_s = &z_batch[s * d..(s + 1) * d];
            for i in 0..d {
                let zi = z_s[i];
                let row_off = i * d;
                for j in i..d {
                    g[row_off + j] += zi * z_s[j];
                }
            }
        }
        for i in 0..d {
            g[i * d + i] += alpha;
            for j in 0..i {
                g[i * d + j] = g[j * d + i];
            }
        }

        // Build RHS B = Z^T Y_scaled [d x c], where Y_scaled[s, 0] = y_norm[s] / num_heads
        let inv_h = 1.0 / (num_heads as f64);
        let mut b = vec![0.0f64; d * c];
        for s in 0..n {
            let val = y_norm[s] * inv_h;
            let z_s = &z_batch[s * d..(s + 1) * d];
            for i in 0..d {
                b[i * c + 0] += z_s[i] * val;
            }
        }

        // Augmented matrix [d x (2d + c)] for G * [P_inv | W] = [I_d | B]
        let m = 2 * d + c;
        let mut aug = vec![0.0f64; d * m];
        for i in 0..d {
            let row_off = i * m;
            for j in 0..d {
                aug[row_off + j] = g[i * d + j];
            }
            aug[row_off + d + i] = 1.0;
            for cl in 0..c {
                aug[row_off + 2 * d + cl] = b[i * c + cl];
            }
        }

        // Gauss-Jordan elimination with partial pivoting
        for col in 0..d {
            let mut max_val = aug[col * m + col].abs();
            let mut pivot_row = col;
            for r in (col + 1)..d {
                let val = aug[r * m + col].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = r;
                }
            }

            if pivot_row != col {
                for k in 0..m {
                    let tmp = aug[col * m + k];
                    aug[col * m + k] = aug[pivot_row * m + k];
                    aug[pivot_row * m + k] = tmp;
                }
            }

            let pivot = aug[col * m + col];
            let inv_pivot = if pivot.abs() > 1e-15 { 1.0 / pivot } else { 1.0 };
            aug[col * m + col] = 1.0;
            for k in (col + 1)..m {
                aug[col * m + k] *= inv_pivot;
            }

            for r in 0..d {
                if r != col {
                    let factor = aug[r * m + col];
                    if factor.abs() > 1e-15 {
                        aug[r * m + col] = 0.0;
                        for k in (col + 1)..m {
                            let sub = factor * aug[col * m + k];
                            aug[r * m + k] -= sub;
                        }
                    }
                }
            }
        }

        // Extract P_inv and W
        for i in 0..d {
            let row_off = i * m;
            for j in 0..d {
                self.p_inv_f64[i * d + j] = aug[row_off + d + j];
            }
            for cl in 0..c {
                self.w_f64[i * c + cl] = aug[row_off + 2 * d + cl];
            }
        }
    }

    #[inline(always)]
    pub fn update_rls_self(
        &mut self,
        residual: &[f64],
        sample_weight: f64,
    ) {
        let d = self.d_circuits;
        let c = self.num_classes;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && d % 16 == 0 {
                let p_ptr = self.p_inv_f64.as_mut_ptr();
                let z_ptr = self.z_f64.as_ptr();
                let k_ptr = self.k_f64.as_mut_ptr();
                let w_ptr = self.w_f64.as_mut_ptr();
                let res_ptr = residual.as_ptr();

                let mut z_dot_k = 0.0f64;
                for i in 0..d {
                    let row_ptr = p_ptr.add(i * d);
                    let mut acc0 = _mm256_setzero_pd();
                    let mut acc1 = _mm256_setzero_pd();
                    let mut acc2 = _mm256_setzero_pd();
                    let mut acc3 = _mm256_setzero_pd();

                    let mut j = 0;
                    while j < d {
                        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(row_ptr.add(j)), _mm256_loadu_pd(z_ptr.add(j)), acc0);
                        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(row_ptr.add(j + 4)), _mm256_loadu_pd(z_ptr.add(j + 4)), acc1);
                        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(row_ptr.add(j + 8)), _mm256_loadu_pd(z_ptr.add(j + 8)), acc2);
                        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(row_ptr.add(j + 12)), _mm256_loadu_pd(z_ptr.add(j + 12)), acc3);
                        j += 16;
                    }

                    let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
                    let hi = _mm256_extractf128_pd(acc, 1);
                    let lo = _mm256_castpd256_pd128(acc);
                    let sum128 = _mm_add_pd(lo, hi);
                    let sum_hi = _mm_unpackhi_pd(sum128, sum128);
                    let row_sum = _mm_cvtsd_f64(_mm_add_sd(sum128, sum_hi));

                    *k_ptr.add(i) = row_sum;
                    z_dot_k += *z_ptr.add(i) * row_sum;
                }

                let w_eff = sample_weight.max(1e-6);
                let base_gamma = 1.0 / ((1.0 / w_eff) + z_dot_k.max(0.0));
                // Few-shot novelty floor when user explicitly flags high-priority/rare sample (sample_weight > 2.0)
                let gamma = if sample_weight > 2.0 {
                    base_gamma.max(0.06 * (sample_weight / 5.0).min(2.0))
                } else {
                    base_gamma
                };

                for i in 0..d {
                    let g_ki = _mm256_set1_pd(gamma * *k_ptr.add(i));
                    let row_ptr = p_ptr.add(i * d);

                    let mut j = 0;
                    while j < d {
                        let p0 = _mm256_loadu_pd(row_ptr.add(j));
                        let k0 = _mm256_loadu_pd(k_ptr.add(j));
                        _mm256_storeu_pd(row_ptr.add(j), _mm256_fnmadd_pd(g_ki, k0, p0));

                        let p1 = _mm256_loadu_pd(row_ptr.add(j + 4));
                        let k1 = _mm256_loadu_pd(k_ptr.add(j + 4));
                        _mm256_storeu_pd(row_ptr.add(j + 4), _mm256_fnmadd_pd(g_ki, k1, p1));

                        let p2 = _mm256_loadu_pd(row_ptr.add(j + 8));
                        let k2 = _mm256_loadu_pd(k_ptr.add(j + 8));
                        _mm256_storeu_pd(row_ptr.add(j + 8), _mm256_fnmadd_pd(g_ki, k2, p2));

                        let p3 = _mm256_loadu_pd(row_ptr.add(j + 12));
                        let k3 = _mm256_loadu_pd(k_ptr.add(j + 12));
                        _mm256_storeu_pd(row_ptr.add(j + 12), _mm256_fnmadd_pd(g_ki, k3, p3));

                        j += 16;
                    }
                }

                for i in 0..d {
                    let g_ki = gamma * *k_ptr.add(i);
                    let w_row = w_ptr.add(i * c);
                    for cl in 0..c {
                        *w_row.add(cl) += g_ki * *res_ptr.add(cl);
                    }
                }
                return;
            }
        }

        // Fallback
        let mut z_dot_k = 0.0f64;
        for i in 0..d {
            let row = &self.p_inv_f64[i * d..(i + 1) * d];
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += row[j] * self.z_f64[j];
            }
            self.k_f64[i] = sum;
            z_dot_k += self.z_f64[i] * sum;
        }

        let w_eff = sample_weight.max(1e-6);
        let base_gamma = 1.0 / ((1.0 / w_eff) + z_dot_k.max(0.0));
        // Few-shot novelty floor when user explicitly flags high-priority/rare sample (sample_weight > 2.0)
        let gamma = if sample_weight > 2.0 {
            base_gamma.max(0.06 * (sample_weight / 5.0).min(2.0))
        } else {
            base_gamma
        };

        for i in 0..d {
            let g_ki = gamma * self.k_f64[i];
            let row = &mut self.p_inv_f64[i * d..(i + 1) * d];
            for j in 0..d {
                row[j] -= g_ki * self.k_f64[j];
            }
        }

        for i in 0..d {
            let g_ki = gamma * self.k_f64[i];
            let w_row = &mut self.w_f64[i * c..(i + 1) * c];
            for cl in 0..c {
                w_row[cl] += g_ki * residual[cl];
            }
        }
    }

    #[inline(always)]
    pub fn update_rls(
        &mut self,
        residual: &[f64],
        z_f64: &[f64],
        sample_weight: f64,
    ) {
        self.z_f64[..self.d_circuits].copy_from_slice(&z_f64[..self.d_circuits]);
        self.update_rls_self(residual, sample_weight);
    }

    pub fn weights(&self) -> &[f64] {
        &self.w_f64
    }

    pub fn output_dim(&self) -> usize {
        self.d_circuits
    }
}
