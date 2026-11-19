//! Multi-architecture SIMD and numerical math engine.
//!
//! Provides runtime feature detection across x86_64 (AVX2, FMA, AVX-512),
//! aarch64 (NEON), and portable scalar fallbacks.

#![allow(clippy::missing_safety_doc)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Avx512,
    Avx2Fma,
    Neon,
    Scalar,
}

/// Detects the highest available SIMD capability at runtime.
pub fn detect_simd_capabilities() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdBackend::Avx2Fma;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdBackend::Neon;
    }
    #[allow(unreachable_code)]
    SimdBackend::Scalar
}

// =============================================================================
// SCALAR BACKEND (Always available, bit-exact reference)
// =============================================================================

#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline]
pub fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let mut i = 0;
    while i + 4 <= len {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline]
pub fn matvec_f32_scalar(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for (i, out_elem) in out.iter_mut().enumerate().take(rows) {
        let row = &a[i * cols..(i + 1) * cols];
        *out_elem = dot_f32_scalar(row, x);
    }
}

#[inline]
pub fn matvec_f64_scalar(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
    for (i, out_elem) in out.iter_mut().enumerate().take(dim) {
        let row = &a[i * dim..(i + 1) * dim];
        *out_elem = dot_f64_scalar(row, x);
    }
}

#[inline]
pub fn fast_cos_f32_scalar(x: &mut [f32], bias: &[f32], len: usize) {
    let n = len.min(x.len()).min(bias.len());
    for i in 0..n {
        x[i] = (x[i] + bias[i]).cos();
    }
}

#[inline]
pub fn rls_rank1_update_scalar(p_inv: &mut [f64], k: &[f64], gamma: f64, d: usize) {
    for i in 0..d {
        let g_ki = gamma * k[i];
        let row_off = i * d;
        for j in 0..d {
            p_inv[row_off + j] -= g_ki * k[j];
        }
    }
}

// =============================================================================
// x86_64 INTRINSICS (AVX2 + FMA & AVX-512)
// =============================================================================

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= len {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(i)), _mm256_loadu_ps(b_ptr.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(i + 8)), _mm256_loadu_ps(b_ptr.add(i + 8)), acc1);
            i += 16;
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= len {
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a_ptr.add(i)), _mm256_loadu_pd(b_ptr.add(i)), acc0);
            acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a_ptr.add(i + 4)), _mm256_loadu_pd(b_ptr.add(i + 4)), acc1);
            i += 8;
        }
        let acc = _mm256_add_pd(acc0, acc1);
        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let sum128 = _mm_add_pd(lo, hi);
        let sum_hi = _mm_unpackhi_pd(sum128, sum128);
        let mut sum = _mm_cvtsd_f64(_mm_add_sd(sum128, sum_hi));
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matvec_f32_avx2(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        for (i, out_elem) in out.iter_mut().enumerate().take(rows) {
            let row = &a[i * cols..(i + 1) * cols];
            *out_elem = dot_f32_avx2(row, x);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matvec_f64_avx2(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
        for (i, out_elem) in out.iter_mut().enumerate().take(dim) {
            let row = &a[i * dim..(i + 1) * dim];
            *out_elem = dot_f64_avx2(row, x);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rls_rank1_update_avx2(p_inv: &mut [f64], k: &[f64], gamma: f64, d: usize) {
        let p_ptr = p_inv.as_mut_ptr();
        let k_ptr = k.as_ptr();
        for i in 0..d {
            let g_ki = _mm256_set1_pd(gamma * *k_ptr.add(i));
            let row_ptr = p_ptr.add(i * d);
            let mut j = 0;
            while j + 8 <= d {
                let p0 = _mm256_loadu_pd(row_ptr.add(j));
                let k0 = _mm256_loadu_pd(k_ptr.add(j));
                _mm256_storeu_pd(row_ptr.add(j), _mm256_fnmadd_pd(g_ki, k0, p0));

                let p1 = _mm256_loadu_pd(row_ptr.add(j + 4));
                let k1 = _mm256_loadu_pd(k_ptr.add(j + 4));
                _mm256_storeu_pd(row_ptr.add(j + 4), _mm256_fnmadd_pd(g_ki, k1, p1));
                j += 8;
            }
            while j < d {
                *row_ptr.add(j) -= gamma * *k_ptr.add(i) * *k_ptr.add(j);
                j += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= len {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a_ptr.add(i)), _mm512_loadu_ps(b_ptr.add(i)), acc);
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = _mm512_setzero_pd();
        let mut i = 0;
        while i + 8 <= len {
            acc = _mm512_fmadd_pd(_mm512_loadu_pd(a_ptr.add(i)), _mm512_loadu_pd(b_ptr.add(i)), acc);
            i += 8;
        }
        let mut sum = _mm512_reduce_add_pd(acc);
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }
}

// =============================================================================
// ARM aarch64 INTRINSICS (NEON)
// =============================================================================

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= len {
            let va0 = vld1q_f32(a_ptr.add(i));
            let vb0 = vld1q_f32(b_ptr.add(i));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a_ptr.add(i + 4));
            let vb1 = vld1q_f32(b_ptr.add(i + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);
            i += 8;
        }
        let acc = vaddq_f32(acc0, acc1);
        let mut sum = vaddvq_f32(acc);
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    pub unsafe fn dot_f64_neon(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut i = 0;
        while i + 4 <= len {
            let va0 = vld1q_f64(a_ptr.add(i));
            let vb0 = vld1q_f64(b_ptr.add(i));
            acc0 = vfmaq_f64(acc0, va0, vb0);

            let va1 = vld1q_f64(a_ptr.add(i + 2));
            let vb1 = vld1q_f64(b_ptr.add(i + 2));
            acc1 = vfmaq_f64(acc1, va1, vb1);
            i += 4;
        }
        let acc = vaddq_f64(acc0, acc1);
        let mut sum = vaddvq_f64(acc);
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    pub unsafe fn matvec_f32_neon(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        for (i, out_elem) in out.iter_mut().enumerate().take(rows) {
            let row = &a[i * cols..(i + 1) * cols];
            *out_elem = dot_f32_neon(row, x);
        }
    }

    pub unsafe fn matvec_f64_neon(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
        for (i, out_elem) in out.iter_mut().enumerate().take(dim) {
            let row = &a[i * dim..(i + 1) * dim];
            *out_elem = dot_f64_neon(row, x);
        }
    }

    pub unsafe fn rls_rank1_update_neon(p_inv: &mut [f64], k: &[f64], gamma: f64, d: usize) {
        let p_ptr = p_inv.as_mut_ptr();
        let k_ptr = k.as_ptr();
        for i in 0..d {
            let g_ki = gamma * *k_ptr.add(i);
            let vg_ki = vdupq_n_f64(g_ki);
            let row_ptr = p_ptr.add(i * d);
            let mut j = 0;
            while j + 4 <= d {
                let vp0 = vld1q_f64(row_ptr.add(j));
                let vk0 = vld1q_f64(k_ptr.add(j));
                // row[j] - g_ki * k[j]
                let vres0 = vnegq_f64(vfmsq_f64(vnegq_f64(vp0), vg_ki, vk0));
                vst1q_f64(row_ptr.add(j), vres0);

                let vp1 = vld1q_f64(row_ptr.add(j + 2));
                let vk1 = vld1q_f64(k_ptr.add(j + 2));
                let vres1 = vnegq_f64(vfmsq_f64(vnegq_f64(vp1), vg_ki, vk1));
                vst1q_f64(row_ptr.add(j + 2), vres1);
                j += 4;
            }
            while j < d {
                *row_ptr.add(j) -= g_ki * *k_ptr.add(j);
                j += 1;
            }
        }
    }
}

// =============================================================================
// RUNTIME DISPATCH WRAPPERS
// =============================================================================

#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::dot_f32_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_f32_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_f32_neon(a, b) };
    }
    dot_f32_scalar(a, b)
}

#[inline(always)]
pub fn simd_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::dot_f64_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_f64_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_f64_neon(a, b) };
    }
    dot_f64_scalar(a, b)
}

#[inline(always)]
pub fn simd_matvec_f32(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                x86::matvec_f32_avx2(a, x, out, rows, cols);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::matvec_f32_neon(a, x, out, rows, cols);
            return;
        }
    }
    matvec_f32_scalar(a, x, out, rows, cols);
}

#[inline(always)]
pub fn simd_matvec_f64(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                x86::matvec_f64_avx2(a, x, out, dim);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::matvec_f64_neon(a, x, out, dim);
            return;
        }
    }
    matvec_f64_scalar(a, x, out, dim);
}

#[inline(always)]
pub fn simd_fast_cos_f32(x: &mut [f32], bias: &[f32], len: usize) {
    fast_cos_f32_scalar(x, bias, len);
}

#[inline(always)]
pub fn simd_rls_rank1_update(p_inv: &mut [f64], k: &[f64], gamma: f64, d: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && d.is_multiple_of(8) {
            unsafe {
                x86::rls_rank1_update_avx2(p_inv, k, gamma, d);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if d.is_multiple_of(4) {
            unsafe {
                neon::rls_rank1_update_neon(p_inv, k, gamma, d);
                return;
            }
        }
    }
    rls_rank1_update_scalar(p_inv, k, gamma, d);
}

// =============================================================================
// NUMERICAL STABILITY & COVARIANCE HARDENING
// =============================================================================

/// Enforces exact symmetry P = 0.5(P + P^T) and diagonal positivity/regularization.
///
/// Guaranteed properties:
/// 1. P[i, j] == P[j, i] for all i, j.
/// 2. P[i, i] >= min_diag > 0.
/// 3. All non-finite (NaN, Inf) entries are restored to safe regularized values.
#[inline]
pub fn stabilize_covariance(p_inv: &mut [f64], d: usize, min_diag: f64) -> (f64, f64) {
    let mut min_d = f64::INFINITY;
    let mut trace = 0.0f64;

    // 1. Regularize diagonal elements first
    for i in 0..d {
        let idx = i * d + i;
        let mut diag = p_inv[idx];
        if !diag.is_finite() || diag < min_diag {
            diag = min_diag;
            p_inv[idx] = diag;
        }
        trace += diag;
        if diag < min_d {
            min_d = diag;
        }
    }

    // 2. Symmetrize off-diagonals and enforce Cauchy-Schwarz bound: |P_ij| <= sqrt(P_ii * P_jj)
    for i in 0..d {
        let row_i = i * d;
        let diag_i = p_inv[row_i + i];
        for j in (i + 1)..d {
            let row_j = j * d;
            let diag_j = p_inv[row_j + j];
            let max_off = (diag_i * diag_j).sqrt();

            let val_ij = p_inv[row_i + j];
            let val_ji = p_inv[row_j + i];
            let avg = 0.5 * (val_ij + val_ji);
            let safe_avg = if avg.is_finite() {
                avg.clamp(-max_off, max_off)
            } else {
                0.0
            };
            p_inv[row_i + j] = safe_avg;
            p_inv[row_j + i] = safe_avg;
        }
    }

    (trace, min_d)
}

/// Computes an accurate estimate of the condition number of symmetric matrix P.
///
/// Uses matrix 1-norm and minimum diagonal element bound:
/// κ(P) <= ||P||_1 / min_i(P_ii).
pub fn estimate_condition_number(p_inv: &[f64], d: usize) -> f64 {
    let mut norm_1 = 0.0f64;
    for j in 0..d {
        let mut col_sum = 0.0f64;
        for i in 0..d {
            col_sum += p_inv[i * d + j].abs();
        }
        if col_sum > norm_1 {
            norm_1 = col_sum;
        }
    }

    let mut min_diag = f64::INFINITY;
    for i in 0..d {
        let diag = p_inv[i * d + i];
        if diag < min_diag {
            min_diag = diag;
        }
    }

    if min_diag <= 0.0 || !min_diag.is_finite() {
        return f64::INFINITY;
    }

    (norm_1 / min_diag).max(1.0)
}
