//! SIMD and fast math utilities for AICore.

#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
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
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

pub fn simd_matvec_f32(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &a[i * cols..(i + 1) * cols];
        out[i] = simd_dot_f32(row, x);
    }
}

pub fn simd_fast_cos_f32(x: &mut [f32], bias: &[f32], len: usize) {
    for i in 0..len {
        x[i] = (x[i] + bias[i]).cos();
    }
}

pub fn simd_matvec_f64(a: &[f64], x: &[f64], out: &mut [f64], dim: usize) {
    for i in 0..dim {
        let mut sum: f64 = 0.0;
        let row = &a[i * dim..(i + 1) * dim];
        for j in 0..dim {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}
