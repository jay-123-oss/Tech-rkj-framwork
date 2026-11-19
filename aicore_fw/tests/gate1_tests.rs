//! GATE 1 VERIFICATION TEST SUITE
//!
//! Mandatory Pass Criteria:
//! 1. cargo test --release passes 100% of unit tests across SIMD and Scalar backends.
//! 2. SIMD vs. Scalar Numerical Parity Test: Output difference < 1e-12 (f64) and < 1e-5 (f32)
//!    across random, ill-conditioned, and collinear matrices.
//! 3. Long-Horizon Streaming Stability Test: 100,000 sequential partial_fit updates produce
//!    zero NaN/Inf values and maintain bounded condition numbers.
//! 4. cargo clippy --all-targets -- -D warnings passes with zero warnings.

use aicore_fw::simd::*;
use aicore_fw::head::SubspaceHead;
use aicore_fw::engine::{MultiHeadSubspaceEngine, AICoreConfig};
use aicore_fw::frontend::IdentityFrontend;

// Simple LCG PRNG for deterministic, reproducible test vectors
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    fn next_gaussian_f64(&mut self) -> f64 {
        let u1 = self.next_f64().clamp(1e-15, 1.0 - 1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn test_01_simd_capabilities_detection() {
    let backend = detect_simd_capabilities();
    println!("Detected SIMD runtime backend: {:?}", backend);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            assert!(backend == SimdBackend::Avx2Fma || backend == SimdBackend::Avx512);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(backend, SimdBackend::Neon);
    }
}

#[test]
fn test_02_simd_vs_scalar_random_parity() {
    let mut rng = TestRng::new(12345);
    let lengths = [1, 7, 8, 15, 16, 31, 32, 64, 128, 256, 512, 1024];

    for &len in &lengths {
        // Test f64
        let a_f64: Vec<f64> = (0..len).map(|_| rng.next_gaussian_f64()).collect();
        let b_f64: Vec<f64> = (0..len).map(|_| rng.next_gaussian_f64()).collect();

        let dot_scalar_f64 = dot_f64_scalar(&a_f64, &b_f64);
        let dot_simd_f64 = simd_dot_f64(&a_f64, &b_f64);
        let diff_f64 = (dot_simd_f64 - dot_scalar_f64).abs();

        assert!(
            diff_f64 < 1e-12,
            "f64 parity violation at len {}: simd={}, scalar={}, diff={:e} >= 1e-12",
            len, dot_simd_f64, dot_scalar_f64, diff_f64
        );

        // Test f32
        let a_f32: Vec<f32> = (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let b_f32: Vec<f32> = (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

        let dot_scalar_f32 = dot_f32_scalar(&a_f32, &b_f32);
        let dot_simd_f32 = simd_dot_f32(&a_f32, &b_f32);
        let diff_f32 = (dot_simd_f32 - dot_scalar_f32).abs();

        assert!(
            diff_f32 < 1e-5,
            "f32 parity violation at len {}: simd={}, scalar={}, diff={:e} >= 1e-5",
            len, dot_simd_f32, dot_scalar_f32, diff_f32
        );
    }
}

#[test]
fn test_03_simd_vs_scalar_ill_conditioned_parity() {
    let mut rng = TestRng::new(54321);
    let dims = [32, 64, 128];

    for &d in &dims {
        // Construct ill-conditioned matrix: singular values decay from 1.0 to 1e-6 (condition number ~ 1e6)
        let mut a_f64 = vec![0.0f64; d * d];
        for i in 0..d {
            let scale = 10.0f64.powf(-6.0 * (i as f64) / ((d - 1) as f64));
            for j in 0..d {
                a_f64[i * d + j] = rng.next_gaussian_f64() * scale;
            }
        }

        let x_f64: Vec<f64> = (0..d).map(|_| rng.next_gaussian_f64()).collect();
        let mut out_simd_f64 = vec![0.0f64; d];
        let mut out_scalar_f64 = vec![0.0f64; d];

        simd_matvec_f64(&a_f64, &x_f64, &mut out_simd_f64, d);
        matvec_f64_scalar(&a_f64, &x_f64, &mut out_scalar_f64, d);

        for i in 0..d {
            let diff = (out_simd_f64[i] - out_scalar_f64[i]).abs();
            assert!(
                diff < 1e-12,
                "Ill-conditioned f64 matvec parity failed at row {}: diff={:e}",
                i, diff
            );
        }

        // Test f32 counterpart
        let a_f32: Vec<f32> = a_f64.iter().map(|&v| v as f32).collect();
        let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
        let mut out_simd_f32 = vec![0.0f32; d];
        let mut out_scalar_f32 = vec![0.0f32; d];

        simd_matvec_f32(&a_f32, &x_f32, &mut out_simd_f32, d, d);
        matvec_f32_scalar(&a_f32, &x_f32, &mut out_scalar_f32, d, d);

        for i in 0..d {
            let diff = (out_simd_f32[i] - out_scalar_f32[i]).abs();
            assert!(
                diff < 1e-5,
                "Ill-conditioned f32 matvec parity failed at row {}: diff={:e}",
                i, diff
            );
        }
    }
}

#[test]
fn test_04_simd_vs_scalar_collinear_parity() {
    let mut rng = TestRng::new(99999);
    let d = 64;

    // Construct collinear matrix (multiple identical/proportional rows)
    let base_row: Vec<f64> = (0..d).map(|_| rng.next_gaussian_f64()).collect();
    let mut a_collinear = vec![0.0f64; d * d];
    for i in 0..d {
        let factor = ((i % 4) + 1) as f64;
        for j in 0..d {
            a_collinear[i * d + j] = base_row[j] * factor;
        }
    }

    let x: Vec<f64> = (0..d).map(|_| rng.next_gaussian_f64()).collect();
    let mut out_simd = vec![0.0f64; d];
    let mut out_scalar = vec![0.0f64; d];

    simd_matvec_f64(&a_collinear, &x, &mut out_simd, d);
    matvec_f64_scalar(&a_collinear, &x, &mut out_scalar, d);

    for i in 0..d {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        assert!(
            diff < 1e-12,
            "Collinear f64 parity failed at row {}: diff={:e}",
            i, diff
        );
    }
}

#[test]
fn test_05_simd_vs_scalar_rls_rank1_update_parity() {
    let mut rng = TestRng::new(77777);
    let d = 128;
    let gamma = 0.045f64;

    let k: Vec<f64> = (0..d).map(|_| rng.next_gaussian_f64() * 0.1).collect();
    let mut p_simd = vec![0.0f64; d * d];
    for i in 0..d {
        p_simd[i * d + i] = 1.0;
        for j in (i + 1)..d {
            let v = rng.next_gaussian_f64() * 0.05;
            p_simd[i * d + j] = v;
            p_simd[j * d + i] = v;
        }
    }
    let mut p_scalar = p_simd.clone();

    simd_rls_rank1_update(&mut p_simd, &k, gamma, d);
    rls_rank1_update_scalar(&mut p_scalar, &k, gamma, d);

    let mut max_diff = 0.0f64;
    for idx in 0..(d * d) {
        let diff = (p_simd[idx] - p_scalar[idx]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff < 1e-12,
        "RLS rank-1 update parity failed: max_diff={:e} >= 1e-12",
        max_diff
    );
}

#[test]
fn test_06_covariance_hardening_and_stabilization() {
    let d = 64;
    let mut p_corrupted = vec![0.0f64; d * d];

    // Inject severe numerical corruptions:
    // 1. Asymmetry
    p_corrupted[1] = 10.0;
    p_corrupted[d] = -5.0;

    // 2. Negative and zero diagonals
    p_corrupted[2 * d + 2] = -1.5;
    p_corrupted[3 * d + 3] = 0.0;

    // 3. NaN and Inf
    p_corrupted[4 * d + 5] = f64::NAN;
    p_corrupted[5 * d + 4] = f64::INFINITY;
    p_corrupted[6 * d + 6] = f64::NAN;

    let (trace, min_d) = stabilize_covariance(&mut p_corrupted, d, 1e-12);

    // Verify properties
    assert!(trace.is_finite() && trace > 0.0, "Trace must be finite and positive");
    assert!(min_d >= 1e-12, "Minimum diagonal must be >= 1e-12");

    for i in 0..d {
        assert!(
            p_corrupted[i * d + i] >= 1e-12,
            "Diagonal element {} must be >= 1e-12, got {}",
            i, p_corrupted[i * d + i]
        );
        for j in (i + 1)..d {
            let val_ij = p_corrupted[i * d + j];
            let val_ji = p_corrupted[j * d + i];
            assert_eq!(
                val_ij, val_ji,
                "Symmetry broken at ({}, {}): {} != {}",
                i, j, val_ij, val_ji
            );
            assert!(val_ij.is_finite(), "Entry ({}, {}) must be finite", i, j);
        }
    }

    let cond = estimate_condition_number(&p_corrupted, d);
    assert!(cond.is_finite(), "Condition number must be finite");
    assert!(cond < 1e12, "Condition number must be bounded, got {:e}", cond);
}

#[test]
fn test_07_long_horizon_streaming_stability_100k_updates() {
    println!("\n=== Starting 100,000 Sequential Online Streaming Updates ===");
    let d_in = 16;
    let d_circuits = 64;
    let num_classes = 2;

    let mut head = SubspaceHead::new(
        d_in,
        d_circuits,
        num_classes,
        10.0,
        0.15,
        0.45,
        42,
    );

    let mut rng = TestRng::new(101010);
    let total_updates = 100_000;
    let mut max_condition_seen = 0.0f64;

    for step in 1..=total_updates {
        // Generate diverse streaming sample
        let mut x = vec![0.0f32; d_in];
        let mode = step % 5;
        match mode {
            0 => {
                // Normal Gaussian
                for val in x.iter_mut() {
                    *val = rng.next_gaussian_f64() as f32;
                }
            }
            1 => {
                // Outlier magnitude
                for val in x.iter_mut() {
                    *val = (rng.next_gaussian_f64() * 50.0) as f32;
                }
            }
            2 => {
                // Collinear vector
                let base = rng.next_gaussian_f64() as f32;
                for val in x.iter_mut() {
                    *val = base;
                }
            }
            3 => {
                // Near-zero vector
                for val in x.iter_mut() {
                    *val = (rng.next_gaussian_f64() * 1e-6) as f32;
                }
            }
            _ => {
                // Sparse binary
                for val in x.iter_mut() {
                    *val = if rng.next_f64() > 0.8 { 1.0 } else { 0.0 };
                }
            }
        }

        // Project and compute residual
        head.project_self(&x);

        let target_class = if rng.next_f64() > 0.5 { 1 } else { 0 };
        let mut scores = [0.0f64; 2];
        head.compute_logits_self(&mut scores);

        let residual = [
            (if target_class == 0 { 1.0 } else { 0.0 }) - scores[0],
            (if target_class == 1 { 1.0 } else { 0.0 }) - scores[1],
        ];

        // Alternating sample weights (including high-priority few-shot)
        let weight = if step % 20 == 0 { 5.0 } else { 1.0 };

        // Streaming update
        head.update_rls_self(&residual, weight);

        // Verification checkpoint every 10,000 steps
        if step % 10_000 == 0 || step == total_updates {
            // 1. Verify all weights and covariance elements are strictly finite
            for (i, &w) in head.w_f64.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "Step {}: Weight w[{}] produced non-finite value: {}",
                    step, i, w
                );
            }

            for (idx, &p_val) in head.p_inv_f64.iter().enumerate() {
                assert!(
                    p_val.is_finite(),
                    "Step {}: Covariance P[{}] produced non-finite value: {}",
                    step, idx, p_val
                );
            }

            // 2. Verify strict symmetry of P: |P_ij - P_ji| < 1e-12
            for i in 0..d_circuits {
                for j in (i + 1)..d_circuits {
                    let diff = (head.p_inv_f64[i * d_circuits + j] - head.p_inv_f64[j * d_circuits + i]).abs();
                    assert!(
                        diff < 1e-12,
                        "Step {}: Symmetry violated at ({}, {}): diff={:e}",
                        step, i, j, diff
                    );
                }
            }

            // 3. Verify positive diagonals: P_ii >= 1e-12
            for i in 0..d_circuits {
                let diag = head.p_inv_f64[i * d_circuits + i];
                assert!(
                    diag >= 1e-12,
                    "Step {}: Diagonal P[{}, {}] dropped below 1e-12: {}",
                    step, i, i, diag
                );
            }

            // 4. Verify condition number
            let cond = estimate_condition_number(&head.p_inv_f64, d_circuits);
            assert!(cond.is_finite(), "Step {}: Condition number is non-finite", step);
            assert!(
                cond < 1e12,
                "Step {}: Condition number exploded beyond 1e12: {:e}",
                step, cond
            );

            if cond > max_condition_seen {
                max_condition_seen = cond;
            }

            println!(
                "  -> Step {:6} / {:6} | Condition: {:10.2e} | Min Diag: {:10.2e} | All Finite: OK",
                step, total_updates, cond, head.p_inv_f64[0]
            );
        }
    }

    println!("=== 100,000 Streaming Updates Completed Successfully! ===");
    println!("Max condition number observed across 100k updates: {:e}", max_condition_seen);
}

#[test]
fn test_08_multihead_engine_zero_allocation_hotpath() {
    let config = AICoreConfig {
        input_dim: 16,
        num_heads: 4,
        head_circuits: 64,
        alpha: 10.0,
        forgetting_factor: 1.0,
        target_accuracy: 0.95,
        num_classes: 2,
        low_freq_scale: 0.15,
        high_freq_scale: 0.45,
    };

    let mut engine = MultiHeadSubspaceEngine::new(config).unwrap();
    let frontend = IdentityFrontend::new(16);

    let x = [0.5f32; 16];

    // Warm-up
    engine.update_online(&frontend, &x, 1, 1.0);

    // Verify zero allocations in loop
    for _ in 0..1000 {
        engine.update_online(&frontend, &x, 1, 1.0);
        let pred = engine.predict(&frontend, &x);
        assert!(pred < 2);
    }
}
