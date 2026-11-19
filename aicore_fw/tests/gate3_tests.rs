//! Phase 3 Gate Verification Test Suite: Native Engine State Serialization and Integrity.

use aicore_fw::engine::{AICoreConfig, MultiHeadSubspaceEngine};
use aicore_fw::frontend::{IdentityFrontend, OnlineZScoreFrontend};

#[test]
fn test_01_rust_engine_bincode_bit_exact_roundtrip() {
    let input_dim = 16;
    let num_heads = 4;
    let head_circuits = 128;
    let num_samples = 100;

    let config = AICoreConfig {
        input_dim,
        num_heads,
        head_circuits,
        alpha: 8.0,
        forgetting_factor: 1.0,
        target_accuracy: 0.95,
        num_classes: 3,
        low_freq_scale: 0.08,
        high_freq_scale: 0.24,
    };

    let mut engine = MultiHeadSubspaceEngine::new(config).unwrap();
    let frontend = IdentityFrontend::new(input_dim);

    // Generate deterministic training batch
    let mut x_batch = vec![0.0f32; num_samples * input_dim];
    let mut y_batch = vec![0usize; num_samples];
    for s in 0..num_samples {
        y_batch[s] = s % 3;
        for d in 0..input_dim {
            x_batch[s * input_dim + d] = ((s * 31 + d * 17) % 100) as f32 / 50.0 - 1.0;
        }
    }

    // Fit batch
    engine.fit_batch(&frontend, &x_batch, &y_batch);

    // Pre-serialization predictions
    let mut orig_preds = vec![0usize; num_samples];
    for s in 0..num_samples {
        let sample = &x_batch[s * input_dim..(s + 1) * input_dim];
        orig_preds[s] = engine.predict(&frontend, sample);
    }

    // Serialize engine via bincode
    let encoded = bincode::serialize(&engine).expect("Failed to serialize engine");
    assert!(!encoded.is_empty());

    // Deserialize engine
    let deserialized: MultiHeadSubspaceEngine =
        bincode::deserialize(&encoded).expect("Failed to deserialize engine");

    // Verify structural equality
    assert_eq!(engine.input_dim, deserialized.input_dim);
    assert_eq!(engine.num_heads, deserialized.num_heads);
    assert_eq!(engine.head_circuits, deserialized.head_circuits);
    assert_eq!(engine.num_classes, deserialized.num_classes);
    assert_eq!(engine.heads.len(), deserialized.heads.len());

    // Verify bit-exact parity across all heads
    for h in 0..engine.heads.len() {
        let h_orig = &engine.heads[h];
        let h_restored = &deserialized.heads[h];

        assert_eq!(h_orig.w_f64, h_restored.w_f64, "Head {} weights drift", h);
        assert_eq!(h_orig.p_inv_f64, h_restored.p_inv_f64, "Head {} P_inv drift", h);
        assert_eq!(h_orig.proj_f32, h_restored.proj_f32, "Head {} projection drift", h);
        assert_eq!(h_orig.bias_f32, h_restored.bias_f32, "Head {} bias drift", h);
    }

    // Verify bit-exact prediction parity
    for s in 0..num_samples {
        let sample = &x_batch[s * input_dim..(s + 1) * input_dim];
        let restored_pred = deserialized.predict(&frontend, sample);
        assert_eq!(
            orig_preds[s], restored_pred,
            "Prediction mismatch on sample {}",
            s
        );
    }
}

#[test]
fn test_02_rust_engine_regression_bincode_roundtrip() {
    let input_dim = 12;
    let num_heads = 4;
    let head_circuits = 96;
    let num_samples = 80;

    let config = AICoreConfig {
        input_dim,
        num_heads,
        head_circuits,
        alpha: 5.0,
        forgetting_factor: 1.0,
        target_accuracy: 0.95,
        num_classes: 1,
        low_freq_scale: 0.1,
        high_freq_scale: 0.3,
    };

    let mut engine = MultiHeadSubspaceEngine::new(config).unwrap();
    let zscore = OnlineZScoreFrontend::new(input_dim);

    let mut x_batch = vec![0.0f32; num_samples * input_dim];
    let mut y_batch = vec![0.0f64; num_samples];
    for s in 0..num_samples {
        y_batch[s] = ((s * 41) % 100) as f64 * 0.5 - 10.0;
        for d in 0..input_dim {
            x_batch[s * input_dim + d] = ((s * 19 + d * 23) % 50) as f32 / 25.0;
        }
    }

    // Fit regression
    engine.fit_regression_batch(&zscore, &x_batch, &y_batch);

    let mut orig_preds = vec![0.0f64; num_samples];
    engine.predict_regression_batch(&zscore, &x_batch, &mut orig_preds);

    let encoded = bincode::serialize(&engine).expect("Failed to serialize regression engine");
    let mut deserialized: MultiHeadSubspaceEngine =
        bincode::deserialize(&encoded).expect("Failed to deserialize regression engine");

    assert_eq!(engine.y_mean, deserialized.y_mean);
    assert_eq!(engine.y_std, deserialized.y_std);

    let mut restored_preds = vec![0.0f64; num_samples];
    deserialized.predict_regression_batch(&zscore, &x_batch, &mut restored_preds);

    for s in 0..num_samples {
        let diff = (orig_preds[s] - restored_preds[s]).abs();
        assert_eq!(diff, 0.0, "Regression prediction drift on sample {}", s);
    }

    // Verify online learning resumes cleanly on restored engine
    let test_sample = &x_batch[0..input_dim];
    deserialized.update_regression_online(&zscore, test_sample, 15.0, 1.0);
    let updated_pred = deserialized.predict_regression(&zscore, test_sample);
    assert!(updated_pred.is_finite());
}

#[test]
fn test_03_rust_corrupt_payload_rejection() {
    // 1. Truncated bincode payload
    let valid_payload = [0u8; 128];
    let truncated = &valid_payload[0..16];
    let res: Result<MultiHeadSubspaceEngine, _> = bincode::deserialize(truncated);
    assert!(res.is_err(), "Truncated payload must return Err");

    // 2. Garbage random bytes
    let garbage = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
    let res2: Result<MultiHeadSubspaceEngine, _> = bincode::deserialize(&garbage);
    assert!(res2.is_err(), "Garbage payload must return Err");
}
