//! aicore_fw: High-performance continual learning subspace engine in Rust.

pub mod simd;
pub mod math_avx2;
pub mod frontend;
pub mod head;
pub mod engine;

pub use engine::{MultiHeadSubspaceEngine, AICoreConfig, AICoreError};
pub use frontend::{FeatureFrontend, IdentityFrontend, OnlineZScoreFrontend, VisionGaborFrontend};
pub use head::SubspaceHead;

