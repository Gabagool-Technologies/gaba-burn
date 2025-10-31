//! Post-Quantum Cryptography for Burn Training Results
//!
//! Provides quantum-resistant encryption for model weights, checkpoints, and training artifacts.
//! Implements BLAKE3 hashing with optional Metal GPU acceleration on Apple Silicon.

pub mod error;
pub mod seal;

#[cfg(feature = "metal")]
pub mod metal_accel;

pub use error::{PqcError, PqcResult};
pub use seal::{Seal, create_seal, verify_seal};

/// Encrypt training checkpoint with PQC
pub async fn encrypt_checkpoint(data: &[u8]) -> PqcResult<Vec<u8>> {
    let seal = create_seal(data).await?;
    Ok(seal.to_bytes())
}

/// Verify and decrypt training checkpoint
pub async fn verify_checkpoint(sealed_data: &[u8], original_data: &[u8]) -> PqcResult<bool> {
    let seal = Seal::from_bytes(sealed_data)?;
    verify_seal(&seal, original_data).await
}
