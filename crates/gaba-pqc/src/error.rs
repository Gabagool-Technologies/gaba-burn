use thiserror::Error;

#[derive(Error, Debug)]
pub enum PqcError {
    #[error("Cryptographic operation failed: {0}")]
    CryptoError(String),

    #[error("Invalid seal format: {0}")]
    InvalidFormat(String),

    #[error("Verification failed")]
    VerificationFailed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[cfg(feature = "metal")]
    #[error("Metal acceleration error: {0}")]
    Metal(String),
}

pub type PqcResult<T> = Result<T, PqcError>;
