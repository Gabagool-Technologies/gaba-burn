extern crate alloc;

use alloc::string::{String, ToString};
use core::fmt;

/// Unified error used by device and related APIs.
///
/// This simple error enum maps low-level device failures to a common type
/// that higher-level crates can convert into their own error domains.
#[derive(Debug, Clone)]
pub enum Error {
    /// Allocation on the device failed.
    AllocFailed,

    /// Copy from host to device failed.
    CopyFailed,

    /// A backend/kernel reported an error with an attached message.
    KernelError(String),

    /// Generic non-specific error with message.
    Other(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AllocFailed => write!(f, "allocation failed"),
            Error::CopyFailed => write!(f, "copy to device failed"),
            Error::KernelError(s) => write!(f, "kernel error: {}", s),
            Error::Other(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
