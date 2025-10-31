//! Zero-Copy Metal Backend for Burn
//!
//! Implements unified memory architecture for Apple Silicon M4 Pro
//! with zero-copy CPU/GPU/Neural Engine data sharing.

#[cfg(feature = "metal")]
pub mod backend;

#[cfg(feature = "metal")]
pub mod kernels;

#[cfg(feature = "metal")]
pub mod unified_memory;

#[cfg(feature = "metal")]
pub use backend::MetalBackend;

#[cfg(feature = "metal")]
pub use unified_memory::UnifiedBuffer;

pub mod error;
pub use error::{MetalError, MetalResult};

#[cfg(not(feature = "metal"))]
compile_error!("gaba-metal-backend requires the 'metal' feature to be enabled");
