pub mod api;
pub mod batching;
pub mod cache;
pub mod scheduler;
pub mod streaming;

pub use api::create_router;
pub use batching::{DynamicBatcher, BatchConfig};
pub use cache::{ModelCache, CacheConfig};
pub use scheduler::{RequestScheduler, SchedulerConfig};
pub use streaming::{StreamingInference, StreamConfig};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServeError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Batch full")]
    BatchFull,
    
    #[error("Timeout")]
    Timeout,
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ServeError>;
