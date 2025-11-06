pub mod vector_store;
pub mod hnsw;
pub mod associations;
pub mod temporal;
pub mod engram;
pub mod hopfield;

pub use vector_store::{MemoryChunk, GabaMemory, MemoryLayer, EngramState};
pub use hnsw::HnswIndex;
pub use associations::AssociativeGraph;
pub use temporal::TemporalMemory;
pub use engram::EngramManager;
pub use hopfield::HopfieldLayer;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Chunk not found: {0}")]
    ChunkNotFound(uuid::Uuid),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
