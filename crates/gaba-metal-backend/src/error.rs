use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetalError {
    #[error("Metal device not available")]
    DeviceNotAvailable,

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Kernel execution failed: {0}")]
    KernelExecution(String),

    #[error("Buffer allocation failed: {0}")]
    BufferAllocation(String),

    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    #[error("Unified memory error: {0}")]
    UnifiedMemory(String),
}

pub type MetalResult<T> = Result<T, MetalError>;
