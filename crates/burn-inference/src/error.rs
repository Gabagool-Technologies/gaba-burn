//! Error types for inference

use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("ONNX runtime error: {0}")]
    OnnxError(#[from] ort::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, InferenceError>;
