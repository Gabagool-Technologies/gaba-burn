pub mod onnx;
pub mod tflite;
pub mod coreml;
pub mod webnn;

pub use onnx::OnnxExporter;
pub use tflite::TfLiteExporter;
pub use coreml::CoreMlExporter;
pub use webnn::WebNnExporter;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Invalid model structure: {0}")]
    InvalidModel(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Export error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ExportError>;

pub trait ModelExporter {
    fn export<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()>;
    fn export_bytes(&self) -> Result<Vec<u8>>;
}
