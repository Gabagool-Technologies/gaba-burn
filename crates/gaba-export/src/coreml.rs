use crate::{ExportError, ModelExporter, Result};
use std::path::Path;

pub struct CoreMlExporter {
    model_name: String,
}

impl CoreMlExporter {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

impl ModelExporter for CoreMlExporter {
    fn export<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(ExportError::Other(
            "CoreML export not yet implemented".to_string(),
        ))
    }

    fn export_bytes(&self) -> Result<Vec<u8>> {
        Err(ExportError::Other(
            "CoreML export not yet implemented".to_string(),
        ))
    }
}
