use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub name: String,
    pub stages: Vec<Stage>,
    pub config: PipelineConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub checkpoint_dir: String,
    pub log_dir: String,
    pub auto_optimize: bool,
    pub distributed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage {
    DataPrep,
    Training,
    Validation,
    Export,
    Deploy,
}

impl Pipeline {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let pipeline = serde_json::from_str(&json)?;
        Ok(pipeline)
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: "./checkpoints".to_string(),
            log_dir: "./logs".to_string(),
            auto_optimize: true,
            distributed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new("test")
            .add_stage(Stage::DataPrep)
            .add_stage(Stage::Training);

        assert_eq!(pipeline.name, "test");
        assert_eq!(pipeline.stages.len(), 2);
    }

    #[test]
    fn test_pipeline_save_load() {
        let pipeline = Pipeline::new("test")
            .add_stage(Stage::DataPrep)
            .add_stage(Stage::Training);

        let temp_file = "/tmp/test_pipeline.json";
        pipeline.save(temp_file).unwrap();

        let loaded = Pipeline::load(temp_file).unwrap();
        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.stages.len(), 2);

        std::fs::remove_file(temp_file).ok();
    }
}
