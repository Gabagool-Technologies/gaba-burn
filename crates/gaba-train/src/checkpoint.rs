//! Checkpoint and resume functionality for training

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Training checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub epoch: usize,
    pub best_loss: f32,
    pub patience_counter: usize,
    pub timestamp: String,
    pub model_state: Vec<u8>,
    pub optimizer_state: Option<Vec<u8>>,
}

/// Checkpoint manager
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    keep_last_n: usize,
}

impl CheckpointManager {
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, keep_last_n: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            keep_last_n,
        }
    }

    pub fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        fs::create_dir_all(&self.checkpoint_dir)
            .context("Failed to create checkpoint directory")?;

        let filename = format!("checkpoint_epoch_{:04}.bin", checkpoint.epoch);
        let path = self.checkpoint_dir.join(&filename);

        let bytes = bincode::serialize(checkpoint).context("Failed to serialize checkpoint")?;

        fs::write(&path, bytes).context("Failed to write checkpoint file")?;

        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    pub fn load_latest(&self) -> Result<Option<Checkpoint>> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        let latest = checkpoints.last().unwrap();
        let bytes = fs::read(latest).context("Failed to read checkpoint file")?;

        let checkpoint =
            bincode::deserialize(&bytes).context("Failed to deserialize checkpoint")?;

        Ok(Some(checkpoint))
    }

    pub fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.len() > self.keep_last_n {
            checkpoints.sort();
            let to_remove = checkpoints.len() - self.keep_last_n;

            for checkpoint in &checkpoints[..to_remove] {
                fs::remove_file(checkpoint).context("Failed to remove old checkpoint")?;
            }
        }

        Ok(())
    }

    fn list_checkpoints(&self) -> Result<Vec<PathBuf>> {
        if !self.checkpoint_dir.exists() {
            return Ok(Vec::new());
        }

        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("checkpoint_epoch_") {
                        checkpoints.push(path);
                    }
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint {
            epoch: 10,
            best_loss: 0.042,
            patience_counter: 0,
            timestamp: "2024-11-05T00:00:00Z".to_string(),
            model_state: vec![1, 2, 3, 4],
            optimizer_state: Some(vec![5, 6, 7, 8]),
        };
        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.model_state.len(), 4);
    }

    #[test]
    fn test_checkpoint_manager() {
        let manager = CheckpointManager::new("/tmp/checkpoints", 5);
        assert_eq!(manager.keep_last_n, 5);
    }

    #[test]
    fn test_checkpoint_save_load() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let manager = CheckpointManager::new(dir.path(), 3);

        let checkpoint = Checkpoint {
            epoch: 5,
            best_loss: 0.123,
            patience_counter: 2,
            timestamp: "2025-11-06T00:00:00Z".to_string(),
            model_state: vec![1, 2, 3, 4, 5],
            optimizer_state: Some(vec![6, 7, 8]),
        };

        manager.save(&checkpoint).unwrap();

        let loaded = manager.load_latest().unwrap().unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.best_loss, 0.123);
        assert_eq!(loaded.model_state, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_checkpoint_cleanup() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let manager = CheckpointManager::new(dir.path(), 2);

        for i in 0..5 {
            let checkpoint = Checkpoint {
                epoch: i,
                best_loss: 0.1,
                patience_counter: 0,
                timestamp: "2025-11-06T00:00:00Z".to_string(),
                model_state: vec![i as u8],
                optimizer_state: None,
            };
            manager.save(&checkpoint).unwrap();
        }

        let checkpoints = manager.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 2);
    }
}
