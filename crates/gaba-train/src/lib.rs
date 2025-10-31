//! Gaba Train: Rust+Zig ML training engine for route optimization
//!
//! High-performance training without Python dependencies.
//! Optimized for edge deployment and federated learning.

pub mod data;
pub mod models;
pub mod optimizers;
pub mod training;

pub use data::{TrafficDataset, RouteDataset};
pub use models::{TrafficModel, RouteModel};
pub use training::{TrainingConfig, Trainer};

use anyhow::Result;

/// Train traffic speed prediction model
pub fn train_traffic_model(
    data_path: &str,
    output_path: &str,
    config: TrainingConfig,
) -> Result<()> {
    let dataset = TrafficDataset::from_csv(data_path)?;
    let model = TrafficModel::new();
    let trainer = Trainer::new(config);
    
    let trained_model = trainer.train(model, dataset)?;
    trained_model.save(output_path)?;
    
    Ok(())
}

/// Train route time prediction model
pub fn train_route_model(
    data_path: &str,
    output_path: &str,
    config: TrainingConfig,
) -> Result<()> {
    let dataset = RouteDataset::from_csv(data_path)?;
    let model = RouteModel::new();
    let trainer = Trainer::new(config);
    
    let trained_model = trainer.train(model, dataset)?;
    trained_model.save(output_path)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }
}
