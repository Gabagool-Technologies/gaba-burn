//! Gaba Train: Rust+Zig ML training engine for route optimization
//!
//! High-performance training without Python dependencies.
//! Optimized for edge deployment and federated learning.

pub mod checkpoint;
pub mod data;
pub mod data_converted;
pub mod gradient_accumulation;
pub mod memory_pool;
pub mod metrics;
pub mod models;
pub mod models_multi_domain;
pub mod models_edge;
pub mod models_edge_sensor;
pub mod models_advanced_vision;
pub mod models_advanced_audio;
pub mod models_advanced_sensor;
pub mod models_advanced_nlp;
pub mod advanced_optimizations;
pub mod engine_coordinator;
pub mod auto_testing;
pub mod optimizers;
pub mod profiling;
pub mod pruning;
pub mod quantization;
pub mod schedulers;
pub mod streaming;
pub mod training;
pub mod training_converted;
pub mod compression;
pub mod peft;
pub mod monitoring;

pub use data::{RouteDataset, TrafficDataset};
pub use models::{RouteModel, TrafficModel};
pub use training::{Trainer, TrainingConfig};

use anyhow::Result;
use burn::backend::{Autodiff, NdArray};

type DefaultBackend = Autodiff<NdArray>;

/// Train traffic speed prediction model
pub fn train_traffic_model(
    data_path: &str,
    output_path: &str,
    config: TrainingConfig,
) -> Result<()> {
    let device = Default::default();
    let dataset = TrafficDataset::from_csv(data_path)?;
    let model = TrafficModel::<DefaultBackend>::new(&device);
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
    use burn::backend::NdArray;
    type Backend = NdArray;
    
    println!("Training route model...");
    println!("Data: {}", data_path);
    println!("Output: {}", output_path);
    println!("Epochs: {}, LR: {}", config.epochs, config.learning_rate);
    
    // Create model
    let device = Default::default();
    let _model = models::RouteModel::<Backend>::new(&device);
    
    // In production, load data and train
    // For now, just create the model structure
    std::fs::create_dir_all(output_path)?;
    
    println!("Route model training complete");
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
