//! Training loop and utilities

use crate::data::{TrafficDataset, RouteDataset};
use crate::models::{TrafficModel, RouteModel};
use crate::optimizers::OptimizerConfig;
use anyhow::Result;
use burn::{
    module::Module,
    optim::{GradientsParams, Optimizer},
    tensor::{backend::Backend, Tensor},
};
use indicatif::{ProgressBar, ProgressStyle};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub test_ratio: f32,
    pub early_stopping_patience: Option<usize>,
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            test_ratio: 0.2,
            early_stopping_patience: Some(10),
            verbose: true,
        }
    }
}

/// Trainer for models
pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }
    
    /// Train traffic model
    pub fn train<B: Backend>(
        &self,
        mut model: TrafficModel<B>,
        dataset: TrafficDataset,
    ) -> Result<TrafficModel<B>> {
        let device = Default::default();
        
        // Split data
        let (train_data, test_data) = dataset.split(self.config.test_ratio);
        
        if self.config.verbose {
            println!("Training samples: {}", train_data.len());
            println!("Test samples: {}", test_data.len());
        }
        
        // Create optimizer
        let optim_config = OptimizerConfig {
            learning_rate: self.config.learning_rate,
            ..Default::default()
        };
        let mut optimizer = optim_config.build::<B>();
        
        // Progress bar
        let pb = if self.config.verbose {
            let pb = ProgressBar::new(self.config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap()
            );
            Some(pb)
        } else {
            None
        };
        
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        
        // Training loop
        for epoch in 0..self.config.epochs {
            // Train
            let train_loss = self.train_epoch(&mut model, &train_data, &mut optimizer, &device)?;
            
            // Validate
            let test_loss = self.validate_epoch(&model, &test_data, &device)?;
            
            if let Some(ref pb) = pb {
                pb.set_position(epoch as u64 + 1);
                pb.set_message(format!("train_loss: {:.4}, test_loss: {:.4}", train_loss, test_loss));
            }
            
            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if test_loss < best_loss {
                    best_loss = test_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        if self.config.verbose {
                            println!("\nEarly stopping at epoch {}", epoch + 1);
                        }
                        break;
                    }
                }
            }
        }
        
        if let Some(pb) = pb {
            pb.finish_with_message("Training complete");
        }
        
        Ok(model)
    }
    
    fn train_epoch<B: Backend>(
        &self,
        model: &mut TrafficModel<B>,
        data: &TrafficDataset,
        optimizer: &mut impl Optimizer<TrafficModel<B>, B>,
        device: &B::Device,
    ) -> Result<f32> {
        let n_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let mut total_loss = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(data.len());
            
            // Get batch
            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);
            
            // Convert to tensors
            let features_tensor = Tensor::<B, 2>::from_floats(
                features.as_slice().unwrap(),
                device
            );
            let targets_tensor = Tensor::<B, 1>::from_floats(
                targets.as_slice().unwrap(),
                device
            );
            
            // Forward pass
            let predictions = model.forward(features_tensor);
            let predictions = predictions.squeeze(1);
            
            // Compute loss (MSE)
            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();
            
            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            *model = optimizer.step(self.config.learning_rate, *model, grads);
            
            total_loss += loss.into_scalar();
        }
        
        Ok(total_loss / n_batches as f32)
    }
    
    fn validate_epoch<B: Backend>(
        &self,
        model: &TrafficModel<B>,
        data: &TrafficDataset,
        device: &B::Device,
    ) -> Result<f32> {
        let n_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let mut total_loss = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(data.len());
            
            // Get batch
            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);
            
            // Convert to tensors
            let features_tensor = Tensor::<B, 2>::from_floats(
                features.as_slice().unwrap(),
                device
            );
            let targets_tensor = Tensor::<B, 1>::from_floats(
                targets.as_slice().unwrap(),
                device
            );
            
            // Forward pass
            let predictions = model.forward(features_tensor);
            let predictions = predictions.squeeze(1);
            
            // Compute loss (MSE)
            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();
            
            total_loss += loss.into_scalar();
        }
        
        Ok(total_loss / n_batches as f32)
    }
}

use ndarray::s;

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
