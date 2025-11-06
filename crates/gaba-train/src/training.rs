//! Training loop and utilities

use crate::data::TrafficDataset;
use crate::models::TrafficModel;
// use crate::optimizers::OptimizerConfig;
use crate::pruning::PruningConfig;
use crate::quantization::QuantizationConfig;
use anyhow::Result;
use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        ElementConversion, Tensor,
    },
};
use gaba_singularity::AdaptiveKernelOrchestrator;
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
    pub use_singularity: bool,
    pub quantization: QuantizationConfig,
    pub pruning: PruningConfig,
    pub enable_profiling: bool,
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
            use_singularity: true,
            quantization: QuantizationConfig::default(),
            pruning: PruningConfig::default(),
            enable_profiling: false,
        }
    }
}

/// Trainer for models
pub struct Trainer {
    config: TrainingConfig,
    singularity: Option<AdaptiveKernelOrchestrator>,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        let singularity = if config.use_singularity {
            Some(AdaptiveKernelOrchestrator::new().with_learning(true))
        } else {
            None
        };

        Self {
            config,
            singularity,
        }
    }

    pub fn with_singularity(mut self, enabled: bool) -> Self {
        self.config.use_singularity = enabled;
        if enabled && self.singularity.is_none() {
            self.singularity = Some(AdaptiveKernelOrchestrator::new().with_learning(true));
        } else if !enabled {
            self.singularity = None;
        }
        self
    }

    pub fn get_singularity_stats(&self) -> Option<usize> {
        self.singularity.as_ref().map(|s| s.history_size())
    }

    /// Train traffic model
    pub fn train<B: AutodiffBackend>(
        &self,
        model: TrafficModel<B>,
        dataset: TrafficDataset,
    ) -> Result<TrafficModel<B>> {
        let device = Default::default();

        // Split data
        let (train_data, test_data) = dataset.split(self.config.test_ratio);

        if self.config.verbose {
            println!("Training samples: {}", train_data.len());
            println!("Test samples: {}", test_data.len());
        }

        // Create optimizer using Burn's SGD
        let mut optimizer = burn::optim::SgdConfig::new().init();

        // Progress bar
        let pb = if self.config.verbose {
            let pb = ProgressBar::new(self.config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };

        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        let mut model = model;

        // Training loop
        for epoch in 0..self.config.epochs {
            // Train
            let (new_model, train_loss) =
                self.train_epoch(model, &train_data, &mut optimizer, &device)?;
            model = new_model;

            // Validate
            let test_loss = self.validate_epoch(&model, &test_data, &device)?;

            if let Some(ref pb) = pb {
                pb.set_position(epoch as u64 + 1);
                pb.set_message(format!(
                    "train_loss: {:.4}, test_loss: {:.4}",
                    train_loss, test_loss
                ));
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

        if self.config.verbose && self.config.use_singularity {
            if let Some(history_size) = self.get_singularity_stats() {
                println!(
                    "Singularity Engine: {} kernel executions learned",
                    history_size
                );
            }
        }

        Ok(model)
    }

    fn train_epoch<B: AutodiffBackend>(
        &self,
        mut model: TrafficModel<B>,
        data: &TrafficDataset,
        optimizer: &mut impl Optimizer<TrafficModel<B>, B>,
        device: &<B as Backend>::Device,
    ) -> Result<(TrafficModel<B>, f32)> {
        let n_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let mut total_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(data.len());

            // Get batch
            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);

            // Convert to tensors
            let batch_size = end - start;
            let features_vec: Vec<f32> = features.iter().copied().collect();
            let targets_vec: Vec<f32> = targets.iter().copied().collect();

            let features_tensor = Tensor::<B, 2>::from_floats(features_vec.as_slice(), device)
                .reshape([batch_size, 22]);
            let targets_tensor = Tensor::<B, 1>::from_floats(targets_vec.as_slice(), device);

            // Forward pass
            let model_ad = model.clone();
            let predictions = model_ad.forward(features_tensor);
            let predictions = predictions.squeeze::<1>();

            // Compute loss (MSE)
            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(self.config.learning_rate, model, grads);

            total_loss += loss.into_scalar().elem::<f32>();
        }

        Ok((model, total_loss / n_batches as f32))
    }

    fn validate_epoch<B: Backend>(
        &self,
        model: &TrafficModel<B>,
        data: &TrafficDataset,
        device: &<B as Backend>::Device,
    ) -> Result<f32> {
        let n_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let mut total_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(data.len());

            // Get batch
            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);

            // Convert to tensors
            let batch_size = end - start;
            let features_vec: Vec<f32> = features.iter().copied().collect();
            let targets_vec: Vec<f32> = targets.iter().copied().collect();

            let features_tensor = Tensor::<B, 2>::from_floats(features_vec.as_slice(), device)
                .reshape([batch_size, 22]);
            let targets_tensor = Tensor::<B, 1>::from_floats(targets_vec.as_slice(), device);

            // Forward pass
            let predictions = model.forward(features_tensor);
            let predictions = predictions.squeeze::<1>();

            // Compute loss (MSE)
            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();

            total_loss += loss.into_scalar().elem::<f32>();
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
