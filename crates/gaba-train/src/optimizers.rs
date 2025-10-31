//! Optimizers for training

use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: Option<f32>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
        }
    }
}

impl OptimizerConfig {
    /// Create Adam optimizer from config
    pub fn build<B: Backend>(&self) -> Adam<B> {
        let mut config = AdamConfig::new();
        
        if let Some(wd) = self.weight_decay {
            config = config.with_weight_decay(Some(wd.into()));
        }
        
        config.init()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer_config() {
        let config = OptimizerConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.beta1, 0.9);
    }
}
