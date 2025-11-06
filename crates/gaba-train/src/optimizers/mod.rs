pub mod adam;
pub mod sgd;

pub use adam::{AdamOptimizer, AdamConfig};
pub use sgd::SGDOptimizer;

use std::collections::HashMap;
use ndarray::Array2;

pub trait Optimizer {
    fn step(&mut self, params: &mut HashMap<String, Array2<f32>>, grads: &HashMap<String, Array2<f32>>);
    fn zero_grad(&mut self);
    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    pub learning_rate: f32,
    pub optimizer_type: OptimizerType,
    pub momentum: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_epsilon: f32,
    pub use_amsgrad: bool,
}

#[derive(Clone, Debug)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            optimizer_type: OptimizerType::Adam,
            momentum: 0.9,
            weight_decay: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            use_amsgrad: false,
        }
    }
}
