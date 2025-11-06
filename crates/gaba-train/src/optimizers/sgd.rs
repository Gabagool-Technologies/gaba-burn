use super::Optimizer;
use std::collections::HashMap;
use ndarray::Array2;

pub struct SGDOptimizer {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: HashMap<String, Array2<f32>>,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: HashMap::new(),
        }
    }
    
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &mut HashMap<String, Array2<f32>>, grads: &HashMap<String, Array2<f32>>) {
        for (name, param) in params.iter_mut() {
            if let Some(grad) = grads.get(name) {
                let mut update = grad.clone();
                
                if self.weight_decay > 0.0 {
                    update = &update + &(param.mapv(|x| x * self.weight_decay));
                }
                
                if self.momentum > 0.0 {
                    let v = self.velocity.entry(name.clone())
                        .or_insert_with(|| Array2::zeros(param.dim()));
                    *v = v.mapv(|x| x * self.momentum) - &update.mapv(|x| x * self.learning_rate);
                    *param = &*param + &*v;
                } else {
                    *param = &*param - &update.mapv(|x| x * self.learning_rate);
                }
            }
        }
    }
    
    fn zero_grad(&mut self) {
        // No-op for SGD
    }
    
    fn get_lr(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_sgd_basic() {
        let mut optimizer = SGDOptimizer::new(0.1);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), arr2(&[[0.1, 0.2], [0.3, 0.4]]));
        
        optimizer.step(&mut params, &grads);
        
        let w = params.get("w").unwrap();
        assert!((w[[0, 0]] - 0.99).abs() < 1e-6);
        assert!((w[[0, 1]] - 1.98).abs() < 1e-6);
    }
    
    #[test]
    fn test_sgd_momentum() {
        let mut optimizer = SGDOptimizer::new(0.1).with_momentum(0.9);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[1.0]]));
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), arr2(&[[1.0]]));
        
        optimizer.step(&mut params, &grads);
        let w1 = params.get("w").unwrap()[[0, 0]];
        
        optimizer.step(&mut params, &grads);
        let w2 = params.get("w").unwrap()[[0, 0]];
        
        assert!(w2 < w1);
    }
}
