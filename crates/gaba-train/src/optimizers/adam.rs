use super::Optimizer;
use std::collections::HashMap;
use ndarray::Array2;

#[derive(Clone)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub amsgrad: bool,
    pub decoupled_weight_decay: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
            decoupled_weight_decay: true,
        }
    }
}

pub struct AdamOptimizer {
    config: AdamConfig,
    m: HashMap<String, Array2<f32>>,
    v: HashMap<String, Array2<f32>>,
    v_hat_max: HashMap<String, Array2<f32>>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            v_hat_max: HashMap::new(),
            t: 0,
        }
    }
    
    pub fn with_lr(learning_rate: f32) -> Self {
        Self::new(AdamConfig {
            learning_rate,
            ..Default::default()
        })
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &mut HashMap<String, Array2<f32>>, grads: &HashMap<String, Array2<f32>>) {
        self.t += 1;
        let _t = self.t as f32;
        
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.t as i32);
        
        for (name, param) in params.iter_mut() {
            if let Some(grad) = grads.get(name) {
                let m = self.m.entry(name.clone())
                    .or_insert_with(|| Array2::zeros(param.dim()));
                let v = self.v.entry(name.clone())
                    .or_insert_with(|| Array2::zeros(param.dim()));
                
                *m = m.mapv(|x| x * self.config.beta1) + grad.mapv(|x| x * (1.0 - self.config.beta1));
                *v = v.mapv(|x| x * self.config.beta2) + grad.mapv(|x| x * x * (1.0 - self.config.beta2));
                
                let m_hat = m.mapv(|x| x / bias_correction1);
                let mut v_hat = v.mapv(|x| x / bias_correction2);
                
                if self.config.amsgrad {
                    let v_max = self.v_hat_max.entry(name.clone())
                        .or_insert_with(|| Array2::zeros(param.dim()));
                    
                    for i in 0..v_hat.nrows() {
                        for j in 0..v_hat.ncols() {
                            v_max[[i, j]] = v_max[[i, j]].max(v_hat[[i, j]]);
                            v_hat[[i, j]] = v_max[[i, j]];
                        }
                    }
                }
                
                if self.config.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                    *param = param.mapv(|x| x * (1.0 - self.config.learning_rate * self.config.weight_decay));
                }
                
                let update = m_hat.iter()
                    .zip(v_hat.iter())
                    .map(|(m_val, v_val)| m_val / (v_val.sqrt() + self.config.epsilon))
                    .collect::<Vec<_>>();
                
                let update_array = Array2::from_shape_vec(param.dim(), update).unwrap();
                *param = &*param - &update_array.mapv(|x| x * self.config.learning_rate);
                
                if !self.config.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                    *param = &*param - &param.mapv(|x| x * self.config.learning_rate * self.config.weight_decay);
                }
            }
        }
    }
    
    fn zero_grad(&mut self) {
        // No-op
    }
    
    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_adam_basic() {
        let config = AdamConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdamOptimizer::new(config);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), arr2(&[[0.1, 0.2], [0.3, 0.4]]));
        
        let initial_w = params.get("w").unwrap().clone();
        optimizer.step(&mut params, &grads);
        
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < initial_w[[0, 0]]);
        assert!(w[[1, 1]] < initial_w[[1, 1]]);
    }
    
    #[test]
    fn test_adam_convergence() {
        let config = AdamConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = AdamOptimizer::new(config);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[5.0]]));
        
        for _ in 0..100 {
            let w = params.get("w").unwrap()[[0, 0]];
            let grad = 2.0 * w;
            
            let mut grads = HashMap::new();
            grads.insert("w".to_string(), arr2(&[[grad]]));
            
            optimizer.step(&mut params, &grads);
        }
        
        let final_w = params.get("w").unwrap()[[0, 0]];
        assert!(final_w.abs() < 0.1);
    }
    
    #[test]
    fn test_adamw_weight_decay() {
        let config = AdamConfig {
            learning_rate: 0.01,
            weight_decay: 0.1,
            decoupled_weight_decay: true,
            ..Default::default()
        };
        let mut optimizer = AdamOptimizer::new(config);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[1.0]]));
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), arr2(&[[0.0]]));
        
        let initial_w = params.get("w").unwrap()[[0, 0]];
        optimizer.step(&mut params, &grads);
        
        let w = params.get("w").unwrap()[[0, 0]];
        assert!(w < initial_w);
    }
    
    #[test]
    fn test_amsgrad() {
        let config = AdamConfig {
            learning_rate: 0.01,
            amsgrad: true,
            ..Default::default()
        };
        let mut optimizer = AdamOptimizer::new(config);
        
        let mut params = HashMap::new();
        params.insert("w".to_string(), arr2(&[[1.0]]));
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), arr2(&[[1.0]]));
        
        optimizer.step(&mut params, &grads);
        optimizer.step(&mut params, &grads);
        
        assert!(optimizer.v_hat_max.contains_key("w"));
    }
}
