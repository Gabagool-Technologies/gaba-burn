#[allow(dead_code)]
use std::collections::HashMap;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AdamOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,

    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl AdamOptimizer {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_params(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        params: &mut HashMap<String, Vec<f32>>,
        grads: &HashMap<String, Vec<f32>>,
    ) {
        self.t += 1;

        for (name, param) in params.iter_mut() {
            let grad = match grads.get(name) {
                Some(g) => g,
                None => continue,
            };

            let m = self
                .m
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);
            let v = self
                .v
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);

            for i in 0..param.len() {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

                let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));

                param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }

    pub fn zero_grad(&mut self) {
        self.m.clear();
        self.v.clear();
    }

    pub fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AdamWOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub t: usize,

    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl AdamWOptimizer {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        params: &mut HashMap<String, Vec<f32>>,
        grads: &HashMap<String, Vec<f32>>,
    ) {
        self.t += 1;

        for (name, param) in params.iter_mut() {
            let grad = match grads.get(name) {
                Some(g) => g,
                None => continue,
            };

            let m = self
                .m
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);
            let v = self
                .v
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; param.len()]);

            for i in 0..param.len() {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

                let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));

                param[i] -= self.lr
                    * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * param[i]);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum LRScheduler {
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
    CosineAnnealingLR { t_max: usize, eta_min: f32 },
}

impl LRScheduler {
    pub fn get_lr(&self, epoch: usize, base_lr: f32) -> f32 {
        match self {
            LRScheduler::StepLR { step_size, gamma } => {
                base_lr * gamma.powi((epoch / step_size) as i32)
            }
            LRScheduler::ExponentialLR { gamma } => base_lr * gamma.powi(epoch as i32),
            LRScheduler::CosineAnnealingLR { t_max, eta_min } => {
                eta_min
                    + (base_lr - eta_min)
                        * (1.0 + (std::f32::consts::PI * epoch as f32 / *t_max as f32).cos())
                        / 2.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::new(0.001);

        let mut params = HashMap::new();
        params.insert("w1".to_string(), vec![1.0, 2.0, 3.0]);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), vec![0.1, 0.2, 0.3]);

        let initial = params["w1"].clone();

        optimizer.step(&mut params, &grads);

        for i in 0..3 {
            assert!(params["w1"][i] < initial[i]);
        }
    }

    #[test]
    fn test_adamw_optimizer() {
        let mut optimizer = AdamWOptimizer::new(0.001, 0.01);

        let mut params = HashMap::new();
        params.insert("w1".to_string(), vec![1.0, 2.0, 3.0]);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), vec![0.1, 0.2, 0.3]);

        optimizer.step(&mut params, &grads);

        assert!(params["w1"][0] < 1.0);
    }

    #[test]
    fn test_lr_scheduler() {
        let scheduler = LRScheduler::StepLR {
            step_size: 10,
            gamma: 0.1,
        };

        let lr0 = scheduler.get_lr(0, 0.1);
        let lr10 = scheduler.get_lr(10, 0.1);
        let lr20 = scheduler.get_lr(20, 0.1);

        assert_eq!(lr0, 0.1);
        assert_eq!(lr10, 0.01);
        assert_eq!(lr20, 0.001);
    }
}
