use crate::kernel_registry::KernelType;
use crate::performance_db::PerformanceVector;
use crate::profiler::WorkloadFeatures;
use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::HashMap;

pub struct KernelSelector {
    q_table: HashMap<(u64, KernelType), f64>,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    min_epsilon: f64,
    epsilon_decay: f64,
}

impl KernelSelector {
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            alpha: 0.1,
            gamma: 0.9,
            epsilon: 0.3,
            min_epsilon: 0.05,
            epsilon_decay: 0.9995,
        }
    }

    pub fn predict(
        &mut self,
        features: &WorkloadFeatures,
        similar_history: &[PerformanceVector],
    ) -> KernelType {
        let state_hash = features.signature();

        if !similar_history.is_empty() && rand::random::<f64>() > self.epsilon {
            return similar_history
                .iter()
                .min_by_key(|v| OrderedFloat(v.execution_time.as_secs_f64()))
                .map(|v| v.kernel_type)
                .unwrap_or(KernelType::RustFallback);
        }

        if rand::random::<f64>() < self.epsilon {
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..KernelType::ALL.len());
            return KernelType::ALL[idx];
        }

        KernelType::ALL
            .iter()
            .max_by_key(|kernel| {
                let q_value = self.q_table.get(&(state_hash, **kernel)).unwrap_or(&0.0);
                OrderedFloat(*q_value)
            })
            .copied()
            .unwrap_or(KernelType::RustFallback)
    }

    pub fn update_q_value(
        &mut self,
        features: &WorkloadFeatures,
        kernel: KernelType,
        execution_time: std::time::Duration,
    ) {
        let state_hash = features.signature();

        let reward = 1.0 / execution_time.as_secs_f64().max(0.000001);

        let current_q = self.q_table.get(&(state_hash, kernel)).unwrap_or(&0.0);

        let max_next_q = KernelType::ALL
            .iter()
            .map(|k| self.q_table.get(&(state_hash, *k)).unwrap_or(&0.0))
            .max_by_key(|q| OrderedFloat(**q))
            .unwrap_or(&0.0);

        let new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q);

        self.q_table.insert((state_hash, kernel), new_q);

        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    pub fn get_best_kernel_for_size(&self, m: usize, n: usize, k: usize) -> KernelType {
        let input_size = m * n * k;

        if input_size < 64 * 64 * 64 {
            KernelType::RustVectorized
        } else if input_size < 128 * 128 * 128 {
            KernelType::Accelerate
        } else if input_size < 512 * 512 * 512 {
            if m >= 128 {
                KernelType::RustParallel
            } else {
                KernelType::Accelerate
            }
        } else {
            KernelType::MetalGPU
        }
    }
}

impl Default for KernelSelector {
    fn default() -> Self {
        Self::new()
    }
}
