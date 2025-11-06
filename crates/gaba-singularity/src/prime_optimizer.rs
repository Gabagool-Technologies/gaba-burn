use std::collections::HashMap;

pub struct PrimeOptimizer {
    prime_sequence: Vec<usize>,
    cache_results: HashMap<String, OptimizationResult>,
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub optimal_batch_size: usize,
    pub optimal_block_size: usize,
    pub optimal_group_size: usize,
    pub learning_rate_schedule: Vec<f32>,
    pub momentum_params: (f32, f32),
    pub cache_hit_rate: f32,
    pub performance_gain: f32,
}

#[derive(Clone, Debug)]
pub struct PrimeConfig {
    pub use_prime_batch_sizes: bool,
    pub use_prime_block_sizes: bool,
    pub use_prime_lr_schedule: bool,
    pub use_prime_momentum: bool,
    pub base_primes: Vec<usize>,
}

impl Default for PrimeConfig {
    fn default() -> Self {
        Self {
            use_prime_batch_sizes: true,
            use_prime_block_sizes: true,
            use_prime_lr_schedule: true,
            use_prime_momentum: true,
            base_primes: vec![13, 19, 42],
        }
    }
}

impl PrimeConfig {
    pub fn extended() -> Self {
        Self {
            use_prime_batch_sizes: true,
            use_prime_block_sizes: true,
            use_prime_lr_schedule: true,
            use_prime_momentum: true,
            base_primes: vec![13, 19, 23, 29, 42, 67],
        }
    }

    pub fn fibonacci() -> Self {
        Self {
            use_prime_batch_sizes: true,
            use_prime_block_sizes: true,
            use_prime_lr_schedule: true,
            use_prime_momentum: true,
            base_primes: vec![13, 21, 34],
        }
    }

    pub fn twin_primes() -> Self {
        Self {
            use_prime_batch_sizes: true,
            use_prime_block_sizes: true,
            use_prime_lr_schedule: true,
            use_prime_momentum: true,
            base_primes: vec![11, 13, 17, 19, 29, 31],
        }
    }
}

impl PrimeOptimizer {
    pub fn new(config: PrimeConfig) -> Self {
        Self {
            prime_sequence: config.base_primes,
            cache_results: HashMap::new(),
        }
    }

    pub fn optimize_batch_size(&self, target_size: usize) -> usize {
        let mut best_size = target_size;
        let mut best_score = 0.0;

        for &prime in &self.prime_sequence {
            for multiplier in 1..=10 {
                let candidate = prime * multiplier;
                if candidate >= target_size / 2 && candidate <= target_size * 2 {
                    let score = self.score_batch_size(candidate, target_size);
                    if score > best_score {
                        best_score = score;
                        best_size = candidate;
                    }
                }
            }
        }

        best_size
    }

    fn score_batch_size(&self, candidate: usize, target: usize) -> f32 {
        let size_diff = ((candidate as f32 - target as f32).abs() / target as f32).min(1.0);
        let cache_alignment = if candidate % 64 == 0 { 0.2 } else { 0.0 };
        let prime_bonus = if self.is_prime_multiple(candidate) { 0.3 } else { 0.0 };
        
        1.0 - size_diff + cache_alignment + prime_bonus
    }

    fn is_prime_multiple(&self, n: usize) -> bool {
        self.prime_sequence.iter().any(|&p| n % p == 0)
    }

    pub fn optimize_block_size(&self, matrix_dim: usize) -> (usize, usize) {
        let mut best_block = (16, 16);
        let mut best_score = 0.0;

        for &prime in &self.prime_sequence {
            if prime <= matrix_dim / 2 {
                let score = self.score_block_size(prime, matrix_dim);
                if score > best_score {
                    best_score = score;
                    best_block = (prime, prime);
                }
            }
        }

        best_block
    }

    fn score_block_size(&self, block_size: usize, matrix_dim: usize) -> f32 {
        let coverage = (matrix_dim as f32 / block_size as f32).fract();
        let cache_fit = if block_size * block_size * 4 <= 32768 { 0.3 } else { 0.0 };
        let alignment = if block_size % 8 == 0 { 0.2 } else { 0.0 };
        
        (1.0 - coverage) + cache_fit + alignment
    }

    pub fn optimize_learning_rate_schedule(
        &self,
        initial_lr: f32,
        total_epochs: usize,
    ) -> Vec<f32> {
        let mut schedule = Vec::with_capacity(total_epochs);
        let decay_points: Vec<usize> = self.prime_sequence
            .iter()
            .filter(|&&p| p < total_epochs)
            .copied()
            .collect();

        let mut current_lr = initial_lr;
        let mut next_decay_idx = 0;

        for epoch in 0..total_epochs {
            if next_decay_idx < decay_points.len() && epoch == decay_points[next_decay_idx] {
                current_lr *= 0.1;
                next_decay_idx += 1;
            }
            schedule.push(current_lr);
        }

        schedule
    }

    pub fn optimize_momentum_params(&self) -> (f32, f32) {
        let beta1 = 0.9 - (self.prime_sequence[0] as f32 / 100.0);
        let beta2 = 0.999 - (self.prime_sequence[1] as f32 / 1000.0);
        
        (beta1.max(0.8).min(0.95), beta2.max(0.99).min(0.999))
    }

    pub fn optimize_quantization_groups(&self, total_params: usize) -> usize {
        for &prime in &self.prime_sequence {
            if total_params % prime == 0 || total_params % (prime * 2) == 0 {
                return prime;
            }
        }
        
        self.prime_sequence[0]
    }

    pub fn optimize_cache_alignment(&self, size: usize) -> usize {
        let cache_line = 64;
        let prime_aligned = self.prime_sequence.iter()
            .map(|&p| ((size + p - 1) / p) * p)
            .find(|&aligned| aligned % cache_line == 0 || (aligned * 4) % cache_line == 0);
        
        prime_aligned.unwrap_or(((size + cache_line - 1) / cache_line) * cache_line)
    }

    pub fn benchmark_configuration(
        &mut self,
        workload_id: &str,
        config: &OptimizationResult,
    ) -> f32 {
        if let Some(cached) = self.cache_results.get(workload_id) {
            return cached.performance_gain;
        }

        let gain = self.simulate_performance(config);
        
        let mut result = config.clone();
        result.performance_gain = gain;
        self.cache_results.insert(workload_id.to_string(), result);
        
        gain
    }

    fn simulate_performance(&self, config: &OptimizationResult) -> f32 {
        let batch_score = if self.is_prime_multiple(config.optimal_batch_size) {
            1.15
        } else {
            1.0
        };
        
        let block_score = if self.is_prime_multiple(config.optimal_block_size) {
            1.10
        } else {
            1.0
        };
        
        let cache_score = 1.0 + (config.cache_hit_rate * 0.5);
        
        batch_score * block_score * cache_score
    }

    pub fn get_optimal_config(&self, workload_size: usize, matrix_dim: usize) -> OptimizationResult {
        let batch_size = self.optimize_batch_size(workload_size);
        let (block_w, block_h) = self.optimize_block_size(matrix_dim);
        let group_size = self.optimize_quantization_groups(matrix_dim * matrix_dim);
        let lr_schedule = self.optimize_learning_rate_schedule(0.001, 100);
        let momentum = self.optimize_momentum_params();

        OptimizationResult {
            optimal_batch_size: batch_size,
            optimal_block_size: block_w,
            optimal_group_size: group_size,
            learning_rate_schedule: lr_schedule,
            momentum_params: momentum,
            cache_hit_rate: 0.85,
            performance_gain: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_optimizer_creation() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        assert_eq!(optimizer.prime_sequence, vec![13, 19, 42]);
    }

    #[test]
    fn test_optimize_batch_size() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let batch_size = optimizer.optimize_batch_size(32);
        assert!(batch_size % 13 == 0 || batch_size % 19 == 0 || batch_size % 42 == 0);
    }

    #[test]
    fn test_optimize_block_size() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let (w, h) = optimizer.optimize_block_size(256);
        assert_eq!(w, h);
        assert!(w == 13 || w == 19 || w == 42);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let schedule = optimizer.optimize_learning_rate_schedule(0.001, 50);
        assert_eq!(schedule.len(), 50);
        assert_eq!(schedule[0], 0.001);
        assert!(schedule[13] < schedule[12]);
        assert!(schedule[19] < schedule[18]);
    }

    #[test]
    fn test_momentum_params() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let (beta1, beta2) = optimizer.optimize_momentum_params();
        assert!(beta1 >= 0.8 && beta1 <= 0.95);
        assert!(beta2 >= 0.99 && beta2 <= 0.999);
    }

    #[test]
    fn test_quantization_groups() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let group_size = optimizer.optimize_quantization_groups(1024);
        assert!(group_size == 13 || group_size == 19 || group_size == 42);
    }

    #[test]
    fn test_cache_alignment() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let aligned = optimizer.optimize_cache_alignment(100);
        assert!(aligned >= 100);
        assert!(aligned % 64 == 0 || (aligned * 4) % 64 == 0);
    }

    #[test]
    fn test_get_optimal_config() {
        let config = PrimeConfig::default();
        let optimizer = PrimeOptimizer::new(config);
        
        let result = optimizer.get_optimal_config(32, 128);
        assert!(result.optimal_batch_size > 0);
        assert!(result.optimal_block_size > 0);
        assert!(result.optimal_group_size > 0);
        assert!(!result.learning_rate_schedule.is_empty());
    }

    #[test]
    fn test_extended_sequence() {
        let config = PrimeConfig::extended();
        let optimizer = PrimeOptimizer::new(config);
        
        assert_eq!(optimizer.prime_sequence, vec![13, 19, 23, 29, 42, 67]);
        
        let batch_size = optimizer.optimize_batch_size(64);
        assert!(batch_size % 23 == 0 || batch_size % 29 == 0 || batch_size % 67 == 0 
                || batch_size % 13 == 0 || batch_size % 19 == 0 || batch_size % 42 == 0);
    }

    #[test]
    fn test_fibonacci_sequence() {
        let config = PrimeConfig::fibonacci();
        let optimizer = PrimeOptimizer::new(config);
        
        assert_eq!(optimizer.prime_sequence, vec![13, 21, 34]);
        
        let (w, h) = optimizer.optimize_block_size(256);
        assert_eq!(w, h);
        assert!(w == 13 || w == 21 || w == 34);
    }

    #[test]
    fn test_twin_primes_sequence() {
        let config = PrimeConfig::twin_primes();
        let optimizer = PrimeOptimizer::new(config);
        
        assert_eq!(optimizer.prime_sequence, vec![11, 13, 17, 19, 29, 31]);
        
        let batch_size = optimizer.optimize_batch_size(32);
        assert!(batch_size % 11 == 0 || batch_size % 13 == 0 || batch_size % 17 == 0
                || batch_size % 19 == 0 || batch_size % 29 == 0 || batch_size % 31 == 0);
    }

    #[test]
    fn test_sequence_comparison() {
        let baseline = PrimeOptimizer::new(PrimeConfig::default());
        let extended = PrimeOptimizer::new(PrimeConfig::extended());
        let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
        
        let target = 128;
        let b1 = baseline.optimize_batch_size(target);
        let b2 = extended.optimize_batch_size(target);
        let b3 = fibonacci.optimize_batch_size(target);
        
        assert!(b1 > 0 && b2 > 0 && b3 > 0);
        assert!(b1 <= target * 2 && b2 <= target * 2 && b3 <= target * 2);
    }
}
