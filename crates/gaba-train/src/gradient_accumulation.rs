//! Gradient accumulation for effective larger batch sizes

use std::marker::PhantomData;

/// Gradient accumulator for simulating larger batch sizes
pub struct GradientAccumulator<B> {
    accumulation_steps: usize,
    current_step: usize,
    _backend: PhantomData<B>,
}

impl<B> GradientAccumulator<B> {
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps,
            current_step: 0,
            _backend: PhantomData,
        }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }

    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    pub fn get_scale_factor(&self) -> f32 {
        1.0 / self.accumulation_steps as f32
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }

    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f32,
    pub monitor: StoppingMetric,
    pub mode: StoppingMode,
}

#[derive(Debug, Clone, Copy)]
pub enum StoppingMetric {
    Loss,
    Accuracy,
}

#[derive(Debug, Clone, Copy)]
pub enum StoppingMode {
    Min,
    Max,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
            monitor: StoppingMetric::Loss,
            mode: StoppingMode::Min,
        }
    }
}

/// Early stopping tracker
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best_value: f32,
    patience_counter: usize,
    best_epoch: usize,
}

impl EarlyStopping {
    pub fn new(config: EarlyStoppingConfig) -> Self {
        let best_value = match config.mode {
            StoppingMode::Min => f32::INFINITY,
            StoppingMode::Max => f32::NEG_INFINITY,
        };

        Self {
            config,
            best_value,
            patience_counter: 0,
            best_epoch: 0,
        }
    }

    pub fn check(&mut self, current_value: f32, epoch: usize) -> bool {
        let improved = match self.config.mode {
            StoppingMode::Min => current_value < self.best_value - self.config.min_delta,
            StoppingMode::Max => current_value > self.best_value + self.config.min_delta,
        };

        if improved {
            self.best_value = current_value;
            self.best_epoch = epoch;
            self.patience_counter = 0;
            false
        } else {
            self.patience_counter += 1;
            self.patience_counter >= self.config.patience
        }
    }

    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    pub fn patience_counter(&self) -> usize {
        self.patience_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = GradientAccumulator::<()>::new(4);

        assert!(!acc.should_update());
        assert_eq!(acc.current_step(), 0);

        for i in 0..3 {
            acc.step();
            assert!(!acc.should_update(), "Should not update at step {}", i + 1);
        }

        acc.step();
        assert!(acc.should_update(), "Should update at step 4");

        acc.reset();
        assert_eq!(acc.current_step(), 0);
    }

    #[test]
    fn test_gradient_accumulator_scale() {
        let acc = GradientAccumulator::<()>::new(4);
        assert_eq!(acc.get_scale_factor(), 0.25);

        let acc2 = GradientAccumulator::<()>::new(8);
        assert_eq!(acc2.get_scale_factor(), 0.125);
    }

    #[test]
    fn test_early_stopping_min() {
        let config = EarlyStoppingConfig {
            patience: 3,
            min_delta: 0.01,
            monitor: StoppingMetric::Loss,
            mode: StoppingMode::Min,
        };

        let mut stopper = EarlyStopping::new(config);

        assert!(!stopper.check(1.0, 0));
        assert!(!stopper.check(0.9, 1));
        assert!(!stopper.check(0.85, 2));
        assert!(!stopper.check(0.86, 3));
        assert!(!stopper.check(0.87, 4));
        assert!(stopper.check(0.88, 5));

        assert_eq!(stopper.best_epoch(), 2);
    }

    #[test]
    fn test_early_stopping_max() {
        let config = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.01,
            monitor: StoppingMetric::Accuracy,
            mode: StoppingMode::Max,
        };

        let mut stopper = EarlyStopping::new(config);

        assert!(!stopper.check(0.5, 0));
        assert!(!stopper.check(0.6, 1));
        assert!(!stopper.check(0.7, 2));
        assert!(!stopper.check(0.69, 3));
        assert!(stopper.check(0.68, 4));

        assert_eq!(stopper.best_epoch(), 2);
    }
}
