use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub enum LRScheduler {
    Constant {
        lr: f32,
    },
    StepDecay {
        initial_lr: f32,
        decay_rate: f32,
        step_size: usize,
    },
    ExponentialDecay {
        initial_lr: f32,
        decay_rate: f32,
    },
    CosineAnnealing {
        initial_lr: f32,
        min_lr: f32,
        t_max: usize,
    },
    CosineAnnealingWarmRestarts {
        initial_lr: f32,
        min_lr: f32,
        t_0: usize,
        t_mult: usize,
    },
    WarmupCosine {
        initial_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
    },
    WarmupStableDecay {
        initial_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        stable_steps: usize,
        decay_steps: usize,
    },
}

impl LRScheduler {
    pub fn get_lr(&self, step: usize) -> f32 {
        match self {
            LRScheduler::Constant { lr } => *lr,
            
            LRScheduler::StepDecay { initial_lr, decay_rate, step_size } => {
                let decay_factor = (step / step_size) as f32;
                initial_lr * decay_rate.powf(decay_factor)
            }
            
            LRScheduler::ExponentialDecay { initial_lr, decay_rate } => {
                initial_lr * decay_rate.powf(step as f32)
            }
            
            LRScheduler::CosineAnnealing { initial_lr, min_lr, t_max } => {
                let t = (step % t_max) as f32;
                let t_max = *t_max as f32;
                min_lr + (initial_lr - min_lr) * (1.0 + (PI * t / t_max).cos()) / 2.0
            }
            
            LRScheduler::CosineAnnealingWarmRestarts { initial_lr, min_lr, t_0, t_mult } => {
                let mut t_cur = step;
                let mut t_i = *t_0;
                let mut _iteration = 0;
                
                while t_cur >= t_i {
                    t_cur -= t_i;
                    t_i *= t_mult;
                    _iteration += 1;
                }
                
                let t_cur = t_cur as f32;
                let t_i = t_i as f32;
                min_lr + (initial_lr - min_lr) * (1.0 + (PI * t_cur / t_i).cos()) / 2.0
            }
            
            LRScheduler::WarmupCosine { initial_lr, min_lr, warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    initial_lr * (step as f32 / *warmup_steps as f32)
                } else {
                    let progress = ((step - warmup_steps) as f32) / ((total_steps - warmup_steps) as f32);
                    min_lr + (initial_lr - min_lr) * (1.0 + (PI * progress).cos()) / 2.0
                }
            }
            
            LRScheduler::WarmupStableDecay { initial_lr, min_lr, warmup_steps, stable_steps, decay_steps } => {
                if step < *warmup_steps {
                    initial_lr * (step as f32 / *warmup_steps as f32)
                } else if step < warmup_steps + stable_steps {
                    *initial_lr
                } else {
                    let decay_progress = ((step - warmup_steps - stable_steps) as f32) / (*decay_steps as f32);
                    let decay_progress = decay_progress.min(1.0);
                    initial_lr - (initial_lr - min_lr) * decay_progress
                }
            }
        }
    }
    
    pub fn step<O: crate::optimizers::Optimizer>(&self, optimizer: &mut O, step: usize) {
        let lr = self.get_lr(step);
        optimizer.set_lr(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_constant_scheduler() {
        let scheduler = LRScheduler::Constant { lr: 0.01 };
        assert_relative_eq!(scheduler.get_lr(0), 0.01);
        assert_relative_eq!(scheduler.get_lr(100), 0.01);
    }
    
    #[test]
    fn test_step_decay() {
        let scheduler = LRScheduler::StepDecay {
            initial_lr: 0.1,
            decay_rate: 0.5,
            step_size: 10,
        };
        
        assert_relative_eq!(scheduler.get_lr(0), 0.1);
        assert_relative_eq!(scheduler.get_lr(9), 0.1);
        assert_relative_eq!(scheduler.get_lr(10), 0.05);
        assert_relative_eq!(scheduler.get_lr(20), 0.025);
    }
    
    #[test]
    fn test_exponential_decay() {
        let scheduler = LRScheduler::ExponentialDecay {
            initial_lr: 0.1,
            decay_rate: 0.95,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr1 = scheduler.get_lr(1);
        let lr10 = scheduler.get_lr(10);
        
        assert_relative_eq!(lr0, 0.1);
        assert!(lr1 < lr0);
        assert!(lr10 < lr1);
    }
    
    #[test]
    fn test_cosine_annealing() {
        let scheduler = LRScheduler::CosineAnnealing {
            initial_lr: 0.1,
            min_lr: 0.001,
            t_max: 100,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr50 = scheduler.get_lr(50);
        let lr100 = scheduler.get_lr(100);
        
        assert_relative_eq!(lr0, 0.1, epsilon = 1e-6);
        // At t=50, cos(pi*50/100) = cos(pi/2) = 0, so lr = (max+min)/2
        assert_relative_eq!(lr50, 0.0505, epsilon = 1e-3);
        assert_relative_eq!(lr100, 0.1, epsilon = 1e-6);
    }
    
    #[test]
    fn test_warmup_cosine() {
        let scheduler = LRScheduler::WarmupCosine {
            initial_lr: 0.1,
            min_lr: 0.001,
            warmup_steps: 10,
            total_steps: 100,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr5 = scheduler.get_lr(5);
        let lr10 = scheduler.get_lr(10);
        let lr100 = scheduler.get_lr(100);
        
        assert_relative_eq!(lr0, 0.0);
        assert!(lr5 > lr0 && lr5 < 0.1);
        assert_relative_eq!(lr10, 0.1, epsilon = 1e-6);
        assert!(lr100 < lr10);
    }
    
    #[test]
    fn test_warmup_stable_decay() {
        let scheduler = LRScheduler::WarmupStableDecay {
            initial_lr: 0.1,
            min_lr: 0.001,
            warmup_steps: 10,
            stable_steps: 50,
            decay_steps: 40,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr10 = scheduler.get_lr(10);
        let lr30 = scheduler.get_lr(30);
        let lr60 = scheduler.get_lr(60);
        let lr80 = scheduler.get_lr(80);
        let lr100 = scheduler.get_lr(100);
        
        assert_relative_eq!(lr0, 0.0);
        assert_relative_eq!(lr10, 0.1, epsilon = 1e-6);
        assert_relative_eq!(lr30, 0.1, epsilon = 1e-6);
        assert_relative_eq!(lr60, 0.1, epsilon = 1e-6); // Decay just started
        assert!(lr80 < 0.1); // Midway through decay
        assert_relative_eq!(lr100, 0.001, epsilon = 1e-3);
    }
    
    #[test]
    fn test_cosine_annealing_warm_restarts() {
        let scheduler = LRScheduler::CosineAnnealingWarmRestarts {
            initial_lr: 0.1,
            min_lr: 0.001,
            t_0: 10,
            t_mult: 2,
        };
        
        let lr0 = scheduler.get_lr(0);
        let lr5 = scheduler.get_lr(5);
        let lr10 = scheduler.get_lr(10);
        
        assert_relative_eq!(lr0, 0.1, epsilon = 1e-6);
        assert!(lr5 < lr0);
        assert_relative_eq!(lr10, 0.1, epsilon = 1e-6);
    }
}
