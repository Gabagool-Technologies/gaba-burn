//! Model pruning for sparsity

use serde::{Deserialize, Serialize};

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub target_sparsity: f32,
    pub pruning_schedule: PruningSchedule,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningSchedule {
    Constant,
    Gradual {
        start_epoch: usize,
        end_epoch: usize,
    },
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            target_sparsity: 0.0,
            pruning_schedule: PruningSchedule::Constant,
            enabled: false,
        }
    }
}

impl PruningConfig {
    pub fn magnitude_based(sparsity: f32) -> Self {
        Self {
            target_sparsity: sparsity,
            pruning_schedule: PruningSchedule::Constant,
            enabled: true,
        }
    }

    pub fn gradual(sparsity: f32, start: usize, end: usize) -> Self {
        Self {
            target_sparsity: sparsity,
            pruning_schedule: PruningSchedule::Gradual {
                start_epoch: start,
                end_epoch: end,
            },
            enabled: true,
        }
    }
}

/// Apply magnitude-based pruning to weights
pub fn prune_weights(weights: &mut [f32], sparsity: f32) {
    if sparsity <= 0.0 || sparsity >= 1.0 {
        return;
    }

    let mut abs_weights: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();

    abs_weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let prune_count = (weights.len() as f32 * sparsity) as usize;
    for i in 0..prune_count {
        weights[abs_weights[i].0] = 0.0;
    }
}

/// Calculate sparsity of weights
pub fn calculate_sparsity(weights: &[f32]) -> f32 {
    let zero_count = weights.iter().filter(|&&w| w == 0.0).count();
    zero_count as f32 / weights.len() as f32
}

/// Compute current sparsity for gradual pruning schedule
pub fn compute_gradual_sparsity(config: &PruningConfig, current_epoch: usize) -> f32 {
    match config.pruning_schedule {
        PruningSchedule::Constant => config.target_sparsity,
        PruningSchedule::Gradual {
            start_epoch,
            end_epoch,
        } => {
            if current_epoch < start_epoch {
                0.0
            } else if current_epoch >= end_epoch {
                config.target_sparsity
            } else {
                let progress =
                    (current_epoch - start_epoch) as f32 / (end_epoch - start_epoch) as f32;
                progress * config.target_sparsity
            }
        }
    }
}

/// Structured pruning - prune entire channels/neurons
pub fn prune_structured(weights: &mut [f32], channels: usize, sparsity: f32) {
    if sparsity <= 0.0 || sparsity >= 1.0 || channels == 0 {
        return;
    }

    let channel_size = weights.len() / channels;
    let mut channel_norms: Vec<(usize, f32)> = Vec::with_capacity(channels);

    for ch in 0..channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let norm: f32 = weights[start..end]
            .iter()
            .map(|&w| w * w)
            .sum::<f32>()
            .sqrt();
        channel_norms.push((ch, norm));
    }

    channel_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let prune_count = (channels as f32 * sparsity) as usize;
    for i in 0..prune_count {
        let ch = channel_norms[i].0;
        let start = ch * channel_size;
        let end = start + channel_size;
        for w in &mut weights[start..end] {
            *w = 0.0;
        }
    }
}

/// Apply pruning mask to weights
pub fn apply_mask(weights: &mut [f32], mask: &[bool]) {
    for (w, &m) in weights.iter_mut().zip(mask.iter()) {
        if !m {
            *w = 0.0;
        }
    }
}

/// Generate pruning mask from weights and sparsity
pub fn generate_mask(weights: &[f32], sparsity: f32) -> Vec<bool> {
    if sparsity <= 0.0 {
        return vec![true; weights.len()];
    }

    let mut abs_weights: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();

    abs_weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let prune_count = (weights.len() as f32 * sparsity) as usize;
    let mut mask = vec![true; weights.len()];

    for i in 0..prune_count {
        mask[abs_weights[i].0] = false;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_config() {
        let config = PruningConfig::magnitude_based(0.5);
        assert_eq!(config.target_sparsity, 0.5);
        assert!(config.enabled);
    }

    #[test]
    fn test_prune_weights() {
        let mut weights = vec![0.5, -0.1, 0.8, -0.05, 0.3];
        prune_weights(&mut weights, 0.4);
        let sparsity = calculate_sparsity(&weights);
        assert!(sparsity >= 0.35 && sparsity <= 0.45);
    }

    #[test]
    fn test_calculate_sparsity() {
        let weights = vec![0.5, 0.0, 0.8, 0.0];
        let sparsity = calculate_sparsity(&weights);
        assert_eq!(sparsity, 0.5);
    }

    #[test]
    fn test_gradual_sparsity() {
        let config = PruningConfig::gradual(0.8, 10, 20);

        assert_eq!(compute_gradual_sparsity(&config, 5), 0.0);
        assert_eq!(compute_gradual_sparsity(&config, 10), 0.0);
        assert_eq!(compute_gradual_sparsity(&config, 15), 0.4);
        assert_eq!(compute_gradual_sparsity(&config, 20), 0.8);
        assert_eq!(compute_gradual_sparsity(&config, 25), 0.8);
    }

    #[test]
    fn test_structured_pruning() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        prune_structured(&mut weights, 2, 0.5);

        let sparsity = calculate_sparsity(&weights);
        assert!(sparsity >= 0.45);
    }

    #[test]
    fn test_pruning_mask() {
        let weights = vec![0.5, -0.1, 0.8, -0.05, 0.3];
        let mask = generate_mask(&weights, 0.4);

        assert_eq!(mask.len(), weights.len());
        let false_count = mask.iter().filter(|&&m| !m).count();
        assert_eq!(false_count, 2);
    }
}
