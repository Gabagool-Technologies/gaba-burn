/// Data Augmentation for Route and Traffic Data
/// Generates synthetic variations to improve model generalization
use rand::Rng;

#[allow(dead_code)]
pub struct AugmentationConfig {
    pub noise_scale: f32,
    pub time_shift_range: f32,
    pub speed_variation: f32,
    pub distance_jitter: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            noise_scale: 0.05,
            time_shift_range: 0.1,
            speed_variation: 0.15,
            distance_jitter: 0.02,
        }
    }
}

#[allow(dead_code)]
pub fn augment_traffic_sample(features: &[f32], config: &AugmentationConfig) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut augmented = features.to_vec();

    // Add Gaussian noise
    for value in augmented.iter_mut() {
        let noise = rng.gen_range(-config.noise_scale..config.noise_scale);
        *value *= 1.0 + noise;
        *value = value.max(0.0); // Ensure non-negative
    }

    augmented
}

#[allow(dead_code)]
pub fn augment_route_sample(features: &[f32], config: &AugmentationConfig) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut augmented = features.to_vec();

    // Assuming feature layout: [distance, time, speed, ...]
    if augmented.len() >= 3 {
        // Distance jitter
        augmented[0] *= 1.0 + rng.gen_range(-config.distance_jitter..config.distance_jitter);

        // Time shift
        augmented[1] *= 1.0 + rng.gen_range(-config.time_shift_range..config.time_shift_range);

        // Speed variation
        augmented[2] *= 1.0 + rng.gen_range(-config.speed_variation..config.speed_variation);
    }

    // Add small noise to remaining features
    for i in 3..augmented.len() {
        let noise = rng.gen_range(-config.noise_scale..config.noise_scale);
        augmented[i] *= 1.0 + noise;
        augmented[i] = augmented[i].max(0.0);
    }

    augmented
}

#[allow(dead_code)]
pub fn mixup_samples(
    sample1: &[f32],
    sample2: &[f32],
    target1: f32,
    target2: f32,
    alpha: f32,
) -> (Vec<f32>, f32) {
    let mut rng = rand::thread_rng();
    let lambda: f32 = if alpha > 0.0 {
        // Beta distribution approximation
        let u: f32 = rng.gen();
        u.powf(1.0 / alpha).max(1.0 - u.powf(1.0 / alpha))
    } else {
        0.5
    };

    let mut mixed = vec![0.0; sample1.len()];
    for i in 0..sample1.len() {
        mixed[i] = lambda * sample1[i] + (1.0 - lambda) * sample2[i];
    }

    let mixed_target = lambda * target1 + (1.0 - lambda) * target2;

    (mixed, mixed_target)
}

#[allow(dead_code)]
pub fn cutmix_samples(
    sample1: &[f32],
    sample2: &[f32],
    _target1: f32,
    _target2: f32,
    _alpha: f32,
) -> (Vec<f32>, f32) {
    let mut rng = rand::thread_rng();
    let lambda: f32 = 0.5;
    let cut_ratio = (1.0_f32 - lambda).sqrt();

    let cut_size = (sample1.len() as f32 * cut_ratio) as usize;
    let cut_start = rng.gen_range(0..=(sample1.len() - cut_size));

    let mut mixed = sample1.to_vec();
    for i in cut_start..(cut_start + cut_size) {
        mixed[i] = sample2[i];
    }

    (mixed, 0.0)
}

#[allow(dead_code)]
pub fn time_series_augmentation(sequence: &[f32], window_size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut augmented = sequence.to_vec();

    // Random time warping
    let warp_factor = rng.gen_range(0.9..1.1);
    if warp_factor != 1.0 {
        let new_len = (sequence.len() as f32 * warp_factor) as usize;
        augmented = interpolate_sequence(sequence, new_len);
        if augmented.len() > sequence.len() {
            augmented.truncate(sequence.len());
        } else {
            augmented.resize(sequence.len(), 0.0);
        }
    }

    // Random window masking
    if window_size > 0 && sequence.len() > window_size {
        let mask_start = rng.gen_range(0..=(sequence.len() - window_size));
        for i in mask_start..(mask_start + window_size) {
            augmented[i] = 0.0;
        }
    }

    augmented
}

#[allow(dead_code)]
fn interpolate_sequence(sequence: &[f32], new_len: usize) -> Vec<f32> {
    if new_len == sequence.len() {
        return sequence.to_vec();
    }

    let mut result = vec![0.0; new_len];
    let ratio = (sequence.len() - 1) as f32 / (new_len - 1) as f32;

    for i in 0..new_len {
        let pos = i as f32 * ratio;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f32;

        if idx + 1 < sequence.len() {
            result[i] = sequence[idx] * (1.0 - frac) + sequence[idx + 1] * frac;
        } else {
            result[i] = sequence[idx];
        }
    }

    result
}

#[allow(dead_code)]
pub fn random_erasing(features: &[f32], erase_prob: f32, area_ratio_range: (f32, f32)) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut result = features.to_vec();

    if rng.gen::<f32>() < erase_prob {
        let area_ratio = rng.gen_range(area_ratio_range.0..area_ratio_range.1);
        let erase_size = (features.len() as f32 * area_ratio) as usize;

        if erase_size > 0 {
            let start = rng.gen_range(0..=(features.len() - erase_size));
            let fill_value = rng.gen_range(0.0..1.0);

            for i in start..(start + erase_size) {
                result[i] = fill_value;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augment_traffic() {
        let features = vec![50.0, 30.0, 60.0, 0.8];
        let config = AugmentationConfig::default();

        let augmented = augment_traffic_sample(&features, &config);

        assert_eq!(augmented.len(), features.len());
        assert!(augmented.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_augment_route() {
        let features = vec![10.0, 15.0, 40.0, 0.5, 0.3];
        let config = AugmentationConfig::default();

        let augmented = augment_route_sample(&features, &config);

        assert_eq!(augmented.len(), features.len());
        assert!(augmented.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_mixup() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![4.0, 5.0, 6.0];

        let (mixed, mixed_target) = mixup_samples(&sample1, &sample2, 1.0, 2.0, 1.0);

        assert_eq!(mixed.len(), sample1.len());
        assert!(mixed_target >= 1.0 && mixed_target <= 2.0);
    }

    #[test]
    fn test_cutmix() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0];
        let sample2 = vec![5.0, 6.0, 7.0, 8.0];

        let (mixed, _) = cutmix_samples(&sample1, &sample2, 1.0, 2.0, 0.5);

        assert_eq!(mixed.len(), sample1.len());
    }

    #[test]
    fn test_time_series_augmentation() {
        let sequence = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let augmented = time_series_augmentation(&sequence, 2);

        assert_eq!(augmented.len(), sequence.len());
    }

    #[test]
    fn test_random_erasing() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let erased = random_erasing(&features, 1.0, (0.2, 0.4));

        assert_eq!(erased.len(), features.len());
    }
}
