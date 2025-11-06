/// Dropout Layer for Regularization
/// Randomly zeros elements during training to prevent overfitting
use rand::Rng;

pub struct DropoutParams {
    pub drop_prob: f32,
    pub training: bool,
}

impl DropoutParams {
    pub fn new(drop_prob: f32) -> Self {
        assert!(
            drop_prob >= 0.0 && drop_prob < 1.0,
            "drop_prob must be in [0, 1)"
        );
        Self {
            drop_prob,
            training: true,
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

pub fn dropout_forward(input: &[f32], params: &DropoutParams) -> (Vec<f32>, Vec<bool>) {
    let mut output = vec![0.0; input.len()];
    let mut mask = vec![false; input.len()];

    if params.training && params.drop_prob > 0.0 {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - params.drop_prob);

        for i in 0..input.len() {
            if rng.gen::<f32>() > params.drop_prob {
                output[i] = input[i] * scale;
                mask[i] = true;
            }
        }
    } else {
        output.copy_from_slice(input);
        mask.fill(true);
    }

    (output, mask)
}

pub fn dropout_backward(grad_output: &[f32], mask: &[bool], drop_prob: f32) -> Vec<f32> {
    let mut grad_input = vec![0.0; grad_output.len()];
    let scale = 1.0 / (1.0 - drop_prob);

    for i in 0..grad_output.len() {
        if mask[i] {
            grad_input[i] = grad_output[i] * scale;
        }
    }

    grad_input
}

pub fn spatial_dropout_2d_forward(
    input: &[f32],
    batch_size: usize,
    num_channels: usize,
    height: usize,
    width: usize,
    drop_prob: f32,
    training: bool,
) -> (Vec<f32>, Vec<bool>) {
    let mut output = vec![0.0; input.len()];
    let spatial_size = height * width;
    let mut channel_mask = vec![false; batch_size * num_channels];

    if training && drop_prob > 0.0 {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - drop_prob);

        // Generate mask per channel (not per pixel)
        for b in 0..batch_size {
            for c in 0..num_channels {
                let idx = b * num_channels + c;
                channel_mask[idx] = rng.gen::<f32>() > drop_prob;
            }
        }

        // Apply mask to entire spatial dimension
        for b in 0..batch_size {
            for c in 0..num_channels {
                let mask_idx = b * num_channels + c;
                if channel_mask[mask_idx] {
                    for i in 0..spatial_size {
                        let data_idx = b * num_channels * spatial_size + c * spatial_size + i;
                        output[data_idx] = input[data_idx] * scale;
                    }
                }
            }
        }
    } else {
        output.copy_from_slice(input);
        channel_mask.fill(true);
    }

    (output, channel_mask)
}

pub fn alpha_dropout_forward(
    input: &[f32],
    drop_prob: f32,
    training: bool,
) -> (Vec<f32>, Vec<bool>) {
    let mut output = vec![0.0; input.len()];
    let mut mask = vec![false; input.len()];

    if training && drop_prob > 0.0 {
        let mut rng = rand::thread_rng();

        // SELU constants
        let alpha = 1.6732632423543772848170429916717;
        let lambda = 1.0507009873554804934193349852946;

        // Alpha dropout parameters
        let alpha_p: f32 = -lambda * alpha;
        let a = ((1.0 - drop_prob) * (1.0 + drop_prob * alpha_p.powi(2))).sqrt();
        let b = -a * alpha_p * drop_prob;

        for i in 0..input.len() {
            if rng.gen::<f32>() > drop_prob {
                output[i] = a * input[i] + b;
                mask[i] = true;
            } else {
                output[i] = alpha_p * a + b;
            }
        }
    } else {
        output.copy_from_slice(input);
        mask.fill(true);
    }

    (output, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_training() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DropoutParams::new(0.5);

        let (output, mask) = dropout_forward(&input, &params);

        assert_eq!(output.len(), input.len());
        assert_eq!(mask.len(), input.len());

        // Check that some elements are dropped
        let dropped = mask.iter().filter(|&&x| !x).count();
        assert!(dropped > 0 || dropped < input.len());

        // Check scaling
        for i in 0..input.len() {
            if mask[i] {
                assert!((output[i] - input[i] * 2.0).abs() < 1e-5);
            } else {
                assert_eq!(output[i], 0.0);
            }
        }
    }

    #[test]
    fn test_dropout_inference() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut params = DropoutParams::new(0.5);
        params.eval();

        let (output, _) = dropout_forward(&input, &params);

        // In inference mode, output should equal input
        assert_eq!(output, input);
    }

    #[test]
    fn test_spatial_dropout() {
        let input = vec![1.0; 32];
        let (output, mask) = spatial_dropout_2d_forward(&input, 2, 2, 2, 2, 0.5, true);

        assert_eq!(output.len(), input.len());
        assert_eq!(mask.len(), 4); // batch_size * num_channels
    }

    #[test]
    fn test_alpha_dropout() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (output, _) = alpha_dropout_forward(&input, 0.5, true);

        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dropout_backward() {
        let grad_output = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];

        let grad_input = dropout_backward(&grad_output, &mask, 0.5);

        assert_eq!(grad_input.len(), grad_output.len());
        assert_eq!(grad_input[0], 2.0); // scaled by 1/(1-0.5)
        assert_eq!(grad_input[1], 0.0); // masked
        assert_eq!(grad_input[2], 6.0); // scaled
        assert_eq!(grad_input[3], 0.0); // masked
    }
}
