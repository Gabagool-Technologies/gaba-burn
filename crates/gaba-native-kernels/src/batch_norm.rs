/// Batch Normalization Layer
/// Normalizes activations to improve training stability

pub struct BatchNorm1dParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub momentum: f32,
    pub epsilon: f32,
}

impl BatchNorm1dParams {
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            momentum: 0.1,
            epsilon: 1e-5,
        }
    }
}

pub fn batch_norm_1d_forward(
    input: &[f32],
    params: &mut BatchNorm1dParams,
    batch_size: usize,
    num_features: usize,
    training: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    if training {
        // Compute batch statistics
        let mut batch_mean = vec![0.0; num_features];
        let mut batch_var = vec![0.0; num_features];
        
        // Calculate mean
        for b in 0..batch_size {
            for f in 0..num_features {
                batch_mean[f] += input[b * num_features + f];
            }
        }
        for f in 0..num_features {
            batch_mean[f] /= batch_size as f32;
        }
        
        // Calculate variance
        for b in 0..batch_size {
            for f in 0..num_features {
                let diff = input[b * num_features + f] - batch_mean[f];
                batch_var[f] += diff * diff;
            }
        }
        for f in 0..num_features {
            batch_var[f] /= batch_size as f32;
        }
        
        // Update running statistics
        for f in 0..num_features {
            params.running_mean[f] = (1.0 - params.momentum) * params.running_mean[f] 
                                   + params.momentum * batch_mean[f];
            params.running_var[f] = (1.0 - params.momentum) * params.running_var[f] 
                                  + params.momentum * batch_var[f];
        }
        
        // Normalize using batch statistics
        for b in 0..batch_size {
            for f in 0..num_features {
                let idx = b * num_features + f;
                let normalized = (input[idx] - batch_mean[f]) / (batch_var[f] + params.epsilon).sqrt();
                output[idx] = params.gamma[f] * normalized + params.beta[f];
            }
        }
    } else {
        // Use running statistics for inference
        for b in 0..batch_size {
            for f in 0..num_features {
                let idx = b * num_features + f;
                let normalized = (input[idx] - params.running_mean[f]) 
                               / (params.running_var[f] + params.epsilon).sqrt();
                output[idx] = params.gamma[f] * normalized + params.beta[f];
            }
        }
    }
    
    output
}

pub struct BatchNorm2dParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub momentum: f32,
    pub epsilon: f32,
}

impl BatchNorm2dParams {
    pub fn new(num_channels: usize) -> Self {
        Self {
            gamma: vec![1.0; num_channels],
            beta: vec![0.0; num_channels],
            running_mean: vec![0.0; num_channels],
            running_var: vec![1.0; num_channels],
            momentum: 0.1,
            epsilon: 1e-5,
        }
    }
}

pub fn batch_norm_2d_forward(
    input: &[f32],
    params: &mut BatchNorm2dParams,
    batch_size: usize,
    num_channels: usize,
    height: usize,
    width: usize,
    training: bool,
) -> Vec<f32> {
    let spatial_size = height * width;
    let mut output = vec![0.0; input.len()];
    
    if training {
        let mut batch_mean = vec![0.0; num_channels];
        let mut batch_var = vec![0.0; num_channels];
        let n = (batch_size * spatial_size) as f32;
        
        // Calculate mean per channel
        for b in 0..batch_size {
            for c in 0..num_channels {
                for i in 0..spatial_size {
                    let idx = b * num_channels * spatial_size + c * spatial_size + i;
                    batch_mean[c] += input[idx];
                }
            }
        }
        for c in 0..num_channels {
            batch_mean[c] /= n;
        }
        
        // Calculate variance per channel
        for b in 0..batch_size {
            for c in 0..num_channels {
                for i in 0..spatial_size {
                    let idx = b * num_channels * spatial_size + c * spatial_size + i;
                    let diff = input[idx] - batch_mean[c];
                    batch_var[c] += diff * diff;
                }
            }
        }
        for c in 0..num_channels {
            batch_var[c] /= n;
        }
        
        // Update running statistics
        for c in 0..num_channels {
            params.running_mean[c] = (1.0 - params.momentum) * params.running_mean[c] 
                                   + params.momentum * batch_mean[c];
            params.running_var[c] = (1.0 - params.momentum) * params.running_var[c] 
                                  + params.momentum * batch_var[c];
        }
        
        // Normalize
        for b in 0..batch_size {
            for c in 0..num_channels {
                for i in 0..spatial_size {
                    let idx = b * num_channels * spatial_size + c * spatial_size + i;
                    let normalized = (input[idx] - batch_mean[c]) / (batch_var[c] + params.epsilon).sqrt();
                    output[idx] = params.gamma[c] * normalized + params.beta[c];
                }
            }
        }
    } else {
        // Use running statistics
        for b in 0..batch_size {
            for c in 0..num_channels {
                for i in 0..spatial_size {
                    let idx = b * num_channels * spatial_size + c * spatial_size + i;
                    let normalized = (input[idx] - params.running_mean[c]) 
                                   / (params.running_var[c] + params.epsilon).sqrt();
                    output[idx] = params.gamma[c] * normalized + params.beta[c];
                }
            }
        }
    }
    
    output
}

pub fn layer_norm_forward(
    input: &[f32],
    normalized_shape: &[usize],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let norm_size: usize = normalized_shape.iter().product();
    let num_groups = input.len() / norm_size;
    
    for g in 0..num_groups {
        let start = g * norm_size;
        let end = start + norm_size;
        let group = &input[start..end];
        
        // Calculate mean
        let mean: f32 = group.iter().sum::<f32>() / norm_size as f32;
        
        // Calculate variance
        let variance: f32 = group.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / norm_size as f32;
        
        // Normalize
        let std = (variance + epsilon).sqrt();
        for i in 0..norm_size {
            let normalized = (group[i] - mean) / std;
            output[start + i] = gamma[i] * normalized + beta[i];
        }
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_norm_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut params = BatchNorm1dParams::new(2);
        
        let output = batch_norm_1d_forward(&input, &mut params, 3, 2, true);
        
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_batch_norm_2d() {
        let input = vec![1.0; 32];
        let mut params = BatchNorm2dParams::new(2);
        
        let output = batch_norm_2d_forward(&input, &mut params, 2, 2, 2, 2, true);
        
        assert_eq!(output.len(), input.len());
    }
    
    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        
        let output = layer_norm_forward(&input, &[2], &gamma, &beta, 1e-5);
        
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
