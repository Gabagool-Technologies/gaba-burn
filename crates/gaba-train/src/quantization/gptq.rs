use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use anyhow::Result;

pub struct GptqQuantizer {
    num_bits: u8,
    group_size: usize,
    damping: f32,
    block_size: usize,
}

pub struct GptqConfig {
    pub num_bits: u8,
    pub group_size: usize,
    pub damping: f32,
    pub block_size: usize,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            num_bits: 4,
            group_size: 128,
            damping: 0.01,
            block_size: 128,
        }
    }
}

impl GptqQuantizer {
    pub fn new(config: GptqConfig) -> Self {
        Self {
            num_bits: config.num_bits,
            group_size: config.group_size,
            damping: config.damping,
            block_size: config.block_size,
        }
    }

    pub fn quantize_layer(
        &self,
        weights: ArrayView2<f32>,
        calibration_data: ArrayView2<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>)> {
        let (out_features, in_features) = weights.dim();
        
        let hessian = self.compute_hessian(&calibration_data)?;
        let hessian_inv = self.compute_hessian_inverse(&hessian)?;
        
        let mut quantized = Array2::<i8>::zeros((out_features, in_features));
        let mut scales = Array1::<f32>::zeros(out_features);
        let mut weights_working = weights.to_owned();
        
        for block_start in (0..in_features).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(in_features);
            let block_size = block_end - block_start;
            
            for i in 0..out_features {
                let block = weights_working.slice(ndarray::s![i, block_start..block_end]);
                
                let scale = self.compute_optimal_scale(block.view());
                scales[i] = scale;
                
                for j in 0..block_size {
                    let idx = block_start + j;
                    let val = weights_working[[i, idx]];
                    let q_val = self.quantize_value(val, scale);
                    quantized[[i, idx]] = q_val;
                    
                    let error = val - self.dequantize_value(q_val, scale);
                    
                    if block_end < in_features {
                        let h_inv_slice = hessian_inv.slice(ndarray::s![idx, block_end..]);
                        for k in 0..(in_features - block_end) {
                            weights_working[[i, block_end + k]] -= error * h_inv_slice[k];
                        }
                    }
                }
            }
        }
        
        Ok((quantized, scales))
    }

    fn compute_hessian(&self, data: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let (batch_size, features) = data.dim();
        let mut hessian = Array2::<f32>::zeros((features, features));
        
        for i in 0..batch_size {
            let x = data.row(i);
            for j in 0..features {
                for k in 0..features {
                    hessian[[j, k]] += x[j] * x[k];
                }
            }
        }
        
        hessian /= batch_size as f32;
        
        Ok(hessian)
    }

    fn compute_hessian_inverse(&self, hessian: &Array2<f32>) -> Result<Array2<f32>> {
        let n = hessian.nrows();
        let mut h_damped = hessian.clone();
        
        for i in 0..n {
            h_damped[[i, i]] += self.damping;
        }
        
        let h_inv = self.cholesky_inverse(&h_damped)?;
        
        Ok(h_inv)
    }

    fn cholesky_inverse(&self, matrix: &Array2<f32>) -> Result<Array2<f32>> {
        let n = matrix.nrows();
        let mut l = Array2::<f32>::zeros((n, n));
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = matrix[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                
                if i == j {
                    l[[i, j]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
        }
        
        let mut l_inv = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            l_inv[[i, i]] = 1.0 / l[[i, i]];
            for j in (0..i).rev() {
                let mut sum = 0.0;
                for k in (j + 1)..=i {
                    sum += l[[i, k]] * l_inv[[k, j]];
                }
                l_inv[[i, j]] = -sum / l[[j, j]];
            }
        }
        
        let mut inv = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    inv[[i, j]] += l_inv[[k, i]] * l_inv[[k, j]];
                }
            }
        }
        
        Ok(inv)
    }

    fn compute_optimal_scale(&self, weights: ArrayView1<f32>) -> f32 {
        let max_val = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let q_max = (1 << (self.num_bits - 1)) - 1;
        max_val / q_max as f32
    }

    fn quantize_value(&self, value: f32, scale: f32) -> i8 {
        let q_max = (1 << (self.num_bits - 1)) - 1;
        let q_min = -(1 << (self.num_bits - 1));
        let q_val = (value / scale).round() as i32;
        q_val.clamp(q_min, q_max) as i8
    }

    fn dequantize_value(&self, q_value: i8, scale: f32) -> f32 {
        q_value as f32 * scale
    }

    pub fn compression_ratio(&self, original_bits: u8) -> f32 {
        original_bits as f32 / self.num_bits as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_gptq_quantizer_creation() {
        let config = GptqConfig::default();
        let quantizer = GptqQuantizer::new(config);
        assert_eq!(quantizer.num_bits, 4);
        assert_eq!(quantizer.group_size, 128);
    }

    #[test]
    fn test_quantize_value() {
        let config = GptqConfig {
            num_bits: 4,
            ..Default::default()
        };
        let quantizer = GptqQuantizer::new(config);
        
        let scale = 0.1;
        let q_val = quantizer.quantize_value(0.7, scale);
        assert_eq!(q_val, 7);
        
        let dq_val = quantizer.dequantize_value(q_val, scale);
        assert!((dq_val - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_compute_hessian() {
        let config = GptqConfig::default();
        let quantizer = GptqQuantizer::new(config);
        
        let data = Array::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]).unwrap();
        
        let hessian = quantizer.compute_hessian(&data.view()).unwrap();
        assert_eq!(hessian.shape(), &[3, 3]);
    }

    #[test]
    fn test_compression_ratio() {
        let config = GptqConfig {
            num_bits: 4,
            ..Default::default()
        };
        let quantizer = GptqQuantizer::new(config);
        assert_eq!(quantizer.compression_ratio(32), 8.0);
    }

    #[test]
    fn test_quantize_small_layer() {
        let config = GptqConfig {
            num_bits: 8,
            block_size: 4,
            ..Default::default()
        };
        let quantizer = GptqQuantizer::new(config);
        
        let weights = Array::from_shape_vec((2, 4), vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ]).unwrap();
        
        let calibration = Array::from_shape_vec((2, 4), vec![
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ]).unwrap();
        
        let result = quantizer.quantize_layer(weights.view(), calibration.view());
        assert!(result.is_ok());
        
        let (quantized, scales) = result.unwrap();
        assert_eq!(quantized.shape(), &[2, 4]);
        assert_eq!(scales.len(), 2);
    }
}
