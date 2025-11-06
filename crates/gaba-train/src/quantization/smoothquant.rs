use ndarray::{Array1, Array2, ArrayView2};
use anyhow::Result;

pub struct SmoothQuantizer {
    num_bits: u8,
    alpha: f32,
    migration_strength: f32,
}

pub struct SmoothQuantConfig {
    pub num_bits: u8,
    pub alpha: f32,
    pub migration_strength: f32,
}

impl Default for SmoothQuantConfig {
    fn default() -> Self {
        Self {
            num_bits: 8,
            alpha: 0.5,
            migration_strength: 0.5,
        }
    }
}

pub struct SmoothedWeights {
    pub weights: Array2<f32>,
    pub smoothing_factors: Array1<f32>,
}

impl SmoothQuantizer {
    pub fn new(config: SmoothQuantConfig) -> Self {
        Self {
            num_bits: config.num_bits,
            alpha: config.alpha,
            migration_strength: config.migration_strength,
        }
    }

    pub fn quantize_layer(
        &self,
        weights: ArrayView2<f32>,
        activations: ArrayView2<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>, Array1<f32>)> {
        let smoothing_factors = self.compute_smoothing_factors(&weights, &activations)?;
        
        let smoothed = self.apply_smoothing(&weights, &smoothing_factors)?;
        
        let (quantized, scales) = self.quantize_smoothed(&smoothed)?;
        
        Ok((quantized, scales, smoothing_factors))
    }

    pub fn compute_smoothing_factors(
        &self,
        weights: &ArrayView2<f32>,
        activations: &ArrayView2<f32>,
    ) -> Result<Array1<f32>> {
        let (_, in_features) = weights.dim();
        
        let weight_max = self.compute_channel_max(weights);
        let activation_max = self.compute_activation_max(activations);
        
        let mut smoothing_factors = Array1::<f32>::ones(in_features);
        
        for i in 0..in_features {
            let w_max = weight_max[i];
            let a_max = activation_max[i];
            
            if a_max > 0.0 && w_max > 0.0 {
                let ratio = (a_max / w_max).powf(self.alpha);
                smoothing_factors[i] = ratio.max(0.01).min(100.0);
            }
        }
        
        Ok(smoothing_factors)
    }

    fn compute_channel_max(&self, weights: &ArrayView2<f32>) -> Array1<f32> {
        let (out_features, in_features) = weights.dim();
        let mut max_vals = Array1::<f32>::zeros(in_features);
        
        for j in 0..in_features {
            let mut max_val = 0.0f32;
            for i in 0..out_features {
                max_val = max_val.max(weights[[i, j]].abs());
            }
            max_vals[j] = max_val;
        }
        
        max_vals
    }

    fn compute_activation_max(&self, activations: &ArrayView2<f32>) -> Array1<f32> {
        let (batch_size, features) = activations.dim();
        let mut max_vals = Array1::<f32>::zeros(features);
        
        for j in 0..features {
            let mut max_val = 0.0f32;
            for i in 0..batch_size {
                max_val = max_val.max(activations[[i, j]].abs());
            }
            max_vals[j] = max_val;
        }
        
        max_vals
    }

    fn apply_smoothing(
        &self,
        weights: &ArrayView2<f32>,
        smoothing_factors: &Array1<f32>,
    ) -> Result<Array2<f32>> {
        let (out_features, in_features) = weights.dim();
        let mut smoothed = Array2::<f32>::zeros((out_features, in_features));
        
        for i in 0..out_features {
            for j in 0..in_features {
                smoothed[[i, j]] = weights[[i, j]] * smoothing_factors[j];
            }
        }
        
        Ok(smoothed)
    }

    fn quantize_smoothed(
        &self,
        weights: &Array2<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>)> {
        let (out_features, in_features) = weights.dim();
        let mut quantized = Array2::<i8>::zeros((out_features, in_features));
        let mut scales = Array1::<f32>::zeros(out_features);
        
        for i in 0..out_features {
            let row = weights.row(i);
            let max_val = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            
            let q_max = (1 << (self.num_bits - 1)) - 1;
            let scale = max_val / q_max as f32;
            scales[i] = scale;
            
            for j in 0..in_features {
                quantized[[i, j]] = self.quantize_value(weights[[i, j]], scale);
            }
        }
        
        Ok((quantized, scales))
    }

    fn quantize_value(&self, value: f32, scale: f32) -> i8 {
        let q_max = (1 << (self.num_bits - 1)) - 1;
        let q_min = -(1 << (self.num_bits - 1));
        
        if scale == 0.0 {
            return 0;
        }
        
        let q_val = (value / scale).round() as i32;
        q_val.clamp(q_min, q_max) as i8
    }

    pub fn dequantize(
        &self,
        quantized: &Array2<i8>,
        scales: &Array1<f32>,
        smoothing_factors: &Array1<f32>,
    ) -> Array2<f32> {
        let (out_features, in_features) = quantized.dim();
        let mut dequantized = Array2::<f32>::zeros((out_features, in_features));
        
        for i in 0..out_features {
            let scale = scales[i];
            for j in 0..in_features {
                let dq_val = quantized[[i, j]] as f32 * scale;
                dequantized[[i, j]] = dq_val / smoothing_factors[j];
            }
        }
        
        dequantized
    }

    pub fn compute_quantization_error(
        &self,
        original: &Array2<f32>,
        dequantized: &Array2<f32>,
    ) -> f32 {
        let diff = original - dequantized;
        let mse: f32 = diff.iter().map(|x| x * x).sum::<f32>() / diff.len() as f32;
        mse.sqrt()
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
    fn test_smoothquant_creation() {
        let config = SmoothQuantConfig::default();
        let quantizer = SmoothQuantizer::new(config);
        assert_eq!(quantizer.num_bits, 8);
        assert_eq!(quantizer.alpha, 0.5);
    }

    #[test]
    fn test_compute_smoothing_factors() {
        let config = SmoothQuantConfig::default();
        let quantizer = SmoothQuantizer::new(config);
        
        let weights = Array::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        
        let activations = Array::from_shape_vec((2, 3), vec![
            10.0, 20.0, 30.0,
            15.0, 25.0, 35.0,
        ]).unwrap();
        
        let factors = quantizer.compute_smoothing_factors(&weights.view(), &activations.view()).unwrap();
        assert_eq!(factors.len(), 3);
        assert!(factors.iter().all(|&f| f > 0.0));
    }

    #[test]
    fn test_apply_smoothing() {
        let config = SmoothQuantConfig::default();
        let quantizer = SmoothQuantizer::new(config);
        
        let weights = Array::from_shape_vec((2, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
        ]).unwrap();
        
        let smoothing_factors = Array::from_vec(vec![2.0, 0.5]);
        
        let smoothed = quantizer.apply_smoothing(&weights.view(), &smoothing_factors).unwrap();
        assert_eq!(smoothed[[0, 0]], 2.0);
        assert_eq!(smoothed[[0, 1]], 1.0);
    }

    #[test]
    fn test_quantize_layer() {
        let config = SmoothQuantConfig {
            num_bits: 8,
            ..Default::default()
        };
        let quantizer = SmoothQuantizer::new(config);
        
        let weights = Array::from_shape_vec((2, 3), vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ]).unwrap();
        
        let activations = Array::from_shape_vec((2, 3), vec![
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]).unwrap();
        
        let result = quantizer.quantize_layer(weights.view(), activations.view());
        assert!(result.is_ok());
        
        let (quantized, scales, smoothing_factors) = result.unwrap();
        assert_eq!(quantized.shape(), &[2, 3]);
        assert_eq!(scales.len(), 2);
        assert_eq!(smoothing_factors.len(), 3);
    }

    #[test]
    fn test_dequantize() {
        let config = SmoothQuantConfig {
            num_bits: 8,
            ..Default::default()
        };
        let quantizer = SmoothQuantizer::new(config);
        
        let quantized = Array::from_shape_vec((2, 2), vec![
            10i8, 20,
            30, 40,
        ]).unwrap();
        
        let scales = Array::from_vec(vec![0.01, 0.01]);
        let smoothing_factors = Array::from_vec(vec![1.0, 1.0]);
        
        let dequantized = quantizer.dequantize(&quantized, &scales, &smoothing_factors);
        assert_eq!(dequantized.shape(), &[2, 2]);
    }

    #[test]
    fn test_compression_ratio() {
        let config = SmoothQuantConfig {
            num_bits: 8,
            ..Default::default()
        };
        let quantizer = SmoothQuantizer::new(config);
        assert_eq!(quantizer.compression_ratio(32), 4.0);
    }
}
