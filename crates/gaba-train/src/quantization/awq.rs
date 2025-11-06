use ndarray::{Array1, Array2, ArrayView2};
use anyhow::Result;

pub struct AwqQuantizer {
    num_bits: u8,
    group_size: usize,
    protection_factor: f32,
    salience_threshold: f32,
}

pub struct AwqConfig {
    pub num_bits: u8,
    pub group_size: usize,
    pub protection_factor: f32,
    pub salience_threshold: f32,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            num_bits: 4,
            group_size: 128,
            protection_factor: 2.0,
            salience_threshold: 0.8,
        }
    }
}

pub struct SalienceScores {
    pub channel_scores: Array1<f32>,
    pub important_channels: Vec<usize>,
}

impl AwqQuantizer {
    pub fn new(config: AwqConfig) -> Self {
        Self {
            num_bits: config.num_bits,
            group_size: config.group_size,
            protection_factor: config.protection_factor,
            salience_threshold: config.salience_threshold,
        }
    }

    pub fn quantize_layer(
        &self,
        weights: ArrayView2<f32>,
        activations: ArrayView2<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>, Array1<f32>)> {
        let salience = self.compute_salience(&weights, &activations)?;
        
        let scales = self.compute_per_channel_scales(&weights, &salience)?;
        
        let (quantized, quant_scales) = self.quantize_with_scales(&weights, &scales)?;
        
        Ok((quantized, scales, quant_scales))
    }

    pub fn compute_salience(
        &self,
        weights: &ArrayView2<f32>,
        activations: &ArrayView2<f32>,
    ) -> Result<SalienceScores> {
        let (out_channels, _) = weights.dim();
        let mut channel_scores = Array1::<f32>::zeros(out_channels);
        
        for i in 0..out_channels {
            let weight_row = weights.row(i);
            
            let mut salience = 0.0;
            for batch_idx in 0..activations.nrows() {
                let act = activations.row(batch_idx);
                let weighted_act: f32 = weight_row
                    .iter()
                    .zip(act.iter())
                    .map(|(w, a)| (w * a).abs())
                    .sum();
                salience += weighted_act;
            }
            
            channel_scores[i] = salience / activations.nrows() as f32;
        }
        
        let max_score = channel_scores.iter().cloned().fold(0.0f32, f32::max);
        channel_scores /= max_score;
        
        let important_channels: Vec<usize> = channel_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= self.salience_threshold)
            .map(|(idx, _)| idx)
            .collect();
        
        Ok(SalienceScores {
            channel_scores,
            important_channels,
        })
    }

    fn compute_per_channel_scales(
        &self,
        weights: &ArrayView2<f32>,
        salience: &SalienceScores,
    ) -> Result<Array1<f32>> {
        let (out_channels, _) = weights.dim();
        let mut scales = Array1::<f32>::ones(out_channels);
        
        for &channel_idx in &salience.important_channels {
            scales[channel_idx] = self.protection_factor;
        }
        
        Ok(scales)
    }

    fn quantize_with_scales(
        &self,
        weights: &ArrayView2<f32>,
        channel_scales: &Array1<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>)> {
        let (out_channels, in_channels) = weights.dim();
        let mut quantized = Array2::<i8>::zeros((out_channels, in_channels));
        let mut quant_scales = Array1::<f32>::zeros(out_channels);
        
        for i in 0..out_channels {
            let row = weights.row(i);
            let channel_scale = channel_scales[i];
            
            let scaled_row: Vec<f32> = row.iter().map(|&w| w * channel_scale).collect();
            
            let max_val = scaled_row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let q_max = (1 << (self.num_bits - 1)) - 1;
            let scale = max_val / q_max as f32;
            quant_scales[i] = scale / channel_scale;
            
            for j in 0..in_channels {
                let q_val = self.quantize_value(scaled_row[j], scale);
                quantized[[i, j]] = q_val;
            }
        }
        
        Ok((quantized, quant_scales))
    }

    fn quantize_value(&self, value: f32, scale: f32) -> i8 {
        let q_max = (1 << (self.num_bits - 1)) - 1;
        let q_min = -(1 << (self.num_bits - 1));
        let q_val = (value / scale).round() as i32;
        q_val.clamp(q_min, q_max) as i8
    }

    pub fn dequantize(
        &self,
        quantized: &Array2<i8>,
        channel_scales: &Array1<f32>,
        quant_scales: &Array1<f32>,
    ) -> Array2<f32> {
        let (out_channels, in_channels) = quantized.dim();
        let mut dequantized = Array2::<f32>::zeros((out_channels, in_channels));
        
        for i in 0..out_channels {
            let scale = quant_scales[i];
            let channel_scale = channel_scales[i];
            
            for j in 0..in_channels {
                dequantized[[i, j]] = quantized[[i, j]] as f32 * scale / channel_scale;
            }
        }
        
        dequantized
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
    fn test_awq_quantizer_creation() {
        let config = AwqConfig::default();
        let quantizer = AwqQuantizer::new(config);
        assert_eq!(quantizer.num_bits, 4);
        assert_eq!(quantizer.group_size, 128);
    }

    #[test]
    fn test_compute_salience() {
        let config = AwqConfig::default();
        let quantizer = AwqQuantizer::new(config);
        
        let weights = Array::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            0.1, 0.2, 0.3, 0.4,
            5.0, 6.0, 7.0, 8.0,
        ]).unwrap();
        
        let activations = Array::from_shape_vec((2, 4), vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]).unwrap();
        
        let salience = quantizer.compute_salience(&weights.view(), &activations.view()).unwrap();
        assert_eq!(salience.channel_scores.len(), 3);
        assert!(salience.channel_scores[2] > salience.channel_scores[1]);
    }

    #[test]
    fn test_quantize_layer() {
        let config = AwqConfig {
            num_bits: 8,
            salience_threshold: 0.5,
            ..Default::default()
        };
        let quantizer = AwqQuantizer::new(config);
        
        let weights = Array::from_shape_vec((2, 4), vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ]).unwrap();
        
        let activations = Array::from_shape_vec((2, 4), vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]).unwrap();
        
        let result = quantizer.quantize_layer(weights.view(), activations.view());
        assert!(result.is_ok());
        
        let (quantized, channel_scales, quant_scales) = result.unwrap();
        assert_eq!(quantized.shape(), &[2, 4]);
        assert_eq!(channel_scales.len(), 2);
        assert_eq!(quant_scales.len(), 2);
    }

    #[test]
    fn test_dequantize() {
        let config = AwqConfig {
            num_bits: 8,
            ..Default::default()
        };
        let quantizer = AwqQuantizer::new(config);
        
        let quantized = Array::from_shape_vec((2, 2), vec![
            10i8, 20,
            30, 40,
        ]).unwrap();
        
        let channel_scales = Array::from_vec(vec![1.0, 1.0]);
        let quant_scales = Array::from_vec(vec![0.01, 0.01]);
        
        let dequantized = quantizer.dequantize(&quantized, &channel_scales, &quant_scales);
        assert_eq!(dequantized.shape(), &[2, 2]);
        assert!((dequantized[[0, 0]] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio() {
        let config = AwqConfig {
            num_bits: 4,
            ..Default::default()
        };
        let quantizer = AwqQuantizer::new(config);
        assert_eq!(quantizer.compression_ratio(32), 8.0);
    }
}
