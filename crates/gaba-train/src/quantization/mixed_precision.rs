use ndarray::{Array1, Array2, ArrayView2};
use anyhow::Result;
use std::collections::HashMap;

pub struct MixedPrecisionQuantizer {
    layer_configs: HashMap<String, LayerQuantConfig>,
    default_bits: u8,
}

#[derive(Clone, Copy)]
pub struct LayerQuantConfig {
    pub num_bits: u8,
    pub layer_type: LayerType,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Attention,
    FeedForward,
    Embedding,
    Output,
}

pub struct SensitivityAnalysis {
    pub layer_sensitivities: HashMap<String, f32>,
    pub bit_assignments: HashMap<String, u8>,
}

impl MixedPrecisionQuantizer {
    pub fn new(default_bits: u8) -> Self {
        Self {
            layer_configs: HashMap::new(),
            default_bits,
        }
    }

    pub fn add_layer_config(&mut self, layer_name: String, config: LayerQuantConfig) {
        self.layer_configs.insert(layer_name, config);
    }

    pub fn auto_configure(&mut self, sensitivity: &SensitivityAnalysis) {
        for (layer_name, &sens) in &sensitivity.layer_sensitivities {
            let bits = if sens > 0.8 {
                16
            } else if sens > 0.5 {
                8
            } else {
                4
            };
            
            let layer_type = self.infer_layer_type(layer_name);
            self.add_layer_config(
                layer_name.clone(),
                LayerQuantConfig {
                    num_bits: bits,
                    layer_type,
                },
            );
        }
    }

    fn infer_layer_type(&self, layer_name: &str) -> LayerType {
        if layer_name.contains("attention") || layer_name.contains("attn") {
            LayerType::Attention
        } else if layer_name.contains("ffn") || layer_name.contains("mlp") {
            LayerType::FeedForward
        } else if layer_name.contains("embed") {
            LayerType::Embedding
        } else if layer_name.contains("output") || layer_name.contains("head") {
            LayerType::Output
        } else {
            LayerType::FeedForward
        }
    }

    pub fn quantize_layer(
        &self,
        layer_name: &str,
        weights: ArrayView2<f32>,
    ) -> Result<(Array2<i8>, Array1<f32>, u8)> {
        let config = self.layer_configs.get(layer_name);
        let num_bits = config.map(|c| c.num_bits).unwrap_or(self.default_bits);
        
        let (quantized, scales) = self.quantize_with_bits(weights, num_bits)?;
        
        Ok((quantized, scales, num_bits))
    }

    fn quantize_with_bits(
        &self,
        weights: ArrayView2<f32>,
        num_bits: u8,
    ) -> Result<(Array2<i8>, Array1<f32>)> {
        let (out_features, in_features) = weights.dim();
        let mut quantized = Array2::<i8>::zeros((out_features, in_features));
        let mut scales = Array1::<f32>::zeros(out_features);
        
        let q_max = (1 << (num_bits - 1)) - 1;
        let q_min = -(1 << (num_bits - 1));
        
        for i in 0..out_features {
            let row = weights.row(i);
            let max_val = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            
            let scale = if max_val > 0.0 {
                max_val / q_max as f32
            } else {
                1.0
            };
            scales[i] = scale;
            
            for j in 0..in_features {
                let val = weights[[i, j]];
                let q_val = (val / scale).round() as i32;
                quantized[[i, j]] = q_val.clamp(q_min, q_max) as i8;
            }
        }
        
        Ok((quantized, scales))
    }

    pub fn analyze_sensitivity(
        &self,
        layer_weights: &HashMap<String, Array2<f32>>,
        calibration_data: &Array2<f32>,
    ) -> Result<SensitivityAnalysis> {
        let mut sensitivities = HashMap::new();
        let mut bit_assignments = HashMap::new();
        
        for (layer_name, weights) in layer_weights {
            let sensitivity = self.compute_layer_sensitivity(weights.view(), calibration_data.view())?;
            sensitivities.insert(layer_name.clone(), sensitivity);
            
            let bits = if sensitivity > 0.8 {
                16
            } else if sensitivity > 0.5 {
                8
            } else {
                4
            };
            bit_assignments.insert(layer_name.clone(), bits);
        }
        
        Ok(SensitivityAnalysis {
            layer_sensitivities: sensitivities,
            bit_assignments,
        })
    }

    fn compute_layer_sensitivity(
        &self,
        weights: ArrayView2<f32>,
        _calibration: ArrayView2<f32>,
    ) -> Result<f32> {
        let variance: f32 = weights.iter().map(|&x| x * x).sum::<f32>() / weights.len() as f32;
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let std_dev = (variance - mean * mean).sqrt();
        
        let sensitivity = (std_dev / (variance.sqrt() + 1e-8)).min(1.0);
        
        Ok(sensitivity)
    }

    pub fn get_layer_config(&self, layer_name: &str) -> Option<LayerQuantConfig> {
        self.layer_configs.get(layer_name).copied()
    }

    pub fn total_model_size(&self, layer_sizes: &HashMap<String, (usize, usize)>) -> usize {
        let mut total_bits = 0;
        
        for (layer_name, &(rows, cols)) in layer_sizes {
            let config = self.layer_configs.get(layer_name);
            let num_bits = config.map(|c| c.num_bits).unwrap_or(self.default_bits);
            total_bits += rows * cols * num_bits as usize;
        }
        
        (total_bits + 7) / 8
    }

    pub fn compression_ratio(&self, layer_sizes: &HashMap<String, (usize, usize)>) -> f32 {
        let original_size: usize = layer_sizes.values().map(|(r, c)| r * c * 32).sum();
        let compressed_size = self.total_model_size(layer_sizes) * 8;
        original_size as f32 / compressed_size as f32
    }
}

pub fn create_default_mixed_precision() -> MixedPrecisionQuantizer {
    let mut quantizer = MixedPrecisionQuantizer::new(8);
    
    quantizer.add_layer_config(
        "attention".to_string(),
        LayerQuantConfig {
            num_bits: 16,
            layer_type: LayerType::Attention,
        },
    );
    
    quantizer.add_layer_config(
        "ffn".to_string(),
        LayerQuantConfig {
            num_bits: 4,
            layer_type: LayerType::FeedForward,
        },
    );
    
    quantizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_mixed_precision_creation() {
        let quantizer = MixedPrecisionQuantizer::new(8);
        assert_eq!(quantizer.default_bits, 8);
    }

    #[test]
    fn test_add_layer_config() {
        let mut quantizer = MixedPrecisionQuantizer::new(8);
        quantizer.add_layer_config(
            "layer1".to_string(),
            LayerQuantConfig {
                num_bits: 4,
                layer_type: LayerType::FeedForward,
            },
        );
        
        let config = quantizer.get_layer_config("layer1");
        assert!(config.is_some());
        assert_eq!(config.unwrap().num_bits, 4);
    }

    #[test]
    fn test_infer_layer_type() {
        let quantizer = MixedPrecisionQuantizer::new(8);
        assert_eq!(quantizer.infer_layer_type("attention_layer"), LayerType::Attention);
        assert_eq!(quantizer.infer_layer_type("ffn_layer"), LayerType::FeedForward);
        assert_eq!(quantizer.infer_layer_type("embedding"), LayerType::Embedding);
        assert_eq!(quantizer.infer_layer_type("output_head"), LayerType::Output);
    }

    #[test]
    fn test_quantize_layer() {
        let mut quantizer = MixedPrecisionQuantizer::new(8);
        quantizer.add_layer_config(
            "test_layer".to_string(),
            LayerQuantConfig {
                num_bits: 4,
                layer_type: LayerType::FeedForward,
            },
        );
        
        let weights = Array::from_shape_vec((2, 4), vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ]).unwrap();
        
        let result = quantizer.quantize_layer("test_layer", weights.view());
        assert!(result.is_ok());
        
        let (quantized, scales, bits) = result.unwrap();
        assert_eq!(quantized.shape(), &[2, 4]);
        assert_eq!(scales.len(), 2);
        assert_eq!(bits, 4);
    }

    #[test]
    fn test_compute_layer_sensitivity() {
        let quantizer = MixedPrecisionQuantizer::new(8);
        
        let weights = Array::from_shape_vec((3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        
        let calibration = Array::from_shape_vec((2, 3), vec![
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]).unwrap();
        
        let sensitivity = quantizer.compute_layer_sensitivity(weights.view(), calibration.view());
        assert!(sensitivity.is_ok());
        let sens_val = sensitivity.unwrap();
        assert!(sens_val >= 0.0 && sens_val <= 1.0);
    }

    #[test]
    fn test_total_model_size() {
        let mut quantizer = MixedPrecisionQuantizer::new(8);
        quantizer.add_layer_config(
            "layer1".to_string(),
            LayerQuantConfig {
                num_bits: 4,
                layer_type: LayerType::FeedForward,
            },
        );
        quantizer.add_layer_config(
            "layer2".to_string(),
            LayerQuantConfig {
                num_bits: 8,
                layer_type: LayerType::Attention,
            },
        );
        
        let mut layer_sizes = HashMap::new();
        layer_sizes.insert("layer1".to_string(), (100, 100));
        layer_sizes.insert("layer2".to_string(), (100, 100));
        
        let size = quantizer.total_model_size(&layer_sizes);
        assert!(size > 0);
    }

    #[test]
    fn test_compression_ratio() {
        let mut quantizer = MixedPrecisionQuantizer::new(4);
        quantizer.add_layer_config(
            "layer1".to_string(),
            LayerQuantConfig {
                num_bits: 4,
                layer_type: LayerType::FeedForward,
            },
        );
        
        let mut layer_sizes = HashMap::new();
        layer_sizes.insert("layer1".to_string(), (100, 100));
        
        let ratio = quantizer.compression_ratio(&layer_sizes);
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_create_default_mixed_precision() {
        let quantizer = create_default_mixed_precision();
        
        let attn_config = quantizer.get_layer_config("attention");
        assert!(attn_config.is_some());
        assert_eq!(attn_config.unwrap().num_bits, 16);
        
        let ffn_config = quantizer.get_layer_config("ffn");
        assert!(ffn_config.is_some());
        assert_eq!(ffn_config.unwrap().num_bits, 4);
    }
}
