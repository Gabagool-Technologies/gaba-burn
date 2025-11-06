use ndarray::{Array1, Array2, ArrayView2};
use anyhow::Result;
use crate::peft::lora::{LoraConfig, LoraLayer};

pub struct QLoraConfig {
    pub lora_config: LoraConfig,
    pub quant_bits: u8,
    pub double_quant: bool,
    pub quant_type: QuantType,
}

#[derive(Clone, Copy)]
pub enum QuantType {
    NF4,
    FP4,
    INT4,
}

impl Default for QLoraConfig {
    fn default() -> Self {
        Self {
            lora_config: LoraConfig::default(),
            quant_bits: 4,
            double_quant: true,
            quant_type: QuantType::NF4,
        }
    }
}

pub struct QLoraLayer {
    pub lora_layer: LoraLayer,
    pub quantized_base: QuantizedWeight,
    pub quant_config: QLoraConfig,
}

pub struct QuantizedWeight {
    pub quantized_data: Vec<u8>,
    pub scales: Array1<f32>,
    pub zero_points: Array1<i8>,
    pub shape: (usize, usize),
    pub quant_type: QuantType,
}

impl QLoraLayer {
    pub fn new(
        base_weight: ArrayView2<f32>,
        in_features: usize,
        out_features: usize,
        config: QLoraConfig,
    ) -> Result<Self> {
        let lora_layer = LoraLayer::new(in_features, out_features, &config.lora_config);
        
        let quantized_base = Self::quantize_base_weight(base_weight, &config)?;
        
        Ok(Self {
            lora_layer,
            quantized_base,
            quant_config: config,
        })
    }

    fn quantize_base_weight(
        weight: ArrayView2<f32>,
        config: &QLoraConfig,
    ) -> Result<QuantizedWeight> {
        let (rows, cols) = weight.dim();
        let group_size = 128;
        let num_groups = (cols + group_size - 1) / group_size;
        
        let mut quantized_data = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        
        for row_idx in 0..rows {
            for group_idx in 0..num_groups {
                let start = group_idx * group_size;
                let end = (start + group_size).min(cols);
                
                let group = weight.slice(ndarray::s![row_idx, start..end]);
                
                let (q_data, scale, zero_point) = match config.quant_type {
                    QuantType::NF4 => Self::quantize_nf4(&group)?,
                    QuantType::FP4 => Self::quantize_fp4(&group)?,
                    QuantType::INT4 => Self::quantize_int4(&group)?,
                };
                
                quantized_data.extend_from_slice(&q_data);
                scales.push(scale);
                zero_points.push(zero_point);
            }
        }
        
        Ok(QuantizedWeight {
            quantized_data,
            scales: Array1::from_vec(scales),
            zero_points: Array1::from_vec(zero_points),
            shape: (rows, cols),
            quant_type: config.quant_type,
        })
    }

    fn quantize_nf4(values: &ndarray::ArrayView1<f32>) -> Result<(Vec<u8>, f32, i8)> {
        let nf4_values = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        
        let max_val = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_val;
        
        let mut quantized = Vec::new();
        for &val in values.iter() {
            let normalized = val / scale;
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            
            for (idx, &nf4_val) in nf4_values.iter().enumerate() {
                let dist = (normalized - nf4_val).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            
            quantized.push(best_idx as u8);
        }
        
        Ok((quantized, scale, 0))
    }

    fn quantize_fp4(values: &ndarray::ArrayView1<f32>) -> Result<(Vec<u8>, f32, i8)> {
        let max_val = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_val / 7.0;
        
        let mut quantized = Vec::new();
        for &val in values.iter() {
            let q_val = (val / scale).round() as i8;
            let clamped = q_val.clamp(-7, 7);
            quantized.push((clamped + 8) as u8);
        }
        
        Ok((quantized, scale, 0))
    }

    fn quantize_int4(values: &ndarray::ArrayView1<f32>) -> Result<(Vec<u8>, f32, i8)> {
        let max_val = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_val / 7.0;
        
        let mut quantized = Vec::new();
        for &val in values.iter() {
            let q_val = (val / scale).round() as i8;
            let clamped = q_val.clamp(-7, 7);
            quantized.push((clamped + 8) as u8);
        }
        
        Ok((quantized, scale, 0))
    }

    pub fn dequantize_base(&self) -> Result<Array2<f32>> {
        let (rows, cols) = self.quantized_base.shape;
        let mut dequantized = Array2::<f32>::zeros((rows, cols));
        
        let group_size = 128;
        let num_groups = (cols + group_size - 1) / group_size;
        
        let mut data_offset = 0;
        for row_idx in 0..rows {
            for group_idx in 0..num_groups {
                let start = group_idx * group_size;
                let end = (start + group_size).min(cols);
                let group_len = end - start;
                
                let scale_idx = row_idx * num_groups + group_idx;
                let scale = self.quantized_base.scales[scale_idx];
                
                for i in 0..group_len {
                    let q_val = self.quantized_base.quantized_data[data_offset + i];
                    let dq_val = match self.quantized_base.quant_type {
                        QuantType::NF4 => Self::dequantize_nf4(q_val, scale),
                        QuantType::FP4 => Self::dequantize_fp4(q_val, scale),
                        QuantType::INT4 => Self::dequantize_int4(q_val, scale),
                    };
                    dequantized[[row_idx, start + i]] = dq_val;
                }
                
                data_offset += group_len;
            }
        }
        
        Ok(dequantized)
    }

    fn dequantize_nf4(q_val: u8, scale: f32) -> f32 {
        let nf4_values = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        
        let idx = (q_val as usize).min(15);
        nf4_values[idx] * scale
    }

    fn dequantize_fp4(q_val: u8, scale: f32) -> f32 {
        let val = (q_val as i8) - 8;
        val as f32 * scale
    }

    fn dequantize_int4(q_val: u8, scale: f32) -> f32 {
        let val = (q_val as i8) - 8;
        val as f32 * scale
    }

    pub fn forward(&self, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        let base_dequantized = self.dequantize_base()?;
        let base_output = input.dot(&base_dequantized.t());
        
        let lora_output = self.lora_layer.forward(input)?;
        
        Ok(base_output + lora_output)
    }

    pub fn memory_footprint(&self) -> usize {
        let base_size = self.quantized_base.quantized_data.len();
        let scales_size = self.quantized_base.scales.len() * 4;
        let lora_size = self.lora_layer.trainable_parameters() * 4;
        
        base_size + scales_size + lora_size
    }

    pub fn compression_ratio(&self) -> f32 {
        let (rows, cols) = self.quantized_base.shape;
        let original_size = rows * cols * 4;
        let compressed_size = self.memory_footprint();
        original_size as f32 / compressed_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_qlora_config_default() {
        let config = QLoraConfig::default();
        assert_eq!(config.quant_bits, 4);
        assert!(config.double_quant);
    }

    #[test]
    fn test_quantize_nf4() {
        let values = Array::from_vec(vec![0.5, -0.3, 0.8, -0.1]);
        let result = QLoraLayer::quantize_nf4(&values.view());
        assert!(result.is_ok());
        
        let (quantized, scale, _) = result.unwrap();
        assert_eq!(quantized.len(), 4);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_quantize_int4() {
        let values = Array::from_vec(vec![0.5, -0.3, 0.8, -0.1]);
        let result = QLoraLayer::quantize_int4(&values.view());
        assert!(result.is_ok());
        
        let (quantized, scale, _) = result.unwrap();
        assert_eq!(quantized.len(), 4);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_qlora_layer_creation() {
        let config = QLoraConfig::default();
        let base_weight = Array2::<f32>::ones((10, 20));
        
        let result = QLoraLayer::new(base_weight.view(), 20, 10, config);
        assert!(result.is_ok());
        
        let layer = result.unwrap();
        assert_eq!(layer.quantized_base.shape, (10, 20));
    }

    #[test]
    fn test_dequantize_base() {
        let config = QLoraConfig {
            quant_bits: 4,
            quant_type: QuantType::INT4,
            ..Default::default()
        };
        let base_weight = Array2::<f32>::from_shape_vec((4, 8), vec![0.1; 32]).unwrap();
        
        let layer = QLoraLayer::new(base_weight.view(), 8, 4, config).unwrap();
        let dequantized = layer.dequantize_base();
        
        assert!(dequantized.is_ok());
        assert_eq!(dequantized.unwrap().shape(), &[4, 8]);
    }

    #[test]
    fn test_memory_footprint() {
        let config = QLoraConfig::default();
        let base_weight = Array2::<f32>::ones((10, 20));
        
        let layer = QLoraLayer::new(base_weight.view(), 20, 10, config).unwrap();
        let footprint = layer.memory_footprint();
        
        assert!(footprint > 0);
        assert!(footprint < 10 * 20 * 4);
    }

    #[test]
    fn test_compression_ratio() {
        let config = QLoraConfig::default();
        let base_weight = Array2::<f32>::ones((100, 200));
        
        let layer = QLoraLayer::new(base_weight.view(), 200, 100, config).unwrap();
        let ratio = layer.compression_ratio();
        
        assert!(ratio > 1.0);
    }
}
