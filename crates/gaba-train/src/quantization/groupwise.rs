use ndarray::{Array2, ArrayView2};
use anyhow::Result;

pub struct GroupwiseQuantizer {
    num_bits: u8,
    group_size: usize,
}

pub struct GroupwiseConfig {
    pub num_bits: u8,
    pub group_size: usize,
}

impl Default for GroupwiseConfig {
    fn default() -> Self {
        Self {
            num_bits: 4,
            group_size: 128,
        }
    }
}

pub struct QuantizedGroups {
    pub quantized: Array2<i8>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub num_groups: usize,
}

impl GroupwiseQuantizer {
    pub fn new(config: GroupwiseConfig) -> Self {
        Self {
            num_bits: config.num_bits,
            group_size: config.group_size,
        }
    }

    pub fn quantize(&self, tensor: ArrayView2<f32>) -> Result<QuantizedGroups> {
        let (rows, cols) = tensor.dim();
        let num_groups = (cols + self.group_size - 1) / self.group_size;
        
        let mut quantized = Array2::<i8>::zeros((rows, cols));
        let mut scales = Vec::with_capacity(rows * num_groups);
        
        for row_idx in 0..rows {
            for group_idx in 0..num_groups {
                let start = group_idx * self.group_size;
                let end = (start + self.group_size).min(cols);
                let group_len = end - start;
                
                let group = tensor.slice(ndarray::s![row_idx, start..end]);
                
                let max_val = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let q_max = (1 << (self.num_bits - 1)) - 1;
                let scale = if max_val > 0.0 {
                    max_val / q_max as f32
                } else {
                    1.0
                };
                
                scales.push(scale);
                
                for i in 0..group_len {
                    let val = tensor[[row_idx, start + i]];
                    quantized[[row_idx, start + i]] = self.quantize_value(val, scale);
                }
            }
        }
        
        Ok(QuantizedGroups {
            quantized,
            scales,
            group_size: self.group_size,
            num_groups,
        })
    }

    pub fn dequantize(&self, groups: &QuantizedGroups) -> Array2<f32> {
        let (rows, cols) = groups.quantized.dim();
        let mut dequantized = Array2::<f32>::zeros((rows, cols));
        
        for row_idx in 0..rows {
            for group_idx in 0..groups.num_groups {
                let start = group_idx * self.group_size;
                let end = (start + self.group_size).min(cols);
                let group_len = end - start;
                
                let scale_idx = row_idx * groups.num_groups + group_idx;
                let scale = groups.scales[scale_idx];
                
                for i in 0..group_len {
                    let q_val = groups.quantized[[row_idx, start + i]];
                    dequantized[[row_idx, start + i]] = q_val as f32 * scale;
                }
            }
        }
        
        dequantized
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

    pub fn pack_int4(&self, tensor: &Array2<i8>) -> Vec<u8> {
        let (rows, cols) = tensor.dim();
        let packed_cols = (cols + 1) / 2;
        let mut packed = Vec::with_capacity(rows * packed_cols);
        
        for row_idx in 0..rows {
            for col_idx in (0..cols).step_by(2) {
                let low = tensor[[row_idx, col_idx]] & 0x0F;
                let high = if col_idx + 1 < cols {
                    (tensor[[row_idx, col_idx + 1]] & 0x0F) << 4
                } else {
                    0
                };
                packed.push((low | high) as u8);
            }
        }
        
        packed
    }

    pub fn unpack_int4(&self, packed: &[u8], rows: usize, cols: usize) -> Array2<i8> {
        let mut unpacked = Array2::<i8>::zeros((rows, cols));
        
        let packed_cols = (cols + 1) / 2;
        for row_idx in 0..rows {
            for col_idx in (0..cols).step_by(2) {
                let packed_idx = row_idx * packed_cols + col_idx / 2;
                let byte = packed[packed_idx];
                
                let low = (byte & 0x0F) as i8;
                unpacked[[row_idx, col_idx]] = if low > 7 { low - 16 } else { low };
                
                if col_idx + 1 < cols {
                    let high = ((byte >> 4) & 0x0F) as i8;
                    unpacked[[row_idx, col_idx + 1]] = if high > 7 { high - 16 } else { high };
                }
            }
        }
        
        unpacked
    }

    pub fn compression_ratio(&self, original_bits: u8) -> f32 {
        original_bits as f32 / self.num_bits as f32
    }

    pub fn memory_savings(&self, original_size: usize) -> usize {
        original_size - (original_size * self.num_bits as usize / 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_groupwise_quantizer_creation() {
        let config = GroupwiseConfig::default();
        let quantizer = GroupwiseQuantizer::new(config);
        assert_eq!(quantizer.num_bits, 4);
        assert_eq!(quantizer.group_size, 128);
    }

    #[test]
    fn test_quantize_small_tensor() {
        let config = GroupwiseConfig {
            num_bits: 4,
            group_size: 4,
        };
        let quantizer = GroupwiseQuantizer::new(config);
        
        let tensor = Array::from_shape_vec((2, 8), vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
        ]).unwrap();
        
        let result = quantizer.quantize(tensor.view());
        assert!(result.is_ok());
        
        let groups = result.unwrap();
        assert_eq!(groups.quantized.shape(), &[2, 8]);
        assert_eq!(groups.num_groups, 2);
        assert_eq!(groups.scales.len(), 4);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let config = GroupwiseConfig {
            num_bits: 8,
            group_size: 4,
        };
        let quantizer = GroupwiseQuantizer::new(config);
        
        let tensor = Array::from_shape_vec((2, 4), vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ]).unwrap();
        
        let groups = quantizer.quantize(tensor.view()).unwrap();
        let dequantized = quantizer.dequantize(&groups);
        
        assert_eq!(dequantized.shape(), tensor.shape());
        
        for i in 0..2 {
            for j in 0..4 {
                let diff = (tensor[[i, j]] - dequantized[[i, j]]).abs();
                assert!(diff < 0.1, "Difference too large: {}", diff);
            }
        }
    }

    #[test]
    fn test_pack_unpack_int4() {
        let config = GroupwiseConfig {
            num_bits: 4,
            group_size: 4,
        };
        let quantizer = GroupwiseQuantizer::new(config);
        
        let tensor = Array::from_shape_vec((2, 4), vec![
            1i8, 2, 3, 4,
            5, 6, 7, -1,
        ]).unwrap();
        
        let packed = quantizer.pack_int4(&tensor);
        assert_eq!(packed.len(), 4);
        
        let unpacked = quantizer.unpack_int4(&packed, 2, 4);
        assert_eq!(unpacked.shape(), tensor.shape());
        
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(unpacked[[i, j]], tensor[[i, j]]);
            }
        }
    }

    #[test]
    fn test_compression_ratio() {
        let config = GroupwiseConfig {
            num_bits: 4,
            group_size: 128,
        };
        let quantizer = GroupwiseQuantizer::new(config);
        assert_eq!(quantizer.compression_ratio(32), 8.0);
    }

    #[test]
    fn test_memory_savings() {
        let config = GroupwiseConfig {
            num_bits: 4,
            group_size: 128,
        };
        let quantizer = GroupwiseQuantizer::new(config);
        let original_size = 1000;
        let savings = quantizer.memory_savings(original_size);
        assert_eq!(savings, 875);
    }
}
