use std::time::Duration;

#[derive(Debug, Clone)]
pub struct PerChannelQuantizationParams {
    pub scales: Vec<f32>,
    pub zero_points: Vec<i8>,
    pub num_channels: usize,
}

impl PerChannelQuantizationParams {
    pub fn from_tensor(tensor: &[f32], num_channels: usize) -> Self {
        let channel_size = tensor.len() / num_channels;
        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);
        
        for ch in 0..num_channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let channel_data = &tensor[start..end];
            
            let min = channel_data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            let range = max - min;
            let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
            let zero_point = if scale > 0.0 {
                (-min / scale).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
            
            scales.push(scale);
            zero_points.push(zero_point);
        }
        
        Self {
            scales,
            zero_points,
            num_channels,
        }
    }
    
    pub fn quantize_channel(&self, value: f32, channel: usize) -> i8 {
        let scale = self.scales[channel];
        let q = (value / scale).round() as i32;
        q.clamp(-128, 127) as i8
    }
    
    pub fn dequantize_channel(&self, value: i8, channel: usize) -> f32 {
        value as f32 * self.scales[channel]
    }
}

pub fn quantize_tensor_per_channel(
    input: &[f32],
    output: &mut [i8],
    num_channels: usize,
) -> PerChannelQuantizationParams {
    let params = PerChannelQuantizationParams::from_tensor(input, num_channels);
    let channel_size = input.len() / num_channels;
    
    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        
        for (i, &val) in input[start..end].iter().enumerate() {
            output[start + i] = params.quantize_channel(val, ch);
        }
    }
    
    params
}

pub fn dequantize_tensor_per_channel(
    input: &[i8],
    output: &mut [f32],
    params: &PerChannelQuantizationParams,
) {
    let channel_size = input.len() / params.num_channels;
    
    for ch in 0..params.num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        
        for (i, &val) in input[start..end].iter().enumerate() {
            output[start + i] = params.dequantize_channel(val, ch);
        }
    }
}

pub fn gemm_per_channel_quantized(
    a_f32: &[f32],
    b_f32: &[f32],
    c_f32: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    a_channels: usize,
    b_channels: usize,
) -> Duration {
    use crate::quantization::gemm_i8;
    
    let start = std::time::Instant::now();
    
    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; k * n];
    let mut c_i32 = vec![0i32; m * n];
    
    let params_a = quantize_tensor_per_channel(a_f32, &mut a_i8, a_channels);
    let params_b = quantize_tensor_per_channel(b_f32, &mut b_i8, b_channels);
    
    gemm_i8(&a_i8, &b_i8, &mut c_i32, m, n, k);
    
    for i in 0..m {
        for j in 0..n {
            let a_ch = i % a_channels;
            let b_ch = j % b_channels;
            let scale_out = params_a.scales[a_ch] * params_b.scales[b_ch];
            c_f32[i * n + j] = c_i32[i * n + j] as f32 * scale_out;
        }
    }
    
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_per_channel_quantization() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            10.0, 20.0, 30.0, 40.0,
        ];
        let mut quantized = vec![0i8; 8];
        let mut output = vec![0.0f32; 8];
        
        let params = quantize_tensor_per_channel(&input, &mut quantized, 2);
        dequantize_tensor_per_channel(&quantized, &mut output, &params);
        
        assert_eq!(params.num_channels, 2);
        assert_eq!(params.scales.len(), 2);
        
        for (i, &val) in input.iter().enumerate() {
            let error = (output[i] - val).abs();
            let relative_error = error / val.abs().max(1.0);
            assert!(relative_error < 0.7, "Value {} vs {} (error: {}, relative: {})", output[i], val, error, relative_error);
        }
    }
    
    #[test]
    fn test_gemm_per_channel() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        
        gemm_per_channel_quantized(&a, &b, &mut c, 2, 2, 2, 2, 2);
        
        for &val in &c {
            assert!(val > 0.0);
        }
    }
}
