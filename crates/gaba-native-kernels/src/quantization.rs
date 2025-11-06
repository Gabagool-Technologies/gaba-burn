use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i8,
}

impl QuantizationParams {
    pub fn from_range(min: f32, max: f32) -> Self {
        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = if scale > 0.0 {
            (-min / scale).round().clamp(-128.0, 127.0) as i8
        } else {
            0
        };
        Self { scale, zero_point }
    }

    pub fn quantize(&self, value: f32) -> i8 {
        let q = (value / self.scale).round() as i32;
        q.clamp(-128, 127) as i8
    }

    pub fn dequantize(&self, value: i8) -> f32 {
        value as f32 * self.scale
    }
}

pub fn quantize_tensor(input: &[f32], output: &mut [i8]) -> QuantizationParams {
    let min = input.iter().copied().fold(f32::INFINITY, f32::min);
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let params = QuantizationParams::from_range(min, max);

    for (i, &val) in input.iter().enumerate() {
        output[i] = params.quantize(val);
    }

    params
}

pub fn dequantize_tensor(input: &[i8], output: &mut [f32], params: &QuantizationParams) {
    for (i, &val) in input.iter().enumerate() {
        output[i] = params.dequantize(val);
    }
}

pub fn gemm_i8(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum: i32 = 0;
            for p in 0..k {
                sum += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = sum;
        }
    }

    start.elapsed()
}

pub fn gemm_i8_vectorized(
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum: i32 = 0;

            let mut p = 0;
            while p + 4 <= k {
                sum += a[i * k + p] as i32 * b[p * n + j] as i32;
                sum += a[i * k + p + 1] as i32 * b[(p + 1) * n + j] as i32;
                sum += a[i * k + p + 2] as i32 * b[(p + 2) * n + j] as i32;
                sum += a[i * k + p + 3] as i32 * b[(p + 3) * n + j] as i32;
                p += 4;
            }

            while p < k {
                sum += a[i * k + p] as i32 * b[p * n + j] as i32;
                p += 1;
            }

            c[i * n + j] = sum;
        }
    }

    start.elapsed()
}

pub fn gemm_quantized(
    a_f32: &[f32],
    b_f32: &[f32],
    c_f32: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    let start = std::time::Instant::now();

    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; k * n];
    let mut c_i32 = vec![0i32; m * n];

    let params_a = quantize_tensor(a_f32, &mut a_i8);
    let params_b = quantize_tensor(b_f32, &mut b_i8);

    gemm_i8(&a_i8, &b_i8, &mut c_i32, m, n, k);

    let scale_out = params_a.scale * params_b.scale;

    for i in 0..m * n {
        c_f32[i] = c_i32[i] as f32 * scale_out;
    }

    start.elapsed()
}

#[cfg(feature = "accelerate")]
pub fn gemm_fp16_accelerate(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    let start = std::time::Instant::now();

    let mut a_f16 = vec![0u16; m * k];
    let mut b_f16 = vec![0u16; k * n];
    let mut c_f16 = vec![0u16; m * n];

    for i in 0..m * k {
        a_f16[i] = f32_to_f16(a[i]);
    }
    for i in 0..k * n {
        b_f16[i] = f32_to_f16(b[i]);
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0u16;
            for p in 0..k {
                let a_val = a_f16[i * k + p];
                let b_val = b_f16[p * n + j];
                sum = f16_add(sum, f16_mul(a_val, b_val));
            }
            c_f16[i * n + j] = sum;
        }
    }

    for i in 0..m * n {
        c[i] = f16_to_f32(c_f16[i]);
    }

    start.elapsed()
}

#[allow(dead_code)]
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7fffff;

    if exp == 0 {
        return sign as u16;
    }

    let exp_f16 = exp - 127 + 15;

    if exp_f16 >= 31 {
        return (sign | 0x7c00) as u16;
    }

    if exp_f16 <= 0 {
        return sign as u16;
    }

    let mantissa_f16 = mantissa >> 13;
    (sign | ((exp_f16 as u32) << 10) | mantissa_f16) as u16
}

#[allow(dead_code)]
fn f16_to_f32(value: u16) -> f32 {
    let sign = (value & 0x8000) as u32;
    let exp = ((value >> 10) & 0x1f) as i32;
    let mantissa = (value & 0x3ff) as u32;

    if exp == 0 {
        return if mantissa == 0 {
            f32::from_bits(sign << 16)
        } else {
            f32::from_bits((sign << 16) | ((mantissa << 13) & 0x7fffff))
        };
    }

    if exp == 31 {
        return if mantissa == 0 {
            if sign != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    let exp_f32 = exp - 15 + 127;
    let bits = (sign << 16) | ((exp_f32 as u32) << 23) | (mantissa << 13);
    f32::from_bits(bits)
}

#[allow(dead_code)]
fn f16_mul(a: u16, b: u16) -> u16 {
    f32_to_f16(f16_to_f32(a) * f16_to_f32(b))
}

#[allow(dead_code)]
fn f16_add(a: u16, b: u16) -> u16 {
    f32_to_f16(f16_to_f32(a) + f16_to_f32(b))
}

#[cfg(target_os = "macos")]
#[cfg(feature = "metal")]
pub fn gemm_i8_metal(
    device: &metal::Device,
    command_queue: &metal::CommandQueue,
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Duration, String> {
    let start = std::time::Instant::now();

    let shader_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_i8_kernel(
    device const char* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device int* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    int sum = 0;
    for (uint k = 0; k < K; k++) {
        sum += int(A[row * K + k]) * int(B[k * N + col]);
    }
    
    C[row * N + col] = sum;
}
"#;

    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(shader_source, &options)
        .map_err(|e| format!("Shader compile failed: {}", e))?;

    let kernel = library
        .get_function("gemm_i8_kernel", None)
        .map_err(|e| format!("Kernel not found: {}", e))?;

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Pipeline failed: {}", e))?;

    let a_size = (m * k * std::mem::size_of::<i8>()) as u64;
    let b_size = (k * n * std::mem::size_of::<i8>()) as u64;
    let c_size = (m * n * std::mem::size_of::<i32>()) as u64;

    let a_buffer = device.new_buffer_with_data(
        a.as_ptr() as *const _,
        a_size,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.new_buffer_with_data(
        b.as_ptr() as *const _,
        b_size,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let c_buffer = device.new_buffer(c_size, metal::MTLResourceOptions::StorageModeShared);

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buffer), 0);
    encoder.set_buffer(1, Some(&b_buffer), 0);
    encoder.set_buffer(2, Some(&c_buffer), 0);

    let dims = [m as u32, n as u32, k as u32];
    encoder.set_bytes(
        3,
        std::mem::size_of_val(&dims) as u64,
        dims.as_ptr() as *const _,
    );

    let threadgroup_size = metal::MTLSize::new(16, 16, 1);
    let threadgroups = metal::MTLSize::new((n as u64 + 15) / 16, (m as u64 + 15) / 16, 1);

    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let ptr = c_buffer.contents() as *const i32;
        std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), m * n);
    }

    Ok(start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_roundtrip() {
        let input = vec![10.0, 20.0, 30.0, -10.0, -20.0];
        let mut quantized = vec![0i8; 5];
        let mut output = vec![0.0f32; 5];

        let params = quantize_tensor(&input, &mut quantized);
        dequantize_tensor(&quantized, &mut output, &params);

        for (i, &val) in input.iter().enumerate() {
            let error = (output[i] - val).abs();
            let relative_error = error / val.abs().max(1.0);
            assert!(
                relative_error < 0.25,
                "Value {} vs {} (error: {}, scale: {})",
                output[i],
                val,
                error,
                params.scale
            );
        }
    }

    #[test]
    fn test_gemm_i8() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1i8, 2, 3, 4];
        let mut c = vec![0i32; 4];

        gemm_i8(&a, &b, &mut c, 2, 2, 2);

        assert_eq!(c[0], 7);
        assert_eq!(c[1], 10);
        assert_eq!(c[2], 15);
        assert_eq!(c[3], 22);
    }

    #[test]
    fn test_fp16_conversion() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0];

        for &val in &values {
            let f16 = f32_to_f16(val);
            let back = f16_to_f32(f16);
            assert!((back - val).abs() < 0.01 * val.abs().max(1.0));
        }
    }
}
