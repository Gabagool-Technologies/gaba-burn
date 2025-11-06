use rayon::prelude::*;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
}

pub fn gemm_relu_fused(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum.max(0.0);
        }
    }

    start.elapsed()
}

pub fn gemm_relu_vectorized(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = if sum > 0.0 { sum } else { 0.0 };
        }
    }

    start.elapsed()
}

pub fn gemm_activation_fused(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    activation: Activation,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }

            c[i * n + j] = match activation {
                Activation::ReLU => sum.max(0.0),
                Activation::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                Activation::Tanh => sum.tanh(),
                Activation::GELU => {
                    let sqrt_2_pi = 0.7978845608028654;
                    0.5 * sum * (1.0 + (sqrt_2_pi * (sum + 0.044715 * sum.powi(3))).tanh())
                }
            };
        }
    }

    start.elapsed()
}

pub fn gemm_batchnorm_fused(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    m: usize,
    n: usize,
    k: usize,
    epsilon: f32,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }

            let normalized = (sum - mean[j]) / (var[j] + epsilon).sqrt();
            c[i * n + j] = gamma[j] * normalized + beta[j];
        }
    }

    start.elapsed()
}

#[cfg(feature = "accelerate")]
pub fn gemm_relu_accelerate(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    use crate::accelerate::gemm_accelerate;

    let gemm_time = gemm_accelerate(a, b, c, m, n, k);

    let start = std::time::Instant::now();
    for val in c.iter_mut() {
        *val = val.max(0.0);
    }
    let relu_time = start.elapsed();

    gemm_time + relu_time
}

#[cfg(target_os = "macos")]
#[cfg(feature = "metal")]
pub fn gemm_relu_metal_fused(
    device: &metal::Device,
    command_queue: &metal::CommandQueue,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Duration, String> {
    let start = std::time::Instant::now();

    let shader_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_relu_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = max(sum, 0.0f);
}
"#;

    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(shader_source, &options)
        .map_err(|e| format!("Shader compile failed: {}", e))?;

    let kernel = library
        .get_function("gemm_relu_kernel", None)
        .map_err(|e| format!("Kernel not found: {}", e))?;

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Pipeline failed: {}", e))?;

    let a_size = (m * k * std::mem::size_of::<f32>()) as u64;
    let b_size = (k * n * std::mem::size_of::<f32>()) as u64;
    let c_size = (m * n * std::mem::size_of::<f32>()) as u64;

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
        let ptr = c_buffer.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), m * n);
    }

    Ok(start.elapsed())
}

pub fn gemm_activation_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    activation: Activation,
) -> Duration {
    let start = std::time::Instant::now();

    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }

            row[j] = match activation {
                Activation::ReLU => sum.max(0.0),
                Activation::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                Activation::Tanh => sum.tanh(),
                Activation::GELU => {
                    let sqrt_2_pi = 0.7978845608028654;
                    0.5 * sum * (1.0 + (sqrt_2_pi * (sum + 0.044715 * sum.powi(3))).tanh())
                }
            };
        }
    });

    start.elapsed()
}

pub fn gemm_relu_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    gemm_activation_parallel(a, b, c, m, n, k, Activation::ReLU)
}

pub fn gemm_batchnorm_relu_fused(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    m: usize,
    n: usize,
    k: usize,
    epsilon: f32,
) -> Duration {
    let start = std::time::Instant::now();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }

            let normalized = (sum - mean[j]) / (var[j] + epsilon).sqrt();
            let bn_output = gamma[j] * normalized + beta[j];
            c[i * n + j] = bn_output.max(0.0);
        }
    }

    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_relu_fused() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];

        gemm_relu_fused(&a, &b, &mut c, 2, 2, 2);

        assert!(
            c.iter().all(|&v| v >= 0.0),
            "All values should be >= 0 after ReLU"
        );
    }

    #[test]
    fn test_gemm_activation_fused() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];

        gemm_activation_fused(&a, &b, &mut c, 2, 2, 2, Activation::ReLU);
        assert!(c.iter().all(|&v| v >= 0.0));

        gemm_activation_fused(&a, &b, &mut c, 2, 2, 2, Activation::Sigmoid);
        assert!(c.iter().all(|&v| v > 0.0 && v < 1.0));
    }

    #[test]
    fn test_gemm_relu_parallel() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];

        gemm_relu_parallel(&a, &b, &mut c, 2, 2, 2);

        assert!(
            c.iter().all(|&v| v >= 0.0),
            "All values should be >= 0 after ReLU"
        );
    }

    #[test]
    fn test_gemm_batchnorm_relu_fused() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let gamma = vec![1.0; 2];
        let beta = vec![0.0; 2];
        let mean = vec![0.0; 2];
        let var = vec![1.0; 2];

        gemm_batchnorm_relu_fused(&a, &b, &mut c, &gamma, &beta, &mean, &var, 2, 2, 2, 1e-5);

        assert!(
            c.iter().all(|&v| v >= 0.0),
            "All values should be >= 0 after BN+ReLU"
        );
    }
}
