//! Minimal native-kernel crate with a Rust GEMM fallback and an FFI shim for a Zig-built GEMM.
//!
//! The Zig kernel is feature-gated behind the `zig` feature. By default the pure-Rust
//! implementation is used so CI and macOS builds don't require Zig or native toolchains.
//!
//! Also includes route optimization kernels for Famiglia Routes application.

pub mod route_optimizer;
pub mod ml_route_optimizer;
pub mod accelerate;
pub mod fusion;
pub mod quantization;
pub mod amx_int8;
pub mod per_channel_quant;
pub mod conv2d;
pub mod lstm;
pub mod batch_norm;
pub mod dropout;
pub mod transformer;

#[cfg(feature = "metal")]
pub mod metal_gpu;

pub use route_optimizer::*;
pub use ml_route_optimizer::MLRouteOptimizer;
pub use accelerate::{gemm_accelerate, detect_amx};
pub use fusion::{Activation, gemm_activation_fused, gemm_relu_fused, gemm_batchnorm_fused};
pub use quantization::{QuantizationParams, quantize_tensor, dequantize_tensor, gemm_quantized, gemm_i8};
pub use amx_int8::{gemm_i8_amx, gemm_quantized_amx};
pub use per_channel_quant::{PerChannelQuantizationParams, quantize_tensor_per_channel, gemm_per_channel_quantized};
pub use conv2d::{Conv2DParams, conv2d_naive, conv2d_im2col, conv2d_relu_fused, conv2d_accelerate};
pub use lstm::{LSTMParams, LSTMState, lstm_cell_forward, lstm_forward};
pub use batch_norm::{BatchNorm1dParams, BatchNorm2dParams, batch_norm_1d_forward, batch_norm_2d_forward, layer_norm_forward};
pub use dropout::{DropoutParams, dropout_forward, dropout_backward, spatial_dropout_2d_forward, alpha_dropout_forward};
pub use transformer::{MultiHeadAttentionParams, TransformerBlockParams, multi_head_attention_forward, transformer_block_forward, positional_encoding};

#[cfg(feature = "metal")]
pub use metal_gpu::MetalGPUExecutor;

#[cfg(feature = "zig")]
extern "C" {
    // Ultra-optimized GEMM with cache blocking and SIMD
    fn gemm_f32_ultra(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
    
    // Original GEMM (fallback)
    fn gemm_f32(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
    
    // Quantized u8 GEMM fast path
    fn gemm_q8_to_i64(a: *const u8, b: *const u8, c: *mut i64, m: usize, n: usize, k: usize);
    
    // ML Kernels
    fn conv2d_3x3_stride1(input: *const f32, kernel: *const f32, output: *mut f32, 
                          in_h: usize, in_w: usize, in_c: usize, out_c: usize);
    fn conv2d_general(input: *const f32, kernel: *const f32, output: *mut f32,
                      in_h: usize, in_w: usize, in_c: usize,
                      kernel_h: usize, kernel_w: usize, out_c: usize, stride: usize);
    fn softmax_f32(input: *const f32, output: *mut f32, size: usize);
    fn layernorm_f32(input: *const f32, gamma: *const f32, beta: *const f32, 
                     output: *mut f32, size: usize, eps: f32);
    fn attention_forward(query: *const f32, key: *const f32, value: *const f32, 
                        output: *mut f32, seq_len: usize, d_model: usize, num_heads: usize);
}

/// Pure Rust reference quantized GEMM: dequantize -> gemm_rust -> requantize
pub fn gemm_q8_rust(a: &[u8], b: &[u8], c: &mut [u8], m: usize, n: usize, k: usize, scale_a: f32, scale_b: f32, scale_out: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // allocate temporary f32 buffers
    let mut a_f = vec![0f32; m * k];
    let mut b_f = vec![0f32; k * n];
    let mut c_f = vec![0f32; m * n];

    for i in 0..(m * k) {
        a_f[i] = (a[i] as f32) * scale_a;
    }
    for i in 0..(k * n) {
        b_f[i] = (b[i] as f32) * scale_b;
    }

    gemm_rust(&a_f, &b_f, &mut c_f, m, n, k);

    for i in 0..(m * n) {
        let q = (c_f[i] / scale_out).round();
        let q = if q < 0.0 { 0.0 } else if q > 255.0 { 255.0 } else { q };
        c[i] = q as u8;
    }
}

/// Pure Rust reference GEMM (row-major): C = A * B
pub fn gemm_rust(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Multi-threaded GEMM using rayon
pub fn gemm_rust_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    use std::sync::Mutex;
    
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let c_mutex = Mutex::new(c);
    let block_size = 64;
    let num_blocks = (m + block_size - 1) / block_size;
    
    (0..num_blocks).into_par_iter().for_each(|block_idx| {
        let i_start = block_idx * block_size;
        let i_end = (i_start + block_size).min(m);
        
        let mut local_results = vec![(0usize, 0usize, 0f32); (i_end - i_start) * n];
        let mut idx = 0;
        
        for i in i_start..i_end {
            for j in 0..n {
                let mut sum = 0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                local_results[idx] = (i, j, sum);
                idx += 1;
            }
        }
        
        let mut c_guard = c_mutex.lock().unwrap();
        for (i, j, val) in local_results {
            c_guard[i * n + j] = val;
        }
    });
}

/// Call the Zig/native kernel when the feature is enabled, otherwise fall back to Rust.
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(feature = "zig")]
    unsafe {
        gemm_f32_ultra(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k);
    }

    #[cfg(not(feature = "zig"))]
    {
        if m >= 256 {
            gemm_rust_parallel(a, b, c, m, n, k);
        } else {
            gemm_rust(a, b, c, m, n, k);
        }
    }
}

/// Call quantized kernel when available; otherwise fallback to Rust dequantize path.
pub fn gemm_q8(a: &[u8], b: &[u8], c: &mut [u8], m: usize, n: usize, k: usize, scale_a: f32, scale_b: f32, scale_out: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(feature = "zig")]
    unsafe {
        // Call Zig to compute integer accumulators into a temporary i64 buffer, then apply scales and requantize in Rust.
        let mut tmp = vec![0i64; m * n];
        gemm_q8_to_i64(a.as_ptr(), b.as_ptr(), tmp.as_mut_ptr(), m, n, k);
        let scale = scale_a * scale_b;
        for i in 0..(m * n) {
            let sumf = (tmp[i] as f32) * scale;
            let q = (sumf / scale_out).round();
            let q = if q < 0.0 { 0.0 } else if q > 255.0 { 255.0 } else { q };
            c[i] = q as u8;
        }
    }

    #[cfg(not(feature = "zig"))]
    {
        gemm_q8_rust(a, b, c, m, n, k, scale_a, scale_b, scale_out);
    }
}

// ML Kernel Wrappers

pub fn conv2d_3x3(input: &[f32], kernel: &[f32], output: &mut [f32], 
                  in_h: usize, in_w: usize, in_c: usize, out_c: usize) {
    let out_h = in_h - 2;
    let out_w = in_w - 2;
    assert_eq!(input.len(), in_h * in_w * in_c);
    assert_eq!(kernel.len(), 3 * 3 * in_c * out_c);
    assert_eq!(output.len(), out_h * out_w * out_c);
    
    #[cfg(feature = "zig")]
    unsafe {
        conv2d_3x3_stride1(input.as_ptr(), kernel.as_ptr(), output.as_mut_ptr(),
                          in_h, in_w, in_c, out_c);
    }
    
    #[cfg(not(feature = "zig"))]
    {
        // Rust fallback
        for oh in 0..out_h {
            for ow in 0..out_w {
                for oc in 0..out_c {
                    let mut sum = 0.0;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            for ic in 0..in_c {
                                let ih = oh + kh;
                                let iw = ow + kw;
                                sum += input[ih * in_w * in_c + iw * in_c + ic] *
                                       kernel[kh * 3 * in_c * out_c + kw * in_c * out_c + ic * out_c + oc];
                            }
                        }
                    }
                    output[oh * out_w * out_c + ow * out_c + oc] = sum;
                }
            }
        }
    }
}

pub fn softmax(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let size = input.len();
    
    #[cfg(feature = "zig")]
    unsafe {
        softmax_f32(input.as_ptr(), output.as_mut_ptr(), size);
    }
    
    #[cfg(not(feature = "zig"))]
    {
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for i in 0..size {
            output[i] = (input[i] - max_val).exp();
            sum += output[i];
        }
        for i in 0..size {
            output[i] /= sum;
        }
    }
}

pub fn layernorm(input: &[f32], gamma: &[f32], beta: &[f32], output: &mut [f32], eps: f32) {
    let size = input.len();
    assert_eq!(gamma.len(), size);
    assert_eq!(beta.len(), size);
    assert_eq!(output.len(), size);
    
    #[cfg(feature = "zig")]
    unsafe {
        layernorm_f32(input.as_ptr(), gamma.as_ptr(), beta.as_ptr(), 
                     output.as_mut_ptr(), size, eps);
    }
    
    #[cfg(not(feature = "zig"))]
    {
        let mean = input.iter().sum::<f32>() / size as f32;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / size as f32;
        let std_dev = (variance + eps).sqrt();
        
        for i in 0..size {
            let normalized = (input[i] - mean) / std_dev;
            output[i] = gamma[i] * normalized + beta[i];
        }
    }
}

pub fn attention(query: &[f32], key: &[f32], value: &[f32], output: &mut [f32],
                seq_len: usize, d_model: usize, num_heads: usize) {
    assert_eq!(query.len(), seq_len * d_model);
    assert_eq!(key.len(), seq_len * d_model);
    assert_eq!(value.len(), seq_len * d_model);
    assert_eq!(output.len(), seq_len * d_model);
    
    #[cfg(feature = "zig")]
    unsafe {
        attention_forward(query.as_ptr(), key.as_ptr(), value.as_ptr(),
                         output.as_mut_ptr(), seq_len, d_model, num_heads);
    }
    
    #[cfg(not(feature = "zig"))]
    {
        // Simple Rust fallback (not optimized)
        let d_k = d_model / num_heads;
        let scale = 1.0 / (d_k as f32).sqrt();
        
        for h in 0..num_heads {
            let head_offset = h * d_k;
            let mut scores = vec![0.0f32; seq_len * seq_len];
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0;
                    for k in 0..d_k {
                        dot += query[i * d_model + head_offset + k] *
                               key[j * d_model + head_offset + k];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
            
            for i in 0..seq_len {
                let row_offset = i * seq_len;
                let max_score = scores[row_offset..row_offset + seq_len]
                    .iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0;
                for j in 0..seq_len {
                    scores[row_offset + j] = (scores[row_offset + j] - max_score).exp();
                    sum_exp += scores[row_offset + j];
                }
                for j in 0..seq_len {
                    scores[row_offset + j] /= sum_exp;
                }
            }
            
            for i in 0..seq_len {
                for k in 0..d_k {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        sum += scores[i * seq_len + j] * value[j * d_model + head_offset + k];
                    }
                    output[i * d_model + head_offset + k] = sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn rand_data(len: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            v.push(((i * 31 + 7) % 101) as f32 * 0.013f32);
        }
        v
    }

    #[test]
    fn gemm_small() {
        let m = 5;
        let n = 7;
        let k = 3;
        let a = rand_data(m * k);
        let b = rand_data(k * n);
        let mut c_ref = vec![0f32; m * n];
        let mut c_tgt = vec![0f32; m * n];

        gemm_rust(&a, &b, &mut c_ref, m, n, k);
        gemm(&a, &b, &mut c_tgt, m, n, k);

        for (r, t) in c_ref.iter().zip(c_tgt.iter()) {
            assert_abs_diff_eq!(r, t, epsilon = 1e-6f32);
        }
    }
}
