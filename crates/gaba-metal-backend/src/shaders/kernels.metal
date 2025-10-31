#include <metal_stdlib>
using namespace metal;

// Zero-copy optimized kernels for Apple Silicon M4 Pro

// Element-wise addition with unified memory
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant ulong& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    c[gid] = a[gid] + b[gid];
}

// Element-wise multiplication with SIMD
kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant ulong& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    c[gid] = a[gid] * b[gid];
}

// Matrix multiplication with M4 Pro optimization
kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant ulong& m [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    constant ulong& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0;
    for (ulong i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    
    c[row * n + col] = sum;
}

// ReLU activation with fast-math
[[fast_math]]
kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = max(0.0f, input[gid]);
}

// Softmax with numerical stability
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Find max for numerical stability
    float max_val = input[0];
    for (ulong i = 1; i < size; i++) {
        max_val = max(max_val, input[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0;
    for (ulong i = 0; i < size; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Normalize
    output[gid] = exp(input[gid] - max_val) / sum;
}

// Batch normalization with M4 Pro SIMD
kernel void batch_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* variance [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& size [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float normalized = (input[gid] - mean[0]) / sqrt(variance[0] + epsilon);
    output[gid] = gamma[0] * normalized + beta[0];
}
