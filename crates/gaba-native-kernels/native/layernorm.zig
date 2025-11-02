const std = @import("std");
const math = std.math;

const SIMD_WIDTH: usize = 8;

export fn layernorm_f32(
    input: [*]const f32,
    gamma: [*]const f32,
    beta: [*]const f32,
    output: [*]f32,
    size: usize,
    eps: f32,
) callconv(.c) void {
    const Vec = @Vector(SIMD_WIDTH, f32);
    
    // Pass 1: Compute mean
    var sum: f32 = 0.0;
    var i: usize = 0;
    
    while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
        var vec: Vec = undefined;
        inline for (0..SIMD_WIDTH) |lane| {
            vec[lane] = input[i + lane];
        }
        sum += @reduce(.Add, vec);
    }
    while (i < size) : (i += 1) {
        sum += input[i];
    }
    const mean = sum / @as(f32, @floatFromInt(size));
    
    // Pass 2: Compute variance
    var var_sum: f32 = 0.0;
    const mean_vec: Vec = @splat(mean);
    i = 0;
    
    while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
        var vec: Vec = undefined;
        inline for (0..SIMD_WIDTH) |lane| {
            vec[lane] = input[i + lane];
        }
        const diff = vec - mean_vec;
        var_sum += @reduce(.Add, diff * diff);
    }
    while (i < size) : (i += 1) {
        const diff = input[i] - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(f32, @floatFromInt(size));
    const std_dev = @sqrt(variance + eps);
    
    // Pass 3: Normalize, scale, shift
    const std_vec: Vec = @splat(std_dev);
    i = 0;
    
    while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
        var input_vec: Vec = undefined;
        var gamma_vec: Vec = undefined;
        var beta_vec: Vec = undefined;
        
        inline for (0..SIMD_WIDTH) |lane| {
            input_vec[lane] = input[i + lane];
            gamma_vec[lane] = gamma[i + lane];
            beta_vec[lane] = beta[i + lane];
        }
        
        const normalized = (input_vec - mean_vec) / std_vec;
        const result = gamma_vec * normalized + beta_vec;
        
        inline for (0..SIMD_WIDTH) |lane| {
            output[i + lane] = result[lane];
        }
    }
    
    while (i < size) : (i += 1) {
        const normalized = (input[i] - mean) / std_dev;
        output[i] = gamma[i] * normalized + beta[i];
    }
}
