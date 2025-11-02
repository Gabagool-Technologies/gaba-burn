const std = @import("std");
const math = std.math;

const SIMD_WIDTH: usize = 8;

export fn softmax_f32(
    input: [*]const f32,
    output: [*]f32,
    size: usize,
) callconv(.c) void {
    const Vec = @Vector(SIMD_WIDTH, f32);
    
    // Pass 1: Find max
    var max_val: f32 = input[0];
    var i: usize = 1;
    
    if (size <= 256) {
        while (i < size) : (i += 1) {
            if (input[i] > max_val) max_val = input[i];
        }
    } else {
        var max_vec: Vec = @splat(input[0]);
        i = 0;
        while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
            var vec: Vec = undefined;
            inline for (0..SIMD_WIDTH) |lane| {
                vec[lane] = input[i + lane];
            }
            max_vec = @max(max_vec, vec);
        }
        max_val = @reduce(.Max, max_vec);
        
        while (i < size) : (i += 1) {
            if (input[i] > max_val) max_val = input[i];
        }
    }
    
    // Pass 2: Exp and sum
    var sum: f32 = 0.0;
    const max_vec: Vec = @splat(max_val);
    
    i = 0;
    while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
        var input_vec: Vec = undefined;
        inline for (0..SIMD_WIDTH) |lane| {
            input_vec[lane] = input[i + lane];
        }
        
        const exp_vec = @exp(input_vec - max_vec);
        inline for (0..SIMD_WIDTH) |lane| {
            output[i + lane] = exp_vec[lane];
        }
        sum += @reduce(.Add, exp_vec);
    }
    
    while (i < size) : (i += 1) {
        const val = @exp(input[i] - max_val);
        output[i] = val;
        sum += val;
    }
    
    // Pass 3: Normalize
    const sum_vec: Vec = @splat(sum);
    i = 0;
    while (i + SIMD_WIDTH <= size) : (i += SIMD_WIDTH) {
        var out_vec: Vec = undefined;
        inline for (0..SIMD_WIDTH) |lane| {
            out_vec[lane] = output[i + lane];
        }
        out_vec = out_vec / sum_vec;
        inline for (0..SIMD_WIDTH) |lane| {
            output[i + lane] = out_vec[lane];
        }
    }
    
    while (i < size) : (i += 1) {
        output[i] /= sum;
    }
}
