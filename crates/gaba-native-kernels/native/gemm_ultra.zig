const std = @import("std");

// Ultra-optimized GEMM for M4 Pro with cache blocking and SIMD
// Target: 2-5x faster than Rust baseline

const BLOCK_M: usize = 64;
const BLOCK_N: usize = 64;
const BLOCK_K: usize = 128;
const SIMD_WIDTH: usize = 8;

export fn gemm_f32_ultra(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) callconv(.c) void {
    @memset(c[0..(m * n)], 0.0);
    
    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK_M) {
        const i_end = @min(ii + BLOCK_M, m);
        
        var jj: usize = 0;
        while (jj < n) : (jj += BLOCK_N) {
            const j_end = @min(jj + BLOCK_N, n);
            
            var kk: usize = 0;
            while (kk < k) : (kk += BLOCK_K) {
                const k_end = @min(kk + BLOCK_K, k);
                
                microKernel(a, b, c, m, n, k, ii, i_end, jj, j_end, kk, k_end);
            }
        }
    }
}

inline fn microKernel(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    _: usize,
    n: usize,
    k: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
) void {
    var i = i_start;
    while (i < i_end) : (i += 1) {
        var j = j_start;
        
        while (j + SIMD_WIDTH <= j_end) : (j += SIMD_WIDTH) {
            var acc: @Vector(SIMD_WIDTH, f32) = @splat(0.0);
            
            var p = k_start;
            while (p + 4 <= k_end) : (p += 4) {
                inline for (0..4) |unroll| {
                    const a_val: @Vector(SIMD_WIDTH, f32) = @splat(a[i * k + p + unroll]);
                    
                    var b_vec: @Vector(SIMD_WIDTH, f32) = undefined;
                    inline for (0..SIMD_WIDTH) |lane| {
                        b_vec[lane] = b[(p + unroll) * n + j + lane];
                    }
                    
                    acc += a_val * b_vec;
                }
            }
            
            while (p < k_end) : (p += 1) {
                const a_val: @Vector(SIMD_WIDTH, f32) = @splat(a[i * k + p]);
                var b_vec: @Vector(SIMD_WIDTH, f32) = undefined;
                inline for (0..SIMD_WIDTH) |lane| {
                    b_vec[lane] = b[p * n + j + lane];
                }
                acc += a_val * b_vec;
            }
            
            inline for (0..SIMD_WIDTH) |lane| {
                c[i * n + j + lane] += acc[lane];
            }
        }
        
        while (j < j_end) : (j += 1) {
            var sum: f32 = 0.0;
            var p = k_start;
            while (p < k_end) : (p += 1) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}
