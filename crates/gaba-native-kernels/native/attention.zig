const std = @import("std");
const math = std.math;

const SIMD_WIDTH: usize = 8;

export fn attention_forward(
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    output: [*]f32,
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
) callconv(.c) void {
    const d_k = d_model / num_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_k)));
    
    @memset(output[0..(seq_len * d_model)], 0.0);
    
    var h: usize = 0;
    while (h < num_heads) : (h += 1) {
        const head_offset = h * d_k;
        
        var scores_buf: [512 * 512]f32 = undefined;
        const scores = scores_buf[0..(seq_len * seq_len)];
        
        // Compute Q @ K^T with scaling
        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            var j: usize = 0;
            while (j < seq_len) : (j += 1) {
                var dot: f32 = 0.0;
                
                var k: usize = 0;
                const Vec = @Vector(SIMD_WIDTH, f32);
                while (k + SIMD_WIDTH <= d_k) : (k += SIMD_WIDTH) {
                    var q_vec: Vec = undefined;
                    var k_vec: Vec = undefined;
                    
                    inline for (0..SIMD_WIDTH) |lane| {
                        q_vec[lane] = query[i * d_model + head_offset + k + lane];
                        k_vec[lane] = key[j * d_model + head_offset + k + lane];
                    }
                    
                    dot += @reduce(.Add, q_vec * k_vec);
                }
                
                while (k < d_k) : (k += 1) {
                    dot += query[i * d_model + head_offset + k] *
                           key[j * d_model + head_offset + k];
                }
                
                scores[i * seq_len + j] = dot * scale;
            }
        }
        
        // Softmax per row
        i = 0;
        while (i < seq_len) : (i += 1) {
            const row_offset = i * seq_len;
            
            var max_score = scores[row_offset];
            var j: usize = 1;
            while (j < seq_len) : (j += 1) {
                if (scores[row_offset + j] > max_score) {
                    max_score = scores[row_offset + j];
                }
            }
            
            var sum_exp: f32 = 0.0;
            j = 0;
            while (j < seq_len) : (j += 1) {
                scores[row_offset + j] = @exp(scores[row_offset + j] - max_score);
                sum_exp += scores[row_offset + j];
            }
            
            j = 0;
            while (j < seq_len) : (j += 1) {
                scores[row_offset + j] /= sum_exp;
            }
        }
        
        // Attention @ Value
        i = 0;
        while (i < seq_len) : (i += 1) {
            var k: usize = 0;
            while (k < d_k) : (k += 1) {
                var sum: f32 = 0.0;
                var j: usize = 0;
                while (j < seq_len) : (j += 1) {
                    sum += scores[i * seq_len + j] *
                           value[j * d_model + head_offset + k];
                }
                output[i * d_model + head_offset + k] = sum;
            }
        }
    }
}
