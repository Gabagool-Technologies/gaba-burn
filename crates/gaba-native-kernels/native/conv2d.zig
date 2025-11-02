const std = @import("std");

// Conv2D kernel with SIMD optimization
// Optimized for common CNN layer sizes

const SIMD_WIDTH: usize = 8;

export fn conv2d_3x3_stride1(
    input: [*]const f32,
    kernel: [*]const f32,
    output: [*]f32,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
) callconv(.c) void {
    const out_h = in_h - 2;
    const out_w = in_w - 2;
    
    @memset(output[0..(out_h * out_w * out_c)], 0.0);
    
    var oh: usize = 0;
    while (oh < out_h) : (oh += 1) {
        var ow: usize = 0;
        while (ow < out_w) : (ow += 1) {
            var oc: usize = 0;
            
            while (oc + SIMD_WIDTH <= out_c) : (oc += SIMD_WIDTH) {
                var acc: @Vector(SIMD_WIDTH, f32) = @splat(0.0);
                
                inline for (0..3) |kh| {
                    inline for (0..3) |kw| {
                        var ic: usize = 0;
                        while (ic < in_c) : (ic += 1) {
                            const ih = oh + kh;
                            const iw = ow + kw;
                            const input_val = input[ih * in_w * in_c + iw * in_c + ic];
                            
                            var kernel_vec: @Vector(SIMD_WIDTH, f32) = undefined;
                            inline for (0..SIMD_WIDTH) |lane| {
                                kernel_vec[lane] = kernel[kh * 3 * in_c * out_c + 
                                                          kw * in_c * out_c + 
                                                          ic * out_c + 
                                                          oc + lane];
                            }
                            
                            acc += @as(@Vector(SIMD_WIDTH, f32), @splat(input_val)) * kernel_vec;
                        }
                    }
                }
                
                inline for (0..SIMD_WIDTH) |lane| {
                    output[oh * out_w * out_c + ow * out_c + oc + lane] = acc[lane];
                }
            }
            
            while (oc < out_c) : (oc += 1) {
                var sum: f32 = 0.0;
                inline for (0..3) |kh| {
                    inline for (0..3) |kw| {
                        var ic: usize = 0;
                        while (ic < in_c) : (ic += 1) {
                            const ih = oh + kh;
                            const iw = ow + kw;
                            sum += input[ih * in_w * in_c + iw * in_c + ic] *
                                   kernel[kh * 3 * in_c * out_c + 
                                         kw * in_c * out_c + 
                                         ic * out_c + 
                                         oc];
                        }
                    }
                }
                output[oh * out_w * out_c + ow * out_c + oc] = sum;
            }
        }
    }
}

export fn conv2d_general(
    input: [*]const f32,
    kernel: [*]const f32,
    output: [*]f32,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_c: usize,
    stride: usize,
) callconv(.c) void {
    const out_h = (in_h - kernel_h) / stride + 1;
    const out_w = (in_w - kernel_w) / stride + 1;
    
    @memset(output[0..(out_h * out_w * out_c)], 0.0);
    
    var oh: usize = 0;
    while (oh < out_h) : (oh += 1) {
        var ow: usize = 0;
        while (ow < out_w) : (ow += 1) {
            var oc: usize = 0;
            while (oc < out_c) : (oc += 1) {
                var sum: f32 = 0.0;
                
                var kh: usize = 0;
                while (kh < kernel_h) : (kh += 1) {
                    var kw: usize = 0;
                    while (kw < kernel_w) : (kw += 1) {
                        var ic: usize = 0;
                        while (ic < in_c) : (ic += 1) {
                            const ih = oh * stride + kh;
                            const iw = ow * stride + kw;
                            sum += input[ih * in_w * in_c + iw * in_c + ic] *
                                   kernel[kh * kernel_w * in_c * out_c + 
                                         kw * in_c * out_c + 
                                         ic * out_c + 
                                         oc];
                        }
                    }
                }
                output[oh * out_w * out_c + ow * out_c + oc] = sum;
            }
        }
    }
}
