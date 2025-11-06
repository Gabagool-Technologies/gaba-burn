use std::time::Duration;

pub struct Conv2DParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

pub fn conv2d_naive(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    params: &Conv2DParams,
    input_h: usize,
    input_w: usize,
) -> Duration {
    let start = std::time::Instant::now();

    let (kh, kw) = params.kernel_size;
    let (sh, sw) = params.stride;
    let (ph, pw) = params.padding;

    let output_h = (input_h + 2 * ph - kh) / sh + 1;
    let output_w = (input_w + 2 * pw - kw) / sw + 1;

    for oc in 0..params.out_channels {
        for oh in 0..output_h {
            for ow in 0..output_w {
                let mut sum = bias[oc];

                for ic in 0..params.in_channels {
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;

                            if ih >= ph && ih < input_h + ph && iw >= pw && iw < input_w + pw {
                                let ih_actual = ih - ph;
                                let iw_actual = iw - pw;

                                let input_idx =
                                    ic * input_h * input_w + ih_actual * input_w + iw_actual;
                                let weight_idx = oc * params.in_channels * kh * kw
                                    + ic * kh * kw
                                    + kh_i * kw
                                    + kw_i;

                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }

                let output_idx = oc * output_h * output_w + oh * output_w + ow;
                output[output_idx] = sum;
            }
        }
    }

    start.elapsed()
}

pub fn conv2d_im2col(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    params: &Conv2DParams,
    input_h: usize,
    input_w: usize,
) -> Duration {
    let start = std::time::Instant::now();

    let (kh, kw) = params.kernel_size;
    let (sh, sw) = params.stride;
    let (ph, pw) = params.padding;

    let output_h = (input_h + 2 * ph - kh) / sh + 1;
    let output_w = (input_w + 2 * pw - kw) / sw + 1;

    let col_size = params.in_channels * kh * kw * output_h * output_w;
    let mut col = vec![0.0f32; col_size];

    let mut col_idx = 0;
    for oh in 0..output_h {
        for ow in 0..output_w {
            for ic in 0..params.in_channels {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let ih = oh * sh + kh_i;
                        let iw = ow * sw + kw_i;

                        if ih >= ph && ih < input_h + ph && iw >= pw && iw < input_w + pw {
                            let ih_actual = ih - ph;
                            let iw_actual = iw - pw;
                            let input_idx =
                                ic * input_h * input_w + ih_actual * input_w + iw_actual;
                            col[col_idx] = input[input_idx];
                        }
                        col_idx += 1;
                    }
                }
            }
        }
    }

    let m = params.out_channels;
    let n = output_h * output_w;
    let k = params.in_channels * kh * kw;

    for i in 0..m {
        for j in 0..n {
            let mut sum = bias[i];
            for p in 0..k {
                sum += weights[i * k + p] * col[p * n + j];
            }
            output[i * n + j] = sum;
        }
    }

    start.elapsed()
}

pub fn conv2d_relu_fused(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    params: &Conv2DParams,
    input_h: usize,
    input_w: usize,
) -> Duration {
    let duration = conv2d_im2col(input, weights, bias, output, params, input_h, input_w);

    for val in output.iter_mut() {
        *val = val.max(0.0);
    }

    duration
}

#[cfg(target_os = "macos")]
pub fn conv2d_accelerate(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    params: &Conv2DParams,
    input_h: usize,
    input_w: usize,
) -> Duration {
    use crate::accelerate::gemm_accelerate;

    let start = std::time::Instant::now();

    let (kh, kw) = params.kernel_size;
    let (sh, sw) = params.stride;
    let (ph, pw) = params.padding;

    let output_h = (input_h + 2 * ph - kh) / sh + 1;
    let output_w = (input_w + 2 * pw - kw) / sw + 1;

    let col_size = params.in_channels * kh * kw * output_h * output_w;
    let mut col = vec![0.0f32; col_size];

    let mut col_idx = 0;
    for oh in 0..output_h {
        for ow in 0..output_w {
            for ic in 0..params.in_channels {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let ih = oh * sh + kh_i;
                        let iw = ow * sw + kw_i;

                        if ih >= ph && ih < input_h + ph && iw >= pw && iw < input_w + pw {
                            let ih_actual = ih - ph;
                            let iw_actual = iw - pw;
                            let input_idx =
                                ic * input_h * input_w + ih_actual * input_w + iw_actual;
                            col[col_idx] = input[input_idx];
                        }
                        col_idx += 1;
                    }
                }
            }
        }
    }

    let m = params.out_channels;
    let n = output_h * output_w;
    let k = params.in_channels * kh * kw;

    let mut gemm_output = vec![0.0f32; m * n];
    gemm_accelerate(weights, &col, &mut gemm_output, m, n, k);

    for i in 0..m {
        for j in 0..n {
            output[i * n + j] = gemm_output[i * n + j] + bias[i];
        }
    }

    start.elapsed()
}

#[cfg(not(target_os = "macos"))]
pub fn conv2d_accelerate(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    params: &Conv2DParams,
    input_h: usize,
    input_w: usize,
) -> Duration {
    conv2d_im2col(input, weights, bias, output, params, input_h, input_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_naive() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let weights = vec![1.0, 0.0, 0.0, 1.0];

        let bias = vec![0.0];
        let mut output = vec![0.0; 4];

        let params = Conv2DParams {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2),
            stride: (1, 1),
            padding: (0, 0),
        };

        conv2d_naive(&input, &weights, &bias, &mut output, &params, 3, 3);

        assert_eq!(output[0], 6.0);
        assert_eq!(output[1], 8.0);
        assert_eq!(output[2], 12.0);
        assert_eq!(output[3], 14.0);
    }

    #[test]
    fn test_conv2d_relu_fused() {
        let input = vec![1.0, -2.0, 3.0, -4.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![-5.0];
        let mut output = vec![0.0; 1];

        let params = Conv2DParams {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2),
            stride: (1, 1),
            padding: (0, 0),
        };

        conv2d_relu_fused(&input, &weights, &bias, &mut output, &params, 2, 2);

        assert!(output[0] >= 0.0);
    }
}
