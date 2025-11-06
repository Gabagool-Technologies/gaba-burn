use std::time::Duration;

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
pub fn gemm_i8_amx(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) -> Duration {
    let start = std::time::Instant::now();

    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut c_f32 = vec![0.0f32; m * n];

    for i in 0..m * k {
        a_f32[i] = a[i] as f32;
    }
    for i in 0..k * n {
        b_f32[i] = b[i] as f32;
    }

    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a_f32.as_ptr(),
            k as i32,
            b_f32.as_ptr(),
            n as i32,
            0.0,
            c_f32.as_mut_ptr(),
            n as i32,
        );
    }

    for i in 0..m * n {
        c[i] = c_f32[i].round() as i32;
    }

    start.elapsed()
}

#[cfg(not(target_os = "macos"))]
pub fn gemm_i8_amx(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) -> Duration {
    crate::quantization::gemm_i8(a, b, c, m, n, k)
}

pub fn gemm_quantized_amx(
    a_f32: &[f32],
    b_f32: &[f32],
    c_f32: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Duration {
    use crate::quantization::quantize_tensor;

    let start = std::time::Instant::now();

    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; k * n];
    let mut c_i32 = vec![0i32; m * n];

    let params_a = quantize_tensor(a_f32, &mut a_i8);
    let params_b = quantize_tensor(b_f32, &mut b_i8);

    gemm_i8_amx(&a_i8, &b_i8, &mut c_i32, m, n, k);

    let scale_out = params_a.scale * params_b.scale;
    for i in 0..m * n {
        c_f32[i] = c_i32[i] as f32 * scale_out;
    }

    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_i8_amx() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1i8, 2, 3, 4];
        let mut c = vec![0i32; 4];

        gemm_i8_amx(&a, &b, &mut c, 2, 2, 2);

        assert_eq!(c[0], 7);
        assert_eq!(c[1], 10);
        assert_eq!(c[2], 15);
        assert_eq!(c[3], 22);
    }

    #[test]
    fn test_gemm_quantized_amx() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];

        gemm_quantized_amx(&a, &b, &mut c, 2, 2, 2);

        for &val in &c {
            assert!(val > 0.0);
        }
    }
}
