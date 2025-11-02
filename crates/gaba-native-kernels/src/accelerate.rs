use std::time::Duration;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
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

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

#[cfg(target_os = "macos")]
pub fn gemm_accelerate(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Duration {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    
    let start = std::time::Instant::now();
    
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    
    start.elapsed()
}

#[cfg(not(target_os = "macos"))]
pub fn gemm_accelerate(_a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) -> Duration {
    panic!("Accelerate framework only available on macOS");
}

pub fn detect_amx() -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.optional.arm.FEAT_DotProd")
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                return s.trim() == "1";
            }
        }
    }
    
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_os = "macos")]
    fn test_accelerate_gemm() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let mut c = vec![0.0; m * n];
        
        gemm_accelerate(&a, &b, &mut c, m, n, k);
        
        for &val in &c {
            assert!((val - 4.0).abs() < 0.001);
        }
    }
}
