use gaba_native_kernels::{gemm, gemm_rust, gemm_q8};
use approx::assert_abs_diff_eq;

#[test]
fn test_gemm_correctness() {
    let m = 64;
    let n = 64;
    let k = 64;
    
    let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.02).collect();
    
    let mut c_ref = vec![0.0f32; m * n];
    let mut c_test = vec![0.0f32; m * n];
    
    gemm_rust(&a, &b, &mut c_ref, m, n, k);
    gemm(&a, &b, &mut c_test, m, n, k);
    
    for i in 0..m*n {
        assert_abs_diff_eq!(c_ref[i], c_test[i], epsilon = 1e-4);
    }
}

#[test]
fn test_gemm_large() {
    let m = 512;
    let n = 512;
    let k = 512;
    
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    let start = std::time::Instant::now();
    gemm(&a, &b, &mut c, m, n, k);
    let elapsed = start.elapsed();
    
    println!("GEMM 512x512: {:?}", elapsed);
    
    // Verify result (should be k for each element)
    for &val in &c {
        assert_abs_diff_eq!(val, k as f32, epsilon = 1e-3);
    }
}

#[test]
fn test_quantized_gemm() {
    let m = 32;
    let n = 32;
    let k = 32;
    
    let a: Vec<u8> = (0..m*k).map(|i| (i % 256) as u8).collect();
    let b: Vec<u8> = (0..k*n).map(|i| ((i * 2) % 256) as u8).collect();
    let mut c = vec![0u8; m * n];
    
    let scale_a = 0.1;
    let scale_b = 0.1;
    let scale_out = 0.01;
    
    gemm_q8(&a, &b, &mut c, m, n, k, scale_a, scale_b, scale_out);
    
    // Verify non-zero output
    let sum: u32 = c.iter().map(|&x| x as u32).sum();
    assert!(sum > 0);
}

#[cfg(feature = "zig")]
#[test]
fn test_zig_simd_performance() {
    let sizes = vec![
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];
    
    for (m, n, k) in sizes {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        
        let start = std::time::Instant::now();
        gemm(&a, &b, &mut c, m, n, k);
        let elapsed = start.elapsed();
        
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / elapsed.as_secs_f64() / 1e9;
        println!("GEMM {}x{}: {:.2}ms ({:.2} GFLOPS)", m, n, 
                 elapsed.as_secs_f64() * 1000.0, gflops);
    }
}
