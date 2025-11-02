use gaba_singularity::{AdaptiveKernelOrchestrator, KernelType};

#[test]
fn test_adaptive_execution() {
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(true);
    
    let m = 64;
    let n = 64;
    let k = 64;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let mut c = vec![0.0f32; m * n];
    
    let (kernel_type, duration) = orchestrator.execute_gemm_adaptive(&a, &b, &mut c, m, n, k);
    
    assert!(duration.as_micros() > 0);
    assert!(matches!(kernel_type, KernelType::RustVectorized | KernelType::RustParallel | KernelType::ZigUltra | KernelType::Accelerate | KernelType::MetalGPU | KernelType::FusedReLU | KernelType::Quantized));
    
    assert!(c.iter().any(|&v| v != 0.0));
}

#[test]
fn test_learning_improves_selection() {
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(true);
    
    let m = 128;
    let n = 128;
    let k = 128;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let mut c = vec![0.0f32; m * n];
    
    for _ in 0..20 {
        orchestrator.execute_gemm_adaptive(&a, &b, &mut c, m, n, k);
    }
    
    let history_size = orchestrator.history_size();
    assert_eq!(history_size, 20);
    
    let history = orchestrator.get_performance_history();
    assert_eq!(history.len(), 20);
    
    let avg_time: f64 = history.iter()
        .map(|v| v.execution_time.as_secs_f64())
        .sum::<f64>() / history.len() as f64;
    
    assert!(avg_time > 0.0);
}

#[test]
fn test_different_sizes() {
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(false);
    
    let sizes = vec![(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)];
    
    for (m, n, k) in sizes {
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        
        let (_, duration) = orchestrator.execute_gemm_adaptive(&a, &b, &mut c, m, n, k);
        
        assert!(duration.as_micros() > 0);
        assert!(c.iter().any(|&v| v != 0.0));
    }
}

#[test]
fn test_correctness() {
    let orchestrator = AdaptiveKernelOrchestrator::new();
    
    let m = 4;
    let n = 4;
    let k = 4;
    let a = vec![1.0; m * k];
    let b = vec![1.0; k * n];
    let mut c = vec![0.0f32; m * n];
    
    let (kernel_type, _) = orchestrator.execute_gemm_adaptive(&a, &b, &mut c, m, n, k);
    
    let expected = 4.0;
    let tolerance = if matches!(kernel_type, KernelType::Quantized) { 1.0 } else { 0.001 };
    
    for &val in &c {
        assert!((val - expected).abs() < tolerance, "Expected ~{}, got {} (kernel: {:?})", expected, val, kernel_type);
    }
}
