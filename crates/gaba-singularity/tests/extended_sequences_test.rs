use gaba_singularity::{PrimeOptimizer, PrimeConfig};

#[test]
fn test_extended_sequence_batch_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::extended());
    
    let batch_32 = optimizer.optimize_batch_size(32);
    assert!(batch_32 >= 16 && batch_32 <= 64);
    
    let batch_64 = optimizer.optimize_batch_size(64);
    assert!(batch_64 >= 32 && batch_64 <= 128);
    
    let batch_128 = optimizer.optimize_batch_size(128);
    assert!(batch_128 >= 64 && batch_128 <= 256);
}

#[test]
fn test_extended_sequence_block_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::extended());
    
    let (w256, h256) = optimizer.optimize_block_size(256);
    assert_eq!(w256, h256);
    assert!(w256 >= 11 && w256 <= 67);
    
    let (w512, h512) = optimizer.optimize_block_size(512);
    assert_eq!(w512, h512);
    assert!(w512 >= 11 && w512 <= 67);
    
    let (w1024, h1024) = optimizer.optimize_block_size(1024);
    assert_eq!(w1024, h1024);
    assert!(w1024 >= 11 && w1024 <= 67);
}

#[test]
fn test_fibonacci_sequence_batch_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::fibonacci());
    
    let batch_32 = optimizer.optimize_batch_size(32);
    assert!(batch_32 % 13 == 0 || batch_32 % 21 == 0 || batch_32 % 34 == 0);
    
    let batch_64 = optimizer.optimize_batch_size(64);
    assert!(batch_64 % 13 == 0 || batch_64 % 21 == 0 || batch_64 % 34 == 0);
    
    let batch_128 = optimizer.optimize_batch_size(128);
    assert!(batch_128 % 13 == 0 || batch_128 % 21 == 0 || batch_128 % 34 == 0);
}

#[test]
fn test_fibonacci_sequence_block_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::fibonacci());
    
    let (w256, h256) = optimizer.optimize_block_size(256);
    assert_eq!(w256, h256);
    assert!(w256 == 13 || w256 == 21 || w256 == 34);
    
    let (w512, h512) = optimizer.optimize_block_size(512);
    assert_eq!(w512, h512);
    assert!(w512 == 13 || w512 == 21 || w512 == 34);
}

#[test]
fn test_twin_primes_batch_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let batch_32 = optimizer.optimize_batch_size(32);
    assert!(batch_32 % 11 == 0 || batch_32 % 13 == 0 || batch_32 % 17 == 0 
            || batch_32 % 19 == 0 || batch_32 % 29 == 0 || batch_32 % 31 == 0);
    
    let batch_64 = optimizer.optimize_batch_size(64);
    assert!(batch_64 % 11 == 0 || batch_64 % 13 == 0 || batch_64 % 17 == 0 
            || batch_64 % 19 == 0 || batch_64 % 29 == 0 || batch_64 % 31 == 0);
}

#[test]
fn test_twin_primes_block_optimization() {
    let optimizer = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let (w256, h256) = optimizer.optimize_block_size(256);
    assert_eq!(w256, h256);
    assert!(w256 == 11 || w256 == 13 || w256 == 17 || w256 == 19 || w256 == 29 || w256 == 31);
}

#[test]
fn test_sequence_learning_rate_schedules() {
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let epochs = 100;
    let lr = 0.001;
    
    let schedule_baseline = baseline.optimize_learning_rate_schedule(lr, epochs);
    let schedule_extended = extended.optimize_learning_rate_schedule(lr, epochs);
    let schedule_fibonacci = fibonacci.optimize_learning_rate_schedule(lr, epochs);
    let schedule_twin = twin.optimize_learning_rate_schedule(lr, epochs);
    
    assert_eq!(schedule_baseline.len(), epochs);
    assert_eq!(schedule_extended.len(), epochs);
    assert_eq!(schedule_fibonacci.len(), epochs);
    assert_eq!(schedule_twin.len(), epochs);
    
    assert!(schedule_baseline[13] < schedule_baseline[12]);
    assert!(schedule_extended[23] < schedule_extended[22]);
    assert!(schedule_fibonacci[21] < schedule_fibonacci[20]);
    assert!(schedule_twin[11] < schedule_twin[10]);
}

#[test]
fn test_sequence_momentum_params() {
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let (b1_base, b2_base) = baseline.optimize_momentum_params();
    let (b1_ext, b2_ext) = extended.optimize_momentum_params();
    let (b1_fib, b2_fib) = fibonacci.optimize_momentum_params();
    let (b1_twin, b2_twin) = twin.optimize_momentum_params();
    
    assert!(b1_base >= 0.8 && b1_base <= 0.95);
    assert!(b2_base >= 0.99 && b2_base <= 0.999);
    
    assert!(b1_ext >= 0.8 && b1_ext <= 0.95);
    assert!(b2_ext >= 0.99 && b2_ext <= 0.999);
    
    assert!(b1_fib >= 0.8 && b1_fib <= 0.95);
    assert!(b2_fib >= 0.99 && b2_fib <= 0.999);
    
    assert!(b1_twin >= 0.8 && b1_twin <= 0.95);
    assert!(b2_twin >= 0.99 && b2_twin <= 0.999);
}

#[test]
fn test_sequence_quantization_groups() {
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    
    let params = 110_000_000;
    
    let group_base = baseline.optimize_quantization_groups(params);
    let group_ext = extended.optimize_quantization_groups(params);
    let group_fib = fibonacci.optimize_quantization_groups(params);
    
    assert!(group_base == 13 || group_base == 19 || group_base == 42);
    assert!(group_ext == 13 || group_ext == 19 || group_ext == 23 || group_ext == 29 || group_ext == 42 || group_ext == 67);
    assert!(group_fib == 13 || group_fib == 21 || group_fib == 34);
}

#[test]
fn test_sequence_performance_comparison() {
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let workload_size = 128;
    let matrix_dim = 512;
    
    let config_base = baseline.get_optimal_config(workload_size, matrix_dim);
    let config_ext = extended.get_optimal_config(workload_size, matrix_dim);
    let config_fib = fibonacci.get_optimal_config(workload_size, matrix_dim);
    let config_twin = twin.get_optimal_config(workload_size, matrix_dim);
    
    assert!(config_base.optimal_batch_size > 0);
    assert!(config_ext.optimal_batch_size > 0);
    assert!(config_fib.optimal_batch_size > 0);
    assert!(config_twin.optimal_batch_size > 0);
    
    assert!(config_base.optimal_block_size > 0);
    assert!(config_ext.optimal_block_size > 0);
    assert!(config_fib.optimal_block_size > 0);
    assert!(config_twin.optimal_block_size > 0);
}

#[test]
fn test_cache_alignment_all_sequences() {
    let sequences = vec![
        PrimeConfig::default(),
        PrimeConfig::extended(),
        PrimeConfig::fibonacci(),
        PrimeConfig::twin_primes(),
    ];
    
    for config in sequences {
        let optimizer = PrimeOptimizer::new(config);
        
        let aligned_100 = optimizer.optimize_cache_alignment(100);
        assert!(aligned_100 >= 100);
        
        let aligned_256 = optimizer.optimize_cache_alignment(256);
        assert!(aligned_256 >= 256);
        
        let aligned_1024 = optimizer.optimize_cache_alignment(1024);
        assert!(aligned_1024 >= 1024);
    }
}
