use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gaba_singularity::{PrimeOptimizer, PrimeConfig};

fn benchmark_batch_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_optimization");
    
    let configs = vec![
        ("baseline_13_19_42", PrimeConfig::default()),
        ("extended_23_29_67", PrimeConfig::extended()),
        ("fibonacci_13_21_34", PrimeConfig::fibonacci()),
        ("twin_primes", PrimeConfig::twin_primes()),
    ];
    
    for (name, config) in configs {
        let optimizer = PrimeOptimizer::new(config);
        
        group.bench_with_input(BenchmarkId::new(name, "32"), &32, |b, &size| {
            b.iter(|| optimizer.optimize_batch_size(black_box(size)));
        });
        
        group.bench_with_input(BenchmarkId::new(name, "64"), &64, |b, &size| {
            b.iter(|| optimizer.optimize_batch_size(black_box(size)));
        });
        
        group.bench_with_input(BenchmarkId::new(name, "128"), &128, |b, &size| {
            b.iter(|| optimizer.optimize_batch_size(black_box(size)));
        });
    }
    
    group.finish();
}

fn benchmark_block_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_size_optimization");
    
    let configs = vec![
        ("baseline_13_19_42", PrimeConfig::default()),
        ("extended_23_29_67", PrimeConfig::extended()),
        ("fibonacci_13_21_34", PrimeConfig::fibonacci()),
        ("twin_primes", PrimeConfig::twin_primes()),
    ];
    
    for (name, config) in configs {
        let optimizer = PrimeOptimizer::new(config);
        
        group.bench_with_input(BenchmarkId::new(name, "256"), &256, |b, &dim| {
            b.iter(|| optimizer.optimize_block_size(black_box(dim)));
        });
        
        group.bench_with_input(BenchmarkId::new(name, "512"), &512, |b, &dim| {
            b.iter(|| optimizer.optimize_block_size(black_box(dim)));
        });
        
        group.bench_with_input(BenchmarkId::new(name, "1024"), &1024, |b, &dim| {
            b.iter(|| optimizer.optimize_block_size(black_box(dim)));
        });
    }
    
    group.finish();
}

fn benchmark_lr_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning_rate_schedule");
    
    let configs = vec![
        ("baseline_13_19_42", PrimeConfig::default()),
        ("extended_23_29_67", PrimeConfig::extended()),
        ("fibonacci_13_21_34", PrimeConfig::fibonacci()),
        ("twin_primes", PrimeConfig::twin_primes()),
    ];
    
    for (name, config) in configs {
        let optimizer = PrimeOptimizer::new(config);
        
        group.bench_with_input(BenchmarkId::new(name, "100_epochs"), &100, |b, &epochs| {
            b.iter(|| optimizer.optimize_learning_rate_schedule(black_box(0.001), black_box(epochs)));
        });
    }
    
    group.finish();
}

fn benchmark_full_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_configuration");
    
    let configs = vec![
        ("baseline_13_19_42", PrimeConfig::default()),
        ("extended_23_29_67", PrimeConfig::extended()),
        ("fibonacci_13_21_34", PrimeConfig::fibonacci()),
        ("twin_primes", PrimeConfig::twin_primes()),
    ];
    
    for (name, config) in configs {
        let optimizer = PrimeOptimizer::new(config);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let _batch = optimizer.optimize_batch_size(black_box(32));
                let _block = optimizer.optimize_block_size(black_box(256));
                let _lr = optimizer.optimize_learning_rate_schedule(black_box(0.001), black_box(100));
                let _momentum = optimizer.optimize_momentum_params();
                let _groups = optimizer.optimize_quantization_groups(black_box(1024));
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_batch_optimization,
    benchmark_block_optimization,
    benchmark_lr_schedule,
    benchmark_full_config
);
criterion_main!(benches);
