use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gaba_singularity::{PrimeOptimizer, PrimeConfig};

fn benchmark_extended_vs_baseline_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("extended_vs_baseline_batch");
    
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let sizes = vec![32, 64, 128, 256];
    
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, &s| {
            b.iter(|| baseline.optimize_batch_size(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("extended", size), &size, |b, &s| {
            b.iter(|| extended.optimize_batch_size(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("fibonacci", size), &size, |b, &s| {
            b.iter(|| fibonacci.optimize_batch_size(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("twin_primes", size), &size, |b, &s| {
            b.iter(|| twin.optimize_batch_size(black_box(s)));
        });
    }
    
    group.finish();
}

fn benchmark_extended_vs_baseline_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("extended_vs_baseline_block");
    
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let dims = vec![256, 512, 1024, 2048];
    
    for dim in dims {
        group.bench_with_input(BenchmarkId::new("baseline", dim), &dim, |b, &d| {
            b.iter(|| baseline.optimize_block_size(black_box(d)));
        });
        
        group.bench_with_input(BenchmarkId::new("extended", dim), &dim, |b, &d| {
            b.iter(|| extended.optimize_block_size(black_box(d)));
        });
        
        group.bench_with_input(BenchmarkId::new("fibonacci", dim), &dim, |b, &d| {
            b.iter(|| fibonacci.optimize_block_size(black_box(d)));
        });
        
        group.bench_with_input(BenchmarkId::new("twin_primes", dim), &dim, |b, &d| {
            b.iter(|| twin.optimize_block_size(black_box(d)));
        });
    }
    
    group.finish();
}

fn benchmark_extended_vs_baseline_lr_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("extended_vs_baseline_lr_schedule");
    
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let epochs_list = vec![50, 100, 200];
    
    for epochs in epochs_list {
        group.bench_with_input(BenchmarkId::new("baseline", epochs), &epochs, |b, &e| {
            b.iter(|| baseline.optimize_learning_rate_schedule(black_box(0.001), black_box(e)));
        });
        
        group.bench_with_input(BenchmarkId::new("extended", epochs), &epochs, |b, &e| {
            b.iter(|| extended.optimize_learning_rate_schedule(black_box(0.001), black_box(e)));
        });
        
        group.bench_with_input(BenchmarkId::new("fibonacci", epochs), &epochs, |b, &e| {
            b.iter(|| fibonacci.optimize_learning_rate_schedule(black_box(0.001), black_box(e)));
        });
        
        group.bench_with_input(BenchmarkId::new("twin_primes", epochs), &epochs, |b, &e| {
            b.iter(|| twin.optimize_learning_rate_schedule(black_box(0.001), black_box(e)));
        });
    }
    
    group.finish();
}

fn benchmark_full_config_all_sequences(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_config_all_sequences");
    
    let configs = vec![
        ("baseline", PrimeConfig::default()),
        ("extended", PrimeConfig::extended()),
        ("fibonacci", PrimeConfig::fibonacci()),
        ("twin_primes", PrimeConfig::twin_primes()),
    ];
    
    for (name, config) in configs {
        let optimizer = PrimeOptimizer::new(config);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                optimizer.get_optimal_config(black_box(128), black_box(512))
            });
        });
    }
    
    group.finish();
}

fn benchmark_quantization_groups(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_groups");
    
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let param_counts = vec![1_000_000, 10_000_000, 110_000_000];
    
    for params in param_counts {
        group.bench_with_input(BenchmarkId::new("baseline", params), &params, |b, &p| {
            b.iter(|| baseline.optimize_quantization_groups(black_box(p)));
        });
        
        group.bench_with_input(BenchmarkId::new("extended", params), &params, |b, &p| {
            b.iter(|| extended.optimize_quantization_groups(black_box(p)));
        });
        
        group.bench_with_input(BenchmarkId::new("fibonacci", params), &params, |b, &p| {
            b.iter(|| fibonacci.optimize_quantization_groups(black_box(p)));
        });
        
        group.bench_with_input(BenchmarkId::new("twin_primes", params), &params, |b, &p| {
            b.iter(|| twin.optimize_quantization_groups(black_box(p)));
        });
    }
    
    group.finish();
}

fn benchmark_cache_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_alignment");
    
    let baseline = PrimeOptimizer::new(PrimeConfig::default());
    let extended = PrimeOptimizer::new(PrimeConfig::extended());
    let fibonacci = PrimeOptimizer::new(PrimeConfig::fibonacci());
    let twin = PrimeOptimizer::new(PrimeConfig::twin_primes());
    
    let sizes = vec![100, 256, 1024, 4096];
    
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("baseline", size), &size, |b, &s| {
            b.iter(|| baseline.optimize_cache_alignment(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("extended", size), &size, |b, &s| {
            b.iter(|| extended.optimize_cache_alignment(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("fibonacci", size), &size, |b, &s| {
            b.iter(|| fibonacci.optimize_cache_alignment(black_box(s)));
        });
        
        group.bench_with_input(BenchmarkId::new("twin_primes", size), &size, |b, &s| {
            b.iter(|| twin.optimize_cache_alignment(black_box(s)));
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_extended_vs_baseline_batch,
    benchmark_extended_vs_baseline_block,
    benchmark_extended_vs_baseline_lr_schedule,
    benchmark_full_config_all_sequences,
    benchmark_quantization_groups,
    benchmark_cache_alignment
);
criterion_main!(benches);
