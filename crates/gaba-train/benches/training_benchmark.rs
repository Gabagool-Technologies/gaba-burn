use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gaba_train::{TrainingConfig, Trainer, TrafficDataset, RouteDataset};
use burn::backend::NdArray;

type Backend = NdArray;

fn benchmark_traffic_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("traffic_training");
    
    for batch_size in [16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                let config = TrainingConfig {
                    epochs: 1,
                    batch_size: size,
                    learning_rate: 0.001,
                    use_singularity: false,
                    enable_profiling: false,
                    verbose: false,
                    ..Default::default()
                };
                
                b.iter(|| {
                    let trainer = Trainer::new(config.clone());
                    black_box(trainer);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_singularity_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("singularity");
    
    let config_no_sing = TrainingConfig {
        epochs: 1,
        batch_size: 32,
        learning_rate: 0.001,
        use_singularity: false,
        enable_profiling: false,
        verbose: false,
        ..Default::default()
    };
    
    let config_with_sing = TrainingConfig {
        use_singularity: true,
        ..config_no_sing.clone()
    };
    
    group.bench_function("without_singularity", |b| {
        b.iter(|| {
            let trainer = Trainer::new(config_no_sing.clone());
            black_box(trainer);
        });
    });
    
    group.bench_function("with_singularity", |b| {
        b.iter(|| {
            let trainer = Trainer::new(config_with_sing.clone());
            black_box(trainer);
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_traffic_training, benchmark_singularity_overhead);
criterion_main!(benches);
