use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gaba_train::pruning::PruningConfig;
use gaba_train::quantization::QuantizationConfig;
use gaba_train::{Trainer, TrainingConfig};

fn benchmark_training_with_singularity(c: &mut Criterion) {
    let mut group = c.benchmark_group("singularity_training");

    for use_singularity in [false, true] {
        let name = if use_singularity {
            "with_singularity"
        } else {
            "without_singularity"
        };

        group.bench_with_input(
            BenchmarkId::new("train_small_model", name),
            &use_singularity,
            |b, &enabled| {
                b.iter(|| {
                    let config = TrainingConfig {
                        epochs: 10,
                        batch_size: 32,
                        learning_rate: 0.001,
                        test_ratio: 0.2,
                        early_stopping_patience: None,
                        verbose: false,
                        use_singularity: enabled,
                        quantization: QuantizationConfig::default(),
                        pruning: PruningConfig::default(),
                        enable_profiling: false,
                    };

                    let trainer = Trainer::new(config);
                    black_box(trainer);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_training_with_singularity);
criterion_main!(benches);
