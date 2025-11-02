use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gaba_singularity::AdaptiveKernelOrchestrator;

fn bench_adaptive_gemm(c: &mut Criterion) {
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(true);
    
    let sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256)];
    
    for (m, n, k) in sizes {
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut result = vec![0.0f32; m * n];
        
        c.bench_with_input(
            BenchmarkId::new("adaptive_gemm", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bencher, &(_m, _n, _k)| {
                bencher.iter(|| {
                    orchestrator.execute_gemm_adaptive(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut result),
                        m, n, k
                    );
                });
            },
        );
    }
}

fn bench_learning_convergence(c: &mut Criterion) {
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(true);
    let m = 256;
    let n = 256;
    let k = 256;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let mut result = vec![0.0f32; m * n];
    
    c.bench_function("learning_convergence_256x256", |bencher| {
        bencher.iter(|| {
            for _ in 0..10 {
                orchestrator.execute_gemm_adaptive(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut result),
                    m, n, k
                );
            }
        });
    });
}

fn bench_cold_start(c: &mut Criterion) {
    let m = 128;
    let n = 128;
    let k = 128;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let mut result = vec![0.0f32; m * n];
    
    c.bench_function("cold_start_128x128", |bencher| {
        bencher.iter(|| {
            let orchestrator = AdaptiveKernelOrchestrator::new();
            orchestrator.execute_gemm_adaptive(
                black_box(&a),
                black_box(&b),
                black_box(&mut result),
                m, n, k
            );
        });
    });
}

criterion_group!(benches, bench_adaptive_gemm, bench_learning_convergence, bench_cold_start);
criterion_main!(benches);
