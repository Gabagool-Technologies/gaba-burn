use criterion::{Criterion, black_box, criterion_group, criterion_main};
use gaba_singularity::AdaptiveKernelOrchestrator;

fn bench_metal_large_matrices(c: &mut Criterion) {
    let orchestrator = AdaptiveKernelOrchestrator::new();

    c.bench_function("metal_512x512x512", |bencher| {
        let m = 512;
        let n = 512;
        let k = 512;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b_mat: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c_mat = vec![0.0f32; m * n];

        bencher.iter(|| {
            orchestrator.execute_gemm_adaptive(
                black_box(&a),
                black_box(&b_mat),
                black_box(&mut c_mat),
                m,
                n,
                k,
            )
        });
    });

    c.bench_function("metal_1024x1024x1024", |bencher| {
        let m = 1024;
        let n = 1024;
        let k = 1024;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b_mat: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c_mat = vec![0.0f32; m * n];

        bencher.iter(|| {
            orchestrator.execute_gemm_adaptive(
                black_box(&a),
                black_box(&b_mat),
                black_box(&mut c_mat),
                m,
                n,
                k,
            )
        });
    });
}

fn bench_metal_vs_accelerate(c: &mut Criterion) {
    let orchestrator = AdaptiveKernelOrchestrator::new();

    let sizes = vec![("256x256", 256), ("512x512", 512), ("768x768", 768)];

    for (name, size) in sizes {
        c.bench_function(&format!("adaptive_{}", name), |bencher| {
            let m = size;
            let n = size;
            let k = size;
            let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
            let b_mat: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
            let mut c_mat = vec![0.0f32; m * n];

            bencher.iter(|| {
                orchestrator.execute_gemm_adaptive(
                    black_box(&a),
                    black_box(&b_mat),
                    black_box(&mut c_mat),
                    m,
                    n,
                    k,
                )
            });
        });
    }
}

criterion_group!(
    benches,
    bench_metal_large_matrices,
    bench_metal_vs_accelerate
);
criterion_main!(benches);
