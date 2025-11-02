use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gaba_native_kernels::{conv2d_3x3, softmax, layernorm, attention};

fn bench_conv2d(c: &mut Criterion) {
    // 224x224x3 -> 222x222x64 (typical first layer)
    let input = vec![0.5f32; 224 * 224 * 3];
    let kernel = vec![0.1f32; 3 * 3 * 3 * 64];
    let mut output = vec![0.0f32; 222 * 222 * 64];
    
    c.bench_function("conv2d_224x224x3_to_64", |b| {
        b.iter(|| {
            conv2d_3x3(
                black_box(&input),
                black_box(&kernel),
                black_box(&mut output),
                224, 224, 3, 64
            );
        })
    });
    
    // 56x56x128 -> 54x54x256 (mid layer)
    let input2 = vec![0.5f32; 56 * 56 * 128];
    let kernel2 = vec![0.1f32; 3 * 3 * 128 * 256];
    let mut output2 = vec![0.0f32; 54 * 54 * 256];
    
    c.bench_function("conv2d_56x56x128_to_256", |b| {
        b.iter(|| {
            conv2d_3x3(
                black_box(&input2),
                black_box(&kernel2),
                black_box(&mut output2),
                56, 56, 128, 256
            );
        })
    });
}

fn bench_softmax(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    
    for &size in &sizes {
        let input = vec![0.5f32; size];
        let mut output = vec![0.0f32; size];
        
        c.bench_function(&format!("softmax_{}", size), |b| {
            b.iter(|| {
                softmax(black_box(&input), black_box(&mut output));
            })
        });
    }
}

fn bench_layernorm(c: &mut Criterion) {
    let sizes = [512, 768, 1024, 2048];
    
    for &size in &sizes {
        let input = vec![0.5f32; size];
        let gamma = vec![1.0f32; size];
        let beta = vec![0.0f32; size];
        let mut output = vec![0.0f32; size];
        
        c.bench_function(&format!("layernorm_{}", size), |b| {
            b.iter(|| {
                layernorm(
                    black_box(&input),
                    black_box(&gamma),
                    black_box(&beta),
                    black_box(&mut output),
                    1e-5
                );
            })
        });
    }
}

fn bench_attention(c: &mut Criterion) {
    // Small attention (seq_len=128, d_model=512, heads=8)
    let seq_len = 128;
    let d_model = 512;
    let num_heads = 8;
    
    let query = vec![0.5f32; seq_len * d_model];
    let key = vec![0.5f32; seq_len * d_model];
    let value = vec![0.5f32; seq_len * d_model];
    let mut output = vec![0.0f32; seq_len * d_model];
    
    c.bench_function("attention_128x512x8", |b| {
        b.iter(|| {
            attention(
                black_box(&query),
                black_box(&key),
                black_box(&value),
                black_box(&mut output),
                seq_len, d_model, num_heads
            );
        })
    });
    
    // Medium attention (seq_len=512, d_model=512, heads=8)
    let seq_len2 = 512;
    let query2 = vec![0.5f32; seq_len2 * d_model];
    let key2 = vec![0.5f32; seq_len2 * d_model];
    let value2 = vec![0.5f32; seq_len2 * d_model];
    let mut output2 = vec![0.0f32; seq_len2 * d_model];
    
    c.bench_function("attention_512x512x8", |b| {
        b.iter(|| {
            attention(
                black_box(&query2),
                black_box(&key2),
                black_box(&value2),
                black_box(&mut output2),
                seq_len2, d_model, num_heads
            );
        })
    });
}

criterion_group!(benches, bench_conv2d, bench_softmax, bench_layernorm, bench_attention);
criterion_main!(benches);
