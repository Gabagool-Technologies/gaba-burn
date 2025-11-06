use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gaba_memory::{GabaMemory, MemoryChunk, MemoryLayer, HnswIndex, AssociativeGraph, HopfieldLayer};
use uuid::Uuid;

fn bench_vector_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_store");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let memory = GabaMemory::new(128);
            
            for i in 0..size {
                let chunk = MemoryChunk::new(
                    format!("chunk_{}", i),
                    vec![0.1; 128],
                    MemoryLayer::Working,
                );
                memory.store(chunk).unwrap();
            }
            
            b.iter(|| {
                let query = vec![0.1; 128];
                black_box(memory.search_similar(&query, 10).unwrap());
            });
        });
    }
    
    group.finish();
}

fn bench_hnsw_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_index");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let index = HnswIndex::new(128, 16, 200);
            
            for i in 0..size {
                let id = Uuid::new_v4();
                let vec = vec![0.1; 128];
                index.insert(id, vec).unwrap();
            }
            
            b.iter(|| {
                let query = vec![0.1; 128];
                black_box(index.search(&query, 10).unwrap());
            });
        });
    }
    
    group.finish();
}

fn bench_associative_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_graph");
    
    let graph = AssociativeGraph::new();
    let ids: Vec<Uuid> = (0..1000).map(|_| Uuid::new_v4()).collect();
    
    for i in 0..999 {
        graph.strengthen_association(ids[i], ids[i + 1], 0.5);
    }
    
    group.bench_function("spread_activation", |b| {
        b.iter(|| {
            black_box(graph.spread_activation(&ids[0..10], 0.1, 3));
        });
    });
    
    group.finish();
}

fn bench_hopfield_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("hopfield_layer");
    
    let layer = HopfieldLayer::new(128);
    let query = vec![0.1; 128];
    
    let candidates: Vec<(Uuid, Vec<f32>, f32)> = (0..100)
        .map(|_| (Uuid::new_v4(), vec![0.1; 128], 0.5))
        .collect();
    
    group.bench_function("rerank_100", |b| {
        b.iter(|| {
            black_box(layer.rerank_candidates(&query, &candidates).unwrap());
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_vector_store,
    bench_hnsw_index,
    bench_associative_graph,
    bench_hopfield_layer
);
criterion_main!(benches);
