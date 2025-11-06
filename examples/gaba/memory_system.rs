use gaba_memory::{HNSWIndex, AssociativeGraph, TemporalMemory, VectorStore};

fn main() {
    println!("GABA Memory System Demo\n");
    
    // HNSW Vector Search
    println!("=== HNSW Vector Search ===");
    let mut hnsw = HNSWIndex::new(16, 32);
    
    // Insert vectors
    for i in 0..100 {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32).sin()).collect();
        hnsw.insert(i, vector);
    }
    
    // Search
    let query: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
    let results = hnsw.search(&query, 5);
    
    println!("Query vector: 128 dimensions");
    println!("Top 5 results:");
    for (id, distance) in results {
        println!("  ID: {}, Distance: {:.4}", id, distance);
    }
    println!();
    
    // Associative Graph
    println!("=== Associative Graph ===");
    let mut graph = AssociativeGraph::new();
    
    // Add concepts
    graph.add_concept(0, vec![1.0, 0.0, 0.0]);
    graph.add_concept(1, vec![0.9, 0.1, 0.0]);
    graph.add_concept(2, vec![0.0, 1.0, 0.0]);
    graph.add_concept(3, vec![0.0, 0.9, 0.1]);
    
    // Spread activation
    let activated = graph.spread_activation(0, 2);
    
    println!("Spreading activation from concept 0:");
    for (id, activation) in activated {
        println!("  Concept {}: {:.4}", id, activation);
    }
    println!();
    
    // Temporal Memory
    println!("=== Temporal Memory ===");
    let mut temporal = TemporalMemory::new();
    
    // Store memories
    temporal.store(0, vec![1.0, 2.0, 3.0], 1.0);
    temporal.store(1, vec![1.1, 2.1, 3.1], 0.8);
    temporal.store(2, vec![4.0, 5.0, 6.0], 0.6);
    
    // Consolidate
    temporal.consolidate();
    
    println!("Stored 3 memories");
    println!("Consolidated based on temporal patterns");
    println!("Memory strength updated");
    println!();
    
    // Vector Store
    println!("=== Vector Store ===");
    let mut store = VectorStore::new(128);
    
    // Add vectors
    for i in 0..50 {
        let vector: Vec<f32> = (0..128).map(|j| ((i * j) as f32).sin()).collect();
        store.add(i, vector);
    }
    
    // Search similar
    let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.5).cos()).collect();
    let similar = store.search_similar(&query, 3);
    
    println!("Query vector: 128 dimensions");
    println!("Top 3 similar vectors:");
    for (id, similarity) in similar {
        println!("  ID: {}, Similarity: {:.4}", id, similarity);
    }
    
    println!("\nAll memory systems operational!");
}
