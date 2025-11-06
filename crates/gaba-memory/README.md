# GABA-Memory

Brain-inspired context and memory system for edge ML applications.

## Features

- **Vector Store**: Multi-layer memory organization with cosine similarity search
- **HNSW Index**: Fast approximate nearest neighbor search
- **Associative Graph**: Spreading activation and Hebbian learning
- **Temporal Memory**: Ebbinghaus forgetting curve with adaptive decay
- **Engram Manager**: Reactivation-driven memory strengthening
- **Hopfield Layer**: Attractor-based pattern completion and reranking

## Quick Start

```rust
use gaba_memory::{GabaMemory, MemoryChunk, MemoryLayer};

// Create memory store
let memory = GabaMemory::new(128);

// Store a memory
let chunk = MemoryChunk::new(
    "Important context".to_string(),
    vec![0.1; 128],  // embedding
    MemoryLayer::Episodic,
);
let id = memory.store(chunk)?;

// Search similar memories
let query = vec![0.1; 128];
let results = memory.search_similar(&query, 10)?;
```

## Performance

- **Latency**: <3ms for 10K vectors (k=10)
- **Memory**: 2GB for 1M vectors
- **Throughput**: 1,250 QPS single-thread, 15K QPS multi-thread

## Comparison

| Feature | GABA-Memory | Pinecone | Qdrant | Chroma |
|---------|-------------|----------|--------|--------|
| Latency | 2.5ms | 50-100ms | 20-40ms | 100-500ms |
| Cost | $0 | $70+/mo | $25+/mo | $0 |
| Brain-inspired | Yes | No | No | No |
| Local-first | Yes | No | Yes | Yes |

## Architecture

### Memory Layers

- **Working**: Short-term, high-precision
- **Episodic**: Time-stamped events
- **Semantic**: Generalized knowledge
- **Procedural**: Skills and procedures
- **Affective**: Emotional associations

### Engram States

- **Active**: Recently accessed, high plasticity
- **Latent**: Not recently accessed, low plasticity
- **Consolidated**: Stable long-term storage

## Advanced Usage

### Associative Recall

```rust
use gaba_memory::AssociativeGraph;

let graph = AssociativeGraph::new();
graph.strengthen_association(chunk_a, chunk_b, 0.8);

// Spread activation across graph
let activated = graph.spread_activation(&[chunk_a], 0.1, 3);
```

### Temporal Dynamics

```rust
use gaba_memory::TemporalMemory;

let temporal = TemporalMemory::new();
let strength = temporal.calculate_strength(&chunk, now);

if temporal.should_consolidate(&chunk, now) {
    // Move to semantic layer
}
```

### Hopfield Reranking

```rust
use gaba_memory::HopfieldLayer;

let hopfield = HopfieldLayer::new(128)
    .with_beta(1.0)
    .with_max_iterations(10);

let reranked = hopfield.rerank_candidates(&query, &candidates)?;
```

## Testing

```bash
# Run tests
cargo test -p gaba-memory

# Run benchmarks
cargo bench -p gaba-memory
```

## Documentation

- [Implementation Guide](../../docs/gaba-memory/general/IMPLEMENTATION.md)
- [Benchmarks](../../docs/gaba-memory/general/BENCHMARKS.md)
- [Market Comparison](../../docs/gaba-memory/general/MARKET_COMPARISON.md)

## Research Foundation

- Ebbinghaus forgetting curve (1885)
- Hebbian learning (1949)
- Spreading activation (Collins & Loftus, 1975)
- Modern Hopfield networks (Ramsauer et al., 2020)
- HNSW algorithm (Malkov & Yashunin, 2018)

## License

MIT OR Apache-2.0
