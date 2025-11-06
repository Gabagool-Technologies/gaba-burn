# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/gaba-burn
cd gaba-burn/gaba-burn

# Build
cargo build --release

# Install CLI
cargo install --path crates/gaba-train-cli
```

## First Steps

### 1. Check System Info

```bash
gaba-train info
```

### 2. Run Benchmarks

```bash
# Edge models
gaba-train benchmark-edge

# All 30 models
gaba-train benchmark-all30

# Singularity engine
gaba-train singularity --iterations 100 --size 256
```

### 3. Generate Data

```bash
gaba-train generate \
  --output ./data \
  --traffic-samples 10000 \
  --route-samples 5000
```

### 4. Train Model

```bash
gaba-train traffic \
  --data ./data/traffic.csv \
  --output ./models \
  --epochs 100 \
  --lr 0.01
```

## Using in Code

### Basic Inference

```rust
use gaba_train::models_edge::*;
use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();
    let model = MicroYOLONano::<Backend>::new(&device);
    
    let input: Tensor<Backend, 4> = Tensor::random(
        [1, 3, 96, 96],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    let output = model.forward(input);
    println!("Output: {:?}", output.dims());
}
```

### With Singularity

```rust
use gaba_singularity::{SingularityEngine, EngineConfig};

fn main() {
    let config = EngineConfig::default();
    let mut engine = SingularityEngine::new(config);
    
    let result = engine.adaptive_gemm(256, 256, 256);
    println!("Optimized execution complete");
}
```

### With Memory

```rust
use gaba_memory::HNSWIndex;

fn main() {
    let mut index = HNSWIndex::new(16, 32);
    
    for i in 0..100 {
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32).sin()).collect();
        index.insert(i, vector);
    }
    
    let query: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
    let results = index.search(&query, 5);
    
    println!("Found {} results", results.len());
}
```

## Next Steps

- Read [Architecture Guide](../architecture.md)
- Explore [API Reference](../api-reference.md)
- Check [Performance Tuning](../performance.md)
- See [Deployment Guide](../deployment.md)
- Try [Examples](../../examples/gaba/)

## Common Issues

### Build Fails

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### Tests Fail

```bash
# Run specific tests
cargo test -p gaba-memory --lib
cargo test -p gaba-singularity --test integration_test
```

### Slow Performance

```bash
# Enable optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Use GPU
cargo build --release --features metal
```

## Getting Help

- Documentation: `docs/`
- Examples: `examples/gaba/`
- Issues: GitHub Issues
- Discussions: GitHub Discussions
