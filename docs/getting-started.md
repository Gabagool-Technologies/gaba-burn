# Getting Started with GABA-BURN

## Installation

### Prerequisites

- Rust 1.70 or later
- Zig 0.11 or later (optional, for native kernels)
- Metal SDK (optional, for GPU acceleration on macOS)

### Build from Source

```bash
git clone https://github.com/your-org/gaba-burn
cd gaba-burn/gaba-burn
cargo build --release
```

### Install CLI

```bash
cargo install --path crates/gaba-train-cli
```

## Quick Start

### 1. Generate Training Data

```bash
gaba-train generate --output ./data --traffic-samples 10000 --route-samples 5000
```

### 2. Train a Model

```bash
gaba-train traffic \
  --data ./data/traffic.csv \
  --output ./models \
  --epochs 100 \
  --lr 0.01
```

### 3. Run Benchmarks

```bash
gaba-train benchmark-all30
```

### 4. Test Singularity Engine

```bash
gaba-train singularity --iterations 100 --size 256
```

## Using in Your Project

Add to `Cargo.toml`:

```toml
[dependencies]
gaba-train = "0.1.0"
gaba-singularity = "0.1.0"
gaba-memory = "0.1.0"
```

### Basic Usage

```rust
use gaba_train::models_edge::*;
use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn main() {
    let device = Default::default();
    
    // Create model
    let model = MicroYOLONano::<Backend>::new(&device);
    
    // Prepare input
    let input: Tensor<Backend, 4> = Tensor::random(
        [1, 3, 96, 96],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device
    );
    
    // Run inference
    let output = model.forward(input);
    
    println!("Detection output: {:?}", output.dims());
}
```

## Next Steps

- Read the [Architecture Guide](./architecture.md)
- Explore [API Reference](./api-reference.md)
- Check [Performance Tuning](./performance.md)
- See [Deployment Guide](./deployment.md)
