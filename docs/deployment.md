# Deployment Guide

## Production Deployment

### Building for Production

```bash
# Optimized release build
cargo build --release --features "metal,zig,pqc"

# Strip binary
strip target/release/gaba-train

# Verify size
ls -lh target/release/gaba-train
```

### Cross-Compilation

#### Linux ARM64

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

#### Embedded ARM

```bash
rustup target add thumbv7em-none-eabihf
cargo build --release --target thumbv7em-none-eabihf --no-default-features
```

## Container Deployment

### Dockerfile

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p gaba-train-cli

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/gaba-train /usr/local/bin/
CMD ["gaba-train", "info"]
```

### Build and Run

```bash
docker build -t gaba-burn:latest .
docker run gaba-burn:latest gaba-train benchmark-all30
```

## Edge Device Deployment

### Raspberry Pi

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build --release

# Run
./target/release/gaba-train info
```

### NVIDIA Jetson

```bash
# Enable CUDA support
cargo build --release --features cuda

# Deploy
scp target/release/gaba-train jetson@device:/usr/local/bin/
```

## Cloud Deployment

### AWS Lambda

Package as Lambda layer with custom runtime.

### Google Cloud Run

Deploy containerized version with auto-scaling.

### Azure Container Instances

Use Docker image for serverless deployment.

## Monitoring

### Logging

```rust
use log::{info, warn, error};

info!("Model loaded: {} params", params);
warn!("High memory usage: {}MB", mem_mb);
error!("Inference failed: {}", err);
```

### Metrics

```rust
use std::time::Instant;

let start = Instant::now();
let output = model.forward(input);
let duration = start.elapsed();

println!("Inference time: {:?}", duration);
```

## Performance Tuning

### CPU Optimization

```bash
# Set thread count
export RAYON_NUM_THREADS=4

# Enable CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Memory Optimization

```rust
// Reuse tensors
let mut buffer = Tensor::zeros([batch, features], &device);
for batch in batches {
    buffer = model.forward(batch);
}
```

### GPU Optimization

```bash
# Enable Metal on macOS
cargo build --release --features metal

# Verify GPU usage
gaba-train bench --metal --large
```

## Security

### Model Encryption

```bash
# Encrypt model
gaba-train encrypt --model model.bin --output model.enc

# Verify integrity
gaba-train verify --encrypted model.enc --original model.bin
```

### Network Security

- Use TLS for model downloads
- Verify checksums
- Implement rate limiting
- Use API keys for access control

## Troubleshooting

### High Memory Usage

- Reduce batch size
- Enable gradient checkpointing
- Use quantization

### Slow Inference

- Enable Metal/CUDA
- Use Zig kernels
- Optimize batch size
- Check CPU governor

### Build Failures

- Update Rust: `rustup update`
- Clean build: `cargo clean`
- Check dependencies: `cargo tree`

## Production Checklist

- [ ] Release build with optimizations
- [ ] Strip debug symbols
- [ ] Enable relevant features
- [ ] Test on target hardware
- [ ] Set up monitoring
- [ ] Configure logging
- [ ] Implement error handling
- [ ] Add health checks
- [ ] Document deployment
- [ ] Create rollback plan
