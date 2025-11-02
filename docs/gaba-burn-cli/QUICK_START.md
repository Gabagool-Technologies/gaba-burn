# Gaba-Burn CLI Quick Start

**Note**: Detailed performance data and optimization techniques are proprietary. Contact licensing@gabagool.tech for commercial access.

## 5-Minute Setup

### 1. Build

```bash
cd gaba-burn
cargo build --release --package gaba-train-cli --features "pqc,zig"
```

### 2. Generate Training Data

```bash
./target/release/gaba-train generate \
  --output ./data \
  --traffic-samples 50000 \
  --route-samples 5000
```

### 3. Train Model

```bash
./target/release/gaba-train traffic \
  --data ./data/traffic.csv \
  --output ./models \
  --epochs 50 \
  --lr 0.01
```

### 4. Encrypt Model (Optional)

```bash
./target/release/gaba-train encrypt \
  --model ./models/model.bin \
  --output ./models/model.encrypted
```

### 5. Benchmark Performance

```bash
./target/release/gaba-train bench --size 128
```

## Expected Output

```
Running performance benchmark...
Benchmark configuration:
  Size: 128
  Metal GPU: disabled
  Large matrices: no

Running benchmarks...

  128x128x128 - RustVectorized
    Time: 1.47 ms
    Performance: 2.86 GFLOPS

  256x256x256 - Quantized
    Time: 9.08 ms
    Performance: 3.70 GFLOPS

  512x512x512 - RustVectorized
    Time: 88.47 ms
    Performance: 3.03 GFLOPS

Benchmark complete!
```

## Additional Commands

### System Information

```bash
./target/release/gaba-train info
```

### Singularity Engine Demo

```bash
./target/release/gaba-train singularity --iterations 100 --size 128
```

## Next Steps

- Read `CLI_USER_GUIDE.md` for detailed documentation
- Check `../model-training/` for training guides
- Explore Zig kernels in `../../crates/gaba-native-kernels/native/`
- Review benchmarks in `../benchmarks/`
