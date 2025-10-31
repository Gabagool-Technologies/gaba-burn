# Gaba-Burn CLI Quick Start

**Note**: Detailed performance data and optimization techniques are proprietary. Contact licensing@gabagool.tech for commercial access.

## 5-Minute Setup

### 1. Build

```bash
cd gaba-burn/crates/gaba-train-cli
cargo build --release --features full
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
./target/release/gaba-train bench --size medium --metal --zig
```

## Expected Output

```
╔═══════════════════════════════════════════════════════════╗
║  GABA-BURN: Hardcore ML Training with Rust+Zig+Metal      ║
║  Zero-copy · PQC · SIMD · Apple Silicon Optimized         ║
╚═══════════════════════════════════════════════════════════╝

Benchmark size: medium
Metal acceleration: enabled
Zig kernels: enabled

Matrix dimensions: 512x512 * 512x512
Zig GEMM: 12.34ms
Performance: 21.85 GFLOPS
```

## Next Steps

- Read `GABA_TRAIN_CLI_GUIDE.md` for detailed documentation
- Check `../IMPLEMENTATION_SUMMARY.md` for architecture details
- Explore Metal shaders in `../../crates/gaba-pqc/src/shaders/`
- Review Zig kernels in `../../crates/gaba-native-kernels/native/`
