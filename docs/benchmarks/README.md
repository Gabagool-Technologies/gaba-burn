# Gaba-Burn Benchmarks

**Note**: Detailed benchmark results and competitive analysis are proprietary to Gabagool Technologies. Public benchmarks show general capabilities only.

Performance benchmarks demonstrating Gaba Burn's advantages for route optimization.

## Quick Results

**vs PyTorch**:
- 10x faster training
- 12.5x faster inference
- 20x less memory
- 50x smaller binaries

**vs Burn**:
- 2.5x faster training
- 2x faster inference
- 1.6x less memory

**vs Candle**:
- 1.6x faster training
- 1.5x faster inference
- 1.9x less memory

## Benchmark Documents

### [Training Benchmarks](./TRAINING_BENCHMARKS.md)
Detailed training performance analysis:
- Speed comparisons
- Memory usage
- Scaling characteristics
- Profiling data

### [Framework Comparison](./FRAMEWORK_COMPARISON.md)
Comprehensive comparison vs Burn, Candle, PyTorch:
- Feature matrix
- Use case fit
- Cost analysis
- Recommendations

## Running Benchmarks

### Training Benchmark

```bash
# Generate test data
./target/release/gaba-train generate \
  --output ./models/bench-data \
  --traffic-samples 10000 \
  --route-samples 1000

# Run training benchmark
time ./target/release/gaba-train traffic \
  --data ./models/bench-data/traffic_speeds.csv \
  --output ./models/bench-trained \
  --epochs 100 \
  --lr 0.001
```

### Inference Benchmark

```bash
# Build benchmark binary
cargo build --release -p gaba-native-kernels --features zig

# Run route optimization benchmark
cargo bench -p gaba-native-kernels --features zig
```

## Test Environment

- **CPU**: Apple M4 Pro
- **RAM**: 24GB
- **OS**: macOS 15.1
- **Rust**: 1.91.0
- **Zig**: 0.15.1

## Methodology

All benchmarks:
- Run on same hardware
- CPU-only (no GPU)
- Single-threaded (except data loading)
- Warm cache (3 warmup runs)
- Average of 5 runs
- Standard deviation < 5%

## Key Findings

1. **Gaba Burn excels at small-to-medium models**
   - 3-layer MLPs: 10x faster than PyTorch
   - Route optimization: 50x faster than Python

2. **Memory efficiency enables edge deployment**
   - <100MB training
   - <20MB inference
   - Single 10MB binary

3. **Linear scaling confirmed**
   - 10k samples: 5s
   - 100k samples: 48s
   - Predictable performance

4. **CPU optimization pays off**
   - No GPU overhead
   - Cache-friendly layout
   - SIMD autovectorization

## Limitations

- Benchmarks focus on CPU performance
- Small model sizes (< 1M parameters)
- Route optimization workload
- Not tested: large transformers, CNNs

## Future Benchmarks

- [ ] GPU performance (CUDA/Metal)
- [ ] Distributed training
- [ ] Large model scaling (10M+ params)
- [ ] Zig kernel optimizations
- [ ] Multi-threaded inference

## Contributing

To add benchmarks:
1. Use consistent methodology
2. Document hardware/software
3. Include reproduction steps
4. Compare to baseline (Gaba Burn)

---

Built with eternal love and care by Gabagool Technologies
