# Contributing to Gaba Burn

Welcome to Gaba Burn!

We appreciate your interest in contributing to our Rust+Zig ML framework for route optimization.

## Focus Areas

Gaba Burn is a performance-first fork of Burn focused on:
- Rust+Zig native kernels for CPU performance
- ML training without Python dependencies
- Edge deployment and route optimization
- Famiglia Routes integration

## How to Contribute

### 1. Review Open Issues

Check the issue tracker for current work and planned features.

### 2. Understand the Architecture

Key components:
- `gaba-native-kernels`: Zig kernels with Rust fallbacks
- `gaba-train-cli`: Pure Rust ML training
- `burn-inference`: ONNX model loading
- `MLRouteOptimizer`: Route optimization with ML

### 3. Fork and Clone

```bash
git clone https://github.com/gabagool-technologies/gaba-burn
cd gaba-burn
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 5. Make Changes

Follow existing code style. Add tests for new features.

### 6. Test Your Changes

```bash
# Run tests
cargo test --workspace

# Check compilation
cargo check --workspace

# Run benchmarks (if applicable)
cargo bench -p gaba-native-kernels --features zig
```

### 7. Submit Pull Request

- Clear description of changes
- Reference related issues
- Include benchmark results for performance changes
- Ensure all tests pass

## Code Standards

- No emojis in code or commits
- Minimal documentation (code should be self-explanatory)
- Performance-first approach
- Zig for hot paths, Rust for safety

## Areas We Welcome Contributions

- Zig kernel optimizations
- ML training improvements
- Route optimization algorithms
- Edge deployment features
- Performance benchmarks
- Bug fixes

## License

By contributing, you agree your contributions will be licensed under MIT OR Apache-2.0.

## Questions?

Open an issue for discussion before starting major work.

---

Built with eternal love and care by Gabagool Technologies
