# GABA-BURN Documentation

## Overview

GABA-BURN is a high-performance edge ML framework built in Rust+Zig with zero Python dependencies. It provides 30 optimized models for computer vision, audio, and sensor applications with sub-millisecond inference times.

## Quick Links

- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
- [Architecture](./architecture.md)
- [Performance](./performance.md)
- [Deployment](./deployment.md)

## Core Components

### gaba-singularity
Adaptive kernel orchestration engine with Q-learning for automatic performance optimization.

### gaba-memory
Multi-modal memory system with HNSW indexing, associative graphs, and temporal consolidation.

### gaba-burn-vector
CPU-parallel vector search with SIMD optimizations.

### gaba-workflows
Pipeline configuration and execution framework.

### gaba-pqc
Post-quantum cryptography for model protection.

### gaba-train
30-model zoo with advanced training techniques.

## Features

- 30 edge-optimized models
- Sub-millisecond inference (0.64ms average)
- Minimal memory footprint (40KB average per model)
- Zero Python dependencies
- Post-quantum cryptography
- Adaptive kernel selection
- Real-time capable

## Performance

- Average inference: 0.64ms
- Average throughput: 1,800 fps
- Total memory: 3.7MB (all 30 models)
- Binary size: <10MB

## Supported Platforms

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (x86_64)
- Embedded (ARM Cortex-M, RISC-V)

## License

Dual licensed under Apache 2.0 and MIT.
