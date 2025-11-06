# API Reference

## Core Modules

### gaba-train

#### Models

**Edge Vision Models**
- `MicroYOLONano` - Object detection (47K params, 0.90ms)
- `EfficientEdgeLite` - Classification (89K params, 0.72ms)
- `SegmentMicro` - Segmentation (62K params, 1.05ms)
- `FaceDetectNano` - Face detection (38K params, 0.62ms)
- `GestureNetMicro` - Gesture recognition (28K params, 0.47ms)

**Edge Audio Models**
- `KeywordSpotMicro` - Wake word detection (15K params, 0.40ms)
- `AudioEventNano` - Audio event detection (19K params, 0.40ms)
- `SpeechEnhanceNano` - Speech enhancement (42K params, 0.68ms)
- `VoiceActivityMicro` - Voice activity detection (24K params, 0.40ms)

**Edge Sensor Models**
- `AnomalyDetectEdge` - Anomaly detection (12K params, 0.40ms)
- `TimeSeriesForecastMicro` - Time series forecasting (18K params, 0.40ms)
- `SensorFusionNano` - Multi-sensor fusion (22K params, 0.40ms)
- `HealthMonitorNano` - Vital signs monitoring (44K params, 0.69ms)

#### Training

```rust
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
}

pub fn train_traffic_model(
    data_path: &str,
    output_path: &str,
    config: TrainingConfig,
) -> Result<()>
```

#### Optimizations

```rust
// Knowledge distillation
pub fn distillation_loss<B: Backend>(
    student_logits: Tensor<B, 2>,
    teacher_logits: Tensor<B, 2>,
    temperature: f32,
    alpha: f32,
) -> Tensor<B, 1>

// Quantization
pub fn fake_quantize<B: Backend>(
    x: Tensor<B, 2>,
    bits: u8,
    symmetric: bool,
) -> Tensor<B, 2>

// Gradient clipping
pub fn clip_gradients<B: Backend>(
    gradients: Vec<Tensor<B, 2>>,
    max_norm: f32,
) -> Vec<Tensor<B, 2>>
```

### gaba-singularity

#### Adaptive Execution

```rust
pub struct SingularityEngine {
    // Adaptive kernel orchestration
}

impl SingularityEngine {
    pub fn new(config: EngineConfig) -> Self
    
    pub fn execute<F>(&mut self, operation: F) -> Result<()>
    where F: Fn() -> Result<()>
    
    pub fn get_stats(&self) -> EngineStats
}
```

### gaba-memory

#### HNSW Index

```rust
pub struct HNSWIndex {
    pub ef_construction: usize,
    pub m: usize,
}

impl HNSWIndex {
    pub fn new(ef_construction: usize, m: usize) -> Self
    
    pub fn insert(&mut self, id: usize, vector: Vec<f32>)
    
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>
}
```

#### Associative Graph

```rust
pub struct AssociativeGraph {
    // Concept association network
}

impl AssociativeGraph {
    pub fn new() -> Self
    
    pub fn add_concept(&mut self, id: usize, embedding: Vec<f32>)
    
    pub fn spread_activation(&self, source: usize, steps: usize) -> Vec<(usize, f32)>
}
```

### gaba-workflows

#### Pipeline

```rust
pub struct Pipeline {
    pub name: String,
    pub steps: Vec<PipelineStep>,
}

impl Pipeline {
    pub fn new(name: String) -> Self
    
    pub fn add_step(&mut self, step: PipelineStep)
    
    pub fn execute(&self) -> Result<()>
    
    pub fn save(&self, path: &Path) -> Result<()>
    
    pub fn load(path: &Path) -> Result<Self>
}
```

### gaba-pqc

#### Encryption

```rust
pub fn encrypt_checkpoint(
    data: &[u8],
    key: &[u8],
) -> Result<Vec<u8>>

pub fn decrypt_checkpoint(
    encrypted: &[u8],
    key: &[u8],
) -> Result<Vec<u8>>

pub fn verify_checkpoint(
    encrypted: &[u8],
    original: &[u8],
) -> Result<bool>
```

## CLI Commands

### Training

```bash
gaba-train traffic --data <PATH> --output <PATH> --epochs <N> --lr <F>
gaba-train route --data <PATH> --output <PATH> --epochs <N> --lr <F>
```

### Data Generation

```bash
gaba-train generate --output <PATH> --traffic-samples <N> --route-samples <N>
```

### Benchmarking

```bash
gaba-train benchmark-edge
gaba-train benchmark-all30
gaba-train bench --size <SIZES> --metal --zig
```

### Utilities

```bash
gaba-train info
gaba-train singularity --iterations <N> --size <N>
gaba-train convert --input <PATH> --output <PATH>
```

## Error Handling

All functions return `Result<T>` or `Result<()>` for proper error propagation.

```rust
use anyhow::Result;

fn example() -> Result<()> {
    let model = create_model()?;
    let output = model.forward(input)?;
    Ok(())
}
```

## Performance Tips

1. Use release builds: `cargo build --release`
2. Enable Metal on macOS: `--features metal`
3. Enable Zig kernels: `--features zig`
4. Batch operations when possible
5. Reuse tensors to minimize allocations
