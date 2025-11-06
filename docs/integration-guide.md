# GABA-BURN Integration Guide

## Overview

GABA-BURN provides multiple integration paths for different use cases and programming languages.

## Integration Methods

### 1. REST API (Recommended for Cross-Language)

**Best for**: Web apps, mobile apps, any language with HTTP client

```bash
# Start API server
gaba-train serve --port 3000 --host 0.0.0.0
```

**Endpoints**:

```http
GET /health
GET /models
GET /models/:name
POST /infer
GET /metrics
```

**Example (JavaScript)**:
```javascript
// List models
const models = await fetch('http://localhost:3000/models').then(r => r.json());

// Run inference
const response = await fetch('http://localhost:3000/infer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'MicroYOLONano',
    input: [[0.1, 0.2, 0.3, ...]]
  })
});
const result = await response.json();
```

**Example (Python)**:
```python
import requests

# List models
models = requests.get('http://localhost:3000/models').json()

# Run inference
response = requests.post('http://localhost:3000/infer', json={
    'model': 'MicroYOLONano',
    'input': [[0.1, 0.2, 0.3, ...]]
})
result = response.json()
```

### 2. CLI Integration

**Best for**: Scripts, automation, CI/CD

```bash
# Generate datasets
gaba-train generate --output ./data --traffic-samples 10000

# Train model
gaba-train traffic --data ./data/traffic.csv --output ./models --epochs 100

# Benchmark
gaba-train benchmark-all30

# Export results
gaba-train benchmark-all30 > results.txt
```

**Shell Script Example**:
```bash
#!/bin/bash
set -e

# Training pipeline
gaba-train generate --output ./data --traffic-samples 50000
gaba-train traffic --data ./data/traffic.csv --output ./models --epochs 200
gaba-train benchmark-edge

echo "Training complete"
```

### 3. Rust Library Integration

**Best for**: Rust applications, maximum performance

```toml
[dependencies]
gaba-train = { path = "../gaba-burn/crates/gaba-train" }
gaba-singularity = { path = "../gaba-burn/crates/gaba-singularity" }
gaba-memory = { path = "../gaba-burn/crates/gaba-memory" }
burn = { version = "0.15", features = ["ndarray"] }
```

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

### 4. FFI Integration (C/C++/Other Languages)

**Best for**: Legacy systems, embedded, non-Rust languages

Create C-compatible wrapper:

```rust
// gaba-ffi/src/lib.rs
use std::os::raw::c_float;
use gaba_train::models_edge::*;

#[no_mangle]
pub extern "C" fn gaba_yolo_create() -> *mut MicroYOLONano<NdArray> {
    let device = Default::default();
    Box::into_raw(Box::new(MicroYOLONano::new(&device)))
}

#[no_mangle]
pub extern "C" fn gaba_yolo_infer(
    model: *mut MicroYOLONano<NdArray>,
    input: *const c_float,
    input_len: usize,
    output: *mut c_float,
) -> i32 {
    // Implementation
    0
}

#[no_mangle]
pub extern "C" fn gaba_yolo_destroy(model: *mut MicroYOLONano<NdArray>) {
    unsafe { drop(Box::from_raw(model)) }
}
```

**C Header**:
```c
// gaba.h
typedef struct GabaModel GabaModel;

GabaModel* gaba_yolo_create(void);
int gaba_yolo_infer(GabaModel* model, const float* input, size_t input_len, float* output);
void gaba_yolo_destroy(GabaModel* model);
```

### 5. WebAssembly Integration

**Best for**: Browser, serverless

```toml
[dependencies]
wasm-bindgen = "0.2"
```

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GabaModel {
    // Model implementation
}

#[wasm_bindgen]
impl GabaModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }
    
    #[wasm_bindgen]
    pub fn infer(&self, input: Vec<f32>) -> Vec<f32> {
        // Inference implementation
        input
    }
}
```

## GABA-ML-Forge Integration

### Architecture

```
┌─────────────────────────────────────────┐
│         GABA-ML-Forge (Tauri)           │
│  ┌───────────────────────────────────┐  │
│  │  React Frontend (TypeScript)      │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌───────────────────────────────────┐  │
│  │  Tauri Backend (Rust)             │  │
│  │  - IPC Commands                   │  │
│  │  - State Management               │  │
│  └───────────────────────────────────┘  │
│              │                           │
└──────────────┼───────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         GABA-BURN Core                  │
│  - Model Zoo (30 models)                │
│  - Singularity Engine                   │
│  - Memory Systems                       │
│  - Training Pipeline                    │
└─────────────────────────────────────────┘
```

### Tauri Integration

**1. Add GABA-BURN to Tauri Backend**:

```toml
# src-tauri/Cargo.toml
[dependencies]
gaba-train = { path = "../../gaba-burn/crates/gaba-train" }
gaba-singularity = { path = "../../gaba-burn/crates/gaba-singularity" }
```

**2. Create Tauri Commands**:

```rust
// src-tauri/src/main.rs
use tauri::State;
use gaba_train::models_edge::*;

#[derive(Default)]
struct AppState {
    // Model cache
}

#[tauri::command]
async fn list_models() -> Result<Vec<String>, String> {
    Ok(vec![
        "MicroYOLONano".to_string(),
        "EfficientEdgeLite".to_string(),
        // ... all 30 models
    ])
}

#[tauri::command]
async fn run_inference(
    model_name: String,
    input: Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>, String> {
    // Run inference
    Ok(input)
}

#[tauri::command]
async fn benchmark_model(model_name: String) -> Result<f64, String> {
    // Benchmark model
    Ok(0.64)
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_models,
            run_inference,
            benchmark_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**3. Frontend Integration (TypeScript)**:

```typescript
// src/services/gabaService.ts
import { invoke } from '@tauri-apps/api/tauri';

export interface ModelInfo {
  name: string;
  params: number;
  inferenceTimeMs: number;
}

export class GabaService {
  async listModels(): Promise<string[]> {
    return await invoke('list_models');
  }
  
  async runInference(modelName: string, input: number[][]): Promise<number[][]> {
    return await invoke('run_inference', { modelName, input });
  }
  
  async benchmarkModel(modelName: string): Promise<number> {
    return await invoke('benchmark_model', { modelName });
  }
}

export const gabaService = new GabaService();
```

**4. React Component Example**:

```typescript
// src/components/ModelBenchmark.tsx
import React, { useState, useEffect } from 'react';
import { gabaService } from '../services/gabaService';

export const ModelBenchmark: React.FC = () => {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [benchmarkResult, setBenchmarkResult] = useState<number | null>(null);
  
  useEffect(() => {
    gabaService.listModels().then(setModels);
  }, []);
  
  const runBenchmark = async () => {
    if (selectedModel) {
      const result = await gabaService.benchmarkModel(selectedModel);
      setBenchmarkResult(result);
    }
  };
  
  return (
    <div>
      <h2>Model Benchmark</h2>
      <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
        {models.map(m => <option key={m} value={m}>{m}</option>)}
      </select>
      <button onClick={runBenchmark}>Run Benchmark</button>
      {benchmarkResult && <p>Inference time: {benchmarkResult}ms</p>}
    </div>
  );
};
```

## API Design Best Practices

### 1. Versioning

```
/v1/models
/v1/infer
/v2/models  # Future version
```

### 2. Error Handling

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'InvalidModel' not found",
    "details": {
      "available_models": ["MicroYOLONano", "EfficientEdgeLite"]
    }
  }
}
```

### 3. Rate Limiting

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1699564800
```

### 4. Authentication

```http
Authorization: Bearer <token>
```

### 5. Pagination

```http
GET /models?page=1&limit=10
```

## Performance Optimization

### 1. Batch Requests

```json
{
  "requests": [
    {"model": "MicroYOLONano", "input": [...]},
    {"model": "EfficientEdgeLite", "input": [...]}
  ]
}
```

### 2. Caching

- Model caching: Load once, reuse
- Result caching: Cache common inputs
- Connection pooling: Reuse HTTP connections

### 3. Async Processing

```javascript
// Non-blocking inference
const jobId = await fetch('/infer/async', {
  method: 'POST',
  body: JSON.stringify({model: 'MicroYOLONano', input: [...]})
}).then(r => r.json());

// Poll for results
const result = await fetch(`/jobs/${jobId}`).then(r => r.json());
```

## Security Considerations

1. **Input Validation**: Validate all inputs
2. **Rate Limiting**: Prevent abuse
3. **Authentication**: Secure endpoints
4. **HTTPS**: Use TLS in production
5. **CORS**: Configure properly
6. **Sanitization**: Clean user inputs

## Monitoring

```rust
use gaba_train::monitoring::Metrics;

let mut metrics = Metrics::new();
metrics.increment_counter("requests_total", 1);
metrics.record_duration("inference_time", duration);
metrics.print_summary();
```

## Next Steps

1. Choose integration method
2. Implement authentication
3. Add monitoring
4. Deploy to production
5. Scale as needed
