use anyhow::Result;
use std::fs;
use std::path::Path;

const CARGO_TOML_TEMPLATE: &str = r#"[package]
name = "{{PROJECT_NAME}}"
version = "0.1.0"
edition = "2021"
description = "{{PROJECT_DESCRIPTION}}"

[dependencies]
gaba-burn = { path = "../../gaba-burn" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[[bin]]
name = "{{PROJECT_NAME}}"
path = "src/main.rs"
"#;

const MAIN_RS_TEMPLATE: &str = r#"use gaba_burn::prelude::*;

fn main() {
    println!("Gaba ML Project: {{PROJECT_NAME}}");
    
    // Initialize Gaba-Burn backend
    let backend = Backend::default();
    println!("Backend initialized: {:?}", backend);
}
"#;

const MODEL_RS_TEMPLATE: &str = r#"use gaba_burn::prelude::*;

pub struct Model {
    // Define your model architecture here
}

impl Model {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Implement forward pass
        input.clone()
    }
}
"#;

const DATASET_RS_TEMPLATE: &str = r#"use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub features: Vec<f32>,
    pub label: f32,
}

impl Dataset {
    pub fn load_csv(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        // Implement CSV loading
        Ok(Self { samples: vec![] })
    }
    
    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;
        let train = Self {
            samples: self.samples[..split_idx].to_vec(),
        };
        let test = Self {
            samples: self.samples[split_idx..].to_vec(),
        };
        (train, test)
    }
}
"#;

const TRAINING_RS_TEMPLATE: &str = r#"use gaba_burn::prelude::*;
use crate::model::Model;
use crate::dataset::Dataset;

pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
        }
    }
}

pub fn train(
    model: &mut Model,
    dataset: &Dataset,
    config: &TrainingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting training...");
    println!("Epochs: {}", config.epochs);
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    
    for epoch in 0..config.epochs {
        // Implement training loop
        if epoch % 10 == 0 {
            println!("Epoch {}/{}", epoch, config.epochs);
        }
    }
    
    println!("Training complete!");
    Ok(())
}
"#;

const EXPORT_RS_TEMPLATE: &str = r#"use crate::model::Model;
use std::path::Path;

pub enum ExportFormat {
    RustBinary,
    Wasm,
    Onnx,
    Embedded,
}

pub fn export_model(
    model: &Model,
    format: ExportFormat,
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        ExportFormat::RustBinary => {
            println!("Exporting as Rust binary...");
            // Implement Rust binary export
        }
        ExportFormat::Wasm => {
            println!("Exporting as WebAssembly...");
            // Implement WASM export
        }
        ExportFormat::Onnx => {
            println!("Exporting as ONNX...");
            // Implement ONNX export
        }
        ExportFormat::Embedded => {
            println!("Exporting for embedded...");
            // Implement embedded export
        }
    }
    Ok(())
}
"#;

const README_TEMPLATE: &str = r#"# {{PROJECT_NAME}}

Gaba ML Project built with Gaba-Burn framework.

## Structure

- `src/main.rs` - Entry point
- `src/model.rs` - Model architecture
- `src/dataset.rs` - Dataset handling
- `src/training.rs` - Training logic
- `src/export.rs` - Model export utilities
- `data/` - Training datasets
- `models/` - Saved models
- `tests/` - Test files

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run
```

## Features

- Rust+Zig ML framework (Gaba-Burn)
- Metal GPU acceleration (macOS)
- Post-quantum cryptography
- Multiple export formats (Rust, WASM, ONNX, Embedded)
"#;

const GITIGNORE_TEMPLATE: &str = r#"target/
Cargo.lock
*.gaba
*.encrypted
.DS_Store
"#;

const WORKSPACE_METADATA_TEMPLATE: &str = r#"{
  "version": "1.0",
  "type": "gaba-ml",
  "created": "{{TIMESTAMP}}",
  "name": "{{PROJECT_NAME}}",
  "description": "{{PROJECT_DESCRIPTION}}"
}
"#;

pub fn create_workspace(base_path: &Path, project_name: &str, description: &str) -> Result<()> {
    validate_project_name(project_name)?;

    let workspace_path = base_path.join(project_name);

    // Create directory structure
    fs::create_dir_all(&workspace_path)?;
    fs::create_dir_all(workspace_path.join("src"))?;
    fs::create_dir_all(workspace_path.join("data"))?;
    fs::create_dir_all(workspace_path.join("models"))?;
    fs::create_dir_all(workspace_path.join("tests"))?;

    let timestamp = chrono::Utc::now().to_rfc3339();

    // Write files
    write_template_with_desc(
        &workspace_path.join("Cargo.toml"),
        CARGO_TOML_TEMPLATE,
        project_name,
        description,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("src/main.rs"),
        MAIN_RS_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("src/model.rs"),
        MODEL_RS_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("src/dataset.rs"),
        DATASET_RS_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("src/training.rs"),
        TRAINING_RS_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("src/export.rs"),
        EXPORT_RS_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join("README.md"),
        README_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template(
        &workspace_path.join(".gitignore"),
        GITIGNORE_TEMPLATE,
        project_name,
        &timestamp,
    )?;
    write_template_with_desc(
        &workspace_path.join(".gaba-workspace"),
        WORKSPACE_METADATA_TEMPLATE,
        project_name,
        description,
        &timestamp,
    )?;

    // Create .gitkeep files
    fs::write(workspace_path.join("data/.gitkeep"), "")?;
    fs::write(workspace_path.join("models/.gitkeep"), "")?;
    fs::write(workspace_path.join("tests/.gitkeep"), "")?;

    Ok(())
}

fn write_template(path: &Path, template: &str, project_name: &str, timestamp: &str) -> Result<()> {
    let content = template
        .replace("{{PROJECT_NAME}}", project_name)
        .replace("{{TIMESTAMP}}", timestamp);
    fs::write(path, content)?;
    Ok(())
}

fn write_template_with_desc(
    path: &Path,
    template: &str,
    project_name: &str,
    description: &str,
    timestamp: &str,
) -> Result<()> {
    let content = template
        .replace("{{PROJECT_NAME}}", project_name)
        .replace("{{PROJECT_DESCRIPTION}}", description)
        .replace("{{TIMESTAMP}}", timestamp);
    fs::write(path, content)?;
    Ok(())
}

fn validate_project_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Project name cannot be empty");
    }

    if name.len() > 64 {
        anyhow::bail!("Project name too long (max 64 characters)");
    }

    let first_char = name.chars().next().unwrap();
    if !first_char.is_alphabetic() {
        anyhow::bail!("Project name must start with a letter");
    }

    for c in name.chars() {
        if !c.is_alphanumeric() && c != '-' && c != '_' {
            anyhow::bail!(
                "Project name can only contain letters, numbers, hyphens, and underscores"
            );
        }
    }

    Ok(())
}
