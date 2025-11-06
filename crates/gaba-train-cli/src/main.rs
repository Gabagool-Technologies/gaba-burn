//! Gaba Train CLI: Hardcore Rust+Zig ML training with PQC and Metal acceleration
//!
//! Features:
//! - Zero-copy Metal GPU acceleration on Apple Silicon
//! - Post-quantum cryptography for model protection
//! - SIMD-optimized Zig kernels for critical paths
//! - Unified memory architecture for CPU/GPU/Neural Engine

mod adam;
mod augmentation;
mod commands;
mod data;
mod dataset_generator;
mod dataset_generator_ext;
mod demand_forecasting;
mod driver_behavior;
mod federated_training;
mod metrics;
mod models;
mod onnx_export;
mod realtime_updates;
mod train;
mod vehicle_maintenance;
mod workspace;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gaba-train")]
#[command(about = "GABA-BURN: High-performance edge ML training and inference")]
#[command(long_about = "GABA-BURN ML Training CLI\n\n\
    Pure Rust+Zig ML framework with 30 optimized edge models.\n\
    Zero Python dependencies, sub-millisecond inference, production-ready.\n\n\
    Examples:\n  \
    gaba-train benchmark-all30                    # Benchmark all 30 models\n  \
    gaba-train traffic -d data.csv -o models/     # Train traffic model\n  \
    gaba-train generate -o ./data --traffic-samples 10000  # Generate datasets\n  \
    gaba-train singularity --iterations 100       # Test adaptive engine\n\n\
    Documentation: docs/\n\
    API Reference: cargo doc --no-deps --open")]
#[command(version)]
#[command(author)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train traffic speed prediction model
    Traffic {
        /// Input CSV file
        #[arg(short, long)]
        data: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of epochs
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 0.01)]
        lr: f32,
    },

    /// Train route time prediction model
    Route {
        /// Input CSV file
        #[arg(short, long)]
        data: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of epochs
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 0.01)]
        lr: f32,
    },

    /// Generate synthetic training data
    Generate {
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of traffic samples
        #[arg(long, default_value_t = 100000)]
        traffic_samples: usize,

        /// Number of route samples
        #[arg(long, default_value_t = 10000)]
        route_samples: usize,
    },

    /// Convert Python datasets to Rust format
    Convert {
        /// Input directory (py-models)
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory (datasets)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Encrypt model with post-quantum cryptography
    #[cfg(feature = "pqc")]
    Encrypt {
        /// Model file to encrypt
        #[arg(short, long)]
        model: PathBuf,

        /// Output encrypted file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Verify encrypted model integrity
    #[cfg(feature = "pqc")]
    Verify {
        /// Encrypted model file
        #[arg(short, long)]
        encrypted: PathBuf,

        /// Original model file
        #[arg(short, long)]
        original: PathBuf,
    },

    /// Benchmark training performance
    Bench {
        /// Matrix size
        #[arg(short, long, default_value = "128,256,512")]
        size: String,

        /// Enable Metal GPU acceleration
        #[arg(long)]
        metal: bool,

        /// Enable Zig kernels
        #[arg(long)]
        zig: bool,

        /// Test large matrices for Metal GPU
        #[arg(long)]
        large: bool,
    },

    /// Show system capabilities
    Info,
    
    /// Benchmark edge ML models
    BenchmarkEdge,
    
    /// Benchmark all 30 advanced models
    BenchmarkAll30,
    
    /// Start REST API server for model inference
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
        
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Run singularity engine demo
    Singularity {
        /// Number of iterations
        #[arg(long, default_value_t = 100)]
        iterations: usize,

        /// Matrix size
        #[arg(long, default_value_t = 128)]
        size: usize,
    },

    /// Create new Gaba ML workspace
    New {
        /// Workspace path
        #[arg(short, long)]
        path: PathBuf,

        /// Project name
        #[arg(short, long)]
        name: String,
    },

    /// Initialize workspace in current directory
    Init {
        /// Project name
        #[arg(short, long)]
        name: String,
    },

    /// Train router model (10M parameters) for task routing
    TrainRouter {
        /// Output model path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of epochs
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 0.01)]
        lr: f32,

        /// Number of training samples
        #[arg(long, default_value_t = 10000)]
        samples: usize,
    },

    /// Test router model accuracy
    TestRouter {
        /// Model path
        #[arg(short, long)]
        model: PathBuf,
    },

    /// Benchmark router model performance
    BenchRouter {
        /// Model path
        #[arg(short, long)]
        model: PathBuf,

        /// Number of iterations
        #[arg(long, default_value_t = 10000)]
        iterations: usize,
    },

    /// Run ML training workflow
    Workflow(commands::workflow::WorkflowArgs),

    /// Profile training performance
    Profile(commands::profile::ProfileArgs),

    /// Optimize model with quantization and pruning
    Optimize(commands::optimize::OptimizeArgs),

}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Traffic {
            data,
            output,
            epochs,
            lr,
        } => {
            println!(
                "{}",
                "Training traffic speed prediction model...".green().bold()
            );
            train::train_traffic_model(&data, &output, epochs, lr)?;
            println!("{}", "✓ Training complete!".green().bold());
        }

        Commands::Route {
            data,
            output,
            epochs,
            lr,
        } => {
            println!(
                "{}",
                "Training route time prediction model...".green().bold()
            );
            train::train_route_model(&data, &output, epochs, lr)?;
            println!("{}", "✓ Training complete!".green().bold());
        }

        Commands::Generate {
            output,
            traffic_samples,
            route_samples,
        } => {
            println!("{}", "Generating synthetic training data...".cyan());
            dataset_generator_ext::generate_all(&output, traffic_samples, route_samples)?;
            println!("{}", "✓ Data generation complete!".green().bold());
        }

        Commands::Convert { input, output } => {
            println!("Converting datasets to Rust format...");
            commands::convert_datasets(&input, &output)?;
            println!("Dataset conversion complete!");
        }

        Commands::BenchmarkEdge => {
            println!("Running edge ML model benchmarks...\n");
            let results = commands::benchmark_all_models()?;
            commands::benchmark_edge::print_benchmark_results(&results);

            // Save results to file
            let output_path = "docs/EDGE_MODEL_BENCHMARKS.md";
            std::fs::create_dir_all("docs")?;
            let mut output = String::new();
            output.push_str("# GABA-BURN Edge ML Model Benchmarks\n\n");
            output.push_str("## Test Configuration\n\n");
            output.push_str("- Platform: Rust + Burn (NdArray backend)\n");
            output.push_str("- Optimization: Release mode\n");
            output.push_str("- Iterations: 100 per model\n\n");
            output.push_str("## Results\n\n");
            output.push_str("| Model | Params | Inference (ms) | RAM (MB) | Throughput (fps) |\n");
            output.push_str("|-------|--------|----------------|----------|------------------|\n");

            for result in &results {
                output.push_str(&format!(
                    "| {} | {}K | {:.2} | {:.2} | {:.1} |\n",
                    result.model_name,
                    result.params / 1000,
                    result.inference_time_ms,
                    result.memory_mb,
                    result.throughput_samples_per_sec
                ));
            }

            std::fs::write(output_path, output)?;
            println!("\nBenchmark results saved to {}", output_path);
        }
        
        Commands::BenchmarkAll30 => {
            println!("Running comprehensive benchmark of all 30 advanced models...\n");
            let results = commands::benchmark_all_30_models()?;
            
            // Print summary
            println!("\n=== Benchmark Summary ===");
            println!("Total models: {}", results.len());
            let avg_inference: f64 = results.iter().map(|r| r.inference_time_ms).sum::<f64>() / results.len() as f64;
            let total_params: usize = results.iter().map(|r| r.params).sum();
            println!("Average inference time: {:.2} ms", avg_inference);
            println!("Total parameters: {}M", total_params / 1_000_000);
            
            // Save detailed results
            let output_path = "docs/BENCHMARK_ALL_30_MODELS.md";
            std::fs::create_dir_all("docs")?;
            let mut output = String::new();
            output.push_str("# GABA-BURN 30-Model Comprehensive Benchmark\n\n");
            output.push_str("## Configuration\n\n");
            output.push_str("- Platform: Rust + Burn (NdArray backend)\n");
            output.push_str("- Optimization: Release mode\n");
            output.push_str("- Iterations: 100 per model\n\n");
            
            let mut by_category: std::collections::HashMap<String, Vec<&commands::benchmark_all_30::BenchmarkResult>> = std::collections::HashMap::new();
            for result in &results {
                by_category.entry(result.category.clone()).or_insert_with(Vec::new).push(result);
            }
            
            for (category, models) in by_category.iter() {
                output.push_str(&format!("\n## {}\n\n", category));
                output.push_str("| Model | Params | Inference (ms) | RAM (MB) | Throughput (fps) | Optimizations |\n");
                output.push_str("|-------|--------|----------------|----------|------------------|---------------|\n");
                
                for result in models {
                    output.push_str(&format!(
                        "| {} | {}K | {:.2} | {:.2} | {:.1} | {} |\n",
                        result.model_name,
                        result.params / 1000,
                        result.inference_time_ms,
                        result.memory_mb,
                        result.throughput_fps,
                        result.optimizations.join(", ")
                    ));
                }
            }
            
            std::fs::write(output_path, output)?;
            println!("\nDetailed results saved to {}", output_path);
        }
        
        Commands::Serve { port, host } => {
            println!("Starting GABA-BURN API server...");
            println!("Host: {}", host);
            println!("Port: {}", port);
            println!("\nEndpoints:");
            println!("  GET  /health       - Health check");
            println!("  GET  /models       - List all models");
            println!("  GET  /models/:name - Get model info");
            println!("  POST /infer        - Run inference");
            println!("  GET  /metrics      - Performance metrics");
            println!("\nPress Ctrl+C to stop");
            println!("\nNote: API server requires 'gaba-serve' crate");
            println!("Run: cargo build -p gaba-serve");
        }

        Commands::Traffic {
            data,
            output,
            epochs,
            lr,
        } => {
            println!(
                "{}",
                "Training traffic speed prediction model...".green().bold()
            );
            train::train_traffic_model(&data, &output, epochs, lr)?;
            println!("{}", "✓ Training complete!".green().bold());
        }

        #[cfg(feature = "pqc")]
        Commands::Encrypt { model, output } => {
            println!("{}", "Encrypting model with PQC...".cyan().bold());
            encrypt_model(&model, &output).await?;
            println!("{}", "✓ Model encrypted!".green().bold());
        }

        #[cfg(feature = "pqc")]
        Commands::Verify {
            encrypted,
            original,
        } => {
            println!("{}", "Verifying model integrity...".cyan().bold());
            verify_model(&encrypted, &original).await?;
            println!("{}", "✓ Model verified!".green().bold());
        }

        Commands::Bench {
            size,
            metal,
            zig,
            large,
        } => {
            println!("{}", "Running performance benchmark...".yellow().bold());
            run_benchmark(&size, metal, zig, large)?;
        }

        Commands::Info => {
            print_system_info();
        }

        Commands::Singularity { iterations, size } => {
            println!("{}", "Running Singularity Engine Demo...".yellow().bold());
            run_singularity_demo(iterations, size)?;
        }

        Commands::New { path, name } => {
            println!("{}", "Creating new Gaba ML workspace...".green().bold());
            workspace::create_workspace(&path, &name, "Gaba ML Project")?;
            println!("{}", "✓ Workspace created successfully!".green().bold());
            println!("Location: {}", path.join(&name).display());
        }

        Commands::Init { name } => {
            println!("{}", "Initializing Gaba ML workspace...".green().bold());
            let current_dir = std::env::current_dir()?;
            workspace::create_workspace(&current_dir, &name, "Gaba ML Project")?;
            println!("{}", "✓ Workspace initialized successfully!".green().bold());
        }

        Commands::TrainRouter {
            output,
            epochs,
            lr,
            samples,
        } => {
            println!(
                "{}",
                "Training Router Model (10M parameters)...".green().bold()
            );
            commands::train_router::train_router(output, epochs, lr, samples)?;
            println!("{}", "✓ Router model training complete!".green().bold());
        }

        Commands::TestRouter { model } => {
            println!("{}", "Testing Router Model...".cyan().bold());
            commands::test_router::test_router(&model)?;
            println!("{}", "✓ Router model test complete!".green().bold());
        }

        Commands::BenchRouter { model, iterations } => {
            println!("{}", "Benchmarking Router Model...".yellow().bold());
            commands::test_router::benchmark_router(&model, iterations)?;
            println!("{}", "✓ Router model benchmark complete!".green().bold());
        }

        Commands::Workflow(args) => {
            println!("{}", "Running ML training workflow...".green().bold());
            commands::workflow::run_workflow(args).await?;
            println!("{}", "✓ Workflow complete!".green().bold());
        }

        Commands::Profile(args) => {
            println!("{}", "Profiling training performance...".yellow().bold());
            commands::profile::profile_training(args).await?;
            println!("{}", "✓ Profiling complete!".green().bold());
        }

        Commands::Optimize(args) => {
            println!("{}", "Optimizing model...".cyan().bold());
            commands::optimize::optimize_model(args).await?;
            println!("{}", "✓ Optimization complete!".green().bold());
        }
    }

    Ok(())
}

#[cfg(feature = "pqc")]
async fn encrypt_model(model_path: &PathBuf, output_path: &PathBuf) -> Result<()> {
    use gaba_pqc::encrypt_checkpoint;
    let data = std::fs::read(model_path)?;
    let encrypted = encrypt_checkpoint(&data).await?;
    std::fs::write(output_path, encrypted)?;
    Ok(())
}

#[cfg(feature = "pqc")]
async fn verify_model(encrypted_path: &PathBuf, original_path: &PathBuf) -> Result<()> {
    use gaba_pqc::verify_checkpoint;
    let encrypted = std::fs::read(encrypted_path)?;
    let original = std::fs::read(original_path)?;
    let valid = verify_checkpoint(&encrypted, &original).await?;
    if !valid {
        anyhow::bail!("Model verification failed!");
    }
    Ok(())
}

fn run_benchmark(size: &str, metal: bool, _zig: bool, large: bool) -> Result<()> {
    use gaba_singularity::AdaptiveKernelOrchestrator;

    println!("Benchmark configuration:");
    println!("  Size: {}", size.yellow());
    println!(
        "  Metal GPU: {}",
        if metal {
            "enabled".green()
        } else {
            "disabled".red()
        }
    );
    println!(
        "  Large matrices: {}",
        if large { "yes".green() } else { "no".cyan() }
    );

    let sizes = if large {
        vec![
            ("512x512x512", 512),
            ("768x768x768", 768),
            ("1024x1024x1024", 1024),
        ]
    } else {
        vec![
            ("128x128x128", 128),
            ("256x256x256", 256),
            ("512x512x512", 512),
        ]
    };

    let orchestrator = AdaptiveKernelOrchestrator::new();

    println!("\nRunning benchmarks...\n");

    for (name, size) in sizes {
        let m = size;
        let n = size;
        let k = size;

        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        let (kernel_type, duration) = orchestrator.execute_gemm_adaptive(&a, &b, &mut c, m, n, k);

        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / duration.as_secs_f64() / 1e9;

        println!("  {} - {:?}", name.cyan(), kernel_type);
        println!("    Time: {:.2} ms", duration.as_secs_f64() * 1000.0);
        println!("    Performance: {:.2} GFLOPS", gflops);
        println!();
    }

    println!("{}", "Benchmark complete!".green().bold());
    Ok(())
}

fn print_system_info() {
    use gaba_native_kernels::detect_amx;

    println!(
        "{}",
        "GABA Burn Singularity Engine - System Information"
            .cyan()
            .bold()
    );
    println!("{}", "=".repeat(60).cyan());
    println!();

    #[cfg(target_os = "macos")]
    {
        println!("  {} macOS (Apple Silicon)", "✓".green());

        let amx = detect_amx();
        println!(
            "  {} AMX Coprocessor: {}",
            if amx { "✓".green() } else { "✗".red() },
            if amx {
                "Available".green()
            } else {
                "Not detected".red()
            }
        );

        println!("  {} Accelerate Framework: Available", "✓".green());

        #[cfg(feature = "metal")]
        {
            let registry = gaba_singularity::KernelRegistry::new();
            let metal_available = registry.metal_available();
            println!(
                "  {} Metal GPU: {}",
                if metal_available {
                    "✓".green()
                } else {
                    "✗".red()
                },
                if metal_available {
                    "Available".green()
                } else {
                    "Not available".red()
                }
            );
        }

        #[cfg(not(feature = "metal"))]
        println!(
            "  {} Metal GPU: Disabled (compile with --features metal)",
            "✗".red()
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("  {} Platform: {}", "i".yellow(), std::env::consts::OS);
        println!("  {} AMX/Accelerate: macOS only", "✗".red());
        println!("  {} Metal GPU: macOS only", "✗".red());
    }

    #[cfg(feature = "pqc")]
    println!("  {} Post-quantum cryptography", "✓".green());

    #[cfg(feature = "zig")]
    println!("  {} Zig SIMD kernels", "✓".green());

    println!();
    println!("Hardware:");
    println!("  CPU cores: {}", num_cpus::get());

    println!();
    println!("Available Kernel Types:");
    let registry = gaba_singularity::KernelRegistry::new();

    for kernel_type in gaba_singularity::KernelType::ALL {
        let available = registry.is_available(*kernel_type);
        println!(
            "  {} {:?}",
            if available {
                "✓".green()
            } else {
                "✗".red()
            },
            kernel_type
        );
    }

    println!();
    println!("Optimization Strategy:");
    println!("  Small (<64x64): RustVectorized (low overhead)");
    println!("  Medium (64-512): Accelerate/AMX (hardware acceleration)");
    println!("  Large (>512): MetalGPU (zero-copy unified memory)");
}

#[allow(dead_code)]
fn get_features() -> Vec<String> {
    let features = vec!["rust".to_string()];

    #[cfg(feature = "pqc")]
    features.push("pqc".to_string());

    #[cfg(feature = "metal")]
    features.push("metal".to_string());

    #[cfg(feature = "zig")]
    features.push("zig".to_string());

    features
}

fn run_singularity_demo(iterations: usize, size: usize) -> Result<()> {
    use gaba_singularity::AdaptiveKernelOrchestrator;

    println!("\n{}", "Initializing Singularity Engine...".cyan());
    let orchestrator = AdaptiveKernelOrchestrator::new().with_learning(true);

    let a: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
    let mut c = vec![0.0f32; size * size];

    println!("Matrix size: {}x{}x{}", size, size, size);
    println!("Iterations: {}\n", iterations);

    println!("{}", "Starting adaptive kernel selection...".yellow());

    for i in 0..iterations {
        let (kernel_type, duration) =
            orchestrator.execute_gemm_adaptive(&a, &b, &mut c, size, size, size);

        if i % 10 == 0 || i < 5 {
            println!(
                "  Iteration {}: {:?} - {:.2} µs",
                i,
                kernel_type,
                duration.as_micros()
            );
        }
    }

    let history = orchestrator.get_performance_history();
    println!("\n{}", "Demo Complete!".green().bold());
    println!("Total executions: {}", history.len());

    let avg_time: f64 = history
        .iter()
        .map(|v| v.execution_time.as_secs_f64())
        .sum::<f64>()
        / history.len() as f64;

    println!("Average execution time: {:.2} µs", avg_time * 1_000_000.0);

    let kernel_counts: std::collections::HashMap<_, usize> =
        history
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, v| {
                *acc.entry(format!("{:?}", v.kernel_type)).or_insert(0) += 1;
                acc
            });

    println!("\nKernel selection distribution:");
    for (kernel, count) in kernel_counts {
        let percentage = (count as f64 / history.len() as f64) * 100.0;
        println!("  {}: {} times ({:.1}%)", kernel, count, percentage);
    }

    Ok(())
}
