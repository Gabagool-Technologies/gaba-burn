//! Gaba Train CLI: Hardcore Rust+Zig ML training with PQC and Metal acceleration
//!
//! Features:
//! - Zero-copy Metal GPU acceleration on Apple Silicon
//! - Post-quantum cryptography for model protection
//! - SIMD-optimized Zig kernels for critical paths
//! - Unified memory architecture for CPU/GPU/Neural Engine

mod data;
mod models;
mod augmentation;
mod onnx_export;
mod dataset_generator;
mod federated_training;
mod realtime_updates;
mod demand_forecasting;
mod vehicle_maintenance;
mod driver_behavior;
mod train;
mod metrics;
mod adam;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gaba-train")]
#[command(about = "Rust+Zig ML training for route optimization", long_about = None)]
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
    
    /// Run singularity engine demo
    Singularity {
        /// Number of iterations
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        
        /// Matrix size
        #[arg(long, default_value_t = 128)]
        size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Traffic { data, output, epochs, lr } => {
            println!("{}", "Training traffic speed prediction model...".green().bold());
            train::train_traffic_model(&data, &output, epochs, lr)?;
            println!("{}", "✓ Training complete!".green().bold());
        }
        
        Commands::Route { data, output, epochs, lr } => {
            println!("{}", "Training route time prediction model...".green().bold());
            train::train_route_model(&data, &output, epochs, lr)?;
            println!("{}", "✓ Training complete!".green().bold());
        }
        
        Commands::Generate { output, traffic_samples, route_samples } => {
            println!("{}", "Generating synthetic training data...".green().bold());
            data::generate_data(&output, traffic_samples, route_samples)?;
            println!("{}", "✓ Data generation complete!".green().bold());
        }
        
        #[cfg(feature = "pqc")]
        Commands::Encrypt { model, output } => {
            println!("{}", "Encrypting model with PQC...".cyan().bold());
            encrypt_model(&model, &output).await?;
            println!("{}", "✓ Model encrypted!".green().bold());
        }
        
        #[cfg(feature = "pqc")]
        Commands::Verify { encrypted, original } => {
            println!("{}", "Verifying model integrity...".cyan().bold());
            verify_model(&encrypted, &original).await?;
            println!("{}", "✓ Model verified!".green().bold());
        }
        
        Commands::Bench { size, metal, zig, large } => {
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
    println!("  Metal GPU: {}", if metal { "enabled".green() } else { "disabled".red() });
    println!("  Large matrices: {}", if large { "yes".green() } else { "no".cyan() });
    
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
    use gaba_native_kernels::{detect_amx};
    
    println!("{}", "GABA Burn Singularity Engine - System Information".cyan().bold());
    println!("{}", "=".repeat(60).cyan());
    println!();
    
    #[cfg(target_os = "macos")]
    {
        println!("  {} macOS (Apple Silicon)", "✓".green());
        
        let amx = detect_amx();
        println!("  {} AMX Coprocessor: {}", 
            if amx { "✓".green() } else { "✗".red() },
            if amx { "Available".green() } else { "Not detected".red() }
        );
        
        println!("  {} Accelerate Framework: Available", "✓".green());
        
        #[cfg(feature = "metal")]
        {
            let registry = gaba_singularity::KernelRegistry::new();
            let metal_available = registry.metal_available();
            println!("  {} Metal GPU: {}", 
                if metal_available { "✓".green() } else { "✗".red() },
                if metal_available { "Available".green() } else { "Not available".red() }
            );
        }
        
        #[cfg(not(feature = "metal"))]
        println!("  {} Metal GPU: Disabled (compile with --features metal)", "✗".red());
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
        println!("  {} {:?}", 
            if available { "✓".green() } else { "✗".red() },
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
    let mut features = vec!["rust".to_string()];
    
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
        let (kernel_type, duration) = orchestrator.execute_gemm_adaptive(&a, &b, &mut c, size, size, size);
        
        if i % 10 == 0 || i < 5 {
            println!("  Iteration {}: {:?} - {:.2} µs", 
                i, 
                kernel_type, 
                duration.as_micros()
            );
        }
    }
    
    let history = orchestrator.get_performance_history();
    println!("\n{}", "Demo Complete!".green().bold());
    println!("Total executions: {}", history.len());
    
    let avg_time: f64 = history.iter()
        .map(|v| v.execution_time.as_secs_f64())
        .sum::<f64>() / history.len() as f64;
    
    println!("Average execution time: {:.2} µs", avg_time * 1_000_000.0);
    
    let kernel_counts: std::collections::HashMap<_, usize> = history.iter()
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
