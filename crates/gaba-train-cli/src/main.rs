//! Gaba Train CLI: Hardcore Rust+Zig ML training with PQC and Metal acceleration
//!
//! Features:
//! - Zero-copy Metal GPU acceleration on Apple Silicon
//! - Post-quantum cryptography for model protection
//! - SIMD-optimized Zig kernels for critical paths
//! - Unified memory architecture for CPU/GPU/Neural Engine

mod data;
mod models;
mod train;

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
        /// Benchmark size (small/medium/large)
        #[arg(short, long, default_value = "medium")]
        size: String,
        
        /// Enable Metal acceleration
        #[arg(long)]
        metal: bool,
        
        /// Enable Zig kernels
        #[arg(long)]
        zig: bool,
    },
    
    /// Show system capabilities
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    print_banner();
    
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
        
        Commands::Bench { size, metal, zig } => {
            println!("{}", "Running performance benchmark...".yellow().bold());
            run_benchmark(&size, metal, zig)?;
        }
        
        Commands::Info => {
            print_system_info();
        }
    }
    
    Ok(())
}

fn print_banner() {
    println!("{}", "╔═══════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║  GABA-BURN: Hardcore ML Training with Rust+Zig+Metal      ║".cyan().bold());
    println!("{}", "║  Zero-copy · PQC · SIMD · Apple Silicon Optimized         ║".cyan());
    println!("{}", "╚═══════════════════════════════════════════════════════════╝".cyan());
    println!();
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

fn run_benchmark(size: &str, _metal: bool, zig: bool) -> Result<()> {
    println!("Benchmark size: {}", size.yellow());
    println!("Zig kernels: {}", if zig { "enabled".green() } else { "multi-threaded Rust".cyan() });
    
    let (m, n, k) = match size {
        "small" => (128, 128, 128),
        "medium" => (512, 512, 512),
        "large" => (2048, 2048, 2048),
        _ => (512, 512, 512),
    };
    
    println!("\nMatrix dimensions: {}x{} * {}x{}", m, k, k, n);
    
    #[cfg(feature = "zig")]
    {
        if zig {
            use gaba_native_kernels::gemm;
            let a = vec![1.0f32; m * k];
            let b = vec![1.0f32; k * n];
            let mut c = vec![0.0f32; m * n];
            
            let start = std::time::Instant::now();
            gemm(&a, &b, &mut c, m, n, k);
            let elapsed = start.elapsed();
            
            println!("Zig GEMM: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            let gflops = (2.0 * m as f64 * n as f64 * k as f64) / elapsed.as_secs_f64() / 1e9;
            println!("Performance: {:.2} GFLOPS", gflops);
            return Ok(());
        }
    }
    
    // Always run Rust benchmark
    use gaba_native_kernels::gemm;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    let start = std::time::Instant::now();
    gemm(&a, &b, &mut c, m, n, k);
    let elapsed = start.elapsed();
    
    println!("Rust GEMM (multi-threaded): {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / elapsed.as_secs_f64() / 1e9;
    println!("Performance: {:.2} GFLOPS", gflops);
    println!("Cores used: {}", num_cpus::get());
    
    Ok(())
}

fn print_system_info() {
    println!("{}", "System Capabilities:".cyan().bold());
    println!();
    
    #[cfg(target_os = "macos")]
    {
        println!("  {} macOS (Apple Silicon optimized)", "✓".green());
        
        #[cfg(feature = "metal")]
        println!("  {} Metal GPU acceleration available", "✓".green());
        
        #[cfg(not(feature = "metal"))]
        println!("  {} Metal GPU acceleration (disabled)", "✗".red());
    }
    
    #[cfg(feature = "pqc")]
    println!("  {} Post-quantum cryptography enabled", "✓".green());
    
    #[cfg(feature = "zig")]
    println!("  {} Zig SIMD kernels enabled", "✓".green());
    
    println!();
    println!("CPU cores: {}", num_cpus::get());
    println!("Features: {}", get_features().join(", "));
}

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
