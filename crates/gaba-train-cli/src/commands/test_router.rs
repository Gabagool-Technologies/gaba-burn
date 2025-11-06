//! Router model testing and benchmarking

#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array1;
use std::path::PathBuf;
use std::time::Instant;

use super::train_router::{generate_training_data, RouterModel};

/// Test router model accuracy
pub fn test_router(model_path: &PathBuf) -> Result<()> {
    println!("Testing Router Model");
    println!("   Model: {:?}", model_path);

    // Load model (for now we'll create a new one as placeholder)
    let model = RouterModel::new();
    println!("   Parameters: {}", model.parameter_count());

    // Generate test data
    println!("\nGenerating test data...");
    let (features, labels) = generate_training_data(1000)?;

    // Test accuracy
    let mut correct = 0;
    let mut total = 0;

    for i in 0..features.nrows() {
        let input = features.row(i).to_owned();
        let target = labels.row(i).to_owned();

        let output = model.forward(&input);

        let pred_class = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_class = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if pred_class == true_class {
            correct += 1;
        }
        total += 1;
    }

    let accuracy = correct as f32 / total as f32 * 100.0;
    println!("\nTest Results:");
    println!("   Accuracy: {:.2}%", accuracy);
    println!("   Correct: {}/{}", correct, total);

    Ok(())
}

/// Benchmark router model performance
pub fn benchmark_router(model_path: &PathBuf, iterations: usize) -> Result<()> {
    println!("Benchmarking Router Model");
    println!("   Model: {:?}", model_path);
    println!("   Iterations: {}", iterations);

    let model = RouterModel::new();

    // Create test input
    let input = Array1::from_vec(vec![0.1; 384]);

    // Warmup
    for _ in 0..10 {
        let _ = model.forward(&input);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward(&input);
    }
    let duration = start.elapsed();

    let avg_latency_us = duration.as_micros() as f64 / iterations as f64;
    let throughput = 1_000_000.0 / avg_latency_us;

    println!("\nBenchmark Results:");
    println!("   Total time: {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("   Average latency: {:.2} us", avg_latency_us);
    println!("   Throughput: {:.0} classifications/sec", throughput);
    println!("   Memory: ~{} KB", model.parameter_count() * 4 / 1024);

    // Compare to target
    let target_latency_us = 100.0; // Target: <100us
    if avg_latency_us < target_latency_us {
        println!(
            "\n   PASS: Latency below target ({:.2} us < {:.2} us)",
            avg_latency_us, target_latency_us
        );
    } else {
        println!(
            "\n   WARNING: Latency above target ({:.2} us > {:.2} us)",
            avg_latency_us, target_latency_us
        );
    }

    Ok(())
}

/// Test router with real-world queries
pub fn test_router_queries() -> Result<()> {
    println!("Testing Router with Real-World Queries\n");

    let model = RouterModel::new();

    let test_queries = vec![
        ("Write a function to sort an array", "code_gen"),
        (
            "Why is my code throwing a NullPointerException?",
            "debugger",
        ),
        (
            "Design a microservices architecture for e-commerce",
            "architect",
        ),
        ("What is the meaning of this text?", "embedder"),
        ("Hello, how are you?", "general"),
    ];

    for (query, expected) in test_queries {
        // In real implementation, would embed the query
        // For now, use random embedding
        let embedding = Array1::from_vec(vec![0.1; 384]);
        let output = model.forward(&embedding);

        let pred_class = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let class_name = match pred_class {
            0 => "code_gen",
            1 => "debugger",
            2 => "architect",
            3 => "embedder",
            4 => "general",
            _ => "unknown",
        };

        let confidence = output[pred_class] * 100.0;

        println!("Query: \"{}\"", query);
        println!("   Expected: {}", expected);
        println!(
            "   Predicted: {} ({:.1}% confidence)",
            class_name, confidence
        );
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_forward_pass() {
        let model = RouterModel::new();
        let input = Array1::from_vec(vec![0.1; 384]);
        let output = model.forward(&input);

        assert_eq!(output.len(), 5);
        assert!((output.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_router_latency() {
        let model = RouterModel::new();
        let input = Array1::from_vec(vec![0.1; 384]);

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = model.forward(&input);
        }
        let duration = start.elapsed();

        let avg_latency_us = duration.as_micros() as f64 / 1000.0;
        assert!(
            avg_latency_us < 100.0,
            "Latency too high: {:.2} us",
            avg_latency_us
        );
    }
}
