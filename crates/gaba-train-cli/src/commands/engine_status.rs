use anyhow::Result;
use gaba_train::engine_coordinator::GabaEngineCoordinator;

pub async fn show_engine_status() -> Result<()> {
    println!("GABA Engine Status\n");
    println!("{}", "=".repeat(80));
    
    let coordinator = GabaEngineCoordinator::new();
    coordinator.initialize().await?;
    
    let health = coordinator.health_check().await;
    let metrics = coordinator.get_metrics().await;
    
    println!("Health Status:");
    println!("  Singularity Engine: {}", if health.singularity_healthy { "OK" } else { "DEGRADED" });
    println!("  Memory System: {}", if health.memory_healthy { "OK" } else { "DEGRADED" });
    println!("  Vector Store: {}", if health.vector_healthy { "OK" } else { "DEGRADED" });
    println!("  Workflows: {}", if health.workflows_healthy { "OK" } else { "DEGRADED" });
    println!("  PQC Security: {}", if health.pqc_healthy { "OK" } else { "DEGRADED" });
    println!("  Overall: {}", if health.overall_healthy { "HEALTHY" } else { "UNHEALTHY" });
    
    println!("\nPerformance Metrics:");
    println!("  Active Kernels: {}", metrics.active_kernels);
    println!("  Memory Usage: {:.2} MB", metrics.memory_usage_mb);
    println!("  Vector Search Latency: {:.2} ms", metrics.vector_search_latency_ms);
    println!("  Workflow Throughput: {:.2} ops/s", metrics.workflow_throughput);
    println!("  Encrypted Models: {}", metrics.encrypted_models);
    println!("  Performance Score: {:.2}", metrics.overall_performance_score);
    
    println!("{}", "=".repeat(80));
    
    Ok(())
}

pub async fn optimize_engine() -> Result<()> {
    println!("Optimizing GABA Engine...\n");
    
    let coordinator = GabaEngineCoordinator::new();
    coordinator.initialize().await?;
    
    let report = coordinator.optimize_all().await?;
    
    println!("Optimization Report:");
    println!("  Singularity Optimized: {}", report.singularity_optimized);
    println!("  Memory Compacted: {}", report.memory_compacted);
    println!("  Vector Reindexed: {}", report.vector_reindexed);
    println!("  Workflows Pruned: {}", report.workflows_pruned);
    
    println!("\nOptimization complete!");
    
    Ok(())
}
