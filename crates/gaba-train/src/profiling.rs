//! Training profiler for performance analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub track_memory: bool,
    pub output_path: Option<String>,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            track_memory: false,
            output_path: None,
        }
    }
}

/// Profiling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMetrics {
    pub kernel_times: HashMap<String, Duration>,
    pub memory_usage: HashMap<String, usize>,
    pub peak_memory: usize,
    pub total_time: Duration,
    pub overhead_percent: f64,
}

/// Flamegraph-compatible stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub name: String,
    pub duration_us: u64,
    pub children: Vec<StackFrame>,
}

/// Training profiler
pub struct Profiler {
    config: ProfilerConfig,
    start_time: Option<Instant>,
    kernel_times: HashMap<String, Duration>,
    memory_usage: HashMap<String, usize>,
    peak_memory: usize,
    profiling_overhead: Duration,
    call_counts: HashMap<String, usize>,
}

impl Profiler {
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            start_time: None,
            kernel_times: HashMap::new(),
            memory_usage: HashMap::new(),
            peak_memory: 0,
            profiling_overhead: Duration::ZERO,
            call_counts: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        if self.config.enabled {
            self.start_time = Some(Instant::now());
        }
    }

    pub fn record_kernel(&mut self, name: &str, duration: Duration) {
        if self.config.enabled {
            let overhead_start = Instant::now();
            *self
                .kernel_times
                .entry(name.to_string())
                .or_insert(Duration::ZERO) += duration;
            *self.call_counts.entry(name.to_string()).or_insert(0) += 1;
            self.profiling_overhead += overhead_start.elapsed();
        }
    }

    pub fn record_memory(&mut self, name: &str, bytes: usize) {
        if self.config.enabled && self.config.track_memory {
            self.memory_usage.insert(name.to_string(), bytes);
            if bytes > self.peak_memory {
                self.peak_memory = bytes;
            }
        }
    }

    pub fn get_metrics(&self) -> ProfileMetrics {
        let total_time = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        let overhead_percent = if total_time.as_secs_f64() > 0.0 {
            (self.profiling_overhead.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        } else {
            0.0
        };

        ProfileMetrics {
            kernel_times: self.kernel_times.clone(),
            memory_usage: self.memory_usage.clone(),
            peak_memory: self.peak_memory,
            total_time,
            overhead_percent,
        }
    }

    pub fn save_json(&self, path: &str) -> anyhow::Result<()> {
        let metrics = self.get_metrics();
        let json = serde_json::to_string_pretty(&metrics)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn save_flamegraph(&self, path: &str) -> anyhow::Result<()> {
        let mut lines = Vec::new();

        for (name, duration) in &self.kernel_times {
            let count = self.call_counts.get(name).unwrap_or(&1);
            let avg_us = duration.as_micros() / (*count as u128);
            lines.push(format!("{}; {} {}", name, count, avg_us));
        }

        std::fs::write(path, lines.join("\n"))?;
        Ok(())
    }

    pub fn print_summary(&self) {
        let metrics = self.get_metrics();

        println!("\nProfiling Summary:");
        println!("Total time: {:.2}s", metrics.total_time.as_secs_f64());
        println!(
            "Peak memory: {:.2} MB",
            metrics.peak_memory as f64 / 1024.0 / 1024.0
        );
        println!("Profiling overhead: {:.2}%", metrics.overhead_percent);

        println!("\nKernel breakdown:");
        let mut sorted: Vec<_> = metrics.kernel_times.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (name, duration) in sorted.iter().take(10) {
            let percent = (duration.as_secs_f64() / metrics.total_time.as_secs_f64()) * 100.0;
            let count = self.call_counts.get(*name).unwrap_or(&1);
            println!(
                "  {}: {:.2}ms ({:.1}%, {} calls)",
                name,
                duration.as_secs_f64() * 1000.0,
                percent,
                count
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_config() {
        let config = ProfilerConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_profiler_metrics() {
        let mut profiler = Profiler::new(ProfilerConfig {
            enabled: true,
            track_memory: true,
            output_path: None,
        });

        profiler.start();
        profiler.record_kernel("forward", Duration::from_millis(10));
        profiler.record_memory("model", 1024);

        let metrics = profiler.get_metrics();
        assert!(metrics.kernel_times.contains_key("forward"));
        assert!(metrics.memory_usage.contains_key("model"));
    }
}
