use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub timestamp: u64,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, Vec<f64>>,
    pub timers: HashMap<String, Vec<Duration>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            timers: HashMap::new(),
        }
    }
    
    pub fn increment_counter(&mut self, name: &str, value: u64) {
        *self.counters.entry(name.to_string()).or_insert(0) += value;
    }
    
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        self.gauges.insert(name.to_string(), value);
    }
    
    pub fn record_value(&mut self, name: &str, value: f64) {
        self.histograms.entry(name.to_string()).or_insert_with(Vec::new).push(value);
    }
    
    pub fn record_duration(&mut self, name: &str, duration: Duration) {
        self.timers.entry(name.to_string()).or_insert_with(Vec::new).push(duration);
    }
    
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or(0)
    }
    
    pub fn get_gauge(&self, name: &str) -> f64 {
        self.gauges.get(name).copied().unwrap_or(0.0)
    }
    
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histograms.get(name).map(|values| {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = sorted.iter().sum();
            let count = sorted.len();
            let mean = sum / count as f64;
            
            let p50 = sorted[count / 2];
            let p95 = sorted[(count * 95) / 100];
            let p99 = sorted[(count * 99) / 100];
            
            HistogramStats {
                count,
                mean,
                min: sorted[0],
                max: sorted[count - 1],
                p50,
                p95,
                p99,
            }
        })
    }
    
    pub fn get_timer_stats(&self, name: &str) -> Option<TimerStats> {
        self.timers.get(name).map(|durations| {
            let mut sorted: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = sorted.iter().sum();
            let count = sorted.len();
            let mean = sum / count as f64;
            
            let p50 = sorted[count / 2];
            let p95 = sorted[(count * 95) / 100];
            let p99 = sorted[(count * 99) / 100];
            
            TimerStats {
                count,
                mean_ms: mean * 1000.0,
                min_ms: sorted[0] * 1000.0,
                max_ms: sorted[count - 1] * 1000.0,
                p50_ms: p50 * 1000.0,
                p95_ms: p95 * 1000.0,
                p99_ms: p99 * 1000.0,
            }
        })
    }
    
    pub fn reset(&mut self) {
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
        self.timers.clear();
    }
    
    pub fn print_summary(&self) {
        println!("\n=== Metrics Summary ===");
        
        if !self.counters.is_empty() {
            println!("\nCounters:");
            for (name, value) in &self.counters {
                println!("  {}: {}", name, value);
            }
        }
        
        if !self.gauges.is_empty() {
            println!("\nGauges:");
            for (name, value) in &self.gauges {
                println!("  {}: {:.2}", name, value);
            }
        }
        
        if !self.histograms.is_empty() {
            println!("\nHistograms:");
            for (name, _) in &self.histograms {
                if let Some(stats) = self.get_histogram_stats(name) {
                    println!("  {}:", name);
                    println!("    Count: {}", stats.count);
                    println!("    Mean: {:.2}", stats.mean);
                    println!("    P50: {:.2}, P95: {:.2}, P99: {:.2}", stats.p50, stats.p95, stats.p99);
                }
            }
        }
        
        if !self.timers.is_empty() {
            println!("\nTimers:");
            for (name, _) in &self.timers {
                if let Some(stats) = self.get_timer_stats(name) {
                    println!("  {}:", name);
                    println!("    Count: {}", stats.count);
                    println!("    Mean: {:.2}ms", stats.mean_ms);
                    println!("    P50: {:.2}ms, P95: {:.2}ms, P99: {:.2}ms", stats.p50_ms, stats.p95_ms, stats.p99_ms);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone)]
pub struct TimerStats {
    pub count: usize,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn stop(self, metrics: &mut Metrics) {
        let duration = self.start.elapsed();
        metrics.record_duration(&self.name, duration);
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
