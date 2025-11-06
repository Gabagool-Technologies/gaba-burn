use std::time::{Duration, Instant};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub throughput_samples_per_sec: f32,
    pub cache_hit_rate: f32,
    pub memory_bandwidth_gbps: f32,
    pub avg_latency_ms: f32,
}

pub struct RuntimeProfiler {
    window_size: usize,
    throughput_history: VecDeque<f32>,
    cache_hit_history: VecDeque<f32>,
    bandwidth_history: VecDeque<f32>,
    latency_history: VecDeque<f32>,
    last_measurement: Instant,
    samples_processed: usize,
}

impl RuntimeProfiler {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            throughput_history: VecDeque::with_capacity(window_size),
            cache_hit_history: VecDeque::with_capacity(window_size),
            bandwidth_history: VecDeque::with_capacity(window_size),
            latency_history: VecDeque::with_capacity(window_size),
            last_measurement: Instant::now(),
            samples_processed: 0,
        }
    }

    pub fn record_batch(&mut self, batch_size: usize, duration: Duration) {
        let elapsed_secs = duration.as_secs_f32();
        let throughput = batch_size as f32 / elapsed_secs;
        
        if self.throughput_history.len() >= self.window_size {
            self.throughput_history.pop_front();
        }
        self.throughput_history.push_back(throughput);
        
        if self.latency_history.len() >= self.window_size {
            self.latency_history.pop_front();
        }
        self.latency_history.push_back(elapsed_secs * 1000.0);
        
        self.samples_processed += batch_size;
        self.last_measurement = Instant::now();
    }

    pub fn record_cache_metrics(&mut self, hit_rate: f32) {
        if self.cache_hit_history.len() >= self.window_size {
            self.cache_hit_history.pop_front();
        }
        self.cache_hit_history.push_back(hit_rate);
    }

    pub fn record_bandwidth(&mut self, bandwidth_gbps: f32) {
        if self.bandwidth_history.len() >= self.window_size {
            self.bandwidth_history.pop_front();
        }
        self.bandwidth_history.push_back(bandwidth_gbps);
    }

    fn add_to_history(&mut self, history: &mut VecDeque<f32>, value: f32) {
        if history.len() >= self.window_size {
            history.pop_front();
        }
        history.push_back(value);
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            throughput_samples_per_sec: self.avg(&self.throughput_history),
            cache_hit_rate: self.avg(&self.cache_hit_history),
            memory_bandwidth_gbps: self.avg(&self.bandwidth_history),
            avg_latency_ms: self.avg(&self.latency_history),
        }
    }

    fn avg(&self, history: &VecDeque<f32>) -> f32 {
        if history.is_empty() {
            return 0.0;
        }
        history.iter().sum::<f32>() / history.len() as f32
    }

    pub fn should_adapt(&self) -> bool {
        self.throughput_history.len() >= self.window_size
    }

    pub fn recommend_batch_size_adjustment(&self, current_batch: usize) -> Option<usize> {
        if !self.should_adapt() {
            return None;
        }

        let metrics = self.get_metrics();
        
        if metrics.cache_hit_rate < 0.85 {
            Some(current_batch.saturating_sub(current_batch / 4))
        } else if metrics.cache_hit_rate > 0.95 && metrics.throughput_samples_per_sec > 1000.0 {
            Some(current_batch + current_batch / 4)
        } else {
            None
        }
    }

    pub fn is_performing_well(&self) -> bool {
        if !self.should_adapt() {
            return true;
        }

        let metrics = self.get_metrics();
        metrics.cache_hit_rate > 0.85 && metrics.throughput_samples_per_sec > 500.0
    }

    pub fn reset(&mut self) {
        self.throughput_history.clear();
        self.cache_hit_history.clear();
        self.bandwidth_history.clear();
        self.latency_history.clear();
        self.samples_processed = 0;
        self.last_measurement = Instant::now();
    }
}

pub struct AdaptiveOptimizer {
    profiler: RuntimeProfiler,
    current_sequence: String,
    adaptation_threshold: f32,
    min_samples_before_adapt: usize,
}

impl AdaptiveOptimizer {
    pub fn new(initial_sequence: String) -> Self {
        Self {
            profiler: RuntimeProfiler::new(100),
            current_sequence: initial_sequence,
            adaptation_threshold: 0.05,
            min_samples_before_adapt: 1000,
        }
    }

    pub fn record_batch(&mut self, batch_size: usize, duration: Duration) {
        self.profiler.record_batch(batch_size, duration);
    }

    pub fn record_cache_metrics(&mut self, hit_rate: f32) {
        self.profiler.record_cache_metrics(hit_rate);
    }

    pub fn should_switch_sequence(&self) -> Option<String> {
        if self.profiler.samples_processed < self.min_samples_before_adapt {
            return None;
        }

        let metrics = self.profiler.get_metrics();
        
        if metrics.cache_hit_rate < 0.80 {
            match self.current_sequence.as_str() {
                "extended" => Some("fibonacci".to_string()),
                "fibonacci" => Some("baseline".to_string()),
                _ => None,
            }
        } else if metrics.cache_hit_rate > 0.95 && self.current_sequence == "baseline" {
            Some("extended".to_string())
        } else {
            None
        }
    }

    pub fn get_current_sequence(&self) -> &str {
        &self.current_sequence
    }

    pub fn set_sequence(&mut self, sequence: String) {
        self.current_sequence = sequence;
        self.profiler.reset();
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.profiler.get_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = RuntimeProfiler::new(100);
        assert_eq!(profiler.window_size, 100);
        assert_eq!(profiler.samples_processed, 0);
    }

    #[test]
    fn test_record_batch() {
        let mut profiler = RuntimeProfiler::new(10);
        profiler.record_batch(32, Duration::from_millis(10));
        
        let metrics = profiler.get_metrics();
        assert!(metrics.throughput_samples_per_sec > 0.0);
        assert!(metrics.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_cache_metrics() {
        let mut profiler = RuntimeProfiler::new(10);
        profiler.record_cache_metrics(0.90);
        profiler.record_cache_metrics(0.92);
        
        let metrics = profiler.get_metrics();
        assert!(metrics.cache_hit_rate > 0.89 && metrics.cache_hit_rate < 0.93);
    }

    #[test]
    fn test_should_adapt() {
        let mut profiler = RuntimeProfiler::new(5);
        assert!(!profiler.should_adapt());
        
        for _ in 0..5 {
            profiler.record_batch(32, Duration::from_millis(10));
        }
        
        assert!(profiler.should_adapt());
    }

    #[test]
    fn test_batch_size_adjustment() {
        let mut profiler = RuntimeProfiler::new(5);
        
        for _ in 0..5 {
            profiler.record_batch(32, Duration::from_millis(10));
            profiler.record_cache_metrics(0.80);
        }
        
        let adjustment = profiler.recommend_batch_size_adjustment(32);
        assert!(adjustment.is_some());
        assert!(adjustment.unwrap() < 32);
    }

    #[test]
    fn test_adaptive_optimizer() {
        let mut optimizer = AdaptiveOptimizer::new("baseline".to_string());
        assert_eq!(optimizer.get_current_sequence(), "baseline");
        
        for _ in 0..1100 {
            optimizer.record_batch(32, Duration::from_millis(10));
            optimizer.record_cache_metrics(0.96);
        }
        
        let switch = optimizer.should_switch_sequence();
        assert_eq!(switch, Some("extended".to_string()));
    }

    #[test]
    fn test_sequence_switching() {
        let mut optimizer = AdaptiveOptimizer::new("extended".to_string());
        
        for _ in 0..1100 {
            optimizer.record_batch(32, Duration::from_millis(10));
            optimizer.record_cache_metrics(0.75);
        }
        
        let switch = optimizer.should_switch_sequence();
        assert_eq!(switch, Some("fibonacci".to_string()));
    }

    #[test]
    fn test_reset() {
        let mut profiler = RuntimeProfiler::new(10);
        profiler.record_batch(32, Duration::from_millis(10));
        profiler.record_cache_metrics(0.90);
        
        profiler.reset();
        assert_eq!(profiler.samples_processed, 0);
        assert_eq!(profiler.throughput_history.len(), 0);
    }
}
