use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadFeatures {
    pub input_size: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub cache_pressure: f64,
    pub memory_bandwidth: f64,
    pub cpu_utilization: f64,
}

impl WorkloadFeatures {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        let input_size = m * n * k;
        Self {
            input_size,
            m,
            n,
            k,
            cache_pressure: Self::estimate_cache_pressure(m, n, k),
            memory_bandwidth: 0.5,
            cpu_utilization: 0.5,
        }
    }

    fn estimate_cache_pressure(m: usize, n: usize, k: usize) -> f64 {
        let total_bytes = (m * k + k * n + m * n) * std::mem::size_of::<f32>();
        let l1_size = 32 * 1024;
        let l2_size = 256 * 1024;
        let l3_size = 8 * 1024 * 1024;

        if total_bytes < l1_size {
            0.1
        } else if total_bytes < l2_size {
            0.3
        } else if total_bytes < l3_size {
            0.6
        } else {
            0.9
        }
    }

    pub fn to_embedding(&self) -> Vec<f64> {
        vec![
            self.input_size as f64,
            self.m as f64,
            self.n as f64,
            self.k as f64,
            self.cache_pressure,
            self.memory_bandwidth,
            self.cpu_utilization,
            (self.m as f64 / self.n as f64).ln(),
            (self.k as f64 / self.m as f64).ln(),
        ]
    }

    pub fn signature(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.m.hash(&mut hasher);
        self.n.hash(&mut hasher);
        self.k.hash(&mut hasher);
        hasher.finish()
    }
}

pub struct HardwareProfiler;

impl HardwareProfiler {
    pub fn new() -> Self {
        Self
    }

    pub fn l1_miss_rate(&self) -> f64 {
        0.05
    }

    pub fn bandwidth_utilization(&self) -> f64 {
        0.5
    }

    pub fn cpu_usage(&self) -> f64 {
        0.5
    }

    pub fn gpu_available(&self) -> bool {
        false
    }

    pub fn temperature(&self) -> f64 {
        50.0
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}
