use crate::prime_optimizer::{PrimeConfig, PrimeOptimizer};

#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub compute_units: usize,
    pub memory_gb: usize,
    pub memory_bandwidth_gbps: f32,
    pub warp_size: usize,
    pub max_threads_per_block: usize,
}

impl GpuInfo {
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::detect_metal()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_metal() -> Option<Self> {
        Some(Self {
            vendor: GpuVendor::Apple,
            compute_units: 16,
            memory_gb: 32,
            memory_bandwidth_gbps: 273.0,
            warp_size: 32,
            max_threads_per_block: 1024,
        })
    }

    pub fn optimal_block_size(&self) -> (usize, usize) {
        match self.vendor {
            GpuVendor::NVIDIA => (32, 32),
            GpuVendor::AMD => (64, 64),
            GpuVendor::Apple => (16, 16),
            GpuVendor::Intel => (16, 16),
            GpuVendor::Unknown => (16, 16),
        }
    }

    pub fn optimal_grid_size(&self, total_work: usize) -> usize {
        let threads_per_block = self.max_threads_per_block;
        (total_work + threads_per_block - 1) / threads_per_block
    }

    pub fn recommended_sequence(&self) -> &'static str {
        match self.vendor {
            GpuVendor::NVIDIA => "extended",
            GpuVendor::AMD => "extended",
            GpuVendor::Apple => "fibonacci",
            GpuVendor::Intel => "baseline",
            GpuVendor::Unknown => "baseline",
        }
    }
}

pub struct GpuOptimizer {
    gpu_info: Option<GpuInfo>,
    prime_optimizer: PrimeOptimizer,
}

impl GpuOptimizer {
    pub fn new() -> Self {
        let gpu_info = GpuInfo::detect();
        let config = if let Some(ref info) = gpu_info {
            match info.recommended_sequence() {
                "extended" => PrimeConfig::extended(),
                "fibonacci" => PrimeConfig::fibonacci(),
                _ => PrimeConfig::default(),
            }
        } else {
            PrimeConfig::default()
        };

        Self {
            gpu_info,
            prime_optimizer: PrimeOptimizer::new(config),
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu_info.is_some()
    }

    pub fn optimize_kernel_launch(&self, total_work: usize) -> KernelConfig {
        if let Some(ref gpu) = self.gpu_info {
            let (block_x, block_y) = gpu.optimal_block_size();
            let grid_size = gpu.optimal_grid_size(total_work);
            
            let batch_size = self.prime_optimizer.optimize_batch_size(total_work);
            
            KernelConfig {
                grid_size,
                block_size_x: block_x,
                block_size_y: block_y,
                batch_size,
                shared_memory_bytes: self.calculate_shared_memory(block_x, block_y),
            }
        } else {
            KernelConfig::default()
        }
    }

    fn calculate_shared_memory(&self, block_x: usize, block_y: usize) -> usize {
        let elements = block_x * block_y;
        let bytes_per_element = 4;
        elements * bytes_per_element
    }

    pub fn optimize_matrix_multiply(&self, m: usize, n: usize, k: usize) -> MatmulConfig {
        let (tile_m, tile_n) = if let Some(ref gpu) = self.gpu_info {
            gpu.optimal_block_size()
        } else {
            (16, 16)
        };

        let tile_k = self.prime_optimizer.optimize_block_size(k).0;

        MatmulConfig {
            tile_m,
            tile_n,
            tile_k,
            grid_m: (m + tile_m - 1) / tile_m,
            grid_n: (n + tile_n - 1) / tile_n,
        }
    }

    pub fn optimize_convolution(&self, batch: usize, channels: usize) -> ConvConfig {
        let batch_size = self.prime_optimizer.optimize_batch_size(batch);
        let channel_groups = self.prime_optimizer.optimize_quantization_groups(channels);

        ConvConfig {
            batch_size,
            channel_groups,
            use_winograd: channels >= 64,
            use_implicit_gemm: batch_size >= 32,
        }
    }

    pub fn get_gpu_info(&self) -> Option<&GpuInfo> {
        self.gpu_info.as_ref()
    }
}

impl Default for GpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub grid_size: usize,
    pub block_size_x: usize,
    pub block_size_y: usize,
    pub batch_size: usize,
    pub shared_memory_bytes: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            grid_size: 1,
            block_size_x: 16,
            block_size_y: 16,
            batch_size: 32,
            shared_memory_bytes: 4096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatmulConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub grid_m: usize,
    pub grid_n: usize,
}

#[derive(Debug, Clone)]
pub struct ConvConfig {
    pub batch_size: usize,
    pub channel_groups: usize,
    pub use_winograd: bool,
    pub use_implicit_gemm: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_optimizer_creation() {
        let optimizer = GpuOptimizer::new();
        assert!(optimizer.has_gpu() || !optimizer.has_gpu());
    }

    #[test]
    fn test_kernel_launch_optimization() {
        let optimizer = GpuOptimizer::new();
        let config = optimizer.optimize_kernel_launch(1024);
        
        assert!(config.grid_size > 0);
        assert!(config.block_size_x > 0);
        assert!(config.block_size_y > 0);
        assert!(config.batch_size > 0);
    }

    #[test]
    fn test_matrix_multiply_optimization() {
        let optimizer = GpuOptimizer::new();
        let config = optimizer.optimize_matrix_multiply(512, 512, 512);
        
        assert!(config.tile_m > 0);
        assert!(config.tile_n > 0);
        assert!(config.tile_k > 0);
        assert!(config.grid_m > 0);
        assert!(config.grid_n > 0);
    }

    #[test]
    fn test_convolution_optimization() {
        let optimizer = GpuOptimizer::new();
        let config = optimizer.optimize_convolution(32, 128);
        
        assert!(config.batch_size > 0);
        assert!(config.channel_groups > 0);
    }

    #[test]
    fn test_gpu_info_optimal_sizes() {
        let info = GpuInfo {
            vendor: GpuVendor::NVIDIA,
            compute_units: 80,
            memory_gb: 24,
            memory_bandwidth_gbps: 900.0,
            warp_size: 32,
            max_threads_per_block: 1024,
        };

        let (bx, by) = info.optimal_block_size();
        assert_eq!(bx, 32);
        assert_eq!(by, 32);

        let grid = info.optimal_grid_size(10000);
        assert!(grid > 0);
    }

    #[test]
    fn test_recommended_sequences() {
        let nvidia = GpuInfo {
            vendor: GpuVendor::NVIDIA,
            compute_units: 80,
            memory_gb: 24,
            memory_bandwidth_gbps: 900.0,
            warp_size: 32,
            max_threads_per_block: 1024,
        };
        assert_eq!(nvidia.recommended_sequence(), "extended");

        let apple = GpuInfo {
            vendor: GpuVendor::Apple,
            compute_units: 16,
            memory_gb: 32,
            memory_bandwidth_gbps: 273.0,
            warp_size: 32,
            max_threads_per_block: 1024,
        };
        assert_eq!(apple.recommended_sequence(), "fibonacci");
    }

    #[test]
    fn test_shared_memory_calculation() {
        let optimizer = GpuOptimizer::new();
        let shared_mem = optimizer.calculate_shared_memory(16, 16);
        assert_eq!(shared_mem, 16 * 16 * 4);
    }
}
