use crate::kernel_registry::{KernelRegistry, KernelType};
use crate::performance_db::{PerformanceVector, PerformanceVectorDB};
use crate::profiler::{HardwareProfiler, WorkloadFeatures};
use crate::selector::KernelSelector;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub struct AdaptiveKernelOrchestrator {
    performance_db: Arc<PerformanceVectorDB>,
    kernels: KernelRegistry,
    selector: parking_lot::RwLock<KernelSelector>,
    #[allow(dead_code)]
    profiler: HardwareProfiler,
    learning_enabled: bool,
}

impl AdaptiveKernelOrchestrator {
    pub fn new() -> Self {
        let orchestrator = Self {
            performance_db: Arc::new(PerformanceVectorDB::new()),
            kernels: KernelRegistry::new(),
            selector: parking_lot::RwLock::new(KernelSelector::new()),
            profiler: HardwareProfiler::new(),
            learning_enabled: true,
        };
        
        orchestrator.warm_up();
        orchestrator
    }
    
    fn warm_up(&self) {
        let a = vec![1.0f32; 64 * 64];
        let b = vec![1.0f32; 64 * 64];
        let mut c = vec![0.0f32; 64 * 64];
        
        for kernel_type in KernelType::ALL {
            if self.kernels.is_available(*kernel_type) {
                let _ = self.kernels.execute_gemm(*kernel_type, &a, &b, &mut c, 64, 64, 64);
            }
        }
    }
    
    pub fn with_learning(mut self, enabled: bool) -> Self {
        self.learning_enabled = enabled;
        self
    }
    
    pub fn select_optimal_kernel(&self, features: &WorkloadFeatures) -> KernelType {
        let similar_workloads = self.performance_db.find_similar(features, 10);
        
        let mut selector = self.selector.write();
        let predicted = selector.predict(features, &similar_workloads);
        
        if !self.kernels.is_available(predicted) {
            return self.fallback_kernel(features);
        }
        
        predicted
    }
    
    fn fallback_kernel(&self, features: &WorkloadFeatures) -> KernelType {
        let input_size = features.input_size;
        
        if input_size > 256 * 256 * 256 {
            KernelType::RustParallel
        } else if input_size > 128 * 128 * 128 {
            KernelType::ZigUltra
        } else {
            KernelType::RustVectorized
        }
    }
    
    pub fn execute_gemm_adaptive(&self, a: &[f32], b: &[f32], c: &mut [f32], 
                                 m: usize, n: usize, k: usize) -> (KernelType, Duration) {
        let features = WorkloadFeatures::new(m, n, k);
        let kernel_type = self.select_optimal_kernel(&features);
        
        let execution_time = self.kernels.execute_gemm(kernel_type, a, b, c, m, n, k);
        
        if self.learning_enabled {
            self.learn_from_execution(&features, kernel_type, execution_time);
        }
        
        (kernel_type, execution_time)
    }
    
    pub fn learn_from_execution(&self, features: &WorkloadFeatures, 
                               kernel_used: KernelType, 
                               execution_time: Duration) {
        self.performance_db.insert(PerformanceVector {
            workload_features: features.clone(),
            kernel_type: kernel_used,
            execution_time,
            timestamp: SystemTime::now(),
        });
        
        let mut selector = self.selector.write();
        selector.update_q_value(features, kernel_used, execution_time);
    }
    
    pub fn get_performance_history(&self) -> Vec<PerformanceVector> {
        self.performance_db.get_all()
    }
    
    pub fn history_size(&self) -> usize {
        self.performance_db.len()
    }
}

impl Default for AdaptiveKernelOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}
