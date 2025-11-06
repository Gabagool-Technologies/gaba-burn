use crate::kernel_registry::KernelType;
use crate::profiler::WorkloadFeatures;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct JITCompiler {
    compiled_kernels: Arc<RwLock<HashMap<WorkloadSignature, CompiledKernel>>>,
    optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkloadSignature {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alignment: usize,
    pub kernel_type: KernelType,
}

impl WorkloadSignature {
    pub fn from_features(features: &WorkloadFeatures, kernel_type: KernelType) -> Self {
        Self {
            m: features.m,
            n: features.n,
            k: features.k,
            alignment: 64,
            kernel_type,
        }
    }
}

#[derive(Clone)]
pub struct CompiledKernel {
    pub signature: WorkloadSignature,
    pub code: Vec<u8>,
    pub entry_point: usize,
    pub compile_time_ms: f64,
}

impl JITCompiler {
    pub fn new() -> Self {
        Self {
            compiled_kernels: Arc::new(RwLock::new(HashMap::new())),
            optimization_level: OptimizationLevel::O2,
        }
    }

    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    pub fn compile_kernel(
        &self,
        signature: WorkloadSignature,
    ) -> Result<Arc<CompiledKernel>, String> {
        {
            let cache = self.compiled_kernels.read();
            if let Some(kernel) = cache.get(&signature) {
                return Ok(Arc::new(kernel.clone()));
            }
        }

        let start = std::time::Instant::now();

        let code = match signature.kernel_type {
            KernelType::RustVectorized => self.compile_vectorized_gemm(&signature)?,
            KernelType::RustParallel => self.compile_parallel_gemm(&signature)?,
            KernelType::FusedReLU => self.compile_fused_relu(&signature)?,
            KernelType::Quantized => self.compile_quantized_gemm(&signature)?,
            _ => return Err("JIT not supported for this kernel type".to_string()),
        };

        let compile_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let kernel = CompiledKernel {
            signature: signature.clone(),
            code,
            entry_point: 0,
            compile_time_ms,
        };

        let kernel_arc = Arc::new(kernel);

        {
            let mut cache = self.compiled_kernels.write();
            cache.insert(signature, (*kernel_arc).clone());
        }

        Ok(kernel_arc)
    }

    fn compile_vectorized_gemm(&self, sig: &WorkloadSignature) -> Result<Vec<u8>, String> {
        let mut code = Vec::new();

        let _tile_m = (sig.m.min(64) / 8) * 8;
        let _tile_n = (sig.n.min(64) / 8) * 8;
        let _tile_k = (sig.k.min(64) / 8) * 8;

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        Ok(code)
    }

    fn compile_parallel_gemm(&self, sig: &WorkloadSignature) -> Result<Vec<u8>, String> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let _chunk_size = (sig.m + num_threads - 1) / num_threads;

        let mut code = Vec::new();
        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        Ok(code)
    }

    fn compile_fused_relu(&self, _sig: &WorkloadSignature) -> Result<Vec<u8>, String> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5, 0x0f, 0x57, 0xc0]);

        Ok(code)
    }

    fn compile_quantized_gemm(&self, _sig: &WorkloadSignature) -> Result<Vec<u8>, String> {
        let mut code = Vec::new();

        code.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]);

        Ok(code)
    }

    pub fn cache_size(&self) -> usize {
        self.compiled_kernels.read().len()
    }

    pub fn clear_cache(&self) {
        self.compiled_kernels.write().clear();
    }
}

impl Default for JITCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = JITCompiler::new();
        assert_eq!(compiler.cache_size(), 0);
    }

    #[test]
    fn test_workload_signature() {
        let sig1 = WorkloadSignature {
            m: 128,
            n: 128,
            k: 128,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let sig2 = WorkloadSignature {
            m: 128,
            n: 128,
            k: 128,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_compile_kernel() {
        let compiler = JITCompiler::new();
        let sig = WorkloadSignature {
            m: 64,
            n: 64,
            k: 64,
            alignment: 64,
            kernel_type: KernelType::RustVectorized,
        };

        let result = compiler.compile_kernel(sig);
        assert!(result.is_ok());
        assert_eq!(compiler.cache_size(), 1);
    }
}
