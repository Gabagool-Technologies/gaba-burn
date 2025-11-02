use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelType {
    RustFallback,
    RustVectorized,
    RustParallel,
    ZigOptimized,
    ZigUltra,
    Accelerate,
    MetalGPU,
    FusedReLU,
    Quantized,
}

impl KernelType {
    pub const ALL: &'static [KernelType] = &[
        KernelType::RustFallback,
        KernelType::RustVectorized,
        KernelType::RustParallel,
        KernelType::ZigOptimized,
        KernelType::ZigUltra,
        KernelType::Accelerate,
        KernelType::MetalGPU,
        KernelType::FusedReLU,
        KernelType::Quantized,
    ];
    
    pub fn from_usize(val: usize) -> Self {
        match val {
            0 => KernelType::RustFallback,
            1 => KernelType::RustVectorized,
            2 => KernelType::RustParallel,
            3 => KernelType::ZigOptimized,
            4 => KernelType::ZigUltra,
            5 => KernelType::Accelerate,
            6 => KernelType::MetalGPU,
            7 => KernelType::FusedReLU,
            8 => KernelType::Quantized,
            _ => KernelType::RustFallback,
        }
    }
    
    pub fn to_usize(self) -> usize {
        self as usize
    }
}

pub struct KernelRegistry {
    zig_available: bool,
    accelerate_available: bool,
    amx_detected: bool,
    #[cfg(feature = "metal")]
    metal_executor: Option<gaba_native_kernels::MetalGPUExecutor>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        let accelerate_available = cfg!(target_os = "macos");
        let amx_detected = if accelerate_available {
            gaba_native_kernels::detect_amx()
        } else {
            false
        };
        
        #[cfg(feature = "metal")]
        let metal_executor = gaba_native_kernels::MetalGPUExecutor::new().ok();
        
        Self {
            zig_available: cfg!(feature = "zig"),
            accelerate_available,
            amx_detected,
            #[cfg(feature = "metal")]
            metal_executor,
        }
    }
    
    pub fn execute_gemm(&self, kernel_type: KernelType, a: &[f32], b: &[f32], 
                       c: &mut [f32], m: usize, n: usize, k: usize) -> Duration {
        match kernel_type {
            KernelType::RustFallback => {
                let start = std::time::Instant::now();
                gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                start.elapsed()
            }
            KernelType::RustVectorized => {
                let start = std::time::Instant::now();
                gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                start.elapsed()
            }
            KernelType::RustParallel => {
                let start = std::time::Instant::now();
                gaba_native_kernels::gemm_rust_parallel(a, b, c, m, n, k);
                start.elapsed()
            }
            KernelType::ZigOptimized | KernelType::ZigUltra => {
                let start = std::time::Instant::now();
                if self.zig_available {
                    gaba_native_kernels::gemm(a, b, c, m, n, k);
                } else {
                    gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                }
                start.elapsed()
            }
            KernelType::Accelerate => {
                if self.accelerate_available {
                    gaba_native_kernels::gemm_accelerate(a, b, c, m, n, k)
                } else {
                    let start = std::time::Instant::now();
                    gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                    start.elapsed()
                }
            }
            KernelType::MetalGPU => {
                #[cfg(feature = "metal")]
                {
                    if let Some(ref executor) = self.metal_executor {
                        executor.gemm_gpu(a, b, c, m, n, k).unwrap_or_else(|_| {
                            let start = std::time::Instant::now();
                            gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                            start.elapsed()
                        })
                    } else {
                        let start = std::time::Instant::now();
                        gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                        start.elapsed()
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    let start = std::time::Instant::now();
                    gaba_native_kernels::gemm_rust(a, b, c, m, n, k);
                    start.elapsed()
                }
            }
            KernelType::FusedReLU => {
                gaba_native_kernels::gemm_relu_fused(a, b, c, m, n, k)
            }
            KernelType::Quantized => {
                gaba_native_kernels::gemm_quantized(a, b, c, m, n, k)
            }
        }
    }
    
    pub fn is_available(&self, kernel_type: KernelType) -> bool {
        match kernel_type {
            KernelType::RustFallback | KernelType::RustVectorized | KernelType::RustParallel => true,
            KernelType::ZigOptimized | KernelType::ZigUltra => self.zig_available,
            KernelType::Accelerate => self.accelerate_available,
            KernelType::FusedReLU | KernelType::Quantized => true,
            KernelType::MetalGPU => {
                #[cfg(feature = "metal")]
                {
                    self.metal_executor.is_some()
                }
                #[cfg(not(feature = "metal"))]
                {
                    false
                }
            }
        }
    }
    
    pub fn amx_available(&self) -> bool {
        self.amx_detected
    }
    
    pub fn metal_available(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.metal_executor.is_some()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
