use std::time::Duration;

#[cfg(target_os = "macos")]
use metal::*;

#[cfg(target_os = "macos")]
pub struct MetalGPUExecutor {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalGPUExecutor {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("Metal not available")?;
        
        let command_queue = device.new_command_queue();
        
        let shader_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}
"#;
        
        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .map_err(|e| format!("Failed to compile shader: {}", e))?;
        
        let kernel = library.get_function("gemm_kernel", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        
        let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;
        
        Ok(Self {
            device,
            command_queue,
            pipeline_state,
        })
    }
    
    pub fn gemm_gpu(&self, a: &[f32], b: &[f32], c: &mut [f32], 
                    m: usize, n: usize, k: usize) -> Result<Duration, String> {
        let start = std::time::Instant::now();
        
        let a_size = (m * k * std::mem::size_of::<f32>()) as u64;
        let b_size = (k * n * std::mem::size_of::<f32>()) as u64;
        let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
        
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const _,
            a_size,
            MTLResourceOptions::StorageModeShared
        );
        
        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const _,
            b_size,
            MTLResourceOptions::StorageModeShared
        );
        
        let c_buffer = self.device.new_buffer(
            c_size,
            MTLResourceOptions::StorageModeShared
        );
        
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        
        let dims = [m as u32, n as u32, k as u32];
        encoder.set_bytes(3, std::mem::size_of_val(&dims) as u64, dims.as_ptr() as *const _);
        
        let threadgroup_size = MTLSize::new(16, 16, 1);
        let threadgroups = MTLSize::new(
            (n as u64 + 15) / 16,
            (m as u64 + 15) / 16,
            1
        );
        
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        unsafe {
            let ptr = c_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), m * n);
        }
        
        Ok(start.elapsed())
    }
    
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalGPUExecutor;

#[cfg(not(target_os = "macos"))]
impl MetalGPUExecutor {
    pub fn new() -> Result<Self, String> {
        Err("Metal only available on macOS".to_string())
    }
    
    pub fn gemm_gpu(&self, _a: &[f32], _b: &[f32], _c: &mut [f32], 
                    _m: usize, _n: usize, _k: usize) -> Result<Duration, String> {
        Err("Metal only available on macOS".to_string())
    }
    
    pub fn is_available() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_gemm() {
        if let Ok(executor) = MetalGPUExecutor::new() {
            let m = 64;
            let n = 64;
            let k = 64;
            let a = vec![1.0; m * k];
            let b = vec![1.0; k * n];
            let mut c = vec![0.0; m * n];
            
            let result = executor.gemm_gpu(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok());
            
            for &val in &c {
                assert!((val - 64.0).abs() < 0.1);
            }
        }
    }
}
