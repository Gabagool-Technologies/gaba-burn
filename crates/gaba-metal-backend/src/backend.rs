//! Metal Backend Implementation with Zero-Copy Architecture

use crate::error::{MetalError, MetalResult};
use crate::unified_memory::{UnifiedBuffer, UnifiedMemoryPool};
use metal::*;
use std::sync::Arc;

/// Zero-copy Metal backend for Burn
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    memory_pool: UnifiedMemoryPool,
    pipelines: PipelineCache,
}

impl MetalBackend {
    pub fn new() -> MetalResult<Self> {
        let device = Device::system_default()
            .ok_or(MetalError::DeviceNotAvailable)?;
        
        let command_queue = device.new_command_queue();
        let memory_pool = UnifiedMemoryPool::new(device.clone());
        let pipelines = PipelineCache::new(device.clone())?;
        
        Ok(Self {
            device,
            command_queue,
            memory_pool,
            pipelines,
        })
    }
    
    /// Get device handle
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }
    
    /// Allocate unified buffer with zero-copy
    pub fn allocate<T>(&mut self, len: usize) -> MetalResult<UnifiedBuffer<T>> {
        self.memory_pool.allocate(len)
    }
    
    /// Execute kernel with zero-copy buffers
    pub fn execute_kernel(
        &self,
        kernel_name: &str,
        buffers: &[&Buffer],
        thread_groups: MTLSize,
        threads_per_group: MTLSize,
    ) -> MetalResult<()> {
        let pipeline = self.pipelines.get(kernel_name)?;
        
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        
        for (idx, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(idx as u64, Some(buffer), 0);
        }
        
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
}

/// Pipeline cache for compiled Metal kernels
struct PipelineCache {
    device: Device,
    library: Library,
    cache: std::collections::HashMap<String, ComputePipelineState>,
}

impl PipelineCache {
    fn new(device: Device) -> MetalResult<Self> {
        let shader_source = include_str!("shaders/kernels.metal");
        
        let library = device.new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;
        
        Ok(Self {
            device,
            library,
            cache: std::collections::HashMap::new(),
        })
    }
    
    fn get(&mut self, name: &str) -> MetalResult<&ComputePipelineState> {
        if !self.cache.contains_key(name) {
            let function = self.library.get_function(name, None)
                .map_err(|e| MetalError::ShaderCompilation(format!("Function {} not found: {}", name, e)))?;
            
            let pipeline = self.device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::ShaderCompilation(format!("Pipeline creation failed: {}", e)))?;
            
            self.cache.insert(name.to_string(), pipeline);
        }
        
        Ok(self.cache.get(name).unwrap())
    }
}
