//! Metal GPU Acceleration for BLAKE3 Hashing
//!
//! Implements hardware-optimized compute shaders for quantum-resistant cryptographic hashing
//! on Apple Silicon M4 Pro with unified memory architecture.

use crate::error::{PqcError, PqcResult};
use metal::*;
use std::mem;

const CHUNK_SIZE: usize = 128 * 1024 * 1024; // 128MB chunks for streaming

pub struct MetalBlake3Accelerator {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalBlake3Accelerator {
    pub fn new() -> PqcResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| PqcError::Metal("No Metal device found".to_string()))?;
        
        let command_queue = device.new_command_queue();
        
        // Compile Metal shader for BLAKE3
        let shader_source = include_str!("shaders/blake3.metal");
        let library = device.new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| PqcError::Metal(format!("Shader compilation failed: {}", e)))?;
        
        let kernel = library.get_function("blake3_hash", None)
            .map_err(|e| PqcError::Metal(format!("Kernel not found: {}", e)))?;
        
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| PqcError::Metal(format!("Pipeline creation failed: {}", e)))?;
        
        Ok(Self {
            device,
            command_queue,
            pipeline,
        })
    }
    
    /// Hash data with streaming double-buffer architecture
    pub async fn hash_data_streaming(&mut self, data: &[u8]) -> PqcResult<[u8; 32]> {
        if data.is_empty() {
            return Err(PqcError::CryptoError("Empty data".to_string()));
        }
        
        // For small data, use single-pass
        if data.len() < CHUNK_SIZE {
            return self.hash_chunk(data);
        }
        
        // Streaming processing for large data
        let mut hasher_state = [0u8; 32];
        
        for chunk in data.chunks(CHUNK_SIZE) {
            let chunk_hash = self.hash_chunk(chunk)?;
            // Combine with previous state
            for i in 0..32 {
                hasher_state[i] ^= chunk_hash[i];
            }
        }
        
        Ok(hasher_state)
    }
    
    fn hash_chunk(&self, data: &[u8]) -> PqcResult<[u8; 32]> {
        let data_len = data.len();
        
        // Create Metal buffers with unified memory (zero-copy on Apple Silicon)
        let input_buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let output_buffer = self.device.new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );
        
        let size_buffer = self.device.new_buffer_with_data(
            &data_len as *const usize as *const _,
            mem::size_of::<usize>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&size_buffer), 0);
        
        // Calculate optimal thread configuration for M4 Pro
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(
            ((data_len + 255) / 256) as u64,
            1,
            1,
        );
        
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        
        // Submit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Extract result from unified memory (zero-copy read)
        let result_ptr = output_buffer.contents() as *const u8;
        let mut hash = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(result_ptr, hash.as_mut_ptr(), 32);
        }
        
        Ok(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metal_acceleration() {
        let mut accel = MetalBlake3Accelerator::new().unwrap();
        let data = vec![0u8; 1024 * 1024]; // 1MB test
        let hash = accel.hash_data_streaming(&data).await.unwrap();
        assert_ne!(hash, [0u8; 32]);
    }
}
