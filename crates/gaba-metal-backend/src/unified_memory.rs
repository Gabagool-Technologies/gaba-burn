//! Unified Memory Architecture for Zero-Copy Operations
//!
//! Leverages Apple Silicon's unified memory to eliminate CPU↔GPU transfers.

use crate::error::{MetalError, MetalResult};
use metal::*;
use std::marker::PhantomData;

/// Zero-copy buffer with unified memory backing
pub struct UnifiedBuffer<T> {
    buffer: Buffer,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> UnifiedBuffer<T> {
    /// Create unified buffer from CPU data (zero-copy on Apple Silicon)
    pub fn from_slice(device: &Device, data: &[T]) -> MetalResult<Self> {
        let byte_len = data.len() * std::mem::size_of::<T>();
        
        // StorageModeShared enables zero-copy unified memory access
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        Ok(Self {
            buffer,
            len: data.len(),
            _phantom: PhantomData,
        })
    }
    
    /// Create empty unified buffer
    pub fn new(device: &Device, len: usize) -> MetalResult<Self> {
        let byte_len = len * std::mem::size_of::<T>();
        
        let buffer = device.new_buffer(
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        Ok(Self {
            buffer,
            len,
            _phantom: PhantomData,
        })
    }
    
    /// Zero-copy read from unified memory
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents() as *const T,
                self.len,
            )
        }
    }
    
    /// Zero-copy mutable access to unified memory
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents() as *mut T,
                self.len,
            )
        }
    }
    
    /// Get Metal buffer for GPU operations
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }
    
    /// Get buffer length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Unified memory pool for efficient allocation
pub struct UnifiedMemoryPool {
    device: Device,
    pools: Vec<Buffer>,
}

impl UnifiedMemoryPool {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            pools: Vec::new(),
        }
    }
    
    /// Allocate from pool with zero-copy semantics
    pub fn allocate<T>(&mut self, len: usize) -> MetalResult<UnifiedBuffer<T>> {
        UnifiedBuffer::new(&self.device, len)
    }
    
    /// Pre-allocate pool buffers for common sizes
    pub fn preallocate(&mut self, sizes: &[usize]) {
        for &size in sizes {
            let buffer = self.device.new_buffer(
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.pools.push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_buffer() {
        let device = Device::system_default().unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        let buffer = UnifiedBuffer::from_slice(&device, &data).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
