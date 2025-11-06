//! Memory pool allocator for efficient buffer reuse

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory pool for reusing allocations
pub struct MemoryPool {
    pools: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>,
    max_pool_size: usize,
    total_allocated: Arc<Mutex<usize>>,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_pool_size,
            total_allocated: Arc::new(Mutex::new(0)),
        }
    }

    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        let mut pools = self.pools.lock().unwrap();

        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                return buffer;
            }
        }

        let mut total = self.total_allocated.lock().unwrap();
        *total += size * std::mem::size_of::<f32>();

        vec![0.0; size]
    }

    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        let size = buffer.len();
        buffer.clear();
        buffer.resize(size, 0.0);

        let mut pools = self.pools.lock().unwrap();
        let pool = pools.entry(size).or_insert_with(Vec::new);

        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }

    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();

        let mut total = self.total_allocated.lock().unwrap();
        *total = 0;
    }

    pub fn total_allocated(&self) -> usize {
        *self.total_allocated.lock().unwrap()
    }

    pub fn pool_sizes(&self) -> HashMap<usize, usize> {
        let pools = self.pools.lock().unwrap();
        pools.iter().map(|(k, v)| (*k, v.len())).collect()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Scoped buffer that automatically returns to pool
pub struct PooledBuffer {
    buffer: Option<Vec<f32>>,
    pool: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>,
}

impl PooledBuffer {
    pub fn new(buffer: Vec<f32>, pool: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        self.buffer.as_ref().unwrap()
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.buffer.as_mut().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.buffer.take() {
            let size = buffer.len();
            buffer.clear();
            buffer.resize(size, 0.0);

            let mut pools = self.pool.lock().unwrap();
            let pool = pools.entry(size).or_insert_with(Vec::new);

            if pool.len() < 10 {
                pool.push(buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new(5);

        let buf1 = pool.get_buffer(100);
        assert_eq!(buf1.len(), 100);

        pool.return_buffer(buf1);

        let buf2 = pool.get_buffer(100);
        assert_eq!(buf2.len(), 100);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPool::new(5);

        for _ in 0..10 {
            let buf = pool.get_buffer(50);
            pool.return_buffer(buf);
        }

        let sizes = pool.pool_sizes();
        assert!(sizes.get(&50).unwrap() <= &5);
    }

    #[test]
    fn test_memory_pool_different_sizes() {
        let pool = MemoryPool::new(5);

        let buf1 = pool.get_buffer(100);
        let buf2 = pool.get_buffer(200);
        let buf3 = pool.get_buffer(100);

        pool.return_buffer(buf1);
        pool.return_buffer(buf2);
        pool.return_buffer(buf3);

        let sizes = pool.pool_sizes();
        assert_eq!(sizes.get(&100), Some(&2));
        assert_eq!(sizes.get(&200), Some(&1));
    }

    #[test]
    fn test_pooled_buffer_drop() {
        let pools = Arc::new(Mutex::new(HashMap::new()));
        let buffer = vec![1.0; 50];

        {
            let _pooled = PooledBuffer::new(buffer, pools.clone());
        }

        let pools_lock = pools.lock().unwrap();
        assert_eq!(pools_lock.get(&50).unwrap().len(), 1);
    }
}
