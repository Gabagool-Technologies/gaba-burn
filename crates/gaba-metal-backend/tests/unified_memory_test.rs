#[cfg(feature = "metal")]
mod tests {
    use gaba_metal_backend::{MetalBackend, UnifiedBuffer};
    
    #[test]
    fn test_unified_buffer_creation() {
        let backend = MetalBackend::new().unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        let buffer = UnifiedBuffer::from_slice(backend.device(), &data).unwrap();
        
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_zero_copy_access() {
        let backend = MetalBackend::new().unwrap();
        let data = vec![1.0f32; 1024];
        
        let mut buffer = UnifiedBuffer::from_slice(backend.device(), &data).unwrap();
        
        // CPU write (zero-copy)
        buffer.as_mut_slice()[0] = 42.0;
        
        // CPU read (zero-copy)
        assert_eq!(buffer.as_slice()[0], 42.0);
    }
    
    #[test]
    fn test_large_buffer() {
        let backend = MetalBackend::new().unwrap();
        let size = 100 * 1024 * 1024 / 4; // 100MB of f32
        
        let mut buffer: UnifiedBuffer<f32> = backend.allocate(size).unwrap();
        
        assert_eq!(buffer.len(), size);
        
        // Write pattern
        for i in 0..1024 {
            buffer.as_mut_slice()[i] = i as f32;
        }
        
        // Verify pattern
        for i in 0..1024 {
            assert_eq!(buffer.as_slice()[i], i as f32);
        }
    }
    
    #[test]
    fn test_backend_initialization() {
        let backend = MetalBackend::new().unwrap();
        
        assert!(backend.device().name().contains("Apple"));
        println!("Metal device: {}", backend.device().name());
    }
}
