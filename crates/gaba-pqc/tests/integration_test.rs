use gaba_pqc::{create_seal, verify_seal, encrypt_checkpoint, verify_checkpoint};

#[tokio::test]
async fn test_seal_creation() {
    let data = b"test data for sealing";
    let seal = create_seal(data).await.unwrap();
    
    assert_eq!(seal.hash.len(), 32);
    assert!(seal.timestamp > 0);
}

#[tokio::test]
async fn test_seal_verification() {
    let data = b"test data for verification";
    let seal = create_seal(data).await.unwrap();
    
    let valid = verify_seal(&seal, data).await.unwrap();
    assert!(valid);
    
    let invalid_data = b"modified data";
    let invalid = verify_seal(&seal, invalid_data).await.unwrap();
    assert!(!invalid);
}

#[tokio::test]
async fn test_seal_serialization() {
    let data = b"serialization test";
    let seal = create_seal(data).await.unwrap();
    
    let bytes = seal.to_bytes();
    let deserialized = gaba_pqc::seal::Seal::from_bytes(&bytes).unwrap();
    
    assert_eq!(seal.hash, deserialized.hash);
    assert_eq!(seal.timestamp, deserialized.timestamp);
}

#[tokio::test]
async fn test_checkpoint_encryption() {
    let model_data = vec![1u8; 1024 * 1024]; // 1MB model
    let encrypted = encrypt_checkpoint(&model_data).await.unwrap();
    
    assert!(encrypted.len() > 40); // At least seal size
    
    let valid = verify_checkpoint(&encrypted, &model_data).await.unwrap();
    assert!(valid);
}

#[tokio::test]
async fn test_large_data_seal() {
    let large_data = vec![42u8; 10 * 1024 * 1024]; // 10MB
    let seal = create_seal(&large_data).await.unwrap();
    
    let valid = verify_seal(&seal, &large_data).await.unwrap();
    assert!(valid);
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn test_metal_acceleration() {
    let large_data = vec![0u8; 100 * 1024 * 1024]; // 100MB - triggers Metal
    
    let start = std::time::Instant::now();
    let seal = create_seal(&large_data).await.unwrap();
    let elapsed = start.elapsed();
    
    println!("Metal-accelerated seal: {:?}", elapsed);
    assert!(seal.metadata.is_some());
    assert_eq!(seal.metadata.unwrap(), "metal-accelerated");
}
