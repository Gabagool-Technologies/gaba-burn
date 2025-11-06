use crate::error::{PqcError, PqcResult};
use blake3::Hasher;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
const METAL_THRESHOLD: usize = 5 * 1024 * 1024; // 5MB

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Seal {
    pub hash: [u8; 32],
    pub timestamp: u64,
    pub metadata: Option<String>,
}

/// Create quantum-resistant seal with adaptive Metal acceleration
pub async fn create_seal(data: &[u8]) -> PqcResult<Seal> {
    #[cfg(feature = "metal")]
    {
        if data.len() >= METAL_THRESHOLD {
            return create_seal_metal(data).await;
        }
    }

    create_seal_cpu(data)
}

/// CPU-based sealing (foundation)
fn create_seal_cpu(data: &[u8]) -> PqcResult<Seal> {
    let mut hasher = Hasher::new();
    hasher.update(data);
    let hash = hasher.finalize();

    Ok(Seal {
        hash: *hash.as_bytes(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        metadata: None,
    })
}

/// Metal-accelerated sealing for large files
#[cfg(feature = "metal")]
async fn create_seal_metal(data: &[u8]) -> PqcResult<Seal> {
    use crate::metal_accel::MetalBlake3Accelerator;

    let mut accelerator = MetalBlake3Accelerator::new()
        .map_err(|e| PqcError::Metal(format!("Failed to initialize Metal: {}", e)))?;

    let hash = accelerator
        .hash_data_streaming(data)
        .await
        .map_err(|e| PqcError::Metal(format!("Metal hashing failed: {}", e)))?;

    Ok(Seal {
        hash,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        metadata: Some("metal-accelerated".to_string()),
    })
}

/// Verify seal against data
pub async fn verify_seal(seal: &Seal, data: &[u8]) -> PqcResult<bool> {
    let current_seal = create_seal(data).await?;
    Ok(current_seal.hash == seal.hash)
}

impl Seal {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.hash);
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        if let Some(ref meta) = self.metadata {
            let meta_bytes = meta.as_bytes();
            bytes.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(meta_bytes);
        } else {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }

        bytes
    }

    pub fn from_bytes(data: &[u8]) -> PqcResult<Self> {
        if data.len() < 40 {
            return Err(PqcError::InvalidFormat("Seal data too short".to_string()));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[0..32]);

        let timestamp = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]);

        let metadata = if data.len() > 40 {
            let meta_len = u32::from_le_bytes([data[40], data[41], data[42], data[43]]) as usize;
            if meta_len > 0 && data.len() >= 44 + meta_len {
                Some(String::from_utf8_lossy(&data[44..44 + meta_len]).to_string())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Seal {
            hash,
            timestamp,
            metadata,
        })
    }
}
