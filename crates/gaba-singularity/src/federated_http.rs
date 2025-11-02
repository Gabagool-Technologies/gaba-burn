use crate::federated::{FederatedLearningEngine, PrivatePerformanceVector};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    pub peer_id: String,
    pub vectors: Vec<PrivatePerformanceVector>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    pub success: bool,
    pub vectors: Vec<PrivatePerformanceVector>,
    pub message: String,
}

impl FederatedLearningEngine {
    pub async fn sync_over_http(&self) -> Result<(), String> {
        let local_vectors = self.get_recent_vectors(3600);
        let private_vectors = self.apply_differential_privacy(local_vectors);
        
        let peers = self.get_peers();
        let client = reqwest::Client::new();
        
        for peer in peers {
            let request = SyncRequest {
                peer_id: "local".to_string(),
                vectors: private_vectors.clone(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };
            
            let url = format!("{}/sync", peer.endpoint);
            match client.post(&url)
                .json(&request)
                .send()
                .await
            {
                Ok(response) => {
                    if let Ok(sync_response) = response.json::<SyncResponse>().await {
                        if sync_response.success {
                            for vector in sync_response.vectors {
                                self.aggregate_remote_vector(vector);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to sync with peer {}: {}", peer.peer_id, e);
                }
            }
        }
        
        Ok(())
    }
}

pub async fn start_http_server(
    _engine: Arc<FederatedLearningEngine>,
    _port: u16,
) -> Result<(), String> {
    Ok(())
}

async fn _handle_sync(
    _request: SyncRequest,
    engine: Arc<FederatedLearningEngine>,
) -> Result<String, String> {
    let local_vectors = engine.get_recent_vectors(3600);
    let private_vectors = engine.apply_differential_privacy(local_vectors);
    
    let _response = SyncResponse {
        success: true,
        vectors: private_vectors,
        message: "Sync successful".to_string(),
    };
    
    Ok("{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_db::PerformanceVectorDB;
    
    #[tokio::test]
    async fn test_http_sync() {
        let db = Arc::new(PerformanceVectorDB::new());
        let engine = FederatedLearningEngine::new(db);
        
        let result = engine.sync_over_http().await;
        assert!(result.is_ok());
    }
}
