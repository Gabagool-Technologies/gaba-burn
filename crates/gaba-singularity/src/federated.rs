use crate::kernel_registry::KernelType;
use crate::performance_db::{PerformanceVector, PerformanceVectorDB};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::SystemTime;

pub struct FederatedLearningEngine {
    local_db: Arc<PerformanceVectorDB>,
    peers: Arc<RwLock<Vec<PeerConnection>>>,
    privacy_budget: f64,
    noise_scale: f64,
}

#[derive(Clone)]
pub struct PeerConnection {
    pub peer_id: String,
    pub endpoint: String,
    pub last_sync: SystemTime,
    pub trust_score: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrivatePerformanceVector {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub kernel_type: KernelType,
    pub execution_time_ms: f64,
    pub noise_added: f64,
}

impl FederatedLearningEngine {
    pub fn new(local_db: Arc<PerformanceVectorDB>) -> Self {
        Self {
            local_db,
            peers: Arc::new(RwLock::new(Vec::new())),
            privacy_budget: 1.0,
            noise_scale: 0.1,
        }
    }

    pub fn with_privacy_budget(mut self, budget: f64) -> Self {
        self.privacy_budget = budget;
        self
    }

    pub fn add_peer(&self, peer: PeerConnection) {
        let mut peers = self.peers.write();
        peers.push(peer);
    }

    pub fn remove_peer(&self, peer_id: &str) {
        let mut peers = self.peers.write();
        peers.retain(|p| p.peer_id != peer_id);
    }

    pub fn apply_differential_privacy(
        &self,
        vectors: Vec<PerformanceVector>,
    ) -> Vec<PrivatePerformanceVector> {
        vectors
            .into_iter()
            .map(|v| {
                let noise = self.generate_laplace_noise();
                let noisy_time = (v.execution_time.as_secs_f64() * 1000.0 + noise).max(0.0);

                PrivatePerformanceVector {
                    m: v.workload_features.m,
                    n: v.workload_features.n,
                    k: v.workload_features.k,
                    kernel_type: v.kernel_type,
                    execution_time_ms: noisy_time,
                    noise_added: noise,
                }
            })
            .collect()
    }

    fn generate_laplace_noise(&self) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let u: f64 = rng.gen_range(-0.5..0.5);
        let sign = if u >= 0.0 { 1.0 } else { -1.0 };

        -sign * self.noise_scale * u.abs().ln()
    }

    pub fn aggregate_knowledge(
        &self,
        private_vectors: Vec<PrivatePerformanceVector>,
    ) -> Vec<AggregatedKnowledge> {
        use std::collections::HashMap;

        let mut aggregates: HashMap<(usize, usize, usize, KernelType), Vec<f64>> = HashMap::new();

        for pv in private_vectors {
            let key = (pv.m, pv.n, pv.k, pv.kernel_type);
            aggregates
                .entry(key)
                .or_insert_with(Vec::new)
                .push(pv.execution_time_ms);
        }

        aggregates
            .into_iter()
            .map(|((m, n, k, kernel_type), times)| {
                let median = {
                    let mut sorted = times.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted[sorted.len() / 2]
                };

                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let variance =
                    times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;

                AggregatedKnowledge {
                    m,
                    n,
                    k,
                    kernel_type,
                    median_time_ms: median,
                    mean_time_ms: mean,
                    variance: variance,
                    sample_count: times.len(),
                }
            })
            .collect()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    pub fn get_recent_vectors(&self, seconds: u64) -> Vec<PerformanceVector> {
        let cutoff = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - seconds;

        self.local_db
            .get_all()
            .into_iter()
            .filter(|v| {
                v.timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    >= cutoff
            })
            .collect()
    }

    pub fn get_peers(&self) -> Vec<PeerConnection> {
        self.peers.read().clone()
    }

    pub fn aggregate_remote_vector(&self, _vector: PrivatePerformanceVector) {
        // Aggregate remote performance data into local knowledge base
        // This would update the local performance predictions
    }
}

#[derive(Clone, Debug)]
pub struct AggregatedKnowledge {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub kernel_type: KernelType,
    pub median_time_ms: f64,
    pub mean_time_ms: f64,
    pub variance: f64,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::WorkloadFeatures;
    use std::time::Duration;

    #[test]
    fn test_federated_engine_creation() {
        let db = Arc::new(PerformanceVectorDB::new());
        let engine = FederatedLearningEngine::new(db);
        assert_eq!(engine.peer_count(), 0);
    }

    #[test]
    fn test_add_remove_peer() {
        let db = Arc::new(PerformanceVectorDB::new());
        let engine = FederatedLearningEngine::new(db);

        let peer = PeerConnection {
            peer_id: "peer1".to_string(),
            endpoint: "localhost:8080".to_string(),
            last_sync: SystemTime::now(),
            trust_score: 1.0,
        };

        engine.add_peer(peer);
        assert_eq!(engine.peer_count(), 1);

        engine.remove_peer("peer1");
        assert_eq!(engine.peer_count(), 0);
    }

    #[test]
    fn test_differential_privacy() {
        let db = Arc::new(PerformanceVectorDB::new());
        let engine = FederatedLearningEngine::new(db);

        let vector = PerformanceVector {
            workload_features: WorkloadFeatures::new(64, 64, 64),
            kernel_type: KernelType::RustVectorized,
            execution_time: Duration::from_millis(10),
            timestamp: SystemTime::now(),
        };

        let private_vectors = engine.apply_differential_privacy(vec![vector]);
        assert_eq!(private_vectors.len(), 1);
        assert!(private_vectors[0].execution_time_ms > 0.0);
    }

    #[test]
    fn test_aggregate_knowledge() {
        let db = Arc::new(PerformanceVectorDB::new());
        let engine = FederatedLearningEngine::new(db);

        let private_vectors = vec![
            PrivatePerformanceVector {
                m: 64,
                n: 64,
                k: 64,
                kernel_type: KernelType::RustVectorized,
                execution_time_ms: 10.0,
                noise_added: 0.1,
            },
            PrivatePerformanceVector {
                m: 64,
                n: 64,
                k: 64,
                kernel_type: KernelType::RustVectorized,
                execution_time_ms: 12.0,
                noise_added: 0.2,
            },
        ];

        let aggregated = engine.aggregate_knowledge(private_vectors);
        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0].sample_count, 2);
    }
}
