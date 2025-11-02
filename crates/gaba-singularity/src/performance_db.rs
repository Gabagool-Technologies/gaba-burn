use crate::kernel_registry::KernelType;
use crate::profiler::WorkloadFeatures;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceVector {
    pub workload_features: WorkloadFeatures,
    pub kernel_type: KernelType,
    pub execution_time: Duration,
    pub timestamp: SystemTime,
}

impl PerformanceVector {
    pub fn distance_to(&self, other: &WorkloadFeatures) -> f64 {
        let self_emb = self.workload_features.to_embedding();
        let other_emb = other.to_embedding();
        
        self_emb.iter()
            .zip(other_emb.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

pub struct PerformanceVectorDB {
    vectors: Arc<DashMap<u64, Vec<PerformanceVector>>>,
    all_vectors: Arc<parking_lot::RwLock<Vec<PerformanceVector>>>,
}

impl PerformanceVectorDB {
    pub fn new() -> Self {
        Self {
            vectors: Arc::new(DashMap::new()),
            all_vectors: Arc::new(parking_lot::RwLock::new(Vec::new())),
        }
    }
    
    pub fn insert(&self, vector: PerformanceVector) {
        let sig = vector.workload_features.signature();
        
        self.vectors.entry(sig)
            .or_insert_with(Vec::new)
            .push(vector.clone());
        
        self.all_vectors.write().push(vector);
    }
    
    pub fn find_similar(&self, features: &WorkloadFeatures, k: usize) -> Vec<PerformanceVector> {
        let sig = features.signature();
        
        if let Some(exact_match) = self.vectors.get(&sig) {
            return exact_match.iter().take(k).cloned().collect();
        }
        
        let all = self.all_vectors.read();
        let mut scored: Vec<_> = all.iter()
            .map(|v| {
                let distance = v.distance_to(features);
                let age_seconds = SystemTime::now()
                    .duration_since(v.timestamp)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs();
                let recency_weight = (-(age_seconds as f64) / 86400.0).exp();
                let score = distance / recency_weight.max(0.1);
                (v.clone(), score)
            })
            .collect();
        
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(v, _)| v).collect()
    }
    
    pub fn len(&self) -> usize {
        self.all_vectors.read().len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn get_all(&self) -> Vec<PerformanceVector> {
        self.all_vectors.read().clone()
    }
}

impl Default for PerformanceVectorDB {
    fn default() -> Self {
        Self::new()
    }
}
