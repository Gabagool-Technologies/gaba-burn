use crate::{Result, MemoryError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use uuid::Uuid;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryLayer {
    Working,
    Episodic,
    Semantic,
    Procedural,
    Affective,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EngramState {
    Active,
    Latent,
    Consolidated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub timestamp: i64,
    pub access_count: u32,
    pub importance: f32,
    pub plasticity_trace: f32,
    pub engram_state: EngramState,
    pub last_recall_ts: i64,
    pub layer: MemoryLayer,
    pub associations: Vec<(Uuid, f32)>,
    pub metadata: HashMap<String, String>,
}

impl MemoryChunk {
    pub fn new(content: String, embedding: Vec<f32>, layer: MemoryLayer) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            timestamp: now,
            access_count: 0,
            importance: 0.5,
            plasticity_trace: 1.0,
            engram_state: EngramState::Active,
            last_recall_ts: now,
            layer,
            associations: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

pub struct GabaMemory {
    dimension: usize,
    chunks: Arc<RwLock<HashMap<Uuid, MemoryChunk>>>,
    temporal_index: Arc<RwLock<BTreeMap<i64, Vec<Uuid>>>>,
    layer_index: Arc<RwLock<HashMap<MemoryLayer, Vec<Uuid>>>>,
}

impl GabaMemory {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            chunks: Arc::new(RwLock::new(HashMap::new())),
            temporal_index: Arc::new(RwLock::new(BTreeMap::new())),
            layer_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn store(&self, chunk: MemoryChunk) -> Result<Uuid> {
        if chunk.embedding.len() != self.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimension,
                actual: chunk.embedding.len(),
            });
        }
        
        let id = chunk.id;
        let timestamp = chunk.timestamp;
        let layer = chunk.layer.clone();
        
        // Store chunk
        self.chunks.write().insert(id, chunk);
        
        // Update temporal index
        self.temporal_index.write()
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(id);
        
        // Update layer index
        self.layer_index.write()
            .entry(layer)
            .or_insert_with(Vec::new)
            .push(id);
        
        Ok(id)
    }
    
    pub fn get(&self, id: &Uuid) -> Result<MemoryChunk> {
        self.chunks.read()
            .get(id)
            .cloned()
            .ok_or(MemoryError::ChunkNotFound(*id))
    }
    
    pub fn update(&self, id: &Uuid, mut updater: impl FnMut(&mut MemoryChunk)) -> Result<()> {
        let mut chunks = self.chunks.write();
        let chunk = chunks.get_mut(id)
            .ok_or(MemoryError::ChunkNotFound(*id))?;
        updater(chunk);
        Ok(())
    }
    
    pub fn search_by_layer(&self, layer: &MemoryLayer, limit: usize) -> Vec<MemoryChunk> {
        let layer_index = self.layer_index.read();
        let chunks = self.chunks.read();
        
        layer_index.get(layer)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| chunks.get(id).cloned())
                    .take(limit)
                    .collect()
            })
            .unwrap_or_default()
    }
    
    pub fn search_by_time_range(&self, start: i64, end: i64, limit: usize) -> Vec<MemoryChunk> {
        let temporal_index = self.temporal_index.read();
        let chunks = self.chunks.read();
        
        temporal_index.range(start..=end)
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|id| chunks.get(id).cloned())
            .take(limit)
            .collect()
    }
    
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    pub fn search_similar(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        if query_embedding.len() != self.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimension,
                actual: query_embedding.len(),
            });
        }
        
        let chunks = self.chunks.read();
        let mut similarities: Vec<(Uuid, f32)> = chunks.iter()
            .map(|(id, chunk)| {
                let sim = Self::cosine_similarity(query_embedding, &chunk.embedding);
                (*id, sim)
            })
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        
        Ok(similarities)
    }
    
    pub fn count(&self) -> usize {
        self.chunks.read().len()
    }
    
    pub fn clear(&self) {
        self.chunks.write().clear();
        self.temporal_index.write().clear();
        self.layer_index.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_store_and_retrieve() {
        let memory = GabaMemory::new(128);
        let chunk = MemoryChunk::new(
            "test content".to_string(),
            vec![0.1; 128],
            MemoryLayer::Working,
        );
        
        let id = memory.store(chunk.clone()).unwrap();
        let retrieved = memory.get(&id).unwrap();
        
        assert_eq!(retrieved.content, "test content");
        assert_eq!(retrieved.layer, MemoryLayer::Working);
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let memory = GabaMemory::new(128);
        let chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 64],  // Wrong dimension
            MemoryLayer::Working,
        );
        
        assert!(memory.store(chunk).is_err());
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((GabaMemory::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((GabaMemory::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_search_similar() {
        let memory = GabaMemory::new(3);
        
        let chunk1 = MemoryChunk::new("a".to_string(), vec![1.0, 0.0, 0.0], MemoryLayer::Working);
        let chunk2 = MemoryChunk::new("b".to_string(), vec![0.9, 0.1, 0.0], MemoryLayer::Working);
        let chunk3 = MemoryChunk::new("c".to_string(), vec![0.0, 1.0, 0.0], MemoryLayer::Working);
        
        memory.store(chunk1).unwrap();
        memory.store(chunk2).unwrap();
        memory.store(chunk3).unwrap();
        
        let query = vec![1.0, 0.0, 0.0];
        let results = memory.search_similar(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].1 > results[1].1);  // First result more similar
    }
}
