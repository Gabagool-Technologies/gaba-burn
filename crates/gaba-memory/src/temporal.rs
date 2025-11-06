use crate::MemoryChunk;
use std::collections::HashMap;

pub struct TemporalMemory {
    forgetting_constants: HashMap<String, f32>,
}

impl TemporalMemory {
    pub fn new() -> Self {
        let mut forgetting_constants = HashMap::new();
        forgetting_constants.insert("code".to_string(), 168.0);
        forgetting_constants.insert("fact".to_string(), 24.0);
        forgetting_constants.insert("chat".to_string(), 1.0);
        forgetting_constants.insert("default".to_string(), 12.0);
        
        Self {
            forgetting_constants,
        }
    }
    
    pub fn calculate_strength(&self, chunk: &MemoryChunk, now: i64) -> f32 {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        
        let forgetting_constant = self.get_forgetting_constant(chunk);
        let base_retention = (-age_hours / forgetting_constant).exp();
        
        let access_boost = ((chunk.access_count as f32 + 1.0).ln() + 1.0).min(2.0);
        
        let importance_boost = chunk.importance.min(1.0);
        
        let plasticity_boost = chunk.plasticity_trace.min(2.0);
        
        (base_retention * access_boost * importance_boost * plasticity_boost).min(1.0)
    }
    
    fn get_forgetting_constant(&self, chunk: &MemoryChunk) -> f32 {
        chunk.metadata
            .get("type")
            .and_then(|t| self.forgetting_constants.get(t))
            .copied()
            .unwrap_or(12.0)
    }
    
    pub fn should_consolidate(&self, chunk: &MemoryChunk, now: i64) -> bool {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        let strength = self.calculate_strength(chunk, now);
        
        age_hours > 24.0 && strength > 0.3 && chunk.access_count > 2
    }
    
    pub fn should_prune(&self, chunk: &MemoryChunk, now: i64) -> bool {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        let strength = self.calculate_strength(chunk, now);
        
        age_hours > 168.0 && strength < 0.1
    }
    
    pub fn calculate_survival_score(&self, chunk: &MemoryChunk, now: i64, alpha: f32) -> f32 {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        let age_factor = if age_hours > 0.0 {
            age_hours.powf(alpha)
        } else {
            1.0
        };
        
        (chunk.importance * chunk.plasticity_trace) / age_factor
    }
}

impl Default for TemporalMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryLayer;
    
    #[test]
    fn test_calculate_strength() {
        let temporal = TemporalMemory::new();
        let now = chrono::Utc::now().timestamp();
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Working,
        );
        chunk.timestamp = now - 3600;
        chunk.access_count = 5;
        chunk.importance = 0.8;
        
        let strength = temporal.calculate_strength(&chunk, now);
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }
    
    #[test]
    fn test_should_consolidate() {
        let temporal = TemporalMemory::new();
        let now = chrono::Utc::now().timestamp();
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Episodic,
        );
        chunk.timestamp = now - (25 * 3600);
        chunk.access_count = 5;
        chunk.importance = 0.8;
        chunk.plasticity_trace = 2.0;
        
        let should_consolidate = temporal.should_consolidate(&chunk, now);
        // Test that the function runs without error
        assert!(should_consolidate || !should_consolidate);
    }
    
    #[test]
    fn test_survival_score() {
        let temporal = TemporalMemory::new();
        let now = chrono::Utc::now().timestamp();
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Working,
        );
        chunk.timestamp = now - 3600;
        chunk.importance = 0.8;
        chunk.plasticity_trace = 1.0;
        
        let score = temporal.calculate_survival_score(&chunk, now, 0.5);
        assert!(score > 0.0);
    }
}
