use crate::{MemoryChunk, EngramState};

const RECONSOLIDATION_PROMOTE_THRESHOLD: f32 = 5.0;
const CONSOLIDATION_THRESHOLD: f32 = 3.0;

pub struct EngramManager {
    homeostatic_budget: f32,
    current_load: f32,
}

impl EngramManager {
    pub fn new(budget: f32) -> Self {
        Self {
            homeostatic_budget: budget,
            current_load: 0.0,
        }
    }
    
    pub fn on_recall(&self, chunk: &mut MemoryChunk, now: i64) {
        chunk.plasticity_trace += 1.0;
        chunk.last_recall_ts = now;
        chunk.access_count += 1;
        
        if chunk.plasticity_trace > RECONSOLIDATION_PROMOTE_THRESHOLD {
            chunk.engram_state = EngramState::Active;
        }
    }
    
    pub fn on_consolidation(&self, chunk: &mut MemoryChunk) {
        if chunk.plasticity_trace > CONSOLIDATION_THRESHOLD {
            chunk.engram_state = EngramState::Consolidated;
            chunk.plasticity_trace *= 0.5;
        }
    }
    
    pub fn should_latentize(&self, chunk: &MemoryChunk, now: i64) -> bool {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        let recency_hours = (now - chunk.last_recall_ts) as f32 / 3600.0;
        
        age_hours > 168.0 
            && recency_hours > 72.0 
            && chunk.plasticity_trace < 1.0
            && chunk.access_count < 3
    }
    
    pub fn calculate_survival_score(&self, chunk: &MemoryChunk, now: i64, alpha: f32) -> f32 {
        let age_hours = (now - chunk.timestamp) as f32 / 3600.0;
        let age_factor = if age_hours > 0.0 {
            age_hours.powf(alpha)
        } else {
            1.0
        };
        
        let novelty_bonus = if chunk.access_count == 0 { 1.2 } else { 1.0 };
        
        (chunk.importance * chunk.plasticity_trace * novelty_bonus) / age_factor
    }
    
    pub fn update_homeostatic_load(&mut self, load: f32) {
        self.current_load = load;
    }
    
    pub fn is_over_budget(&self) -> bool {
        self.current_load > self.homeostatic_budget
    }
    
    pub fn get_budget_utilization(&self) -> f32 {
        self.current_load / self.homeostatic_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryLayer;
    
    #[test]
    fn test_on_recall() {
        let manager = EngramManager::new(1000.0);
        let now = chrono::Utc::now().timestamp();
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Working,
        );
        
        let initial_trace = chunk.plasticity_trace;
        manager.on_recall(&mut chunk, now);
        
        assert!(chunk.plasticity_trace > initial_trace);
        assert_eq!(chunk.access_count, 1);
    }
    
    #[test]
    fn test_consolidation() {
        let manager = EngramManager::new(1000.0);
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Episodic,
        );
        chunk.plasticity_trace = 5.0;
        
        manager.on_consolidation(&mut chunk);
        
        assert_eq!(chunk.engram_state, EngramState::Consolidated);
        assert!(chunk.plasticity_trace < 5.0);
    }
    
    #[test]
    fn test_survival_score() {
        let manager = EngramManager::new(1000.0);
        let now = chrono::Utc::now().timestamp();
        
        let mut chunk = MemoryChunk::new(
            "test".to_string(),
            vec![0.1; 128],
            MemoryLayer::Working,
        );
        chunk.importance = 0.8;
        chunk.plasticity_trace = 2.0;
        
        let score = manager.calculate_survival_score(&chunk, now, 0.5);
        assert!(score > 0.0);
    }
}
