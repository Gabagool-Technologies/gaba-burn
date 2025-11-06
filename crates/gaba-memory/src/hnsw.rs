use crate::{Result, MemoryError};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use uuid::Uuid;
use parking_lot::RwLock;

#[derive(Clone)]
struct Neighbor {
    id: Uuid,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct HnswIndex {
    dimension: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    vectors: RwLock<HashMap<Uuid, Vec<f32>>>,
    graph: RwLock<HashMap<Uuid, Vec<Vec<Uuid>>>>,
    entry_point: RwLock<Option<Uuid>>,
}

impl HnswIndex {
    pub fn new(dimension: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dimension,
            m,
            ef_construction,
            ef_search: ef_construction,
            vectors: RwLock::new(HashMap::new()),
            graph: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
        }
    }
    
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }
    
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            1.0 - (dot / (norm_a * norm_b))
        }
    }
    
    fn get_random_level(&self) -> usize {
        let ml = 1.0 / (self.m as f32).ln();
        let r: f32 = rand::random();
        (-r.ln() * ml).floor() as usize
    }
    
    pub fn insert(&self, id: Uuid, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        
        let level = self.get_random_level();
        
        self.vectors.write().insert(id, vector.clone());
        
        let mut graph = self.graph.write();
        graph.entry(id).or_insert_with(|| vec![Vec::new(); level + 1]);
        
        let mut entry_point = self.entry_point.write();
        if entry_point.is_none() {
            *entry_point = Some(id);
            return Ok(());
        }
        
        Ok(())
    }
    
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        if query.len() != self.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        
        let entry_point = self.entry_point.read();
        if entry_point.is_none() {
            return Ok(Vec::new());
        }
        
        let vectors = self.vectors.read();
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        
        for (id, vec) in vectors.iter() {
            let dist = self.distance(query, vec);
            candidates.push(Neighbor {
                id: *id,
                distance: dist,
            });
            visited.insert(*id);
        }
        
        let results: Vec<(Uuid, f32)> = candidates
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect();
        
        Ok(results)
    }
    
    pub fn count(&self) -> usize {
        self.vectors.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_insert_and_search() {
        let index = HnswIndex::new(3, 16, 200);
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        
        index.insert(id1, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(id2, vec![0.9, 0.1, 0.0]).unwrap();
        index.insert(id3, vec![0.0, 1.0, 0.0]).unwrap();
        
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(index.count(), 3);
    }
}
