use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;
use uuid::Uuid;
use parking_lot::RwLock;

pub struct AssociativeGraph {
    graph: RwLock<Graph<Uuid, f32>>,
    node_map: RwLock<HashMap<Uuid, NodeIndex>>,
}

impl AssociativeGraph {
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(Graph::new()),
            node_map: RwLock::new(HashMap::new()),
        }
    }
    
    pub fn add_node(&self, id: Uuid) -> NodeIndex {
        let mut graph = self.graph.write();
        let mut node_map = self.node_map.write();
        
        if let Some(&idx) = node_map.get(&id) {
            return idx;
        }
        
        let idx = graph.add_node(id);
        node_map.insert(id, idx);
        idx
    }
    
    pub fn strengthen_association(&self, a: Uuid, b: Uuid, amount: f32) {
        let idx_a = self.add_node(a);
        let idx_b = self.add_node(b);
        
        let mut graph = self.graph.write();
        
        if let Some(edge) = graph.find_edge(idx_a, idx_b) {
            if let Some(weight) = graph.edge_weight_mut(edge) {
                *weight += amount;
                *weight = weight.min(1.0);
            }
        } else {
            graph.add_edge(idx_a, idx_b, amount);
        }
    }
    
    pub fn spread_activation(
        &self,
        seeds: &[Uuid],
        threshold: f32,
        max_hops: usize,
    ) -> Vec<(Uuid, f32)> {
        let graph = self.graph.read();
        let node_map = self.node_map.read();
        
        let mut activations: HashMap<Uuid, f32> = HashMap::new();
        
        for seed in seeds {
            activations.insert(*seed, 1.0);
        }
        
        for hop in 0..max_hops {
            let mut new_activations = activations.clone();
            
            for (chunk_id, activation) in &activations {
                if *activation < threshold {
                    continue;
                }
                
                if let Some(&node_idx) = node_map.get(chunk_id) {
                    let mut neighbors = graph.neighbors(node_idx).detach();
                    while let Some((edge_idx, neighbor_idx)) = neighbors.next(&*graph) {
                        if let Some(&weight) = graph.edge_weight(edge_idx) {
                            let neighbor_id = graph[neighbor_idx];
                            let decay = 0.7_f32.powi(hop as i32);
                            let new_activation = activation * weight * decay;
                            
                            new_activations
                                .entry(neighbor_id)
                                .and_modify(|a| *a = a.max(new_activation))
                                .or_insert(new_activation);
                        }
                    }
                }
            }
            
            activations = new_activations;
        }
        
        let mut results: Vec<_> = activations.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
    
    pub fn decay_associations(&self, decay_rate: f32) {
        let mut graph = self.graph.write();
        let edges: Vec<_> = graph.edge_indices().collect();
        
        for edge in edges {
            if let Some(weight) = graph.edge_weight_mut(edge) {
                *weight *= 1.0 - decay_rate;
                
                if *weight < 0.01 {
                    graph.remove_edge(edge);
                }
            }
        }
    }
    
    pub fn get_associations(&self, id: &Uuid) -> Vec<(Uuid, f32)> {
        let graph = self.graph.read();
        let node_map = self.node_map.read();
        
        if let Some(&node_idx) = node_map.get(id) {
            let mut neighbors = graph.neighbors(node_idx).detach();
            let mut results = Vec::new();
            
            while let Some((edge_idx, neighbor_idx)) = neighbors.next(&*graph) {
                if let Some(&weight) = graph.edge_weight(edge_idx) {
                    let neighbor_id = graph[neighbor_idx];
                    results.push((neighbor_id, weight));
                }
            }
            
            results
        } else {
            Vec::new()
        }
    }
    
    pub fn node_count(&self) -> usize {
        self.node_map.read().len()
    }
    
    pub fn edge_count(&self) -> usize {
        self.graph.read().edge_count()
    }
}

impl Default for AssociativeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strengthen_association() {
        let graph = AssociativeGraph::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        graph.strengthen_association(id1, id2, 0.5);
        
        let assocs = graph.get_associations(&id1);
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].0, id2);
        assert!((assocs[0].1 - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_spread_activation() {
        let graph = AssociativeGraph::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        
        graph.strengthen_association(id1, id2, 0.8);
        graph.strengthen_association(id2, id3, 0.6);
        
        let results = graph.spread_activation(&[id1], 0.1, 2);
        
        assert!(results.len() >= 2);
        assert_eq!(results[0].0, id1);
    }
    
    #[test]
    fn test_decay_associations() {
        let graph = AssociativeGraph::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        graph.strengthen_association(id1, id2, 0.5);
        graph.decay_associations(0.5);
        
        let assocs = graph.get_associations(&id1);
        assert!((assocs[0].1 - 0.25).abs() < 1e-6);
    }
}
