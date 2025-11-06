use crate::Result;
use ndarray::{Array1, Array2};
use uuid::Uuid;

pub struct HopfieldLayer {
    dimension: usize,
    beta: f32,
    max_iterations: usize,
}

impl HopfieldLayer {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            beta: 1.0,
            max_iterations: 10,
        }
    }
    
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }
    
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    
    fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x: Array1<f32> = x.mapv(|v| ((v - max_val) * self.beta).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
    
    pub fn retrieve(
        &self,
        query: &[f32],
        memory_matrix: &Array2<f32>,
    ) -> Result<Array1<f32>> {
        let query_arr = Array1::from_vec(query.to_vec());
        let mut state = query_arr.clone();
        
        for _ in 0..self.max_iterations {
            let scores = memory_matrix.dot(&state);
            let attention = self.softmax(&scores);
            let new_state = memory_matrix.t().dot(&attention);
            
            let diff = (&new_state - &state).mapv(|x| x.abs()).sum();
            state = new_state;
            
            if diff < 1e-6 {
                break;
            }
        }
        
        Ok(state)
    }
    
    pub fn rerank_candidates(
        &self,
        query: &[f32],
        candidates: &[(Uuid, Vec<f32>, f32)],
    ) -> Result<Vec<(Uuid, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = candidates.len();
        let mut memory_matrix = Array2::zeros((n, self.dimension));
        
        for (i, (_, vec, _)) in candidates.iter().enumerate() {
            for (j, &val) in vec.iter().enumerate() {
                memory_matrix[[i, j]] = val;
            }
        }
        
        let retrieved = self.retrieve(query, &memory_matrix)?;
        
        let mut results: Vec<(Uuid, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(_i, (id, vec, _))| {
                let similarity: f32 = vec.iter()
                    .zip(retrieved.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (*id, similarity)
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_softmax() {
        let layer = HopfieldLayer::new(3);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = layer.softmax(&x);
        
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_retrieve() {
        let layer = HopfieldLayer::new(3);
        let query = vec![1.0, 0.0, 0.0];
        let memory = Array2::from_shape_vec((2, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]).unwrap();
        
        let result = layer.retrieve(&query, &memory).unwrap();
        assert_eq!(result.len(), 3);
    }
    
    #[test]
    fn test_rerank_candidates() {
        let layer = HopfieldLayer::new(3);
        let query = vec![1.0, 0.0, 0.0];
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        let candidates = vec![
            (id1, vec![1.0, 0.0, 0.0], 0.9),
            (id2, vec![0.0, 1.0, 0.0], 0.5),
        ];
        
        let results = layer.rerank_candidates(&query, &candidates).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1);
    }
}
