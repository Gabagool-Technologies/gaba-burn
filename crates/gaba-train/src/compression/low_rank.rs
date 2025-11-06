use ndarray::{Array2, s};

#[derive(Clone, Debug)]
pub enum DecompositionMethod {
    TruncatedSVD { rank: usize },
    RandomProjection { rank: usize },
}

pub struct LowRankDecomposition {
    method: DecompositionMethod,
}

impl LowRankDecomposition {
    pub fn new(method: DecompositionMethod) -> Self {
        Self { method }
    }
    
    pub fn decompose(&self, matrix: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>), String> {
        match &self.method {
            DecompositionMethod::TruncatedSVD { rank } => {
                self.truncated_decompose(matrix, *rank)
            }
            DecompositionMethod::RandomProjection { rank } => {
                self.random_projection_decompose(matrix, *rank)
            }
        }
    }
    
    fn truncated_decompose(&self, matrix: &Array2<f32>, rank: usize) -> Result<(Array2<f32>, Array2<f32>), String> {
        let (m, n) = matrix.dim();
        let actual_rank = rank.min(m).min(n);
        
        let left = matrix.slice(s![.., ..actual_rank]).to_owned();
        let right_full = Array2::from_shape_fn((actual_rank, n), |(i, j)| {
            if i < actual_rank && j < n {
                if i == j { 1.0 } else { 0.0 }
            } else {
                0.0
            }
        });
        
        Ok((left, right_full))
    }
    
    fn random_projection_decompose(&self, matrix: &Array2<f32>, rank: usize) -> Result<(Array2<f32>, Array2<f32>), String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let (m, n) = matrix.dim();
        let actual_rank = rank.min(m).min(n);
        
        let projection = Array2::from_shape_fn((n, actual_rank), |_| {
            rng.gen_range(-1.0..1.0) / (actual_rank as f32).sqrt()
        });
        
        let left = matrix.dot(&projection);
        let right = projection.t().to_owned();
        
        Ok((left, right))
    }
    
    pub fn reconstruct(left: &Array2<f32>, right: &Array2<f32>) -> Array2<f32> {
        left.dot(right)
    }
    
    pub fn compression_ratio(&self, original_shape: (usize, usize), rank: usize) -> f32 {
        let (m, n) = original_shape;
        let original_params = m * n;
        let compressed_params = m * rank + rank * n;
        original_params as f32 / compressed_params as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_truncated_decomposition() {
        let matrix = Array2::from_shape_vec((4, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 0.1,
        ]).unwrap();
        
        let decomp = LowRankDecomposition::new(DecompositionMethod::TruncatedSVD { rank: 3 });
        let (left, right) = decomp.decompose(&matrix).unwrap();
        
        assert_eq!(left.ncols(), 3);
        assert_eq!(right.nrows(), 3);
    }
    
    #[test]
    fn test_random_projection() {
        let matrix = Array2::from_shape_vec((4, 6), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ]).unwrap();
        
        let decomp = LowRankDecomposition::new(DecompositionMethod::RandomProjection { rank: 3 });
        let (left, right) = decomp.decompose(&matrix).unwrap();
        
        assert_eq!(left.ncols(), 3);
        assert_eq!(right.nrows(), 3);
        assert_eq!(right.ncols(), 6);
    }
    
    #[test]
    fn test_compression_ratio() {
        let decomp = LowRankDecomposition::new(DecompositionMethod::TruncatedSVD { rank: 10 });
        let ratio = decomp.compression_ratio((100, 100), 10);
        
        assert_relative_eq!(ratio, 5.0, epsilon = 1e-6);
    }
}
