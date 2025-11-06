use crate::{Result, ServeError};
use ndarray::Array2;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_wait_time_ms: u64,
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_time_ms: 10,
            min_batch_size: 1,
        }
    }
}

pub struct InferenceRequest {
    pub id: u64,
    pub input: Array2<f32>,
    pub timestamp: Instant,
}

pub struct InferenceResponse {
    pub id: u64,
    pub output: Array2<f32>,
    pub latency_ms: u64,
}

pub struct DynamicBatcher {
    config: BatchConfig,
    queue: Arc<Mutex<VecDeque<InferenceRequest>>>,
    notify: Arc<Notify>,
    next_id: Arc<Mutex<u64>>,
}

impl DynamicBatcher {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn submit_request(&self, input: Array2<f32>) -> Result<u64> {
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let request = InferenceRequest {
            id,
            input,
            timestamp: Instant::now(),
        };

        {
            let mut queue = self.queue.lock().unwrap();
            if queue.len() >= self.config.max_batch_size * 2 {
                return Err(ServeError::BatchFull);
            }
            queue.push_back(request);
        }

        self.notify.notify_one();
        Ok(id)
    }

    pub async fn get_batch(&self) -> Result<Vec<InferenceRequest>> {
        let start = Instant::now();
        let max_wait = Duration::from_millis(self.config.max_wait_time_ms);

        loop {
            {
                let mut queue = self.queue.lock().unwrap();
                if queue.len() >= self.config.min_batch_size {
                    let batch_size = queue.len().min(self.config.max_batch_size);
                    let batch: Vec<_> = queue.drain(..batch_size).collect();
                    return Ok(batch);
                }
            }

            if start.elapsed() >= max_wait {
                let mut queue = self.queue.lock().unwrap();
                if !queue.is_empty() {
                    let batch_size = queue.len().min(self.config.max_batch_size);
                    let batch: Vec<_> = queue.drain(..batch_size).collect();
                    return Ok(batch);
                }
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    pub fn queue_size(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    pub fn adaptive_batch_size(&self) -> usize {
        let queue_size = self.queue_size();
        
        if queue_size < self.config.min_batch_size {
            self.config.min_batch_size
        } else if queue_size > self.config.max_batch_size {
            self.config.max_batch_size
        } else {
            queue_size
        }
    }

    pub fn throughput_estimate(&self, avg_latency_ms: f64) -> f64 {
        let batch_size = self.adaptive_batch_size() as f64;
        (batch_size / avg_latency_ms) * 1000.0
    }
}

pub fn combine_batch_inputs(requests: &[InferenceRequest]) -> Result<Array2<f32>> {
    if requests.is_empty() {
        return Err(ServeError::Other("Empty batch".to_string()));
    }

    let (_, input_dim) = requests[0].input.dim();
    let batch_size = requests.len();
    
    let mut combined = Array2::<f32>::zeros((batch_size, input_dim));
    
    for (i, request) in requests.iter().enumerate() {
        if request.input.ncols() != input_dim {
            return Err(ServeError::Other("Input dimension mismatch".to_string()));
        }
        combined.row_mut(i).assign(&request.input.row(0));
    }
    
    Ok(combined)
}

pub fn split_batch_outputs(
    output: Array2<f32>,
    requests: &[InferenceRequest],
) -> Vec<InferenceResponse> {
    let mut responses = Vec::new();
    
    for (i, request) in requests.iter().enumerate() {
        let row = output.row(i).to_owned().insert_axis(ndarray::Axis(0));
        let latency_ms = request.timestamp.elapsed().as_millis() as u64;
        
        responses.push(InferenceResponse {
            id: request.id,
            output: row,
            latency_ms,
        });
    }
    
    responses
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[tokio::test]
    async fn test_batcher_creation() {
        let config = BatchConfig::default();
        let batcher = DynamicBatcher::new(config);
        assert_eq!(batcher.queue_size(), 0);
    }

    #[tokio::test]
    async fn test_submit_request() {
        let config = BatchConfig::default();
        let batcher = DynamicBatcher::new(config);
        
        let input = Array::from_shape_vec((1, 10), vec![0.1; 10]).unwrap();
        let result = batcher.submit_request(input).await;
        
        assert!(result.is_ok());
        assert_eq!(batcher.queue_size(), 1);
    }

    #[tokio::test]
    async fn test_get_batch() {
        let config = BatchConfig {
            max_batch_size: 4,
            max_wait_time_ms: 100,
            min_batch_size: 2,
        };
        let batcher = DynamicBatcher::new(config);
        
        for _ in 0..3 {
            let input = Array::from_shape_vec((1, 10), vec![0.1; 10]).unwrap();
            batcher.submit_request(input).await.unwrap();
        }
        
        let batch = batcher.get_batch().await.unwrap();
        assert!(batch.len() >= 2);
        assert!(batch.len() <= 4);
    }

    #[test]
    fn test_combine_batch_inputs() {
        let requests = vec![
            InferenceRequest {
                id: 0,
                input: Array::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
                timestamp: Instant::now(),
            },
            InferenceRequest {
                id: 1,
                input: Array::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap(),
                timestamp: Instant::now(),
            },
        ];
        
        let combined = combine_batch_inputs(&requests).unwrap();
        assert_eq!(combined.shape(), &[2, 3]);
        assert_eq!(combined[[0, 0]], 1.0);
        assert_eq!(combined[[1, 0]], 4.0);
    }

    #[test]
    fn test_split_batch_outputs() {
        let requests = vec![
            InferenceRequest {
                id: 0,
                input: Array::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
                timestamp: Instant::now(),
            },
            InferenceRequest {
                id: 1,
                input: Array::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap(),
                timestamp: Instant::now(),
            },
        ];
        
        let output = Array::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let responses = split_batch_outputs(output, &requests);
        
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].id, 0);
        assert_eq!(responses[1].id, 1);
    }

    #[test]
    fn test_adaptive_batch_size() {
        let config = BatchConfig {
            max_batch_size: 10,
            min_batch_size: 2,
            ..Default::default()
        };
        let batcher = DynamicBatcher::new(config);
        
        assert_eq!(batcher.adaptive_batch_size(), 2);
    }
}
