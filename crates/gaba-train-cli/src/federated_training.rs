//! Federated Learning Implementation
//! Distributed training with differential privacy

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[allow(dead_code)]
pub struct FederatedTrainer {
    node_id: String,
    aggregation_server: String,
    local_model: Vec<f32>,
    privacy_budget: f64,
    update_interval: Duration,
}

impl FederatedTrainer {
    pub fn new(node_id: String, server: String) -> Self {
        Self {
            node_id,
            aggregation_server: server,
            local_model: Vec::new(),
            privacy_budget: 1.0,
            update_interval: Duration::from_secs(60),
        }
    }

    pub fn train_local(&mut self, data: &[Vec<f32>], labels: &[f32], epochs: usize) -> Result<()> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (features, label) in data.iter().zip(labels.iter()) {
                let prediction = self.forward(features);
                let loss = (prediction - label).powi(2);
                total_loss += loss;
                
                self.backward(features, prediction, *label);
            }
            
            if epoch % 10 == 0 {
                println!("Epoch {}: loss={:.4}", epoch, total_loss / data.len() as f32);
            }
        }
        
        Ok(())
    }

    fn forward(&self, features: &[f32]) -> f32 {
        if self.local_model.is_empty() {
            return 0.0;
        }
        
        features.iter()
            .zip(self.local_model.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    fn backward(&mut self, features: &[f32], prediction: f32, target: f32) {
        if self.local_model.is_empty() {
            self.local_model = vec![0.01; features.len()];
        }
        
        let error = prediction - target;
        let lr = 0.01;
        
        for (i, feature) in features.iter().enumerate() {
            self.local_model[i] -= lr * error * feature;
        }
    }

    pub fn add_differential_privacy(&mut self, noise_scale: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for weight in self.local_model.iter_mut() {
            let noise = rng.gen_range(-noise_scale..noise_scale) as f32;
            *weight += noise;
        }
        
        self.privacy_budget -= noise_scale;
    }

    pub fn get_model_update(&self) -> Vec<f32> {
        self.local_model.clone()
    }

    pub fn apply_global_update(&mut self, global_model: Vec<f32>) {
        self.local_model = global_model;
    }
}

#[allow(dead_code)]
pub struct FederatedAggregator {
    global_model: Arc<Mutex<Vec<f32>>>,
    node_updates: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    min_nodes: usize,
}

impl FederatedAggregator {
    pub fn new(min_nodes: usize) -> Self {
        Self {
            global_model: Arc::new(Mutex::new(Vec::new())),
            node_updates: Arc::new(Mutex::new(HashMap::new())),
            min_nodes,
        }
    }

    pub fn receive_update(&self, node_id: String, model: Vec<f32>) -> Result<()> {
        let mut updates = self.node_updates.lock().unwrap();
        updates.insert(node_id, model);
        
        if updates.len() >= self.min_nodes {
            self.aggregate_updates()?;
        }
        
        Ok(())
    }

    fn aggregate_updates(&self) -> Result<()> {
        let updates = self.node_updates.lock().unwrap();
        
        if updates.is_empty() {
            return Ok(());
        }
        
        let model_size = updates.values().next().unwrap().len();
        let mut aggregated = vec![0.0; model_size];
        
        for model in updates.values() {
            for (i, weight) in model.iter().enumerate() {
                aggregated[i] += weight;
            }
        }
        
        let num_models = updates.len() as f32;
        for weight in aggregated.iter_mut() {
            *weight /= num_models;
        }
        
        let mut global = self.global_model.lock().unwrap();
        *global = aggregated;
        
        Ok(())
    }

    pub fn get_global_model(&self) -> Vec<f32> {
        self.global_model.lock().unwrap().clone()
    }

    pub fn clear_updates(&self) {
        self.node_updates.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_trainer() {
        let mut trainer = FederatedTrainer::new("node1".to_string(), "localhost:8080".to_string());
        
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let labels = vec![6.0, 9.0, 12.0];
        
        trainer.train_local(&data, &labels, 10).unwrap();
        
        let update = trainer.get_model_update();
        assert!(!update.is_empty());
    }

    #[test]
    fn test_federated_aggregator() {
        let aggregator = FederatedAggregator::new(2);
        
        aggregator.receive_update("node1".to_string(), vec![1.0, 2.0, 3.0]).unwrap();
        aggregator.receive_update("node2".to_string(), vec![2.0, 3.0, 4.0]).unwrap();
        
        let global = aggregator.get_global_model();
        assert_eq!(global, vec![1.5, 2.5, 3.5]);
    }
}
