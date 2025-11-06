use std::collections::HashMap;
use std::time::Duration;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_metrics: HashMap<String, f32>,
    pub val_metrics: HashMap<String, f32>,
    pub learning_rate: f32,
    pub duration: Duration,
}

pub struct MetricsTracker {
    metrics: HashMap<String, Vec<f32>>,
    best_metrics: HashMap<String, f32>,
    history: Vec<EpochMetrics>,
    mode: HashMap<String, MetricMode>,
}

#[derive(Clone, Copy, Debug)]
pub enum MetricMode {
    Min,
    Max,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            best_metrics: HashMap::new(),
            history: Vec::new(),
            mode: HashMap::new(),
        }
    }
    
    pub fn register_metric(&mut self, name: &str, mode: MetricMode) {
        self.metrics.insert(name.to_string(), Vec::new());
        self.mode.insert(name.to_string(), mode);
        
        let initial_best = match mode {
            MetricMode::Min => f32::INFINITY,
            MetricMode::Max => f32::NEG_INFINITY,
        };
        self.best_metrics.insert(name.to_string(), initial_best);
    }
    
    pub fn update(&mut self, name: &str, value: f32) {
        if let Some(values) = self.metrics.get_mut(name) {
            values.push(value);
            
            if let Some(mode) = self.mode.get(name) {
                let current_best = self.best_metrics.get(name).copied().unwrap();
                let is_better = match mode {
                    MetricMode::Min => value < current_best,
                    MetricMode::Max => value > current_best,
                };
                
                if is_better {
                    self.best_metrics.insert(name.to_string(), value);
                }
            }
        }
    }
    
    pub fn add_epoch(&mut self, metrics: EpochMetrics) {
        self.history.push(metrics);
    }
    
    pub fn get_best(&self, name: &str) -> Option<f32> {
        self.best_metrics.get(name).copied()
    }
    
    pub fn get_history(&self, name: &str) -> Option<&[f32]> {
        self.metrics.get(name).map(|v| v.as_slice())
    }
    
    pub fn get_epoch_history(&self) -> &[EpochMetrics] {
        &self.history
    }
    
    pub fn save_json(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.history)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

pub fn mse(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    sum / predictions.len() as f32
}

pub fn mae(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let sum: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    sum / predictions.len() as f32
}

pub fn rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    mse(predictions, targets).sqrt()
}

pub fn r2_score(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    
    let mean_target: f32 = targets.iter().sum::<f32>() / targets.len() as f32;
    
    let ss_res: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    
    let ss_tot: f32 = targets.iter()
        .map(|t| (t - mean_target).powi(2))
        .sum();
    
    1.0 - (ss_res / ss_tot)
}

pub fn accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let correct = predictions.iter()
        .zip(targets.iter())
        .filter(|(p, t)| {
            let pred_class = if **p > threshold { 1.0 } else { 0.0 };
            let target_class = if **t > threshold { 1.0 } else { 0.0 };
            pred_class == target_class
        })
        .count();
    correct as f32 / predictions.len() as f32
}

pub fn precision(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let mut true_positives = 0;
    let mut false_positives = 0;
    
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let pred_positive = *p > threshold;
        let target_positive = *t > threshold;
        
        if pred_positive && target_positive {
            true_positives += 1;
        } else if pred_positive && !target_positive {
            false_positives += 1;
        }
    }
    
    if true_positives + false_positives == 0 {
        0.0
    } else {
        true_positives as f32 / (true_positives + false_positives) as f32
    }
}

pub fn recall(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let mut true_positives = 0;
    let mut false_negatives = 0;
    
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let pred_positive = *p > threshold;
        let target_positive = *t > threshold;
        
        if pred_positive && target_positive {
            true_positives += 1;
        } else if !pred_positive && target_positive {
            false_negatives += 1;
        }
    }
    
    if true_positives + false_negatives == 0 {
        0.0
    } else {
        true_positives as f32 / (true_positives + false_negatives) as f32
    }
}

pub fn f1_score(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    let p = precision(predictions, targets, threshold);
    let r = recall(predictions, targets, threshold);
    
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * (p * r) / (p + r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 2.9, 4.2];
        let result = mse(&predictions, &targets);
        assert_relative_eq!(result, 0.0175, epsilon = 1e-6);
    }
    
    #[test]
    fn test_mae() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 2.9, 4.2];
        let result = mae(&predictions, &targets);
        assert_relative_eq!(result, 0.125, epsilon = 1e-6);
    }
    
    #[test]
    fn test_rmse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let result = rmse(&predictions, &targets);
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_r2_score() {
        let predictions = vec![3.0, -0.5, 2.0, 7.0];
        let targets = vec![2.5, 0.0, 2.0, 8.0];
        let result = r2_score(&predictions, &targets);
        assert!(result > 0.9);
    }
    
    #[test]
    fn test_accuracy() {
        let predictions = vec![0.9, 0.1, 0.8, 0.2];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let result = accuracy(&predictions, &targets, 0.5);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_precision_recall_f1() {
        let predictions = vec![0.9, 0.8, 0.3, 0.2];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        
        let p = precision(&predictions, &targets, 0.5);
        let r = recall(&predictions, &targets, 0.5);
        let f1 = f1_score(&predictions, &targets, 0.5);
        
        assert_relative_eq!(p, 1.0, epsilon = 1e-6);
        assert_relative_eq!(r, 1.0, epsilon = 1e-6);
        assert_relative_eq!(f1, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricMode::Min);
        tracker.register_metric("accuracy", MetricMode::Max);
        
        tracker.update("loss", 1.0);
        tracker.update("loss", 0.5);
        tracker.update("loss", 0.8);
        
        tracker.update("accuracy", 0.7);
        tracker.update("accuracy", 0.9);
        tracker.update("accuracy", 0.8);
        
        assert_relative_eq!(tracker.get_best("loss").unwrap(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(tracker.get_best("accuracy").unwrap(), 0.9, epsilon = 1e-6);
        
        let loss_history = tracker.get_history("loss").unwrap();
        assert_eq!(loss_history.len(), 3);
    }
}
