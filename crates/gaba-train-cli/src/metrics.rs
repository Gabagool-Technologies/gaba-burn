#[allow(dead_code)]
use std::fmt;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Metrics {
    pub mse: f64,
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    pub mape: f64,
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSE: {:.4}, MAE: {:.4}, RMSE: {:.4}, R²: {:.4}, MAPE: {:.2}%",
            self.mse, self.mae, self.rmse, self.r2, self.mape * 100.0
        )
    }
}

#[allow(dead_code)]
pub fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    
    let sum: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let diff = (*p as f64) - (*t as f64);
            diff * diff
        })
        .sum();
    
    sum / predictions.len() as f64
}

pub fn mean_absolute_error(predictions: &[f32], targets: &[f32]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    
    let sum: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| ((*p as f64) - (*t as f64)).abs())
        .sum();
    
    sum / predictions.len() as f64
}

pub fn root_mean_squared_error(predictions: &[f32], targets: &[f32]) -> f64 {
    mean_squared_error(predictions, targets).sqrt()
}

pub fn r_squared(predictions: &[f32], targets: &[f32]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    
    let mean_target: f64 = targets.iter().map(|&t| t as f64).sum::<f64>() / targets.len() as f64;
    
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let diff = (*t as f64) - (*p as f64);
            diff * diff
        })
        .sum();
    
    let ss_tot: f64 = targets
        .iter()
        .map(|&t| {
            let diff = (t as f64) - mean_target;
            diff * diff
        })
        .sum();
    
    if ss_tot == 0.0 {
        return 0.0;
    }
    
    1.0 - (ss_res / ss_tot)
}

pub fn mean_absolute_percentage_error(predictions: &[f32], targets: &[f32]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    
    let sum: f64 = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(_, &t)| t != 0.0)
        .map(|(p, t)| {
            let t_f64 = *t as f64;
            let p_f64 = *p as f64;
            ((t_f64 - p_f64) / t_f64).abs()
        })
        .sum();
    
    let count = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(_, &t)| t != 0.0)
        .count();
    
    if count == 0 {
        return 0.0;
    }
    
    sum / count as f64
}

pub fn evaluate_model(predictions: &[f32], targets: &[f32]) -> Metrics {
    Metrics {
        mse: mean_squared_error(predictions, targets),
        mae: mean_absolute_error(predictions, targets),
        rmse: root_mean_squared_error(predictions, targets),
        r2: r_squared(predictions, targets),
        mape: mean_absolute_percentage_error(predictions, targets),
    }
}

pub fn confusion_matrix(predictions: &[f32], targets: &[f32], threshold: f32) -> (usize, usize, usize, usize) {
    assert_eq!(predictions.len(), targets.len());
    
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let pred_positive = *p >= threshold;
        let actual_positive = *t >= threshold;
        
        match (pred_positive, actual_positive) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (true, false) => fp += 1,
            (false, true) => fn_count += 1,
        }
    }
    
    (tp, tn, fp, fn_count)
}

pub fn accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f64 {
    let (tp, tn, fp, fn_count) = confusion_matrix(predictions, targets, threshold);
    let total = tp + tn + fp + fn_count;
    
    if total == 0 {
        return 0.0;
    }
    
    (tp + tn) as f64 / total as f64
}

pub fn precision(predictions: &[f32], targets: &[f32], threshold: f32) -> f64 {
    let (tp, _, fp, _) = confusion_matrix(predictions, targets, threshold);
    
    if tp + fp == 0 {
        return 0.0;
    }
    
    tp as f64 / (tp + fp) as f64
}

pub fn recall(predictions: &[f32], targets: &[f32], threshold: f32) -> f64 {
    let (tp, _, _, fn_count) = confusion_matrix(predictions, targets, threshold);
    
    if tp + fn_count == 0 {
        return 0.0;
    }
    
    tp as f64 / (tp + fn_count) as f64
}

pub fn f1_score(predictions: &[f32], targets: &[f32], threshold: f32) -> f64 {
    let p = precision(predictions, targets, threshold);
    let r = recall(predictions, targets, threshold);
    
    if p + r == 0.0 {
        return 0.0;
    }
    
    2.0 * (p * r) / (p + r)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 2.9, 4.2];
        
        let mse = mean_squared_error(&predictions, &targets);
        assert!(mse < 0.1);
    }
    
    #[test]
    fn test_mae() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 2.9, 4.2];
        
        let mae = mean_absolute_error(&predictions, &targets);
        assert!(mae < 0.2);
    }
    
    #[test]
    fn test_r2() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        
        let r2 = r_squared(&predictions, &targets);
        assert!((r2 - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_evaluate_model() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.1, 2.1, 2.9, 4.2];
        
        let metrics = evaluate_model(&predictions, &targets);
        assert!(metrics.mse < 0.1);
        assert!(metrics.mae < 0.2);
        assert!(metrics.r2 > 0.9);
    }
}
