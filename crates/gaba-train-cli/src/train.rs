//! Training functions

use crate::data::{TrafficDataset, RouteDataset};
use crate::models::{TrafficModel, RouteModel};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

pub fn train_traffic_model(
    data_path: &Path,
    output_path: &Path,
    epochs: usize,
    learning_rate: f32,
) -> Result<()> {
    println!("Loading data from {:?}...", data_path);
    let dataset = TrafficDataset::from_csv(data_path)?;
    let (train_data, test_data) = dataset.split(0.2);
    
    println!("Train samples: {}, Test samples: {}", train_data.len(), test_data.len());
    println!("Learning rate: {}", learning_rate);
    
    let mut model = TrafficModel::new();
    
    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );
    
    let batch_size = 1024.min(train_data.len());
    let n_batches = (train_data.len() + batch_size - 1) / batch_size;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        // Mini-batch training
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_data.len());
            
            let batch_features = train_data.features.slice(ndarray::s![start..end, ..]).to_owned();
            let batch_targets = train_data.targets.slice(ndarray::s![start..end]).to_owned();
            
            let fwd = model.forward_cached(&batch_features);
            let predictions = fwd.output.column(0).to_owned();
            let errors = &predictions - &batch_targets;
            let batch_loss = (&errors * &errors).mean().unwrap();
            epoch_loss += batch_loss;
            
            let grads = model.backward(&batch_features, &fwd, &batch_targets);
            model.update(&grads, learning_rate);
        }
        
        let train_loss = epoch_loss / n_batches as f32;
        
        // Test loss (also use batching for large test sets)
        let test_batch_size = 1024.min(test_data.len());
        let test_n_batches = (test_data.len() + test_batch_size - 1) / test_batch_size;
        let mut test_loss_sum = 0.0;
        
        for batch_idx in 0..test_n_batches {
            let start = batch_idx * test_batch_size;
            let end = (start + test_batch_size).min(test_data.len());
            
            let batch_features = test_data.features.slice(ndarray::s![start..end, ..]);
            let batch_targets = test_data.targets.slice(ndarray::s![start..end]);
            
            let test_pred = model.forward(&batch_features.to_owned());
            let test_errors = &test_pred - &batch_targets;
            test_loss_sum += (&test_errors * &test_errors).mean().unwrap();
        }
        
        let test_loss = test_loss_sum / test_n_batches as f32;
        
        pb.set_position(epoch as u64 + 1);
        pb.set_message(format!("train: {:.4}, test: {:.4}", train_loss, test_loss));
    }
    
    pb.finish_with_message("Training complete");
    
    println!("Saving model to {:?}...", output_path);
    std::fs::create_dir_all(output_path)?;
    
    let model_path = output_path.join("traffic_model.bin");
    model.save(&model_path)?;
    println!("Model saved to {:?}", model_path);
    
    Ok(())
}

pub fn train_route_model(
    data_path: &Path,
    output_path: &Path,
    epochs: usize,
    learning_rate: f32,
) -> Result<()> {
    println!("Loading data from {:?}...", data_path);
    let dataset = RouteDataset::from_csv(data_path)?;
    let (train_data, test_data) = dataset.split(0.2);
    
    println!("Train samples: {}, Test samples: {}", train_data.len(), test_data.len());
    println!("Learning rate: {}", learning_rate);
    
    let mut model = RouteModel::new();
    
    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );
    
    let batch_size = 1024.min(train_data.len());
    let n_batches = (train_data.len() + batch_size - 1) / batch_size;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_data.len());
            
            let batch_features = train_data.features.slice(ndarray::s![start..end, ..]).to_owned();
            let batch_targets = train_data.targets.slice(ndarray::s![start..end]).to_owned();
            
            let fwd = model.forward_cached(&batch_features);
            let predictions = fwd.output.column(0).to_owned();
            let errors = &predictions - &batch_targets;
            let batch_loss = (&errors * &errors).mean().unwrap();
            epoch_loss += batch_loss;
            
            let grads = model.backward(&batch_features, &fwd, &batch_targets);
            model.update(&grads, learning_rate);
        }
        
        let train_loss = epoch_loss / n_batches as f32;
        
        let test_batch_size = 1024.min(test_data.len());
        let test_n_batches = (test_data.len() + test_batch_size - 1) / test_batch_size;
        let mut test_loss_sum = 0.0;
        
        for batch_idx in 0..test_n_batches {
            let start = batch_idx * test_batch_size;
            let end = (start + test_batch_size).min(test_data.len());
            
            let batch_features = test_data.features.slice(ndarray::s![start..end, ..]);
            let batch_targets = test_data.targets.slice(ndarray::s![start..end]);
            
            let test_pred = model.forward(&batch_features.to_owned());
            let test_errors = &test_pred - &batch_targets;
            test_loss_sum += (&test_errors * &test_errors).mean().unwrap();
        }
        
        let test_loss = test_loss_sum / test_n_batches as f32;
        
        pb.set_position(epoch as u64 + 1);
        pb.set_message(format!("train: {:.4}, test: {:.4}", train_loss, test_loss));
    }
    
    pb.finish_with_message("Training complete");
    
    println!("Saving model to {:?}...", output_path);
    std::fs::create_dir_all(output_path)?;
    
    let model_path = output_path.join("route_model.bin");
    model.save(&model_path)?;
    println!("Model saved to {:?}", model_path);
    
    Ok(())
}
