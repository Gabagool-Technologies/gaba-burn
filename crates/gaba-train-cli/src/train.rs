use anyhow::Result;
use burn::backend::{Autodiff, NdArray};
use gaba_train::data_converted::ConvertedTrafficDataset;
use gaba_train::models::TrafficModel;
use gaba_train::training_converted::ConvertedTrainer;
use std::path::Path;

type Backend = Autodiff<NdArray>;

pub fn train_traffic_model(
    data_path: &Path,
    _output_path: &Path,
    epochs: usize,
    learning_rate: f32,
) -> Result<()> {
    println!("Loading data from {:?}...", data_path);
    let dataset = ConvertedTrafficDataset::from_csv(data_path)?;
    
    println!("Total samples: {}", dataset.len());
    println!("Epochs: {}", epochs);
    println!("Learning rate: {}", learning_rate);
    
    let device = Default::default();
    let model: TrafficModel<Backend> = TrafficModel::new(&device);
    
    let trainer = ConvertedTrainer::new(epochs, learning_rate as f64);
    let _trained_model = trainer.train(model, dataset)?;
    
    println!("Training complete!");
    Ok(())
}

pub fn train_route_model(
    _data_path: &Path,
    _output_path: &Path,
    _epochs: usize,
    _learning_rate: f32,
) -> Result<()> {
    println!("Route model training not yet implemented for converted format");
    Ok(())
}
