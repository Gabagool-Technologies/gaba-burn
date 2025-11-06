use crate::data_converted::ConvertedTrafficDataset;
use crate::models::TrafficModel;
use anyhow::Result;
use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        ElementConversion, Tensor,
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::s;

pub struct ConvertedTrainer {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub test_ratio: f32,
    pub verbose: bool,
}

impl Default for ConvertedTrainer {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            test_ratio: 0.2,
            verbose: true,
        }
    }
}

impl ConvertedTrainer {
    pub fn new(epochs: usize, learning_rate: f64) -> Self {
        Self {
            epochs,
            learning_rate,
            ..Default::default()
        }
    }

    pub fn train<B: AutodiffBackend>(
        &self,
        model: TrafficModel<B>,
        dataset: ConvertedTrafficDataset,
    ) -> Result<TrafficModel<B>> {
        let device = Default::default();
        let (train_data, test_data) = dataset.split(self.test_ratio);

        if self.verbose {
            println!("Training samples: {}", train_data.len());
            println!("Test samples: {}", test_data.len());
            println!("Batch size: {}", self.batch_size);
            println!("Learning rate: {}", self.learning_rate);
        }

        let mut optimizer = burn::optim::SgdConfig::new().init();

        let pb = if self.verbose {
            let pb = ProgressBar::new(self.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };

        let mut model = model;
        let mut best_loss = f32::INFINITY;

        for epoch in 0..self.epochs {
            let (new_model, train_loss) =
                self.train_epoch(model, &train_data, &mut optimizer, &device)?;
            model = new_model;

            let test_loss = self.validate_epoch(&model, &test_data, &device)?;

            if let Some(ref pb) = pb {
                pb.set_position(epoch as u64 + 1);
                pb.set_message(format!(
                    "train_loss: {:.4}, test_loss: {:.4}",
                    train_loss, test_loss
                ));
            }

            if test_loss < best_loss {
                best_loss = test_loss;
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message(format!("Training complete. Best loss: {:.4}", best_loss));
        }

        Ok(model)
    }

    fn train_epoch<B: AutodiffBackend>(
        &self,
        mut model: TrafficModel<B>,
        data: &ConvertedTrafficDataset,
        optimizer: &mut impl Optimizer<TrafficModel<B>, B>,
        device: &<B as Backend>::Device,
    ) -> Result<(TrafficModel<B>, f32)> {
        let n_batches = (data.len() + self.batch_size - 1) / self.batch_size;
        let mut total_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let start = batch_idx * self.batch_size;
            let end = (start + self.batch_size).min(data.len());

            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);

            let batch_size = end - start;
            let features_vec: Vec<f32> = features.iter().copied().collect();
            let targets_vec: Vec<f32> = targets.iter().copied().collect();

            let features_tensor = Tensor::<B, 1>::from_floats(features_vec.as_slice(), device)
                .reshape([batch_size, 21]);
            let targets_tensor = Tensor::<B, 1>::from_floats(targets_vec.as_slice(), device);

            let model_ad = model.clone();
            let predictions = model_ad.forward(features_tensor);
            let predictions = predictions.squeeze::<1>();

            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(self.learning_rate, model, grads);

            total_loss += loss.into_scalar().elem::<f32>();
        }

        Ok((model, total_loss / n_batches as f32))
    }

    fn validate_epoch<B: Backend>(
        &self,
        model: &TrafficModel<B>,
        data: &ConvertedTrafficDataset,
        device: &<B as Backend>::Device,
    ) -> Result<f32> {
        let n_batches = (data.len() + self.batch_size - 1) / self.batch_size;
        let mut total_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let start = batch_idx * self.batch_size;
            let end = (start + self.batch_size).min(data.len());

            let features = data.features.slice(s![start..end, ..]);
            let targets = data.targets.slice(s![start..end]);

            let batch_size = end - start;
            let features_vec: Vec<f32> = features.iter().copied().collect();
            let targets_vec: Vec<f32> = targets.iter().copied().collect();

            let features_tensor = Tensor::<B, 1>::from_floats(features_vec.as_slice(), device)
                .reshape([batch_size, 21]);
            let targets_tensor = Tensor::<B, 1>::from_floats(targets_vec.as_slice(), device);

            let predictions = model.forward(features_tensor);
            let predictions = predictions.squeeze::<1>();

            let diff = predictions.clone() - targets_tensor;
            let loss = (diff.clone() * diff).mean();

            total_loss += loss.into_scalar().elem::<f32>();
        }

        Ok(total_loss / n_batches as f32)
    }
}
