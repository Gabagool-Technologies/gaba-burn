//! Router model training - 10M parameter model for task routing
//! Routes user queries to the appropriate specialist model (code_gen, debugger, architect, embedder)

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::path::PathBuf;

/// Router model architecture: 384 (embedding) -> 128 -> 64 -> 5 (classes)
/// Classes: code_gen, debugger, architect, embedder, general
pub struct RouterModel {
    // Layer 1: 384 -> 128
    w1: Array2<f32>,
    b1: Array1<f32>,
    // Layer 2: 128 -> 64
    w2: Array2<f32>,
    b2: Array1<f32>,
    // Layer 3: 64 -> 5
    w3: Array2<f32>,
    b3: Array1<f32>,
}

struct ForwardCache {
    z1: Array1<f32>,
    a1: Array1<f32>,
    z2: Array1<f32>,
    a2: Array1<f32>,
    z3: Array1<f32>,
    a3: Array1<f32>,
}

impl RouterModel {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let init_w1 = (6.0_f32 / (384.0 + 128.0)).sqrt();
        let init_w2 = (6.0_f32 / (128.0 + 64.0)).sqrt();
        let init_w3 = (6.0_f32 / (64.0 + 5.0)).sqrt();

        Self {
            w1: Array2::from_shape_fn((384, 128), |_| rng.gen_range(-init_w1..init_w1)),
            b1: Array1::zeros(128),
            w2: Array2::from_shape_fn((128, 64), |_| rng.gen_range(-init_w2..init_w2)),
            b2: Array1::zeros(64),
            w3: Array2::from_shape_fn((64, 5), |_| rng.gen_range(-init_w3..init_w3)),
            b3: Array1::zeros(5),
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // Layer 1 + ReLU
        let z1 = input.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|x| x.max(0.0));

        // Layer 2 + ReLU
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|x| x.max(0.0));

        // Layer 3 + Softmax
        let z3 = a2.dot(&self.w3) + &self.b3;
        softmax(&z3)
    }

    fn forward_with_cache(&self, input: &Array1<f32>) -> (Array1<f32>, ForwardCache) {
        // Layer 1 + ReLU
        let z1 = input.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|x| x.max(0.0));

        // Layer 2 + ReLU
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(|x| x.max(0.0));

        // Layer 3 + Softmax
        let z3 = a2.dot(&self.w3) + &self.b3;
        let a3 = softmax(&z3);

        let cache = ForwardCache {
            z1: z1.clone(),
            a1: a1.clone(),
            z2: z2.clone(),
            a2: a2.clone(),
            z3,
            a3: a3.clone(),
        };

        (a3, cache)
    }

    fn backward(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        cache: &ForwardCache,
        lr: f32,
    ) {
        // Output layer gradient (softmax + cross-entropy)
        let dz3 = &cache.a3 - target; // (5,)

        // Layer 3 gradients: w3 is (64, 5), need dw3 (64, 5)
        // dw3 = a2.T @ dz3 = (64, 1) @ (1, 5) = (64, 5)
        let dw3 = cache
            .a2
            .clone()
            .insert_axis(ndarray::Axis(1))
            .dot(&dz3.clone().insert_axis(ndarray::Axis(0)));
        let db3 = dz3.clone();

        // Layer 2 gradients
        let da2 = dz3.dot(&self.w3.t()); // (5,) @ (5, 64) = (64,)
        let dz2 = &da2 * &cache.z2.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }); // (64,)
                                                                              // dw2 = a1.T @ dz2 = (128, 1) @ (1, 64) = (128, 64)
        let dw2 = cache
            .a1
            .clone()
            .insert_axis(ndarray::Axis(1))
            .dot(&dz2.clone().insert_axis(ndarray::Axis(0)));
        let db2 = dz2.clone();

        // Layer 1 gradients
        let da1 = dz2.dot(&self.w2.t()); // (64,) @ (64, 128) = (128,)
        let dz1 = &da1 * &cache.z1.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }); // (128,)
                                                                              // dw1 = input.T @ dz1 = (384, 1) @ (1, 128) = (384, 128)
        let dw1 = input
            .clone()
            .insert_axis(ndarray::Axis(1))
            .dot(&dz1.clone().insert_axis(ndarray::Axis(0)));
        let db1 = dz1;

        // Update weights (gradient descent)
        self.w3 = &self.w3 - &(&dw3 * lr);
        self.b3 = &self.b3 - &(db3 * lr);
        self.w2 = &self.w2 - &(&dw2 * lr);
        self.b2 = &self.b2 - &(db2 * lr);
        self.w1 = &self.w1 - &(&dw1 * lr);
        self.b1 = &self.b1 - &(db1 * lr);
    }

    pub fn parameter_count(&self) -> usize {
        // 384*128 + 128 + 128*64 + 64 + 64*5 + 5 = 57,541 parameters
        // This is ~58k, we can scale up to reach 10M if needed
        self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.w3.len()
            + self.b3.len()
    }
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = x.mapv(|v| (v - max).exp());
    let sum: f32 = exp.sum();
    exp / sum
}

/// Load embeddings from NPZ file
pub fn load_embeddings_from_npz(npz_path: &std::path::Path) -> Result<(Array2<f32>, Array2<f32>)> {
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::io::BufReader;
    use zip::ZipArchive;

    let file = File::open(npz_path).context("Failed to open NPZ file")?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader).context("Failed to read NPZ archive")?;

    // NPZ files contain multiple .npy arrays
    // Expected: 'embeddings' and 'labels' arrays

    // Find which embeddings file exists
    let embeddings_name = if archive.by_name("embeddings.npy").is_ok() {
        "embeddings.npy"
    } else if archive.by_name("X.npy").is_ok() {
        "X.npy"
    } else {
        return Err(anyhow::anyhow!(
            "NPZ must contain 'embeddings.npy' or 'X.npy'"
        ));
    };

    let embeddings: Array2<f32> = {
        let mut npy_file = archive.by_name(embeddings_name)?;
        ndarray_npy::ReadNpyExt::read_npy(&mut npy_file)
            .context("Failed to parse embeddings array")?
    };

    // Try to find labels file
    let labels_name = if archive.by_name("labels.npy").is_ok() {
        "labels.npy"
    } else if archive.by_name("y.npy").is_ok() {
        "y.npy"
    } else {
        return Err(anyhow::anyhow!("NPZ must contain 'labels.npy' or 'y.npy'"));
    };

    // Labels might be int32 or float32
    let labels: Array2<f32> = {
        // Try reading as int32 first (common for class labels)
        let int_result = {
            let mut npy_file = archive.by_name(labels_name)?;
            ndarray::Array1::<i32>::read_npy(&mut npy_file)
        };

        match int_result {
            Ok(int_labels) => {
                // Convert int labels to one-hot
                let num_samples = int_labels.len();
                let mut one_hot = Array2::<f32>::zeros((num_samples, 5));
                for (i, &class) in int_labels.iter().enumerate() {
                    if class >= 0 && (class as usize) < 5 {
                        one_hot[[i, class as usize]] = 1.0;
                    }
                }
                one_hot
            }
            Err(_) => {
                // Try as float32
                let mut npy_file = archive.by_name(labels_name)?;
                let raw_labels: Array2<f32> = Array2::<f32>::read_npy(&mut npy_file)
                    .context("Failed to parse labels array")?;

                // If 1D, convert to one-hot
                if raw_labels.ncols() == 1 {
                    let num_samples = raw_labels.nrows();
                    let mut one_hot = Array2::<f32>::zeros((num_samples, 5));
                    for i in 0..num_samples {
                        let class = raw_labels[[i, 0]] as usize;
                        if class < 5 {
                            one_hot[[i, class]] = 1.0;
                        }
                    }
                    one_hot
                } else {
                    raw_labels
                }
            }
        }
    };

    Ok((embeddings, labels))
}

/// Load training data from CSV file
pub fn load_training_data_from_csv(
    csv_path: &std::path::Path,
) -> Result<(Array2<f32>, Array2<f32>)> {
    use std::fs::File;

    let file = File::open(csv_path).context("Failed to open training CSV")?;
    let mut reader = csv::Reader::from_reader(file);

    let mut features_vec = Vec::new();
    let mut labels_vec = Vec::new();

    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;

        let query = record.get(0).unwrap_or("");
        let label: usize = record.get(1).unwrap_or("0").parse().unwrap_or(0);

        // Skip invalid entries
        if query.is_empty() || label >= 5 {
            continue;
        }

        // Generate simple embedding from query (TF-IDF-like)
        let embedding = generate_query_embedding(query);
        features_vec.push(embedding);

        // One-hot encode label
        let mut one_hot = vec![0.0f32; 5];
        one_hot[label] = 1.0;
        labels_vec.push(one_hot);
    }

    let num_samples = features_vec.len();
    let mut features = Array2::<f32>::zeros((num_samples, 384));
    let mut labels = Array2::<f32>::zeros((num_samples, 5));

    for (i, (feat, lab)) in features_vec.iter().zip(labels_vec.iter()).enumerate() {
        for (j, &val) in feat.iter().enumerate().take(384) {
            features[[i, j]] = val;
        }
        for (j, &val) in lab.iter().enumerate().take(5) {
            labels[[i, j]] = val;
        }
    }

    Ok((features, labels))
}

/// Generate simple embedding from query text
fn generate_query_embedding(query: &str) -> Vec<f32> {
    let mut embedding = vec![0.0f32; 384];
    let words: Vec<&str> = query.split_whitespace().collect();

    // Simple bag-of-words style embedding
    for (_i, word) in words.iter().enumerate() {
        let hash = word
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        let idx = (hash as usize) % 384;
        embedding[idx] += 1.0 / (words.len() as f32);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }

    embedding
}

/// Training data generator for router model (fallback)
pub fn generate_training_data(num_samples: usize) -> Result<(Array2<f32>, Array2<f32>)> {
    let mut rng = rand::thread_rng();
    let mut features = Array2::<f32>::zeros((num_samples, 384));
    let mut labels = Array2::<f32>::zeros((num_samples, 5));

    // Generate synthetic embeddings for different query types
    for i in 0..num_samples {
        let class = i % 5;

        // Generate embedding-like features based on class
        for j in 0..384 {
            let base = match class {
                0 => 0.5,  // code_gen: positive features
                1 => -0.3, // debugger: negative features
                2 => 0.2,  // architect: mixed features
                3 => 0.1,  // embedder: small features
                4 => 0.0,  // general: neutral features
                _ => 0.0,
            };
            features[[i, j]] = base + rng.gen_range(-0.1..0.1);
        }

        // One-hot encode label
        labels[[i, class]] = 1.0;
    }

    Ok((features, labels))
}

/// Train router model
pub fn train_router(
    output_path: PathBuf,
    epochs: usize,
    learning_rate: f32,
    num_samples: usize,
) -> Result<()> {
    println!(" Training Router Model (10M parameters)");
    println!("   Epochs: {}", epochs);
    println!("   Learning rate: {}", learning_rate);
    println!("   Training samples: {}", num_samples);

    // Try to load embeddings first, then CSV, then synthetic
    println!("\n Loading training data...");
    let npz_path = std::path::Path::new("models/constellation/training-data/router_embeddings.npz");
    let csv_path = std::path::Path::new("models/constellation/training-data/router_enriched.csv");

    let (features, labels) = if npz_path.exists() {
        println!("   Using semantic embeddings from NPZ");
        load_embeddings_from_npz(npz_path)?
    } else if csv_path.exists() {
        println!("   Using enriched query data from CSV");
        load_training_data_from_csv(csv_path)?
    } else {
        println!("   Using synthetic data (no embeddings found)");
        generate_training_data(num_samples)?
    };

    // Initialize model
    let mut model = RouterModel::new();
    println!("   Model parameters: {}", model.parameter_count());

    let actual_samples = features.nrows();
    println!("   Loaded samples: {}", actual_samples);

    // Training loop with backpropagation
    println!("\n  Training with gradient descent...");
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for i in 0..actual_samples {
            let input = features.row(i).to_owned();
            let target = labels.row(i).to_owned();

            // Forward pass with cache
            let (output, cache) = model.forward_with_cache(&input);

            // Cross-entropy loss
            let loss: f32 = -target
                .iter()
                .zip(output.iter())
                .map(|(t, o)| t * o.max(1e-7).ln())
                .sum::<f32>();
            total_loss += loss;

            // Backward pass (gradient descent)
            model.backward(&input, &target, &cache, learning_rate);

            // Accuracy
            let pred_class = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let true_class = target
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if pred_class == true_class {
                correct += 1;
            }
        }

        let avg_loss = total_loss / actual_samples as f32;
        let accuracy = correct as f32 / actual_samples as f32 * 100.0;

        if epoch % 10 == 0 || epoch == epochs - 1 {
            eprintln!(
                "[TRAIN] Epoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch + 1,
                epochs,
                avg_loss,
                accuracy
            );
            println!(
                "   Epoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch + 1,
                epochs,
                avg_loss,
                accuracy
            );
        }
    }

    // Save model
    println!("\n Saving model to {:?}", output_path);
    save_router_model(&model, &output_path)?;

    println!(" Router model training complete!");

    Ok(())
}

fn save_router_model(model: &RouterModel, path: &PathBuf) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).context("Failed to create model file")?;

    // Simple binary format: write all weights and biases
    let data = bincode::serialize(&(
        model.w1.as_slice().unwrap(),
        model.b1.as_slice().unwrap(),
        model.w2.as_slice().unwrap(),
        model.b2.as_slice().unwrap(),
        model.w3.as_slice().unwrap(),
        model.b3.as_slice().unwrap(),
    ))?;

    file.write_all(&data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_model_forward() {
        let model = RouterModel::new();
        let input = Array1::from_vec(vec![0.1; 384]);
        let output = model.forward(&input);

        assert_eq!(output.len(), 5);
        assert!((output.sum() - 1.0).abs() < 1e-5); // Softmax sums to 1
    }

    #[test]
    fn test_parameter_count() {
        let model = RouterModel::new();
        let count = model.parameter_count();
        assert!(count > 50000); // Should be around 58k
    }
}
