use ndarray::{Array2, ArrayView2};
use anyhow::Result;
use std::collections::HashMap;

pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
            ],
        }
    }
}

pub struct LoraLayer {
    pub lora_a: Array2<f32>,
    pub lora_b: Array2<f32>,
    pub scaling: f32,
    pub rank: usize,
    pub merged: bool,
}

impl LoraLayer {
    pub fn new(in_features: usize, out_features: usize, config: &LoraConfig) -> Self {
        let lora_a = Self::initialize_lora_a(in_features, config.rank);
        let lora_b = Array2::<f32>::zeros((config.rank, out_features));
        
        let scaling = config.alpha / config.rank as f32;
        
        Self {
            lora_a,
            lora_b,
            scaling,
            rank: config.rank,
            merged: false,
        }
    }

    fn initialize_lora_a(in_features: usize, rank: usize) -> Array2<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let std = 1.0 / (in_features as f32).sqrt();
        Array2::from_shape_fn((rank, in_features), |_| {
            rng.gen::<f32>() * std * 2.0 - std
        })
    }

    pub fn forward(&self, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        let hidden = input.dot(&self.lora_a.t());
        let output = hidden.dot(&self.lora_b.t());
        Ok(output * self.scaling)
    }

    pub fn merge_with_base(&mut self, base_weight: &mut Array2<f32>) -> Result<()> {
        if self.merged {
            return Ok(());
        }
        
        let lora_weight = self.lora_b.t().dot(&self.lora_a) * self.scaling;
        *base_weight = base_weight.clone() + lora_weight;
        self.merged = true;
        
        Ok(())
    }

    pub fn unmerge_from_base(&mut self, base_weight: &mut Array2<f32>) -> Result<()> {
        if !self.merged {
            return Ok(());
        }
        
        let lora_weight = self.lora_b.t().dot(&self.lora_a) * self.scaling;
        *base_weight = base_weight.clone() - lora_weight;
        self.merged = false;
        
        Ok(())
    }

    pub fn trainable_parameters(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    pub fn parameter_reduction_ratio(&self, base_params: usize) -> f32 {
        let lora_params = self.trainable_parameters();
        base_params as f32 / lora_params as f32
    }
}

pub struct LoraModel {
    pub base_layers: HashMap<String, Array2<f32>>,
    pub lora_layers: HashMap<String, LoraLayer>,
    pub config: LoraConfig,
}

impl LoraModel {
    pub fn new(config: LoraConfig) -> Self {
        Self {
            base_layers: HashMap::new(),
            lora_layers: HashMap::new(),
            config,
        }
    }

    pub fn add_base_layer(&mut self, name: String, weight: Array2<f32>) {
        self.base_layers.insert(name, weight);
    }

    pub fn add_lora_layer(&mut self, name: String, in_features: usize, out_features: usize) {
        if self.config.target_modules.iter().any(|m| name.contains(m)) {
            let lora = LoraLayer::new(in_features, out_features, &self.config);
            self.lora_layers.insert(name, lora);
        }
    }

    pub fn forward(&self, layer_name: &str, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        let base_weight = self.base_layers.get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer not found: {}", layer_name))?;
        
        let mut output = input.dot(&base_weight.t());
        
        if let Some(lora_layer) = self.lora_layers.get(layer_name) {
            let lora_output = lora_layer.forward(input)?;
            output = output + lora_output;
        }
        
        Ok(output)
    }

    pub fn merge_all(&mut self) -> Result<()> {
        for (name, lora_layer) in &mut self.lora_layers {
            if let Some(base_weight) = self.base_layers.get_mut(name) {
                lora_layer.merge_with_base(base_weight)?;
            }
        }
        Ok(())
    }

    pub fn unmerge_all(&mut self) -> Result<()> {
        for (name, lora_layer) in &mut self.lora_layers {
            if let Some(base_weight) = self.base_layers.get_mut(name) {
                lora_layer.unmerge_from_base(base_weight)?;
            }
        }
        Ok(())
    }

    pub fn total_trainable_parameters(&self) -> usize {
        self.lora_layers.values().map(|l| l.trainable_parameters()).sum()
    }

    pub fn total_base_parameters(&self) -> usize {
        self.base_layers.values().map(|w| w.len()).sum()
    }

    pub fn parameter_efficiency(&self) -> f32 {
        let trainable = self.total_trainable_parameters();
        let total = self.total_base_parameters();
        (trainable as f32 / total as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
    }

    #[test]
    fn test_lora_layer_creation() {
        let config = LoraConfig::default();
        let layer = LoraLayer::new(128, 256, &config);
        
        assert_eq!(layer.lora_a.shape(), &[8, 128]);
        assert_eq!(layer.lora_b.shape(), &[8, 256]);
        assert_eq!(layer.rank, 8);
        assert!(!layer.merged);
    }

    #[test]
    fn test_lora_forward() {
        let config = LoraConfig {
            rank: 4,
            ..Default::default()
        };
        let layer = LoraLayer::new(10, 20, &config);
        
        let input = Array::from_shape_vec((2, 10), vec![0.1; 20]).unwrap();
        let output = layer.forward(input.view());
        
        assert!(output.is_ok());
        assert_eq!(output.unwrap().shape(), &[2, 20]);
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let config = LoraConfig {
            rank: 4,
            ..Default::default()
        };
        let mut layer = LoraLayer::new(10, 10, &config);
        let mut base_weight = Array2::<f32>::ones((10, 10));
        let original = base_weight.clone();
        
        layer.merge_with_base(&mut base_weight).unwrap();
        assert!(layer.merged);
        assert_ne!(base_weight, original);
        
        layer.unmerge_from_base(&mut base_weight).unwrap();
        assert!(!layer.merged);
        
        for i in 0..10 {
            for j in 0..10 {
                assert!((base_weight[[i, j]] - original[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_trainable_parameters() {
        let config = LoraConfig {
            rank: 8,
            ..Default::default()
        };
        let layer = LoraLayer::new(128, 256, &config);
        
        let expected = 8 * 128 + 8 * 256;
        assert_eq!(layer.trainable_parameters(), expected);
    }

    #[test]
    fn test_parameter_reduction_ratio() {
        let config = LoraConfig {
            rank: 8,
            ..Default::default()
        };
        let layer = LoraLayer::new(128, 256, &config);
        
        let base_params = 128 * 256;
        let ratio = layer.parameter_reduction_ratio(base_params);
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_lora_model() {
        let config = LoraConfig {
            rank: 4,
            target_modules: vec!["layer1".to_string()],
            ..Default::default()
        };
        let mut model = LoraModel::new(config);
        
        let weight = Array2::<f32>::ones((10, 10));
        model.add_base_layer("layer1".to_string(), weight);
        model.add_lora_layer("layer1".to_string(), 10, 10);
        
        assert_eq!(model.lora_layers.len(), 1);
        assert_eq!(model.base_layers.len(), 1);
    }

    #[test]
    fn test_parameter_efficiency() {
        let config = LoraConfig {
            rank: 8,
            target_modules: vec!["layer".to_string()],
            ..Default::default()
        };
        let mut model = LoraModel::new(config);
        
        let weight = Array2::<f32>::ones((128, 256));
        model.add_base_layer("layer".to_string(), weight);
        model.add_lora_layer("layer".to_string(), 128, 256);
        
        let efficiency = model.parameter_efficiency();
        assert!(efficiency < 100.0);
        assert!(efficiency > 0.0);
    }
}
