use ndarray::Array2;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use crate::peft::lora::LoraLayer;

#[derive(Clone)]
pub struct AdapterConfig {
    pub name: String,
    pub task: String,
    pub enabled: bool,
}

pub struct AdapterManager {
    adapters: HashMap<String, Adapter>,
    active_adapter: Option<String>,
}

pub struct Adapter {
    pub config: AdapterConfig,
    pub lora_layers: HashMap<String, LoraLayer>,
}

impl AdapterManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
        }
    }

    pub fn add_adapter(&mut self, adapter: Adapter) {
        let name = adapter.config.name.clone();
        self.adapters.insert(name, adapter);
    }

    pub fn set_active_adapter(&mut self, name: &str) -> Result<()> {
        if !self.adapters.contains_key(name) {
            return Err(anyhow::anyhow!("Adapter not found: {}", name));
        }
        self.active_adapter = Some(name.to_string());
        Ok(())
    }

    pub fn get_active_adapter(&self) -> Option<&Adapter> {
        self.active_adapter.as_ref().and_then(|name| self.adapters.get(name))
    }

    pub fn get_active_adapter_mut(&mut self) -> Option<&mut Adapter> {
        self.active_adapter.as_ref().and_then(|name| self.adapters.get_mut(name))
    }

    pub fn disable_all_adapters(&mut self) {
        for adapter in self.adapters.values_mut() {
            adapter.config.enabled = false;
        }
        self.active_adapter = None;
    }

    pub fn enable_adapter(&mut self, name: &str) -> Result<()> {
        let adapter = self.adapters.get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Adapter not found: {}", name))?;
        adapter.config.enabled = true;
        self.active_adapter = Some(name.to_string());
        Ok(())
    }

    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    pub fn save_adapter(&self, name: &str, path: &Path) -> Result<()> {
        let adapter = self.adapters.get(name)
            .ok_or_else(|| anyhow::anyhow!("Adapter not found: {}", name))?;
        
        let serialized = serde_json::to_string_pretty(&adapter.config)?;
        std::fs::write(path.join(format!("{}_config.json", name)), serialized)?;
        
        Ok(())
    }

    pub fn load_adapter(&mut self, path: &Path) -> Result<String> {
        let config_files: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        
        if config_files.is_empty() {
            return Err(anyhow::anyhow!("No adapter config found"));
        }
        
        let config_path = &config_files[0].path();
        let config_str = std::fs::read_to_string(config_path)?;
        let config: AdapterConfig = serde_json::from_str(&config_str)?;
        
        let adapter = Adapter {
            config: config.clone(),
            lora_layers: HashMap::new(),
        };
        
        let name = config.name.clone();
        self.add_adapter(adapter);
        
        Ok(name)
    }

    pub fn merge_adapters(&mut self, adapter_names: &[String], weights: &[f32]) -> Result<Adapter> {
        if adapter_names.len() != weights.len() {
            return Err(anyhow::anyhow!("Adapter names and weights length mismatch"));
        }
        
        if adapter_names.is_empty() {
            return Err(anyhow::anyhow!("No adapters to merge"));
        }
        
        let weight_sum: f32 = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            return Err(anyhow::anyhow!("Weights must sum to 1.0"));
        }
        
        let merged_config = AdapterConfig {
            name: "merged_adapter".to_string(),
            task: "merged".to_string(),
            enabled: true,
        };
        
        let merged_adapter = Adapter {
            config: merged_config,
            lora_layers: HashMap::new(),
        };
        
        Ok(merged_adapter)
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Adapter {
    pub fn new(config: AdapterConfig) -> Self {
        Self {
            config,
            lora_layers: HashMap::new(),
        }
    }

    pub fn add_lora_layer(&mut self, name: String, layer: LoraLayer) {
        self.lora_layers.insert(name, layer);
    }

    pub fn get_lora_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.lora_layers.get(name)
    }

    pub fn get_lora_layer_mut(&mut self, name: &str) -> Option<&mut LoraLayer> {
        self.lora_layers.get_mut(name)
    }

    pub fn merge_all_layers(&mut self, base_weights: &mut HashMap<String, Array2<f32>>) -> Result<()> {
        for (name, lora_layer) in &mut self.lora_layers {
            if let Some(base_weight) = base_weights.get_mut(name) {
                lora_layer.merge_with_base(base_weight)?;
            }
        }
        Ok(())
    }

    pub fn unmerge_all_layers(&mut self, base_weights: &mut HashMap<String, Array2<f32>>) -> Result<()> {
        for (name, lora_layer) in &mut self.lora_layers {
            if let Some(base_weight) = base_weights.get_mut(name) {
                lora_layer.unmerge_from_base(base_weight)?;
            }
        }
        Ok(())
    }

    pub fn total_parameters(&self) -> usize {
        self.lora_layers.values().map(|l| l.trainable_parameters()).sum()
    }
}

impl serde::Serialize for AdapterConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("AdapterConfig", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("task", &self.task)?;
        state.serialize_field("enabled", &self.enabled)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for AdapterConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct AdapterConfigHelper {
            name: String,
            task: String,
            enabled: bool,
        }
        
        let helper = AdapterConfigHelper::deserialize(deserializer)?;
        Ok(AdapterConfig {
            name: helper.name,
            task: helper.task,
            enabled: helper.enabled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peft::lora::{LoraConfig, LoraLayer};

    #[test]
    fn test_adapter_manager_creation() {
        let manager = AdapterManager::new();
        assert!(manager.active_adapter.is_none());
        assert_eq!(manager.list_adapters().len(), 0);
    }

    #[test]
    fn test_add_adapter() {
        let mut manager = AdapterManager::new();
        let config = AdapterConfig {
            name: "test_adapter".to_string(),
            task: "classification".to_string(),
            enabled: true,
        };
        let adapter = Adapter::new(config);
        
        manager.add_adapter(adapter);
        assert_eq!(manager.list_adapters().len(), 1);
    }

    #[test]
    fn test_set_active_adapter() {
        let mut manager = AdapterManager::new();
        let config = AdapterConfig {
            name: "test_adapter".to_string(),
            task: "classification".to_string(),
            enabled: true,
        };
        let adapter = Adapter::new(config);
        
        manager.add_adapter(adapter);
        let result = manager.set_active_adapter("test_adapter");
        assert!(result.is_ok());
        assert_eq!(manager.active_adapter, Some("test_adapter".to_string()));
    }

    #[test]
    fn test_disable_all_adapters() {
        let mut manager = AdapterManager::new();
        let config = AdapterConfig {
            name: "test_adapter".to_string(),
            task: "classification".to_string(),
            enabled: true,
        };
        let adapter = Adapter::new(config);
        
        manager.add_adapter(adapter);
        manager.set_active_adapter("test_adapter").unwrap();
        manager.disable_all_adapters();
        
        assert!(manager.active_adapter.is_none());
    }

    #[test]
    fn test_adapter_with_lora_layers() {
        let config = AdapterConfig {
            name: "test_adapter".to_string(),
            task: "classification".to_string(),
            enabled: true,
        };
        let mut adapter = Adapter::new(config);
        
        let lora_config = LoraConfig::default();
        let lora_layer = LoraLayer::new(128, 256, &lora_config);
        
        adapter.add_lora_layer("layer1".to_string(), lora_layer);
        assert_eq!(adapter.lora_layers.len(), 1);
        assert!(adapter.get_lora_layer("layer1").is_some());
    }

    #[test]
    fn test_adapter_total_parameters() {
        let config = AdapterConfig {
            name: "test_adapter".to_string(),
            task: "classification".to_string(),
            enabled: true,
        };
        let mut adapter = Adapter::new(config);
        
        let lora_config = LoraConfig {
            rank: 8,
            ..Default::default()
        };
        let lora_layer = LoraLayer::new(128, 256, &lora_config);
        
        adapter.add_lora_layer("layer1".to_string(), lora_layer);
        let params = adapter.total_parameters();
        assert_eq!(params, 8 * 128 + 8 * 256);
    }
}
