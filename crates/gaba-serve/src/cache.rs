use crate::{Result, ServeError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub struct CacheConfig {
    pub max_models: usize,
    pub ttl_seconds: u64,
    pub enable_warmup: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            ttl_seconds: 3600,
            enable_warmup: true,
        }
    }
}

pub struct CachedModel<T> {
    pub model: T,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub memory_bytes: usize,
}

pub struct ModelCache<T> {
    config: CacheConfig,
    cache: Arc<RwLock<HashMap<String, CachedModel<T>>>>,
    lru_order: Arc<RwLock<Vec<String>>>,
}

impl<T> ModelCache<T> {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn insert(&self, key: String, model: T, memory_bytes: usize) -> Result<()> {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();

        if cache.len() >= self.config.max_models && !cache.contains_key(&key) {
            if let Some(evict_key) = lru_order.first().cloned() {
                cache.remove(&evict_key);
                lru_order.retain(|k| k != &evict_key);
            }
        }

        cache.insert(
            key.clone(),
            CachedModel {
                model,
                last_accessed: Instant::now(),
                access_count: 0,
                memory_bytes,
            },
        );

        lru_order.retain(|k| k != &key);
        lru_order.push(key);

        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<Arc<T>> 
    where
        T: Clone,
    {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();

        if let Some(cached) = cache.get_mut(key) {
            cached.last_accessed = Instant::now();
            cached.access_count += 1;

            lru_order.retain(|k| k != key);
            lru_order.push(key.to_string());

            return Some(Arc::new(cached.model.clone()));
        }

        None
    }

    pub fn remove(&self, key: &str) -> Option<T> {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();

        lru_order.retain(|k| k != key);
        cache.remove(key).map(|cached| cached.model)
    }

    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();

        cache.clear();
        lru_order.clear();
    }

    pub fn evict_expired(&self) {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        let now = Instant::now();

        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();

        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, cached)| now.duration_since(cached.last_accessed) > ttl)
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            cache.remove(&key);
            lru_order.retain(|k| k != &key);
        }
    }

    pub fn size(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    pub fn total_memory(&self) -> usize {
        self.cache
            .read()
            .unwrap()
            .values()
            .map(|cached| cached.memory_bytes)
            .sum()
    }

    pub fn get_stats(&self, key: &str) -> Option<CacheStats> {
        let cache = self.cache.read().unwrap();
        cache.get(key).map(|cached| CacheStats {
            access_count: cached.access_count,
            last_accessed: cached.last_accessed,
            memory_bytes: cached.memory_bytes,
        })
    }

    pub fn list_keys(&self) -> Vec<String> {
        self.cache.read().unwrap().keys().cloned().collect()
    }
}

pub struct CacheStats {
    pub access_count: u64,
    pub last_accessed: Instant,
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestModel {
        name: String,
    }

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache: ModelCache<TestModel> = ModelCache::new(config);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        let model = TestModel {
            name: "test_model".to_string(),
        };

        cache.insert("model1".to_string(), model, 1000).unwrap();
        assert_eq!(cache.size(), 1);

        let retrieved = cache.get("model1");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_models: 2,
            ..Default::default()
        };
        let cache = ModelCache::new(config);

        cache
            .insert(
                "model1".to_string(),
                TestModel {
                    name: "model1".to_string(),
                },
                1000,
            )
            .unwrap();
        cache
            .insert(
                "model2".to_string(),
                TestModel {
                    name: "model2".to_string(),
                },
                1000,
            )
            .unwrap();
        cache
            .insert(
                "model3".to_string(),
                TestModel {
                    name: "model3".to_string(),
                },
                1000,
            )
            .unwrap();

        assert_eq!(cache.size(), 2);
        assert!(cache.get("model1").is_none());
        assert!(cache.get("model2").is_some());
        assert!(cache.get("model3").is_some());
    }

    #[test]
    fn test_remove() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        cache
            .insert(
                "model1".to_string(),
                TestModel {
                    name: "model1".to_string(),
                },
                1000,
            )
            .unwrap();

        let removed = cache.remove("model1");
        assert!(removed.is_some());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_clear() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        for i in 0..5 {
            cache
                .insert(
                    format!("model{}", i),
                    TestModel {
                        name: format!("model{}", i),
                    },
                    1000,
                )
                .unwrap();
        }

        assert_eq!(cache.size(), 5);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_total_memory() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        cache
            .insert(
                "model1".to_string(),
                TestModel {
                    name: "model1".to_string(),
                },
                1000,
            )
            .unwrap();
        cache
            .insert(
                "model2".to_string(),
                TestModel {
                    name: "model2".to_string(),
                },
                2000,
            )
            .unwrap();

        assert_eq!(cache.total_memory(), 3000);
    }

    #[test]
    fn test_get_stats() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        cache
            .insert(
                "model1".to_string(),
                TestModel {
                    name: "model1".to_string(),
                },
                1000,
            )
            .unwrap();

        cache.get("model1");
        cache.get("model1");

        let stats = cache.get_stats("model1");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().access_count, 2);
    }

    #[test]
    fn test_list_keys() {
        let config = CacheConfig::default();
        let cache = ModelCache::new(config);

        cache
            .insert(
                "model1".to_string(),
                TestModel {
                    name: "model1".to_string(),
                },
                1000,
            )
            .unwrap();
        cache
            .insert(
                "model2".to_string(),
                TestModel {
                    name: "model2".to_string(),
                },
                1000,
            )
            .unwrap();

        let keys = cache.list_keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"model1".to_string()));
        assert!(keys.contains(&"model2".to_string()));
    }
}
