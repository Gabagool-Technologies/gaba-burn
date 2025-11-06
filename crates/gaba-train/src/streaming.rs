//! Streaming data loader for memory-efficient training

use anyhow::{Context, Result};
use memmap2::Mmap;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Streaming data loader configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub batch_size: usize,
    pub prefetch_batches: usize,
    pub num_workers: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            prefetch_batches: 2,
            num_workers: 4,
        }
    }
}

/// Streaming batch iterator
pub struct StreamingDataLoader {
    config: StreamConfig,
    total_samples: usize,
    current_batch: usize,
}

impl StreamingDataLoader {
    pub fn new(config: StreamConfig, total_samples: usize) -> Self {
        Self {
            config,
            total_samples,
            current_batch: 0,
        }
    }

    pub fn num_batches(&self) -> usize {
        (self.total_samples + self.config.batch_size - 1) / self.config.batch_size
    }

    pub fn reset(&mut self) {
        self.current_batch = 0;
    }
}

/// Memory-mapped dataset for large files
pub struct MmapDataset {
    mmap: Mmap,
    num_samples: usize,
    feature_dim: usize,
    sample_size_bytes: usize,
    header_offset: usize,
}

impl MmapDataset {
    pub fn from_binary<P: AsRef<Path>>(path: P, feature_dim: usize) -> Result<Self> {
        let file = File::open(path.as_ref()).context("Failed to open dataset file")?;

        let mmap = unsafe { Mmap::map(&file)? };

        let header_offset = 16;
        let sample_size_bytes = (feature_dim + 1) * std::mem::size_of::<f32>();
        let data_size = mmap.len() - header_offset;
        let num_samples = data_size / sample_size_bytes;

        Ok(Self {
            mmap,
            num_samples,
            feature_dim,
            sample_size_bytes,
            header_offset,
        })
    }

    pub fn len(&self) -> usize {
        self.num_samples
    }

    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }

    pub fn get_sample(&self, idx: usize) -> Result<(Vec<f32>, f32)> {
        if idx >= self.num_samples {
            anyhow::bail!("Sample index out of bounds");
        }

        let offset = self.header_offset + idx * self.sample_size_bytes;
        let sample_bytes = &self.mmap[offset..offset + self.sample_size_bytes];

        let mut features = Vec::with_capacity(self.feature_dim);
        for i in 0..self.feature_dim {
            let byte_offset = i * std::mem::size_of::<f32>();
            let bytes = &sample_bytes[byte_offset..byte_offset + 4];
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            features.push(value);
        }

        let target_offset = self.feature_dim * std::mem::size_of::<f32>();
        let target_bytes = &sample_bytes[target_offset..target_offset + 4];
        let target = f32::from_le_bytes([
            target_bytes[0],
            target_bytes[1],
            target_bytes[2],
            target_bytes[3],
        ]);

        Ok((features, target))
    }

    pub fn get_batch(&self, start: usize, end: usize) -> Result<(Array2<f32>, Array1<f32>)> {
        let batch_size = end - start;
        let mut features = Array2::zeros((batch_size, self.feature_dim));
        let mut targets = Array1::zeros(batch_size);

        for (i, idx) in (start..end).enumerate() {
            let (feat, target) = self.get_sample(idx)?;
            for (j, &val) in feat.iter().enumerate() {
                features[[i, j]] = val;
            }
            targets[i] = target;
        }

        Ok((features, targets))
    }

    pub fn get_batch_parallel(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<f32>)> {
        let batch_size = indices.len();
        let samples: Vec<_> = indices
            .par_iter()
            .map(|&idx| self.get_sample(idx))
            .collect::<Result<Vec<_>>>()?;

        let mut features = Array2::zeros((batch_size, self.feature_dim));
        let mut targets = Array1::zeros(batch_size);

        for (i, (feat, target)) in samples.into_iter().enumerate() {
            for (j, val) in feat.into_iter().enumerate() {
                features[[i, j]] = val;
            }
            targets[i] = target;
        }

        Ok((features, targets))
    }
}

/// Parallel data loader with prefetching
pub struct ParallelDataLoader {
    dataset: Arc<MmapDataset>,
    batch_size: usize,
    prefetch_buffer: Arc<Mutex<VecDeque<(Array2<f32>, Array1<f32>)>>>,
    num_workers: usize,
}

impl ParallelDataLoader {
    pub fn new(dataset: MmapDataset, batch_size: usize, prefetch_batches: usize) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            prefetch_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(prefetch_batches))),
            num_workers: num_cpus::get(),
        }
    }

    pub fn get_batch(&self, batch_idx: usize) -> Result<(Array2<f32>, Array1<f32>)> {
        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.dataset.len());

        let indices: Vec<usize> = (start..end).collect();
        self.dataset.get_batch_parallel(&indices)
    }

    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config() {
        let config = StreamConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.prefetch_batches, 2);
    }

    #[test]
    fn test_streaming_loader() {
        let config = StreamConfig::default();
        let loader = StreamingDataLoader::new(config, 100);
        assert_eq!(loader.num_batches(), 4);
    }
}
