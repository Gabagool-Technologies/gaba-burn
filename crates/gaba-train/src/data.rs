//! Data loading and preprocessing for route optimization models

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::path::Path;

/// Traffic speed data point
#[derive(Debug, Clone, Deserialize)]
pub struct TrafficRecord {
    pub timestamp: String,
    pub road_segment_id: String,
    pub speed_mph: f32,
    pub day_of_week: u32,
    pub hour: u32,
    pub season: String,
    pub weather_condition: String,
    pub is_holiday: bool,
    pub base_speed: f32,
}

/// Route completion data point
#[derive(Debug, Clone, Deserialize)]
pub struct RouteRecord {
    pub route_id: String,
    pub stops_count: u32,
    pub total_distance_miles: f32,
    pub predicted_time_minutes: u32,
    pub actual_time_minutes: u32,
    pub traffic_delay_minutes: u32,
    pub start_time: String,
    pub weather: String,
    pub season: String,
    pub hour: u32,
    pub day_of_week: u32,
}

/// Traffic dataset for training
pub struct TrafficDataset {
    pub features: Array2<f32>,
    pub targets: Array1<f32>,
    pub feature_names: Vec<String>,
}

impl TrafficDataset {
    /// Load from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path).context("Failed to open CSV file")?;

        let mut records = Vec::new();
        for result in reader.deserialize() {
            let record: TrafficRecord = result.context("Failed to parse record")?;
            records.push(record);
        }

        let (features, targets) = Self::encode_features(&records)?;
        let feature_names = Self::feature_names();

        Ok(Self {
            features,
            targets,
            feature_names,
        })
    }

    /// Encode features from raw records
    fn encode_features(records: &[TrafficRecord]) -> Result<(Array2<f32>, Array1<f32>)> {
        let n_samples = records.len();
        let n_features = 22; // Cyclical time + one-hot encodings

        let mut features = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        for (i, record) in records.iter().enumerate() {
            let mut feat_idx = 0;

            // Cyclical time encoding
            let hour_rad = 2.0 * std::f32::consts::PI * record.hour as f32 / 24.0;
            features[[i, feat_idx]] = hour_rad.sin();
            feat_idx += 1;
            features[[i, feat_idx]] = hour_rad.cos();
            feat_idx += 1;

            let dow_rad = 2.0 * std::f32::consts::PI * record.day_of_week as f32 / 7.0;
            features[[i, feat_idx]] = dow_rad.sin();
            feat_idx += 1;
            features[[i, feat_idx]] = dow_rad.cos();
            feat_idx += 1;

            // Holiday
            features[[i, feat_idx]] = if record.is_holiday { 1.0 } else { 0.0 };
            feat_idx += 1;

            // Base speed
            features[[i, feat_idx]] = record.base_speed;
            feat_idx += 1;

            // Season one-hot
            features[[i, feat_idx]] = if record.season == "winter" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "spring" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "summer" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "fall" { 1.0 } else { 0.0 };
            feat_idx += 1;

            // Weather one-hot
            features[[i, feat_idx]] = if record.weather_condition == "clear" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather_condition == "rain" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather_condition == "heavy_rain" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather_condition == "snow" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather_condition == "heavy_snow" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather_condition == "fog" {
                1.0
            } else {
                0.0
            };
            feat_idx += 1;

            // Road segment one-hot (6 segments)
            let segment_idx = match record.road_segment_id.as_str() {
                "GSP_Exit_83_90" => 0,
                "GSP_Exit_117_120" => 1,
                "GSP_Exit_29_30" => 2,
                "Turnpike_I95" => 3,
                "Route_1" => 4,
                "Route_9" => 5,
                _ => 0,
            };
            for j in 0..6 {
                features[[i, feat_idx + j]] = if j == segment_idx { 1.0 } else { 0.0 };
            }

            // Target
            targets[i] = record.speed_mph;
        }

        Ok((features, targets))
    }

    fn feature_names() -> Vec<String> {
        vec![
            "hour_sin".to_string(),
            "hour_cos".to_string(),
            "dow_sin".to_string(),
            "dow_cos".to_string(),
            "is_holiday".to_string(),
            "base_speed".to_string(),
            "season_winter".to_string(),
            "season_spring".to_string(),
            "season_summer".to_string(),
            "season_fall".to_string(),
            "weather_clear".to_string(),
            "weather_rain".to_string(),
            "weather_heavy_rain".to_string(),
            "weather_snow".to_string(),
            "weather_heavy_snow".to_string(),
            "weather_fog".to_string(),
            "segment_gsp_83_90".to_string(),
            "segment_gsp_117_120".to_string(),
            "segment_gsp_29_30".to_string(),
            "segment_turnpike".to_string(),
            "segment_route1".to_string(),
            "segment_route9".to_string(),
        ]
    }

    /// Split into train/test sets
    pub fn split(&self, test_ratio: f32) -> (Self, Self) {
        let n_samples = self.features.nrows();
        let n_test = (n_samples as f32 * test_ratio) as usize;
        let n_train = n_samples - n_test;

        let train_features = self.features.slice(s![..n_train, ..]).to_owned();
        let train_targets = self.targets.slice(s![..n_train]).to_owned();

        let test_features = self.features.slice(s![n_train.., ..]).to_owned();
        let test_targets = self.targets.slice(s![n_train..]).to_owned();

        let train = Self {
            features: train_features,
            targets: train_targets,
            feature_names: self.feature_names.clone(),
        };

        let test = Self {
            features: test_features,
            targets: test_targets,
            feature_names: self.feature_names.clone(),
        };

        (train, test)
    }

    pub fn len(&self) -> usize {
        self.features.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Route dataset for training
pub struct RouteDataset {
    pub features: Array2<f32>,
    pub targets: Array1<f32>,
    pub feature_names: Vec<String>,
}

impl RouteDataset {
    /// Load from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path).context("Failed to open CSV file")?;

        let mut records = Vec::new();
        for result in reader.deserialize() {
            let record: RouteRecord = result.context("Failed to parse record")?;
            records.push(record);
        }

        let (features, targets) = Self::encode_features(&records)?;
        let feature_names = Self::feature_names();

        Ok(Self {
            features,
            targets,
            feature_names,
        })
    }

    /// Encode features from raw records
    fn encode_features(records: &[RouteRecord]) -> Result<(Array2<f32>, Array1<f32>)> {
        let n_samples = records.len();
        let n_features = 15;

        let mut features = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        for (i, record) in records.iter().enumerate() {
            let mut feat_idx = 0;

            // Basic features
            features[[i, feat_idx]] = record.stops_count as f32;
            feat_idx += 1;
            features[[i, feat_idx]] = record.total_distance_miles;
            feat_idx += 1;

            // Derived features
            let stops_per_mile = record.stops_count as f32 / record.total_distance_miles.max(0.1);
            let avg_stop_distance = record.total_distance_miles / record.stops_count as f32;
            features[[i, feat_idx]] = stops_per_mile;
            feat_idx += 1;
            features[[i, feat_idx]] = avg_stop_distance;
            feat_idx += 1;

            // Cyclical time
            let hour_rad = 2.0 * std::f32::consts::PI * record.hour as f32 / 24.0;
            features[[i, feat_idx]] = hour_rad.sin();
            feat_idx += 1;
            features[[i, feat_idx]] = hour_rad.cos();
            feat_idx += 1;

            let dow_rad = 2.0 * std::f32::consts::PI * record.day_of_week as f32 / 7.0;
            features[[i, feat_idx]] = dow_rad.sin();
            feat_idx += 1;
            features[[i, feat_idx]] = dow_rad.cos();
            feat_idx += 1;

            // Season one-hot
            features[[i, feat_idx]] = if record.season == "winter" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "spring" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "summer" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.season == "fall" { 1.0 } else { 0.0 };
            feat_idx += 1;

            // Weather one-hot
            features[[i, feat_idx]] = if record.weather == "clear" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather == "rain" { 1.0 } else { 0.0 };
            feat_idx += 1;
            features[[i, feat_idx]] = if record.weather == "snow" { 1.0 } else { 0.0 };

            // Target
            targets[i] = record.actual_time_minutes as f32;
        }

        Ok((features, targets))
    }

    fn feature_names() -> Vec<String> {
        vec![
            "stops_count".to_string(),
            "total_distance".to_string(),
            "stops_per_mile".to_string(),
            "avg_stop_distance".to_string(),
            "hour_sin".to_string(),
            "hour_cos".to_string(),
            "dow_sin".to_string(),
            "dow_cos".to_string(),
            "season_winter".to_string(),
            "season_spring".to_string(),
            "season_summer".to_string(),
            "season_fall".to_string(),
            "weather_clear".to_string(),
            "weather_rain".to_string(),
            "weather_snow".to_string(),
        ]
    }

    /// Split into train/test sets
    pub fn split(&self, test_ratio: f32) -> (Self, Self) {
        let n_samples = self.features.nrows();
        let n_test = (n_samples as f32 * test_ratio) as usize;
        let n_train = n_samples - n_test;

        let train_features = self.features.slice(s![..n_train, ..]).to_owned();
        let train_targets = self.targets.slice(s![..n_train]).to_owned();

        let test_features = self.features.slice(s![n_train.., ..]).to_owned();
        let test_targets = self.targets.slice(s![n_train..]).to_owned();

        let train = Self {
            features: train_features,
            targets: train_targets,
            feature_names: self.feature_names.clone(),
        };

        let test = Self {
            features: test_features,
            targets: test_targets,
            feature_names: self.feature_names.clone(),
        };

        (train, test)
    }

    pub fn len(&self) -> usize {
        self.features.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Import ndarray slicing macro
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclical_encoding() {
        let hour = 6u32;
        let rad = 2.0 * std::f32::consts::PI * hour as f32 / 24.0;
        let sin = rad.sin();
        let cos = rad.cos();

        assert!((sin - 1.0).abs() < 0.01);
        assert!(cos.abs() < 0.01);
    }
}
