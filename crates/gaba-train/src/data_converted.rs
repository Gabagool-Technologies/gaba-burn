use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ConvertedTrafficRecord {
    pub hour_sin: f32,
    pub hour_cos: f32,
    pub dow_sin: f32,
    pub dow_cos: f32,
    pub base_speed: f32,
    pub season_winter: u8,
    pub season_spring: u8,
    pub season_summer: u8,
    pub season_fall: u8,
    pub weather_clear: u8,
    pub weather_rain: u8,
    pub weather_heavy_rain: u8,
    pub weather_snow: u8,
    pub weather_heavy_snow: u8,
    pub weather_fog: u8,
    #[serde(rename = "segment_GSP_Exit_83_90")]
    pub segment_gsp_exit_83_90: u8,
    #[serde(rename = "segment_GSP_Exit_117_120")]
    pub segment_gsp_exit_117_120: u8,
    #[serde(rename = "segment_GSP_Exit_29_30")]
    pub segment_gsp_exit_29_30: u8,
    #[serde(rename = "segment_Turnpike_I95")]
    pub segment_turnpike_i95: u8,
    #[serde(rename = "segment_Route_1")]
    pub segment_route_1: u8,
    #[serde(rename = "segment_Route_9")]
    pub segment_route_9: u8,
    pub speed_mph: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConvertedRouteRecord {
    pub stops_count: f32,
    pub total_distance_miles: f32,
    pub stops_per_mile: f32,
    pub avg_stop_distance: f32,
    pub hour_sin: f32,
    pub hour_cos: f32,
    pub dow_sin: f32,
    pub dow_cos: f32,
    pub season_winter: u8,
    pub season_spring: u8,
    pub season_summer: u8,
    pub season_fall: u8,
    pub weather_clear: u8,
    pub weather_rain: u8,
    pub weather_snow: u8,
    pub actual_time_minutes: f32,
}

pub struct ConvertedTrafficDataset {
    pub features: Array2<f32>,
    pub targets: Array1<f32>,
}

impl ConvertedTrafficDataset {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path).context("Failed to open CSV file")?;
        
        let mut features_vec = Vec::new();
        let mut targets_vec = Vec::new();
        
        for result in reader.deserialize() {
            let record: ConvertedTrafficRecord = result.context("Failed to parse record")?;
            
            let features = vec![
                record.hour_sin,
                record.hour_cos,
                record.dow_sin,
                record.dow_cos,
                record.base_speed,
                record.season_winter as f32,
                record.season_spring as f32,
                record.season_summer as f32,
                record.season_fall as f32,
                record.weather_clear as f32,
                record.weather_rain as f32,
                record.weather_heavy_rain as f32,
                record.weather_snow as f32,
                record.weather_heavy_snow as f32,
                record.weather_fog as f32,
                record.segment_gsp_exit_83_90 as f32,
                record.segment_gsp_exit_117_120 as f32,
                record.segment_gsp_exit_29_30 as f32,
                record.segment_turnpike_i95 as f32,
                record.segment_route_1 as f32,
                record.segment_route_9 as f32,
            ];
            
            features_vec.extend(features);
            targets_vec.push(record.speed_mph);
        }
        
        let n_samples = targets_vec.len();
        let n_features = 21;
        
        let features = Array2::from_shape_vec((n_samples, n_features), features_vec)?;
        let targets = Array1::from_vec(targets_vec);
        
        Ok(Self { features, targets })
    }
    
    pub fn len(&self) -> usize {
        self.targets.len()
    }
    
    pub fn split(&self, test_ratio: f32) -> (Self, Self) {
        let n_test = (self.len() as f32 * test_ratio) as usize;
        let n_train = self.len() - n_test;
        
        let train_features = self.features.slice(ndarray::s![..n_train, ..]).to_owned();
        let train_targets = self.targets.slice(ndarray::s![..n_train]).to_owned();
        
        let test_features = self.features.slice(ndarray::s![n_train.., ..]).to_owned();
        let test_targets = self.targets.slice(ndarray::s![n_train..]).to_owned();
        
        (
            Self {
                features: train_features,
                targets: train_targets,
            },
            Self {
                features: test_features,
                targets: test_targets,
            },
        )
    }
}

pub struct ConvertedRouteDataset {
    pub features: Array2<f32>,
    pub targets: Array1<f32>,
}

impl ConvertedRouteDataset {
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path).context("Failed to open CSV file")?;
        
        let mut features_vec = Vec::new();
        let mut targets_vec = Vec::new();
        
        for result in reader.deserialize() {
            let record: ConvertedRouteRecord = result.context("Failed to parse record")?;
            
            let features = vec![
                record.stops_count,
                record.total_distance_miles,
                record.stops_per_mile,
                record.avg_stop_distance,
                record.hour_sin,
                record.hour_cos,
                record.dow_sin,
                record.dow_cos,
                record.season_winter as f32,
                record.season_spring as f32,
                record.season_summer as f32,
                record.season_fall as f32,
                record.weather_clear as f32,
                record.weather_rain as f32,
                record.weather_snow as f32,
            ];
            
            features_vec.extend(features);
            targets_vec.push(record.actual_time_minutes);
        }
        
        let n_samples = targets_vec.len();
        let n_features = 15;
        
        let features = Array2::from_shape_vec((n_samples, n_features), features_vec)?;
        let targets = Array1::from_vec(targets_vec);
        
        Ok(Self { features, targets })
    }
    
    pub fn len(&self) -> usize {
        self.targets.len()
    }
    
    pub fn split(&self, test_ratio: f32) -> (Self, Self) {
        let n_test = (self.len() as f32 * test_ratio) as usize;
        let n_train = self.len() - n_test;
        
        let train_features = self.features.slice(ndarray::s![..n_train, ..]).to_owned();
        let train_targets = self.targets.slice(ndarray::s![..n_train]).to_owned();
        
        let test_features = self.features.slice(ndarray::s![n_train.., ..]).to_owned();
        let test_targets = self.targets.slice(ndarray::s![n_train..]).to_owned();
        
        (
            Self {
                features: train_features,
                targets: train_targets,
            },
            Self {
                features: test_features,
                targets: test_targets,
            },
        )
    }
}
