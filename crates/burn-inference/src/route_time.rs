//! Route time prediction

use crate::{encode_cyclical, InferenceError, Result, RouteFeatures, Season, WeatherCondition};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;

/// Route time predictor using ONNX model
pub struct RouteTimePredictor {
    session: Session,
}

impl RouteTimePredictor {
    /// Load model from ONNX file
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {:?}",
                path
            )));
        }

        let session = Session::builder()
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Predict route completion time from features
    pub fn predict(&mut self, features: &RouteFeatures) -> Result<f64> {
        let feature_vec = self.encode_features(features);
        let len = feature_vec.len();

        // Create input tensor - ort expects (shape, data) tuple
        let input_value = ort::value::Value::from_array(([1, len], feature_vec))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs!["float_input" => &input_value])?;

        // Extract prediction
        let output_tensor = outputs[0].try_extract_array::<f32>()?;

        let time_minutes = output_tensor[[0, 0]] as f64;

        // Clamp to reasonable range
        Ok(time_minutes.max(10.0).min(480.0))
    }

    /// Encode features into vector for model input
    fn encode_features(&self, features: &RouteFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(15);

        // Basic route features
        vec.push(features.stops_count as f32);
        vec.push(features.total_distance_miles);

        // Derived features
        let stops_per_mile = features.stops_count as f32 / features.total_distance_miles.max(0.1);
        let avg_stop_distance = features.total_distance_miles / features.stops_count as f32;
        vec.push(stops_per_mile);
        vec.push(avg_stop_distance);

        // Cyclical time encoding
        let (hour_sin, hour_cos) = encode_cyclical(features.hour as f32, 24.0);
        vec.push(hour_sin);
        vec.push(hour_cos);

        let (dow_sin, dow_cos) = encode_cyclical(features.day_of_week as f32, 7.0);
        vec.push(dow_sin);
        vec.push(dow_cos);

        // Season one-hot
        vec.push(if features.season == Season::Winter {
            1.0
        } else {
            0.0
        });
        vec.push(if features.season == Season::Spring {
            1.0
        } else {
            0.0
        });
        vec.push(if features.season == Season::Summer {
            1.0
        } else {
            0.0
        });
        vec.push(if features.season == Season::Fall {
            1.0
        } else {
            0.0
        });

        // Weather one-hot
        vec.push(if features.weather == WeatherCondition::Clear {
            1.0
        } else {
            0.0
        });
        vec.push(if features.weather == WeatherCondition::Rain {
            1.0
        } else {
            0.0
        });
        vec.push(if features.weather == WeatherCondition::Snow {
            1.0
        } else {
            0.0
        });

        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_encoding() {
        let predictor = RouteTimePredictor {
            session: unsafe { std::mem::zeroed() }, // Mock for testing
        };

        let features = RouteFeatures {
            stops_count: 10,
            total_distance_miles: 15.5,
            hour: 8,
            day_of_week: 1,
            season: Season::Summer,
            weather: WeatherCondition::Clear,
        };

        let encoded = predictor.encode_features(&features);

        // Should have all features
        assert!(encoded.len() >= 15);

        // Check some specific values
        assert_eq!(encoded[0], 10.0); // Stops count
        assert_eq!(encoded[1], 15.5); // Distance
    }
}
