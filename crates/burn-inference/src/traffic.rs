//! Traffic speed prediction

use crate::{InferenceError, Result, TrafficFeatures, Season, WeatherCondition, RoadSegment, encode_cyclical};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

/// Traffic speed predictor using ONNX model
pub struct TrafficPredictor {
    session: Session,
}

impl TrafficPredictor {
    /// Load model from ONNX file
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(InferenceError::ModelLoadError(
                format!("Model file not found: {:?}", path)
            ));
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
    
    /// Predict traffic speed from features
    pub fn predict(&mut self, features: &TrafficFeatures) -> Result<f32> {
        let feature_vec = self.encode_features(features);
        let len = feature_vec.len();
        
        // Create input tensor - ort expects (shape, data) tuple
        let input_value = ort::value::Value::from_array(([1, len], feature_vec))?;
        
        // Run inference
        let outputs = self.session.run(ort::inputs!["features" => &input_value])?;
        
        // Extract prediction
        let output_tensor = outputs["speed"]
            .try_extract_array::<f32>()?;
        
        let speed = output_tensor[[0, 0]];
        
        // Clamp to reasonable range
        Ok(speed.max(5.0).min(80.0))
    }
    
    /// Encode features into vector for model input
    fn encode_features(&self, features: &TrafficFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(20);
        
        // Cyclical time encoding
        let (hour_sin, hour_cos) = encode_cyclical(features.hour as f32, 24.0);
        vec.push(hour_sin);
        vec.push(hour_cos);
        
        let (dow_sin, dow_cos) = encode_cyclical(features.day_of_week as f32, 7.0);
        vec.push(dow_sin);
        vec.push(dow_cos);
        
        // Holiday
        vec.push(if features.is_holiday { 1.0 } else { 0.0 });
        
        // Base speed
        vec.push(features.base_speed);
        
        // Season one-hot
        vec.push(if features.season == Season::Winter { 1.0 } else { 0.0 });
        vec.push(if features.season == Season::Spring { 1.0 } else { 0.0 });
        vec.push(if features.season == Season::Summer { 1.0 } else { 0.0 });
        vec.push(if features.season == Season::Fall { 1.0 } else { 0.0 });
        
        // Weather one-hot
        vec.push(if features.weather == WeatherCondition::Clear { 1.0 } else { 0.0 });
        vec.push(if features.weather == WeatherCondition::Rain { 1.0 } else { 0.0 });
        vec.push(if features.weather == WeatherCondition::HeavyRain { 1.0 } else { 0.0 });
        vec.push(if features.weather == WeatherCondition::Snow { 1.0 } else { 0.0 });
        vec.push(if features.weather == WeatherCondition::HeavySnow { 1.0 } else { 0.0 });
        vec.push(if features.weather == WeatherCondition::Fog { 1.0 } else { 0.0 });
        
        // Road segment one-hot
        vec.push(if features.road_segment == RoadSegment::GspExit83_90 { 1.0 } else { 0.0 });
        vec.push(if features.road_segment == RoadSegment::GspExit117_120 { 1.0 } else { 0.0 });
        vec.push(if features.road_segment == RoadSegment::GspExit29_30 { 1.0 } else { 0.0 });
        vec.push(if features.road_segment == RoadSegment::TurnpikeI95 { 1.0 } else { 0.0 });
        vec.push(if features.road_segment == RoadSegment::Route1 { 1.0 } else { 0.0 });
        vec.push(if features.road_segment == RoadSegment::Route9 { 1.0 } else { 0.0 });
        
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_encoding() {
        let predictor = TrafficPredictor {
            session: unsafe { std::mem::zeroed() }, // Mock for testing
        };
        
        let features = TrafficFeatures {
            hour: 8,
            day_of_week: 1,
            season: Season::Summer,
            weather: WeatherCondition::Clear,
            is_holiday: false,
            base_speed: 55.0,
            road_segment: RoadSegment::GspExit117_120,
        };
        
        let encoded = predictor.encode_features(&features);
        
        // Should have all features
        assert!(encoded.len() >= 20);
        
        // Check some specific values
        assert_eq!(encoded[4], 0.0); // Not holiday
        assert_eq!(encoded[5], 55.0); // Base speed
    }
}
