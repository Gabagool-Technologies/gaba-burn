//! ML-enhanced route optimizer with traffic prediction

use crate::route_optimizer::{GeoPoint, build_distance_matrix, solve_tsp_optimized};
use chrono::{DateTime, Utc};

#[cfg(feature = "ml-inference")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "ml-inference")]
use burn_inference::{TrafficPredictor, RouteTimePredictor, RouteFeatures, WeatherCondition};

/// ML-enhanced route optimizer
pub struct MLRouteOptimizer {
    #[cfg(feature = "ml-inference")]
    traffic_predictor: Option<Arc<Mutex<TrafficPredictor>>>,
    #[cfg(feature = "ml-inference")]
    time_predictor: Option<Arc<Mutex<RouteTimePredictor>>>,
}

impl MLRouteOptimizer {
    /// Create new ML optimizer
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ml-inference")]
            traffic_predictor: None,
            #[cfg(feature = "ml-inference")]
            time_predictor: None,
        }
    }
    
    /// Load ML models
    #[cfg(feature = "ml-inference")]
    pub fn with_models(
        traffic_model_path: &str,
        time_model_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let traffic_predictor = TrafficPredictor::load(traffic_model_path)?;
        let time_predictor = RouteTimePredictor::load(time_model_path)?;
        
        Ok(Self {
            traffic_predictor: Some(Arc::new(Mutex::new(traffic_predictor))),
            time_predictor: Some(Arc::new(Mutex::new(time_predictor))),
        })
    }
    
    /// Optimize route with ML enhancements
    pub fn optimize(
        &self,
        points: &[GeoPoint],
        start_idx: usize,
        max_iterations: usize,
        #[allow(unused_variables)]
        timestamp: Option<DateTime<Utc>>,
        #[allow(unused_variables)]
        #[cfg(feature = "ml-inference")]
        weather: Option<WeatherCondition>,
        #[allow(unused_variables)]
        #[cfg(not(feature = "ml-inference"))]
        weather: Option<u8>,
    ) -> (Vec<usize>, f64) {
        #[cfg(feature = "ml-inference")]
        {
            if let (Some(_traffic_pred), Some(time_pred)) = 
                (&self.traffic_predictor, &self.time_predictor) 
            {
                return self.optimize_with_ml(
                    points, 
                    start_idx, 
                    max_iterations,
                    timestamp.unwrap_or_else(Utc::now),
                    weather.unwrap_or(WeatherCondition::Clear),
                    time_pred,
                );
            }
        }
        
        // Fallback to baseline optimizer
        self.optimize_baseline(points, start_idx, max_iterations)
    }
    
    /// Baseline optimization without ML
    fn optimize_baseline(
        &self,
        points: &[GeoPoint],
        start_idx: usize,
        max_iterations: usize,
    ) -> (Vec<usize>, f64) {
        let matrix = build_distance_matrix(points);
        solve_tsp_optimized(&matrix, points.len(), start_idx, max_iterations)
    }
    
    /// ML-enhanced optimization
    #[cfg(feature = "ml-inference")]
    fn optimize_with_ml(
        &self,
        points: &[GeoPoint],
        start_idx: usize,
        max_iterations: usize,
        timestamp: DateTime<Utc>,
        weather: WeatherCondition,
        time_pred: &Arc<Mutex<RouteTimePredictor>>,
    ) -> (Vec<usize>, f64) {
        // Build distance matrix
        let matrix = build_distance_matrix(points);
        
        // Run TSP solver
        let (route, distance) = solve_tsp_optimized(&matrix, points.len(), start_idx, max_iterations);
        
        // Predict route time
        let features = RouteFeatures::from_route(
            points.len() as u32,
            distance as f32,
            timestamp,
            weather,
        );
        
        let _predicted_time = time_pred
            .lock()
            .ok()
            .and_then(|mut p| p.predict(&features).ok())
            .unwrap_or(distance * 2.0);
        
        // For now, return the optimized route
        // In future, could use traffic predictions to adjust edge weights
        (route, distance)
    }
}

impl Default for MLRouteOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_baseline_optimization() {
        let optimizer = MLRouteOptimizer::new();
        
        let points = vec![
            GeoPoint::new(40.7357, -74.1724),
            GeoPoint::new(40.7420, -74.1726),
            GeoPoint::new(40.7489, -74.1717),
        ];
        
        let (route, distance) = optimizer.optimize(&points, 0, 100, None, None);
        
        assert_eq!(route.len(), points.len());
        assert!(distance > 0.0);
    }
}
