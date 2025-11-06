//! ML inference for route optimization
//!
//! Provides ONNX-based traffic prediction and route time estimation.

use chrono::{DateTime, Datelike, Timelike, Utc};

pub mod error;
pub mod route_time;
pub mod traffic;

pub use error::{InferenceError, Result};
pub use route_time::RouteTimePredictor;
pub use traffic::TrafficPredictor;

/// Feature vector for traffic prediction
#[derive(Debug, Clone)]
pub struct TrafficFeatures {
    pub hour: u32,
    pub day_of_week: u32,
    pub season: Season,
    pub weather: WeatherCondition,
    pub is_holiday: bool,
    pub base_speed: f32,
    pub road_segment: RoadSegment,
}

/// Season enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Season {
    Winter,
    Spring,
    Summer,
    Fall,
}

impl Season {
    pub fn from_month(month: u32) -> Self {
        match month {
            12 | 1 | 2 => Season::Winter,
            3 | 4 | 5 => Season::Spring,
            6 | 7 | 8 => Season::Summer,
            _ => Season::Fall,
        }
    }
}

/// Weather condition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherCondition {
    Clear,
    Rain,
    HeavyRain,
    Snow,
    HeavySnow,
    Fog,
}

/// Road segment type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadSegment {
    GspExit83_90,
    GspExit117_120,
    GspExit29_30,
    TurnpikeI95,
    Route1,
    Route9,
}

impl RoadSegment {
    pub fn base_speed(&self) -> f32 {
        match self {
            RoadSegment::GspExit83_90 => 45.0,
            RoadSegment::GspExit117_120 => 55.0,
            RoadSegment::GspExit29_30 => 50.0,
            RoadSegment::TurnpikeI95 => 60.0,
            RoadSegment::Route1 => 35.0,
            RoadSegment::Route9 => 40.0,
        }
    }
}

/// Feature vector for route time prediction
#[derive(Debug, Clone)]
pub struct RouteFeatures {
    pub stops_count: u32,
    pub total_distance_miles: f32,
    pub hour: u32,
    pub day_of_week: u32,
    pub season: Season,
    pub weather: WeatherCondition,
}

impl RouteFeatures {
    pub fn from_route(
        stops_count: u32,
        total_distance_miles: f32,
        timestamp: DateTime<Utc>,
        weather: WeatherCondition,
    ) -> Self {
        Self {
            stops_count,
            total_distance_miles,
            hour: timestamp.hour(),
            day_of_week: timestamp.weekday().num_days_from_monday(),
            season: Season::from_month(timestamp.month()),
            weather,
        }
    }
}

/// Helper to encode cyclical features
pub fn encode_cyclical(value: f32, max_value: f32) -> (f32, f32) {
    let angle = 2.0 * std::f32::consts::PI * value / max_value;
    (angle.sin(), angle.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_season_from_month() {
        assert_eq!(Season::from_month(1), Season::Winter);
        assert_eq!(Season::from_month(4), Season::Spring);
        assert_eq!(Season::from_month(7), Season::Summer);
        assert_eq!(Season::from_month(10), Season::Fall);
    }

    #[test]
    fn test_cyclical_encoding() {
        let (sin, cos) = encode_cyclical(0.0, 24.0);
        assert!((sin - 0.0).abs() < 0.001);
        assert!((cos - 1.0).abs() < 0.001);

        let (sin, cos) = encode_cyclical(6.0, 24.0);
        assert!((sin - 1.0).abs() < 0.001);
        assert!((cos - 0.0).abs() < 0.001);
    }
}
