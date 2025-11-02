#[allow(dead_code)]
/// Advanced Dataset Generator for ML Training
/// Creates diverse synthetic datasets based on real-world patterns

use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[allow(dead_code)]
pub struct DatasetConfig {
    pub name: String,
    pub samples: usize,
    pub noise_level: f32,
    pub complexity: Complexity,
}

#[allow(dead_code)]
pub enum Complexity {
    Linear,
    Polynomial,
    Exponential,
    Seasonal,
    Mixed,
}

#[allow(dead_code)]
pub fn generate_urban_traffic_dataset(output: &Path, samples: usize) -> anyhow::Result<()> {
    let mut file = File::create(output)?;
    writeln!(file, "timestamp,road_segment_id,speed_mph,day_of_week,hour,season,weather_condition,is_holiday,base_speed")?;
    
    let mut rng = rand::thread_rng();
    let seasons = ["spring", "summer", "fall", "winter"];
    let weather = ["clear", "rain", "snow", "fog"];
    
    for i in 0..samples {
        let hour = rng.gen_range(0..24);
        let day = rng.gen_range(0..7);
        let season = seasons[rng.gen_range(0..4)];
        let weather_cond = weather[rng.gen_range(0..4)];
        let is_holiday = rng.gen::<f32>() < 0.05;
        
        let base_speed: f32 = 45.0;
        let rush_hour_penalty: f32 = if hour >= 7 && hour <= 9 || hour >= 17 && hour <= 19 {
            rng.gen_range(15.0..25.0)
        } else {
            0.0
        };
        
        let weather_penalty: f32 = match weather_cond {
            "rain" => rng.gen_range(5.0..10.0),
            "snow" => rng.gen_range(10.0..20.0),
            "fog" => rng.gen_range(3.0..8.0),
            _ => 0.0,
        };
        
        let speed = (base_speed - rush_hour_penalty - weather_penalty).max(5.0);
        
        writeln!(
            file,
            "2025-01-{:02}T{:02}:00:00,seg_{},{}. {},day_{},hour_{},{},{},{:.1}",
            (i % 30) + 1, i % 100, speed, i % 100, day, hour, season, weather_cond, is_holiday, base_speed
        )?;
    }
    
    Ok(())
}

#[allow(dead_code)]
pub fn generate_highway_traffic_dataset(output: &Path, samples: usize) -> anyhow::Result<()> {
    let mut file = File::create(output)?;
    writeln!(file, "timestamp,road_segment_id,speed_mph,day_of_week,hour,season,weather_condition,is_holiday,base_speed")?;
    
    let mut rng = rand::thread_rng();
    let seasons = ["spring", "summer", "fall", "winter"];
    let weather = ["clear", "rain", "snow", "fog"];
    
    for i in 0..samples {
        let hour = rng.gen_range(0..24);
        let day = rng.gen_range(0..7);
        let season = seasons[rng.gen_range(0..4)];
        let weather_cond = weather[rng.gen_range(0..4)];
        let is_holiday = rng.gen::<f32>() < 0.05;
        
        let base_speed = 65.0;
        let traffic_factor = if hour >= 6 && hour <= 20 { 0.9 } else { 1.0 };
        let weather_factor = match weather_cond {
            "rain" => 0.85,
            "snow" => 0.7,
            "fog" => 0.8,
            _ => 1.0,
        };
        
        let speed = base_speed * traffic_factor * weather_factor;
        
        writeln!(
            file,
            "2025-01-{:02}T{:02}:00:00,highway_{},{:.1},{},{},{},{},{},{:.1}",
            (i % 30) + 1, hour, i % 50, speed, day, hour, season, weather_cond, is_holiday, base_speed
        )?;
    }
    
    Ok(())
}

#[allow(dead_code)]
pub fn generate_rural_route_dataset(output: &Path, samples: usize) -> anyhow::Result<()> {
    let mut file = File::create(output)?;
    writeln!(file, "route_id,stops_count,total_distance_miles,predicted_time_minutes,actual_time_minutes,traffic_delay_minutes,start_time,weather,season,hour,day_of_week,is_holiday")?;
    
    let mut rng = rand::thread_rng();
    let weather = ["clear", "rain", "snow"];
    let seasons = ["spring", "summer", "fall", "winter"];
    
    for i in 0..samples {
        let stops = rng.gen_range(5..20);
        let distance = stops as f32 * rng.gen_range(2.0..5.0);
        let hour = rng.gen_range(6..18);
        let day = rng.gen_range(0..7);
        let season = seasons[rng.gen_range(0..4)];
        let weather_cond = weather[rng.gen_range(0..3)];
        let is_holiday = rng.gen::<f32>() < 0.03;
        
        let base_time = distance * 2.0;
        let stop_time = stops as f32 * rng.gen_range(3.0..6.0);
        let weather_delay = match weather_cond {
            "rain" => rng.gen_range(5.0..15.0),
            "snow" => rng.gen_range(15.0..30.0),
            _ => 0.0,
        };
        
        let predicted = (base_time + stop_time) as u32;
        let actual = (predicted as f32 + weather_delay) as u32;
        let delay = (weather_delay) as u32;
        
        writeln!(
            file,
            "rural_{},{},{:.1},{},{},{},{}:00,{},{},{},{},{}",
            i, stops, distance, predicted, actual, delay, hour, weather_cond, season, hour, day, is_holiday
        )?;
    }
    
    Ok(())
}

#[allow(dead_code)]
pub fn generate_dense_urban_route_dataset(output: &Path, samples: usize) -> anyhow::Result<()> {
    let mut file = File::create(output)?;
    writeln!(file, "route_id,stops_count,total_distance_miles,predicted_time_minutes,actual_time_minutes,traffic_delay_minutes,start_time,weather,season,hour,day_of_week,is_holiday")?;
    
    let mut rng = rand::thread_rng();
    let weather = ["clear", "rain", "snow", "fog"];
    let seasons = ["spring", "summer", "fall", "winter"];
    
    for i in 0..samples {
        let stops = rng.gen_range(15..40);
        let distance = stops as f32 * rng.gen_range(0.3..0.8);
        let hour = rng.gen_range(5..22);
        let day = rng.gen_range(0..7);
        let season = seasons[rng.gen_range(0..4)];
        let weather_cond = weather[rng.gen_range(0..4)];
        let is_holiday = rng.gen::<f32>() < 0.08;
        
        let base_time = distance * 3.5;
        let stop_time = stops as f32 * rng.gen_range(2.0..4.0);
        
        let rush_hour_delay = if hour >= 7 && hour <= 9 || hour >= 17 && hour <= 19 {
            rng.gen_range(20.0..40.0)
        } else {
            rng.gen_range(5.0..15.0)
        };
        
        let weather_delay = match weather_cond {
            "rain" => rng.gen_range(10.0..20.0),
            "snow" => rng.gen_range(25.0..45.0),
            "fog" => rng.gen_range(8.0..15.0),
            _ => 0.0,
        };
        
        let predicted = (base_time + stop_time) as u32;
        let actual = (predicted as f32 + rush_hour_delay + weather_delay) as u32;
        let delay = (rush_hour_delay + weather_delay) as u32;
        
        writeln!(
            file,
            "urban_{},{},{:.1},{},{},{},{}:00,{},{},{},{},{}",
            i, stops, distance, predicted, actual, delay, hour, weather_cond, season, hour, day, is_holiday
        )?;
    }
    
    Ok(())
}

#[allow(dead_code)]
pub fn generate_mixed_terrain_dataset(output: &Path, samples: usize) -> anyhow::Result<()> {
    let mut file = File::create(output)?;
    writeln!(file, "route_id,stops_count,total_distance_miles,predicted_time_minutes,actual_time_minutes,traffic_delay_minutes,start_time,weather,season,hour,day_of_week,is_holiday")?;
    
    let mut rng = rand::thread_rng();
    let weather = ["clear", "rain", "snow", "fog", "wind"];
    let seasons = ["spring", "summer", "fall", "winter"];
    let terrains = ["flat", "hilly", "mountain"];
    
    for i in 0..samples {
        let stops = rng.gen_range(8..25);
        let distance = stops as f32 * rng.gen_range(1.5..4.0);
        let hour = rng.gen_range(5..20);
        let day = rng.gen_range(0..7);
        let season = seasons[rng.gen_range(0..4)];
        let weather_cond = weather[rng.gen_range(0..5)];
        let terrain = terrains[rng.gen_range(0..3)];
        let is_holiday = rng.gen::<f32>() < 0.05;
        
        let terrain_factor = match terrain {
            "flat" => 1.0,
            "hilly" => 1.3,
            "mountain" => 1.6,
            _ => 1.0,
        };
        
        let base_time = distance * 2.2 * terrain_factor;
        let stop_time = stops as f32 * rng.gen_range(3.0..5.0);
        
        let weather_delay = match weather_cond {
            "rain" => rng.gen_range(8.0..18.0),
            "snow" => rng.gen_range(20.0..40.0),
            "fog" => rng.gen_range(10.0..20.0),
            "wind" => rng.gen_range(5.0..12.0),
            _ => 0.0,
        };
        
        let predicted = (base_time + stop_time) as u32;
        let actual = (predicted as f32 + weather_delay) as u32;
        let delay = weather_delay as u32;
        
        writeln!(
            file,
            "mixed_{},{},{:.1},{},{},{},{}:00,{},{},{},{},{}",
            i, stops, distance, predicted, actual, delay, hour, weather_cond, season, hour, day, is_holiday
        )?;
    }
    
    Ok(())
}
