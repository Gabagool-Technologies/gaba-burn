use anyhow::Result;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub fn convert_datasets(input_dir: &Path, output_dir: &Path) -> Result<()> {
    println!("Converting Python datasets to Rust format...");
    
    fs::create_dir_all(output_dir)?;
    
    // Convert traffic data
    if let Ok(traffic_path) = input_dir.join("training-data/traffic_speeds.csv").canonicalize() {
        convert_traffic_data(&traffic_path, &output_dir.join("traffic"))?;
    }
    
    // Convert route data
    if let Ok(route_path) = input_dir.join("training-data/route_completions.csv").canonicalize() {
        convert_route_data(&route_path, &output_dir.join("route"))?;
    }
    
    // Generate sensor data
    generate_sensor_data(&output_dir.join("sensor"), 50000)?;
    
    // Generate anomaly data
    generate_anomaly_data(&output_dir.join("anomaly"), 30000)?;
    
    println!("Dataset conversion complete!");
    Ok(())
}

fn convert_traffic_data(input: &Path, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Skip header
    lines.next();
    
    let output_path = output_dir.join("traffic_features.csv");
    let mut output = File::create(&output_path)?;
    
    // Write header
    writeln!(output, "hour_sin,hour_cos,dow_sin,dow_cos,base_speed,season_winter,season_spring,season_summer,season_fall,weather_clear,weather_rain,weather_heavy_rain,weather_snow,weather_heavy_snow,weather_fog,segment_GSP_Exit_83_90,segment_GSP_Exit_117_120,segment_GSP_Exit_29_30,segment_Turnpike_I95,segment_Route_1,segment_Route_9,speed_mph")?;
    
    let mut count = 0;
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        
        if fields.len() < 9 {
            continue;
        }
        
        let hour: f32 = fields[4].parse().unwrap_or(0.0);
        let dow: f32 = fields[3].parse().unwrap_or(0.0);
        let speed: f32 = fields[2].parse().unwrap_or(0.0);
        let base_speed: f32 = fields[8].parse().unwrap_or(55.0);
        let season = fields[5];
        let weather = fields[6];
        let segment = fields[1];
        
        // Cyclical encoding
        let hour_sin = (2.0 * std::f32::consts::PI * hour / 24.0).sin();
        let hour_cos = (2.0 * std::f32::consts::PI * hour / 24.0).cos();
        let dow_sin = (2.0 * std::f32::consts::PI * dow / 7.0).sin();
        let dow_cos = (2.0 * std::f32::consts::PI * dow / 7.0).cos();
        
        // One-hot season
        let (s_winter, s_spring, s_summer, s_fall) = match season {
            "winter" => (1, 0, 0, 0),
            "spring" => (0, 1, 0, 0),
            "summer" => (0, 0, 1, 0),
            _ => (0, 0, 0, 1),
        };
        
        // One-hot weather
        let (w_clear, w_rain, w_heavy_rain, w_snow, w_heavy_snow, w_fog) = match weather {
            "clear" => (1, 0, 0, 0, 0, 0),
            "rain" => (0, 1, 0, 0, 0, 0),
            "heavy_rain" => (0, 0, 1, 0, 0, 0),
            "snow" => (0, 0, 0, 1, 0, 0),
            "heavy_snow" => (0, 0, 0, 0, 1, 0),
            _ => (0, 0, 0, 0, 0, 1),
        };
        
        // One-hot segment
        let (seg1, seg2, seg3, seg4, seg5, seg6) = match segment {
            "GSP_Exit_83_90" => (1, 0, 0, 0, 0, 0),
            "GSP_Exit_117_120" => (0, 1, 0, 0, 0, 0),
            "GSP_Exit_29_30" => (0, 0, 1, 0, 0, 0),
            "Turnpike_I95" => (0, 0, 0, 1, 0, 0),
            "Route_1" => (0, 0, 0, 0, 1, 0),
            _ => (0, 0, 0, 0, 0, 1),
        };
        
        writeln!(output, "{:.4},{:.4},{:.4},{:.4},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
            hour_sin, hour_cos, dow_sin, dow_cos, base_speed,
            s_winter, s_spring, s_summer, s_fall,
            w_clear, w_rain, w_heavy_rain, w_snow, w_heavy_snow, w_fog,
            seg1, seg2, seg3, seg4, seg5, seg6,
            speed
        )?;
        
        count += 1;
    }
    
    println!("Converted traffic data: {} samples -> {}", count, output_path.display());
    Ok(())
}

fn convert_route_data(input: &Path, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    lines.next(); // Skip header
    
    let output_path = output_dir.join("route_features.csv");
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "stops_count,total_distance_miles,stops_per_mile,avg_stop_distance,hour_sin,hour_cos,dow_sin,dow_cos,season_winter,season_spring,season_summer,season_fall,weather_clear,weather_rain,weather_snow,actual_time_minutes")?;
    
    let mut count = 0;
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        
        if fields.len() < 11 {
            continue;
        }
        
        let stops: f32 = fields[1].parse().unwrap_or(0.0);
        let distance: f32 = fields[2].parse().unwrap_or(0.0);
        let actual_time: f32 = fields[4].parse().unwrap_or(0.0);
        let hour: f32 = fields[8].parse().unwrap_or(0.0);
        let dow: f32 = fields[9].parse().unwrap_or(0.0);
        let season = fields[7];
        let weather = fields[6];
        
        let stops_per_mile = if distance > 0.0 { stops / distance } else { 0.0 };
        let avg_stop_distance = if stops > 0.0 { distance / stops } else { 0.0 };
        
        let hour_sin = (2.0 * std::f32::consts::PI * hour / 24.0).sin();
        let hour_cos = (2.0 * std::f32::consts::PI * hour / 24.0).cos();
        let dow_sin = (2.0 * std::f32::consts::PI * dow / 7.0).sin();
        let dow_cos = (2.0 * std::f32::consts::PI * dow / 7.0).cos();
        
        let (s_winter, s_spring, s_summer, s_fall) = match season {
            "winter" => (1, 0, 0, 0),
            "spring" => (0, 1, 0, 0),
            "summer" => (0, 0, 1, 0),
            _ => (0, 0, 0, 1),
        };
        
        let (w_clear, w_rain, w_snow) = match weather {
            "clear" => (1, 0, 0),
            "rain" => (0, 1, 0),
            _ => (0, 0, 1),
        };
        
        writeln!(output, "{:.0},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{:.0}",
            stops, distance, stops_per_mile, avg_stop_distance,
            hour_sin, hour_cos, dow_sin, dow_cos,
            s_winter, s_spring, s_summer, s_fall,
            w_clear, w_rain, w_snow,
            actual_time
        )?;
        
        count += 1;
    }
    
    println!("Converted route data: {} samples -> {}", count, output_path.display());
    Ok(())
}

fn generate_sensor_data(output_dir: &Path, num_samples: usize) -> Result<()> {
    use rand::Rng;
    
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("sensor_features.csv");
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "temperature_f,vibration_g,pressure_psi,rpm,runtime_hours,failure_probability")?;
    
    let mut rng = rand::thread_rng();
    
    for _ in 0..num_samples {
        let temperature = rng.gen_range(40.0..110.0);
        let vibration = rng.gen_range(0.0..10.0);
        let pressure = rng.gen_range(70.0..130.0);
        let rpm = rng.gen_range(1000.0..2000.0);
        let runtime_hours = rng.gen_range(0..10000);
        
        let mut failure_score = 0.0;
        if temperature > 90.0 {
            failure_score += (temperature - 90.0) * 0.01;
        }
        if vibration > 5.0 {
            failure_score += (vibration - 5.0) * 0.05;
        }
        if pressure < 80.0 || pressure > 120.0 {
            failure_score += ((pressure - 100.0) as f32).abs() * 0.005;
        }
        
        failure_score = failure_score.max(0.0).min(1.0);
        
        writeln!(output, "{:.2},{:.3},{:.2},{:.0},{},{:.4}",
            temperature, vibration, pressure, rpm, runtime_hours, failure_score
        )?;
    }
    
    println!("Generated sensor data: {} samples -> {}", num_samples, output_path.display());
    Ok(())
}

fn generate_anomaly_data(output_dir: &Path, num_samples: usize) -> Result<()> {
    use rand::Rng;
    
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("anomaly_features.csv");
    let mut output = File::create(&output_path)?;
    
    writeln!(output, "metric1,metric2,metric3,metric4,metric5,is_anomaly")?;
    
    let mut rng = rand::thread_rng();
    
    for _ in 0..num_samples {
        let is_anomaly = rng.gen::<f32>() > 0.9;
        
        let (m1, m2, m3) = if is_anomaly {
            (
                rng.gen_range(50.0..150.0),
                rng.gen_range(20.0..80.0),
                rng.gen_range(10.0..30.0),
            )
        } else {
            (
                rng.gen_range(40.0..60.0),
                rng.gen_range(20.0..30.0),
                rng.gen_range(3.0..7.0),
            )
        };
        
        let m4 = rng.gen_range(-2.0..2.0);
        let m5 = rng.gen_range(-1.0..1.0);
        
        writeln!(output, "{:.2},{:.2},{:.2},{:.3},{:.3},{}",
            m1, m2, m3, m4, m5, if is_anomaly { 1 } else { 0 }
        )?;
    }
    
    println!("Generated anomaly data: {} samples -> {}", num_samples, output_path.display());
    Ok(())
}
