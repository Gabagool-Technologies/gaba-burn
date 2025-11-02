#[allow(dead_code)]
use burn::tensor::{backend::Backend, Tensor};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ClimateZone {
    Tropical,
    Arid,
    Temperate,
    Continental,
    Polar,
    Mediterranean,
    Subtropical,
    Monsoon,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TerrainType {
    Flat,
    Hilly,
    Mountainous,
    Coastal,
    Desert,
    Urban,
    Rural,
    Mixed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RoadQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Unpaved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRouteFeatures {
    pub distance_km: f32,
    pub num_stops: usize,
    pub avg_stop_duration_min: f32,
    pub traffic_density: f32,
    pub hour_of_day: f32,
    pub day_of_week: f32,
    pub is_weekend: bool,
    pub is_holiday: bool,
    
    pub weather_temp_c: f32,
    pub weather_precipitation_mm: f32,
    pub weather_wind_kmh: f32,
    pub weather_humidity_pct: f32,
    pub weather_visibility_km: f32,
    pub weather_snow_depth_cm: f32,
    
    pub road_type_highway_pct: f32,
    pub road_type_urban_pct: f32,
    pub road_type_rural_pct: f32,
    pub road_quality: RoadQuality,
    
    pub elevation_gain_m: f32,
    pub elevation_loss_m: f32,
    pub max_elevation_m: f32,
    pub min_elevation_m: f32,
    pub terrain_type: TerrainType,
    
    pub num_traffic_lights: usize,
    pub num_stop_signs: usize,
    pub num_roundabouts: usize,
    pub num_toll_booths: usize,
    pub num_railroad_crossings: usize,
    
    pub avg_speed_limit_kmh: f32,
    pub min_speed_limit_kmh: f32,
    pub max_speed_limit_kmh: f32,
    
    pub historical_avg_time_min: f32,
    pub historical_std_time_min: f32,
    pub historical_min_time_min: f32,
    pub historical_max_time_min: f32,
    
    pub population_density: f32,
    pub is_rush_hour: bool,
    pub is_school_zone: bool,
    pub construction_zones: usize,
    pub accident_risk_score: f32,
    pub route_complexity_score: f32,
    
    pub vehicle_type: VehicleType,
    pub cargo_weight_kg: f32,
    pub driver_experience_years: f32,
    pub fuel_efficiency_factor: f32,
    pub maintenance_status: f32,
    pub tire_condition: f32,
    pub vehicle_age_years: f32,
    
    pub climate_zone: ClimateZone,
    pub latitude: f32,
    pub longitude: f32,
    pub altitude_m: f32,
    
    pub time_of_year: f32,
    pub daylight_hours: f32,
    pub moon_phase: f32,
    
    pub border_crossings: usize,
    pub ferry_crossings: usize,
    pub bridge_crossings: usize,
    pub tunnel_crossings: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VehicleType {
    SmallVan,
    MediumTruck,
    LargeTruck,
    Motorcycle,
    ElectricVan,
    HybridTruck,
    RefrigeratedTruck,
    TankerTruck,
}

impl EnhancedRouteFeatures {
    pub fn to_tensor_vec(&self) -> Vec<f32> {
        vec![
            self.distance_km,
            self.num_stops as f32,
            self.avg_stop_duration_min,
            self.traffic_density,
            self.hour_of_day,
            self.day_of_week,
            if self.is_weekend { 1.0 } else { 0.0 },
            if self.is_holiday { 1.0 } else { 0.0 },
            
            self.weather_temp_c,
            self.weather_precipitation_mm,
            self.weather_wind_kmh,
            self.weather_humidity_pct,
            self.weather_visibility_km,
            self.weather_snow_depth_cm,
            
            self.road_type_highway_pct,
            self.road_type_urban_pct,
            self.road_type_rural_pct,
            match self.road_quality {
                RoadQuality::Excellent => 1.0,
                RoadQuality::Good => 0.8,
                RoadQuality::Fair => 0.6,
                RoadQuality::Poor => 0.4,
                RoadQuality::Unpaved => 0.2,
            },
            
            self.elevation_gain_m,
            self.elevation_loss_m,
            self.max_elevation_m,
            self.min_elevation_m,
            match self.terrain_type {
                TerrainType::Flat => 0.0,
                TerrainType::Hilly => 1.0,
                TerrainType::Mountainous => 2.0,
                TerrainType::Coastal => 3.0,
                TerrainType::Desert => 4.0,
                TerrainType::Urban => 5.0,
                TerrainType::Rural => 6.0,
                TerrainType::Mixed => 7.0,
            },
            
            self.num_traffic_lights as f32,
            self.num_stop_signs as f32,
            self.num_roundabouts as f32,
            self.num_toll_booths as f32,
            self.num_railroad_crossings as f32,
            
            self.avg_speed_limit_kmh,
            self.min_speed_limit_kmh,
            self.max_speed_limit_kmh,
            
            self.historical_avg_time_min,
            self.historical_std_time_min,
            self.historical_min_time_min,
            self.historical_max_time_min,
            
            self.population_density,
            if self.is_rush_hour { 1.0 } else { 0.0 },
            if self.is_school_zone { 1.0 } else { 0.0 },
            self.construction_zones as f32,
            self.accident_risk_score,
            self.route_complexity_score,
            
            match self.vehicle_type {
                VehicleType::SmallVan => 0.0,
                VehicleType::MediumTruck => 1.0,
                VehicleType::LargeTruck => 2.0,
                VehicleType::Motorcycle => 3.0,
                VehicleType::ElectricVan => 4.0,
                VehicleType::HybridTruck => 5.0,
                VehicleType::RefrigeratedTruck => 6.0,
                VehicleType::TankerTruck => 7.0,
            },
            self.cargo_weight_kg,
            self.driver_experience_years,
            self.fuel_efficiency_factor,
            self.maintenance_status,
            self.tire_condition,
            self.vehicle_age_years,
            
            match self.climate_zone {
                ClimateZone::Tropical => 0.0,
                ClimateZone::Arid => 1.0,
                ClimateZone::Temperate => 2.0,
                ClimateZone::Continental => 3.0,
                ClimateZone::Polar => 4.0,
                ClimateZone::Mediterranean => 5.0,
                ClimateZone::Subtropical => 6.0,
                ClimateZone::Monsoon => 7.0,
            },
            self.latitude,
            self.longitude,
            self.altitude_m,
            
            self.time_of_year,
            self.daylight_hours,
            self.moon_phase,
            
            self.border_crossings as f32,
            self.ferry_crossings as f32,
            self.bridge_crossings as f32,
            self.tunnel_crossings as f32,
        ]
    }
    
    pub fn generate_synthetic(climate: ClimateZone, terrain: TerrainType) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let (base_temp, base_precip, base_humidity) = match climate {
            ClimateZone::Tropical => (28.0, 15.0, 80.0),
            ClimateZone::Arid => (32.0, 2.0, 30.0),
            ClimateZone::Temperate => (15.0, 8.0, 60.0),
            ClimateZone::Continental => (10.0, 6.0, 55.0),
            ClimateZone::Polar => (-10.0, 3.0, 70.0),
            ClimateZone::Mediterranean => (20.0, 5.0, 50.0),
            ClimateZone::Subtropical => (25.0, 12.0, 75.0),
            ClimateZone::Monsoon => (27.0, 20.0, 85.0),
        };
        
        let (base_elevation, elevation_variance) = match terrain {
            TerrainType::Flat => (100.0, 50.0),
            TerrainType::Hilly => (300.0, 200.0),
            TerrainType::Mountainous => (1500.0, 800.0),
            TerrainType::Coastal => (50.0, 100.0),
            TerrainType::Desert => (400.0, 300.0),
            TerrainType::Urban => (150.0, 80.0),
            TerrainType::Rural => (250.0, 150.0),
            TerrainType::Mixed => (400.0, 400.0),
        };
        
        let road_quality = match terrain {
            TerrainType::Urban => RoadQuality::Excellent,
            TerrainType::Flat | TerrainType::Hilly => RoadQuality::Good,
            TerrainType::Rural | TerrainType::Coastal => RoadQuality::Fair,
            TerrainType::Mountainous | TerrainType::Desert => RoadQuality::Poor,
            TerrainType::Mixed => RoadQuality::Fair,
        };
        
        Self {
            distance_km: rng.gen_range(10.0..150.0),
            num_stops: rng.gen_range(2..20),
            avg_stop_duration_min: rng.gen_range(2.0..10.0),
            traffic_density: rng.gen_range(0.1..0.95),
            hour_of_day: rng.gen_range(0.0..24.0),
            day_of_week: rng.gen_range(0.0..7.0),
            is_weekend: rng.gen_bool(2.0 / 7.0),
            is_holiday: rng.gen_bool(0.05),
            
            weather_temp_c: base_temp + rng.gen_range(-15.0..15.0),
            weather_precipitation_mm: base_precip + rng.gen_range(-5.0..10.0).max(0.0),
            weather_wind_kmh: rng.gen_range(0.0..60.0),
            weather_humidity_pct: base_humidity + rng.gen_range(-20.0..20.0),
            weather_visibility_km: rng.gen_range(1.0..50.0),
            weather_snow_depth_cm: if base_temp < 0.0 { rng.gen_range(0.0..30.0) } else { 0.0 },
            
            road_type_highway_pct: rng.gen_range(0.0..0.7),
            road_type_urban_pct: rng.gen_range(0.1..0.8),
            road_type_rural_pct: rng.gen_range(0.0..0.5),
            road_quality,
            
            elevation_gain_m: rng.gen_range(0.0..elevation_variance),
            elevation_loss_m: rng.gen_range(0.0..elevation_variance),
            max_elevation_m: base_elevation + rng.gen_range(0.0..elevation_variance),
            min_elevation_m: base_elevation - rng.gen_range(0.0..elevation_variance / 2.0),
            terrain_type: terrain,
            
            num_traffic_lights: rng.gen_range(0..40),
            num_stop_signs: rng.gen_range(0..25),
            num_roundabouts: rng.gen_range(0..8),
            num_toll_booths: rng.gen_range(0..5),
            num_railroad_crossings: rng.gen_range(0..3),
            
            avg_speed_limit_kmh: rng.gen_range(30.0..110.0),
            min_speed_limit_kmh: rng.gen_range(20.0..50.0),
            max_speed_limit_kmh: rng.gen_range(80.0..130.0),
            
            historical_avg_time_min: rng.gen_range(15.0..180.0),
            historical_std_time_min: rng.gen_range(3.0..30.0),
            historical_min_time_min: rng.gen_range(10.0..60.0),
            historical_max_time_min: rng.gen_range(60.0..240.0),
            
            population_density: rng.gen_range(50.0..10000.0),
            is_rush_hour: rng.gen_bool(0.3),
            is_school_zone: rng.gen_bool(0.15),
            construction_zones: rng.gen_range(0..4),
            accident_risk_score: rng.gen_range(0.0..1.0),
            route_complexity_score: rng.gen_range(0.0..1.0),
            
            vehicle_type: match rng.gen_range(0..8) {
                0 => VehicleType::SmallVan,
                1 => VehicleType::MediumTruck,
                2 => VehicleType::LargeTruck,
                3 => VehicleType::Motorcycle,
                4 => VehicleType::ElectricVan,
                5 => VehicleType::HybridTruck,
                6 => VehicleType::RefrigeratedTruck,
                _ => VehicleType::TankerTruck,
            },
            cargo_weight_kg: rng.gen_range(0.0..8000.0),
            driver_experience_years: rng.gen_range(1.0..40.0),
            fuel_efficiency_factor: rng.gen_range(0.6..1.4),
            maintenance_status: rng.gen_range(0.4..1.0),
            tire_condition: rng.gen_range(0.5..1.0),
            vehicle_age_years: rng.gen_range(0.0..15.0),
            
            climate_zone: climate,
            latitude: rng.gen_range(-90.0..90.0),
            longitude: rng.gen_range(-180.0..180.0),
            altitude_m: base_elevation,
            
            time_of_year: rng.gen_range(0.0..365.0),
            daylight_hours: rng.gen_range(8.0..16.0),
            moon_phase: rng.gen_range(0.0..1.0),
            
            border_crossings: rng.gen_range(0..3),
            ferry_crossings: rng.gen_range(0..2),
            bridge_crossings: rng.gen_range(0..10),
            tunnel_crossings: rng.gen_range(0..5),
        }
    }
}

#[derive(Module, Debug, Serialize, Deserialize)]
pub struct UltraRouteModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    fc5: Linear<B>,
    fc6: Linear<B>,
}

impl<B: Backend> UltraRouteModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_size = 73;
        Self {
            fc1: LinearConfig::new(input_size, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),
            fc3: LinearConfig::new(256, 128).init(device),
            fc4: LinearConfig::new(128, 64).init(device),
            fc5: LinearConfig::new(64, 32).init(device),
            fc6: LinearConfig::new(32, 1).init(device),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = x.relu();
        let x = self.fc2.forward(x);
        let x = x.relu();
        let x = self.fc3.forward(x);
        let x = x.relu();
        let x = self.fc4.forward(x);
        let x = x.relu();
        let x = self.fc5.forward(x);
        let x = x.relu();
        self.fc6.forward(x)
    }
}

pub fn generate_ultra_route_data(num_samples: usize) -> Vec<(EnhancedRouteFeatures, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_samples);
    
    let climates = [
        ClimateZone::Tropical,
        ClimateZone::Arid,
        ClimateZone::Temperate,
        ClimateZone::Continental,
        ClimateZone::Polar,
        ClimateZone::Mediterranean,
        ClimateZone::Subtropical,
        ClimateZone::Monsoon,
    ];
    
    let terrains = [
        TerrainType::Flat,
        TerrainType::Hilly,
        TerrainType::Mountainous,
        TerrainType::Coastal,
        TerrainType::Desert,
        TerrainType::Urban,
        TerrainType::Rural,
        TerrainType::Mixed,
    ];
    
    for _ in 0..num_samples {
        let climate = climates[rng.gen_range(0..climates.len())];
        let terrain = terrains[rng.gen_range(0..terrains.len())];
        let features = EnhancedRouteFeatures::generate_synthetic(climate, terrain);
        
        let base_time = features.distance_km / features.avg_speed_limit_kmh * 60.0;
        let stop_time = features.num_stops as f32 * features.avg_stop_duration_min;
        let traffic_factor = 1.0 + features.traffic_density * 0.6;
        let weather_factor = 1.0 + (features.weather_precipitation_mm / 20.0) * 0.4;
        let elevation_factor = 1.0 + (features.elevation_gain_m / 1000.0) * 0.2;
        let terrain_factor = match terrain {
            TerrainType::Flat => 1.0,
            TerrainType::Hilly => 1.15,
            TerrainType::Mountainous => 1.35,
            TerrainType::Coastal => 1.05,
            TerrainType::Desert => 1.2,
            TerrainType::Urban => 1.25,
            TerrainType::Rural => 1.1,
            TerrainType::Mixed => 1.15,
        };
        let road_quality_factor = match features.road_quality {
            RoadQuality::Excellent => 1.0,
            RoadQuality::Good => 1.05,
            RoadQuality::Fair => 1.15,
            RoadQuality::Poor => 1.3,
            RoadQuality::Unpaved => 1.5,
        };
        
        let predicted_time = base_time * traffic_factor * weather_factor * 
                           elevation_factor * terrain_factor * road_quality_factor + stop_time;
        let noise = rng.gen_range(-8.0..8.0);
        
        data.push((features, predicted_time + noise));
    }
    
    data
}
