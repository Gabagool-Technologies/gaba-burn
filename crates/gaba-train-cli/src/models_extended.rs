#[allow(dead_code)]
use burn::tensor::{backend::Backend, Tensor};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug, Serialize, Deserialize)]
pub struct GlobalRouteModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    fc5: Linear<B>,
}

impl<B: Backend> GlobalRouteModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(35, 256).init(device),
            fc2: LinearConfig::new(256, 128).init(device),
            fc3: LinearConfig::new(128, 64).init(device),
            fc4: LinearConfig::new(64, 32).init(device),
            fc5: LinearConfig::new(32, 1).init(device),
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
        self.fc5.forward(x)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRouteFeatures {
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
    pub road_type_highway_pct: f32,
    pub road_type_urban_pct: f32,
    pub road_type_rural_pct: f32,
    pub elevation_gain_m: f32,
    pub elevation_loss_m: f32,
    pub num_traffic_lights: usize,
    pub num_stop_signs: usize,
    pub num_roundabouts: usize,
    pub avg_speed_limit_kmh: f32,
    pub historical_avg_time_min: f32,
    pub historical_std_time_min: f32,
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
    pub region_type: RegionType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VehicleType {
    SmallVan,
    MediumTruck,
    LargeTruck,
    Motorcycle,
    ElectricVan,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RegionType {
    NorthAmerica,
    Europe,
    Asia,
    SouthAmerica,
    Africa,
    Oceania,
}

impl GlobalRouteFeatures {
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
            self.road_type_highway_pct,
            self.road_type_urban_pct,
            self.road_type_rural_pct,
            self.elevation_gain_m,
            self.elevation_loss_m,
            self.num_traffic_lights as f32,
            self.num_stop_signs as f32,
            self.num_roundabouts as f32,
            self.avg_speed_limit_kmh,
            self.historical_avg_time_min,
            self.historical_std_time_min,
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
            },
            self.cargo_weight_kg,
            self.driver_experience_years,
            self.fuel_efficiency_factor,
            self.maintenance_status,
            self.tire_condition,
            match self.region_type {
                RegionType::NorthAmerica => 0.0,
                RegionType::Europe => 1.0,
                RegionType::Asia => 2.0,
                RegionType::SouthAmerica => 3.0,
                RegionType::Africa => 4.0,
                RegionType::Oceania => 5.0,
            },
        ]
    }
    
    pub fn generate_synthetic(region: RegionType) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let (base_distance, base_traffic, base_temp) = match region {
            RegionType::NorthAmerica => (25.0, 0.6, 15.0),
            RegionType::Europe => (18.0, 0.7, 12.0),
            RegionType::Asia => (15.0, 0.85, 25.0),
            RegionType::SouthAmerica => (30.0, 0.55, 22.0),
            RegionType::Africa => (35.0, 0.45, 28.0),
            RegionType::Oceania => (28.0, 0.5, 20.0),
        };
        
        Self {
            distance_km: base_distance + rng.gen_range(-10.0..20.0),
            num_stops: rng.gen_range(3..15),
            avg_stop_duration_min: rng.gen_range(2.0..8.0),
            traffic_density: base_traffic + rng.gen_range(-0.2..0.2),
            hour_of_day: rng.gen_range(0.0..24.0),
            day_of_week: rng.gen_range(0.0..7.0),
            is_weekend: rng.gen_bool(2.0 / 7.0),
            is_holiday: rng.gen_bool(0.05),
            weather_temp_c: base_temp + rng.gen_range(-10.0..10.0),
            weather_precipitation_mm: rng.gen_range(0.0..20.0),
            weather_wind_kmh: rng.gen_range(0.0..40.0),
            road_type_highway_pct: rng.gen_range(0.0..0.6),
            road_type_urban_pct: rng.gen_range(0.2..0.8),
            road_type_rural_pct: rng.gen_range(0.0..0.4),
            elevation_gain_m: rng.gen_range(0.0..500.0),
            elevation_loss_m: rng.gen_range(0.0..500.0),
            num_traffic_lights: rng.gen_range(0..30),
            num_stop_signs: rng.gen_range(0..20),
            num_roundabouts: rng.gen_range(0..5),
            avg_speed_limit_kmh: rng.gen_range(30.0..90.0),
            historical_avg_time_min: rng.gen_range(20.0..120.0),
            historical_std_time_min: rng.gen_range(5.0..20.0),
            population_density: rng.gen_range(100.0..5000.0),
            is_rush_hour: rng.gen_bool(0.3),
            is_school_zone: rng.gen_bool(0.15),
            construction_zones: rng.gen_range(0..3),
            accident_risk_score: rng.gen_range(0.0..1.0),
            route_complexity_score: rng.gen_range(0.0..1.0),
            vehicle_type: match rng.gen_range(0..5) {
                0 => VehicleType::SmallVan,
                1 => VehicleType::MediumTruck,
                2 => VehicleType::LargeTruck,
                3 => VehicleType::Motorcycle,
                _ => VehicleType::ElectricVan,
            },
            cargo_weight_kg: rng.gen_range(0.0..5000.0),
            driver_experience_years: rng.gen_range(1.0..30.0),
            fuel_efficiency_factor: rng.gen_range(0.7..1.3),
            maintenance_status: rng.gen_range(0.5..1.0),
            tire_condition: rng.gen_range(0.6..1.0),
            region_type: region,
        }
    }
}

pub fn generate_global_route_data(num_samples: usize) -> Vec<(GlobalRouteFeatures, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_samples);
    
    let regions = [
        RegionType::NorthAmerica,
        RegionType::Europe,
        RegionType::Asia,
        RegionType::SouthAmerica,
        RegionType::Africa,
        RegionType::Oceania,
    ];
    
    for _ in 0..num_samples {
        let region = regions[rng.gen_range(0..regions.len())];
        let features = GlobalRouteFeatures::generate_synthetic(region);
        
        let base_time = features.distance_km / features.avg_speed_limit_kmh * 60.0;
        let stop_time = features.num_stops as f32 * features.avg_stop_duration_min;
        let traffic_factor = 1.0 + features.traffic_density * 0.5;
        let weather_factor = 1.0 + (features.weather_precipitation_mm / 20.0) * 0.3;
        let rush_hour_factor = if features.is_rush_hour { 1.3 } else { 1.0 };
        let vehicle_factor = match features.vehicle_type {
            VehicleType::SmallVan => 1.0,
            VehicleType::MediumTruck => 1.1,
            VehicleType::LargeTruck => 1.25,
            VehicleType::Motorcycle => 0.85,
            VehicleType::ElectricVan => 1.05,
        };
        
        let predicted_time = base_time * traffic_factor * weather_factor * rush_hour_factor * vehicle_factor + stop_time;
        let noise = rng.gen_range(-5.0..5.0);
        
        data.push((features, predicted_time + noise));
    }
    
    data
}
