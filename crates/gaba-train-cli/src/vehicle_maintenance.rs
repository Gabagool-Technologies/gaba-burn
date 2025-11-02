//! Vehicle Maintenance Prediction
//! Predictive maintenance for fleet vehicles

use anyhow::Result;

#[allow(dead_code)]
pub struct MaintenancePredictor {
    weights: Vec<f32>,
    threshold: f32,
}

impl MaintenancePredictor {
    pub fn new(input_size: usize) -> Self {
        Self {
            weights: vec![0.01; input_size],
            threshold: 0.7,
        }
    }

    pub fn predict_maintenance_risk(&self, vehicle_data: &VehicleData) -> f32 {
        let features = vehicle_data.to_features();
        
        let score: f32 = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();
        
        1.0 / (1.0 + (-score).exp())
    }

    pub fn needs_maintenance(&self, vehicle_data: &VehicleData) -> bool {
        self.predict_maintenance_risk(vehicle_data) > self.threshold
    }

    pub fn train(&mut self, data: &[(VehicleData, bool)], epochs: usize) -> Result<()> {
        let lr = 0.01;
        
        for epoch in 0..epochs {
            let mut correct = 0;
            
            for (vehicle, needs_maintenance) in data {
                let prediction = self.predict_maintenance_risk(vehicle);
                let target = if *needs_maintenance { 1.0 } else { 0.0 };
                let error = prediction - target;
                
                let features = vehicle.to_features();
                for (i, feature) in features.iter().enumerate() {
                    self.weights[i] -= lr * error * feature;
                }
                
                if (prediction > self.threshold) == *needs_maintenance {
                    correct += 1;
                }
            }
            
            if epoch % 10 == 0 {
                let accuracy = correct as f32 / data.len() as f32;
                println!("Epoch {}: accuracy={:.2}%", epoch, accuracy * 100.0);
            }
        }
        
        Ok(())
    }
}

#[allow(dead_code)]
pub struct VehicleData {
    pub mileage: f32,
    pub engine_hours: f32,
    pub avg_speed: f32,
    pub harsh_braking_count: u32,
    pub rapid_acceleration_count: u32,
    pub idle_time_hours: f32,
    pub days_since_last_service: u32,
    pub oil_life_percent: f32,
    pub tire_pressure_avg: f32,
    pub battery_voltage: f32,
}

impl VehicleData {
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            self.mileage / 100000.0,
            self.engine_hours / 5000.0,
            self.avg_speed / 100.0,
            self.harsh_braking_count as f32 / 100.0,
            self.rapid_acceleration_count as f32 / 100.0,
            self.idle_time_hours / 500.0,
            self.days_since_last_service as f32 / 180.0,
            (100.0 - self.oil_life_percent) / 100.0,
            (35.0 - self.tire_pressure_avg) / 35.0,
            (14.0 - self.battery_voltage) / 14.0,
        ]
    }
}

#[allow(dead_code)]
pub fn generate_maintenance_data(samples: usize) -> Vec<(VehicleData, bool)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    
    for _ in 0..samples {
        let mileage = rng.gen_range(10000.0..150000.0);
        let days_since_service = rng.gen_range(0..365);
        let oil_life = rng.gen_range(0.0..100.0);
        
        let needs_maintenance = 
            mileage > 100000.0 ||
            days_since_service > 180 ||
            oil_life < 20.0;
        
        let vehicle = VehicleData {
            mileage,
            engine_hours: mileage / 50.0,
            avg_speed: rng.gen_range(30.0..70.0),
            harsh_braking_count: rng.gen_range(0..50),
            rapid_acceleration_count: rng.gen_range(0..50),
            idle_time_hours: rng.gen_range(50.0..500.0),
            days_since_last_service: days_since_service,
            oil_life_percent: oil_life,
            tire_pressure_avg: rng.gen_range(28.0..35.0),
            battery_voltage: rng.gen_range(12.0..14.5),
        };
        
        data.push((vehicle, needs_maintenance));
    }
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maintenance_predictor() {
        let mut predictor = MaintenancePredictor::new(10);
        let data = generate_maintenance_data(100);
        
        predictor.train(&data, 20).unwrap();
        
        let test_vehicle = VehicleData {
            mileage: 120000.0,
            engine_hours: 2400.0,
            avg_speed: 45.0,
            harsh_braking_count: 30,
            rapid_acceleration_count: 25,
            idle_time_hours: 200.0,
            days_since_last_service: 200,
            oil_life_percent: 15.0,
            tire_pressure_avg: 30.0,
            battery_voltage: 13.5,
        };
        
        let risk = predictor.predict_maintenance_risk(&test_vehicle);
        assert!(risk > 0.0 && risk < 1.0);
    }
}
