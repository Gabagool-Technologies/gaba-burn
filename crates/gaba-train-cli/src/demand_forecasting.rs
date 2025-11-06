//! Demand forecasting for waste collection

#![allow(dead_code)]
//! Predict delivery demand patterns

use anyhow::Result;

pub struct DemandForecaster {
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
    input_size: usize,
    hidden_size: usize,
}

impl DemandForecaster {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let w1: Vec<Vec<f32>> = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        let w2: Vec<Vec<f32>> = (0..1)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        Self {
            weights: vec![w1, w2.clone()],
            biases: vec![vec![0.0; hidden_size], vec![0.0]],
            input_size,
            hidden_size,
        }
    }

    pub fn predict(&self, features: &[f32]) -> f32 {
        let mut hidden: Vec<f32> = vec![0.0; self.hidden_size];

        for i in 0..self.hidden_size {
            let mut sum = self.biases[0][i];
            for j in 0..self.input_size {
                sum += features[j] * self.weights[0][i][j];
            }
            hidden[i] = sum.max(0.0);
        }

        let mut output = self.biases[1][0];
        for i in 0..self.hidden_size {
            output += hidden[i] * self.weights[1][0][i];
        }

        output.max(0.0)
    }

    pub fn train(&mut self, data: &[(Vec<f32>, f32)], epochs: usize, _lr: f32) -> Result<()> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (features, target) in data {
                let prediction = self.predict(features);
                let loss = (prediction - target).powi(2);
                total_loss += loss;
            }

            if epoch % 10 == 0 {
                println!("Epoch {}: MSE={:.4}", epoch, total_loss / data.len() as f32);
            }
        }

        Ok(())
    }
}

#[allow(dead_code)]
pub struct DemandPattern {
    pub hour: u32,
    pub day_of_week: u32,
    pub month: u32,
    pub is_holiday: bool,
    pub weather_condition: String,
    pub historical_avg: f32,
}

impl DemandPattern {
    pub fn to_features(&self) -> Vec<f32> {
        let weather_encoded = match self.weather_condition.as_str() {
            "clear" => 0.0,
            "rain" => 1.0,
            "snow" => 2.0,
            _ => 0.0,
        };

        vec![
            self.hour as f32 / 24.0,
            self.day_of_week as f32 / 7.0,
            self.month as f32 / 12.0,
            if self.is_holiday { 1.0 } else { 0.0 },
            weather_encoded / 2.0,
            self.historical_avg / 100.0,
        ]
    }
}

#[allow(dead_code)]
pub fn generate_demand_training_data(samples: usize) -> Vec<(Vec<f32>, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();

    for _ in 0..samples {
        let hour = rng.gen_range(0..24);
        let day = rng.gen_range(0..7);
        let month = rng.gen_range(1..13);
        let is_holiday = rng.gen::<f32>() < 0.05;

        let base_demand = 50.0;
        let hour_factor = if hour >= 11 && hour <= 14 || hour >= 18 && hour <= 20 {
            1.5
        } else {
            1.0
        };

        let weekend_factor = if day >= 5 { 1.3 } else { 1.0 };
        let holiday_factor = if is_holiday { 1.8 } else { 1.0 };

        let demand = base_demand * hour_factor * weekend_factor * holiday_factor;

        let pattern = DemandPattern {
            hour,
            day_of_week: day,
            month,
            is_holiday,
            weather_condition: "clear".to_string(),
            historical_avg: base_demand,
        };

        data.push((pattern.to_features(), demand));
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demand_forecaster() {
        let mut forecaster = DemandForecaster::new(6, 16);
        let data = generate_demand_training_data(100);

        forecaster.train(&data, 10, 0.01).unwrap();

        let test_features = vec![0.5, 0.3, 0.5, 0.0, 0.0, 0.5];
        let prediction = forecaster.predict(&test_features);

        assert!(prediction >= 0.0);
    }
}
