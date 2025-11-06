//! Demand Forecasting Example
//! Predicts delivery demand based on time and conditions

use anyhow::Result;

fn main() -> Result<()> {
    println!("Demand Forecasting Example");
    println!("==========================\n");

    let mut model = vec![vec![0.1; 6]; 16];
    let mut output_weights = vec![0.1; 16];

    let training_data = generate_demand_data(100);

    println!("Training on {} samples...\n", training_data.len());

    for epoch in 0..50 {
        let mut total_loss = 0.0;

        for (features, target) in &training_data {
            let mut hidden = vec![0.0; 16];
            
            for i in 0..16 {
                let mut sum = 0.0;
                for j in 0..6 {
                    sum += features[j] * model[i][j];
                }
                hidden[i] = sum.max(0.0);
            }

            let prediction: f32 = hidden.iter()
                .zip(output_weights.iter())
                .map(|(h, w)| h * w)
                .sum();

            let loss = (prediction - target).powi(2);
            total_loss += loss;
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: MSE={:.4}", epoch, total_loss / training_data.len() as f32);
        }
    }

    println!("\nTesting predictions:");
    let test_cases = vec![
        ("Weekday lunch rush", vec![0.5, 0.3, 0.5, 0.0, 0.0, 0.5]),
        ("Weekend evening", vec![0.8, 0.9, 0.5, 0.0, 0.0, 0.6]),
        ("Holiday morning", vec![0.3, 0.1, 0.5, 1.0, 0.0, 0.5]),
    ];

    for (desc, features) in test_cases {
        let mut hidden = vec![0.0; 16];
        for i in 0..16 {
            let mut sum = 0.0;
            for j in 0..6 {
                sum += features[j] * model[i][j];
            }
            hidden[i] = sum.max(0.0);
        }

        let prediction: f32 = hidden.iter()
            .zip(output_weights.iter())
            .map(|(h, w)| h * w)
            .sum::<f32>()
            .max(0.0);

        println!("  {}: {:.1} deliveries", desc, prediction);
    }

    Ok(())
}

fn generate_demand_data(samples: usize) -> Vec<(Vec<f32>, f32)> {
    use std::f32::consts::PI;
    let mut data = Vec::new();

    for i in 0..samples {
        let hour = (i % 24) as f32 / 24.0;
        let day = ((i / 24) % 7) as f32 / 7.0;
        let month = ((i / 168) % 12) as f32 / 12.0;
        let is_holiday = if i % 50 == 0 { 1.0 } else { 0.0 };

        let base_demand = 50.0;
        let hour_factor = 1.0 + 0.5 * (2.0 * PI * hour).sin();
        let weekend_factor = if day > 0.7 { 1.3 } else { 1.0 };
        let holiday_factor = 1.0 + is_holiday * 0.8;

        let demand = base_demand * hour_factor * weekend_factor * holiday_factor;

        data.push((
            vec![hour, day, month, is_holiday, 0.0, base_demand / 100.0],
            demand,
        ));
    }

    data
}
