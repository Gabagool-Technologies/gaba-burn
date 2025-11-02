//! Vehicle Maintenance Prediction Example
//! Predicts when vehicles need maintenance

use anyhow::Result;

fn main() -> Result<()> {
    println!("Vehicle Maintenance Prediction Example");
    println!("=======================================\n");

    let mut model = vec![0.1; 10];
    let threshold = 0.7;

    let training_data = generate_maintenance_data(200);

    println!("Training on {} vehicles...\n", training_data.len());

    let lr = 0.01;
    for epoch in 0..100 {
        let mut correct = 0;

        for (features, needs_maintenance) in &training_data {
            let score: f32 = features.iter()
                .zip(model.iter())
                .map(|(f, w)| f * w)
                .sum();

            let prediction = 1.0 / (1.0 + (-score).exp());
            let target = if *needs_maintenance { 1.0 } else { 0.0 };
            let error = prediction - target;

            for (i, feature) in features.iter().enumerate() {
                model[i] -= lr * error * feature;
            }

            if (prediction > threshold) == *needs_maintenance {
                correct += 1;
            }
        }

        if epoch % 20 == 0 {
            let accuracy = correct as f32 / training_data.len() as f32;
            println!("Epoch {}: accuracy={:.1}%", epoch, accuracy * 100.0);
        }
    }

    println!("\nTesting on sample vehicles:\n");

    let test_vehicles = vec![
        ("High mileage truck", vec![0.9, 0.8, 0.5, 0.3, 0.2, 0.4, 0.9, 0.8, 0.1, 0.2]),
        ("New delivery van", vec![0.1, 0.1, 0.4, 0.05, 0.05, 0.1, 0.1, 0.05, 0.0, 0.0]),
        ("Overdue service", vec![0.5, 0.5, 0.6, 0.2, 0.2, 0.3, 0.95, 0.9, 0.3, 0.4]),
    ];

    for (desc, features) in test_vehicles {
        let score: f32 = features.iter()
            .zip(model.iter())
            .map(|(f, w)| f * w)
            .sum();

        let risk = 1.0 / (1.0 + (-score).exp());
        let needs_maintenance = risk > threshold;

        println!("  {}", desc);
        println!("    Risk score: {:.1}%", risk * 100.0);
        println!("    Recommendation: {}\n",
                 if needs_maintenance { "Schedule maintenance" } else { "OK" });
    }

    Ok(())
}

fn generate_maintenance_data(samples: usize) -> Vec<(Vec<f32>, bool)> {
    let mut data = Vec::new();

    for i in 0..samples {
        let mileage = ((i * 1000) as f32 / 100000.0).min(1.0);
        let engine_hours = mileage * 0.8;
        let days_since_service = ((i * 2) as f32 / 365.0).min(1.0);
        let oil_life = 1.0 - days_since_service;

        let needs_maintenance = mileage > 0.7 || days_since_service > 0.5 || oil_life < 0.2;

        let features = vec![
            mileage,
            engine_hours,
            0.5,
            (i % 10) as f32 / 100.0,
            (i % 8) as f32 / 100.0,
            (i % 20) as f32 / 500.0,
            days_since_service,
            1.0 - oil_life,
            0.1,
            0.1,
        ];

        data.push((features, needs_maintenance));
    }

    data
}
