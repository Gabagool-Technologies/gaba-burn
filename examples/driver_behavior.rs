//! Driver Behavior Analysis Example
//! Analyzes and scores driver performance

use anyhow::Result;

fn main() -> Result<()> {
    println!("Driver Behavior Analysis Example");
    println!("=================================\n");

    let weights = vec![-0.5, -0.4, -0.3, 0.2, 0.3, -0.2, 0.4, -0.1];

    let drivers = vec![
        ("Excellent Driver", vec![0.1, 0.1, 0.0, 0.9, 0.85, 0.05, 0.98, 0.0]),
        ("Average Driver", vec![0.3, 0.3, 0.2, 0.6, 0.65, 0.15, 0.85, 0.0]),
        ("Needs Improvement", vec![0.7, 0.6, 0.5, 0.3, 0.45, 0.35, 0.65, 0.1]),
        ("High Risk", vec![0.9, 0.8, 0.8, 0.2, 0.35, 0.45, 0.55, 0.2]),
    ];

    println!("Analyzing {} drivers:\n", drivers.len());

    for (name, features) in drivers {
        let raw_score: f32 = features.iter()
            .zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        let normalized_score = ((raw_score + 5.0) / 10.0 * 100.0).clamp(0.0, 100.0);

        let rating = match normalized_score {
            s if s >= 90.0 => "Excellent",
            s if s >= 75.0 => "Good",
            s if s >= 60.0 => "Average",
            s if s >= 40.0 => "Below Average",
            _ => "Poor",
        };

        let safety_score = calculate_safety(&features);
        let efficiency_score = calculate_efficiency(&features);
        let reliability_score = features[6] * 100.0;

        println!("Driver: {}", name);
        println!("  Overall Score: {:.1}/100 ({})", normalized_score, rating);
        println!("  Safety: {:.1}/100", safety_score);
        println!("  Efficiency: {:.1}/100", efficiency_score);
        println!("  Reliability: {:.1}/100", reliability_score);

        let recommendations = generate_recommendations(&features);
        if !recommendations.is_empty() {
            println!("  Recommendations:");
            for rec in recommendations {
                println!("    - {}", rec);
            }
        }
        println!();
    }

    Ok(())
}

fn calculate_safety(features: &[f32]) -> f32 {
    let base = 100.0;
    let penalties = features[0] * 20.0 + features[1] * 15.0 + features[2] * 50.0 + features[7] * 200.0;
    (base - penalties).max(0.0)
}

fn calculate_efficiency(features: &[f32]) -> f32 {
    let fuel_score = features[4] * 100.0;
    let idle_penalty = features[5] * 50.0;
    (fuel_score - idle_penalty).max(0.0)
}

fn generate_recommendations(features: &[f32]) -> Vec<String> {
    let mut recs = Vec::new();

    if features[0] > 0.5 {
        recs.push("Reduce harsh braking".to_string());
    }
    if features[1] > 0.5 {
        recs.push("Smooth acceleration".to_string());
    }
    if features[2] > 0.3 {
        recs.push("Maintain speed limits".to_string());
    }
    if features[5] > 0.2 {
        recs.push("Reduce idle time".to_string());
    }
    if features[4] < 0.6 {
        recs.push("Improve fuel efficiency".to_string());
    }
    if features[6] < 0.9 {
        recs.push("Better time management".to_string());
    }

    recs
}
