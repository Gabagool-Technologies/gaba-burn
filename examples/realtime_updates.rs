//! Real-time Model Updates Example
//! Demonstrates online learning with streaming data

use anyhow::Result;
use std::collections::VecDeque;

fn main() -> Result<()> {
    println!("Real-time Model Updates Example");
    println!("================================\n");

    let mut model = vec![0.1, 0.1, 0.1];
    let mut buffer: VecDeque<(Vec<f32>, f32)> = VecDeque::new();
    let buffer_size = 10;
    let lr = 0.01;

    println!("Streaming data samples...\n");

    for i in 0..50 {
        let features = vec![
            (i as f32) / 10.0,
            (i as f32 + 1.0) / 10.0,
            (i as f32 + 2.0) / 10.0,
        ];
        let label = features.iter().sum::<f32>();

        buffer.push_back((features.clone(), label));
        if buffer.len() > buffer_size {
            buffer.pop_front();
        }

        if i % 10 == 0 && !buffer.is_empty() {
            for (feat, lbl) in buffer.iter() {
                let prediction: f32 = feat.iter()
                    .zip(model.iter())
                    .map(|(f, w)| f * w)
                    .sum();

                let error = prediction - lbl;
                for (j, f) in feat.iter().enumerate() {
                    model[j] -= lr * error * f;
                }
            }

            println!("Sample {}: model=[{:.3}, {:.3}, {:.3}]",
                     i, model[0], model[1], model[2]);
        }
    }

    println!("\nFinal model after streaming updates:");
    println!("  [{:.3}, {:.3}, {:.3}]", model[0], model[1], model[2]);

    Ok(())
}
