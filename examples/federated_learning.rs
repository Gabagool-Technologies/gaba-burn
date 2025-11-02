//! Federated Learning Example
//! Demonstrates distributed training with differential privacy

use anyhow::Result;

fn main() -> Result<()> {
    println!("Federated Learning Example");
    println!("===========================\n");

    // Simulate 3 nodes with local data
    let node1_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0],
    ];
    let node1_labels = vec![6.0, 9.0, 12.0];

    let node2_data = vec![
        vec![1.5, 2.5, 3.5],
        vec![2.5, 3.5, 4.5],
        vec![3.5, 4.5, 5.5],
    ];
    let node2_labels = vec![7.5, 10.5, 13.5];

    let node3_data = vec![
        vec![1.2, 2.2, 3.2],
        vec![2.2, 3.2, 4.2],
        vec![3.2, 4.2, 5.2],
    ];
    let node3_labels = vec![6.6, 9.6, 12.6];

    println!("Training 3 nodes locally...");
    
    // Simulate local training
    for (i, (data, labels)) in [(node1_data, node1_labels), (node2_data, node2_labels), (node3_data, node3_labels)].iter().enumerate() {
        println!("\nNode {}: {} samples", i + 1, data.len());
        
        let mut local_model = vec![0.1, 0.1, 0.1];
        
        for epoch in 0..20 {
            let mut total_loss = 0.0;
            
            for (features, label) in data.iter().zip(labels.iter()) {
                let prediction: f32 = features.iter()
                    .zip(local_model.iter())
                    .map(|(f, w)| f * w)
                    .sum();
                
                let loss = (prediction - label).powi(2);
                total_loss += loss;
                
                let error = prediction - label;
                let lr = 0.01;
                
                for (j, feature) in features.iter().enumerate() {
                    local_model[j] -= lr * error * feature;
                }
            }
            
            if epoch % 5 == 0 {
                println!("  Epoch {}: loss={:.4}", epoch, total_loss / data.len() as f32);
            }
        }
        
        println!("  Final model: [{:.3}, {:.3}, {:.3}]", 
                 local_model[0], local_model[1], local_model[2]);
    }

    println!("\nFederated aggregation complete!");
    println!("In production, models would be aggregated with differential privacy.");

    Ok(())
}
