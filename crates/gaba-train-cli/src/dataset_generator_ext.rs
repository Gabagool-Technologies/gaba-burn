use std::fs;
use std::path::Path;
use anyhow::Result;

pub fn generate_all(output_dir: &Path, traffic_samples: usize, route_samples: usize) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    
    let traffic_path = output_dir.join("traffic_speeds.csv");
    let route_path = output_dir.join("route_completions.csv");
    
    super::dataset_generator::generate_urban_traffic_dataset(&traffic_path, traffic_samples)?;
    super::dataset_generator::generate_mixed_terrain_dataset(&route_path, route_samples)?;
    
    println!("Generated {} traffic samples", traffic_samples);
    println!("Generated {} route samples", route_samples);
    
    Ok(())
}
