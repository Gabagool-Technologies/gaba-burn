// Example: Route optimization for Famiglia Routes
// Demonstrates Rust-Zig integration for TSP solving

use gaba_native_kernels::route_optimizer::{build_distance_matrix, solve_tsp_optimized, GeoPoint};

fn main() {
    // Example: Waste management route in NJ/NY area
    let stops = vec![
        GeoPoint::new(40.7128, -74.0060), // NYC (depot)
        GeoPoint::new(40.7580, -73.9855), // Times Square
        GeoPoint::new(40.7489, -73.9680), // Queens
        GeoPoint::new(40.6782, -73.9442), // Brooklyn
        GeoPoint::new(40.7614, -73.9776), // Central Park
        GeoPoint::new(40.7061, -74.0087), // Jersey City
    ];

    println!("Famiglia Routes - Route Optimization Demo");
    println!("==========================================\n");
    println!("Optimizing route for {} stops...\n", stops.len());

    // Build distance matrix
    let distance_matrix = build_distance_matrix(&stops);

    // Solve TSP starting from depot (index 0)
    let start_idx = 0;
    let max_iterations = 100;

    let (route, total_distance) =
        solve_tsp_optimized(&distance_matrix, stops.len(), start_idx, max_iterations);

    // Display results
    println!("Optimized Route:");
    println!("----------------");
    for (i, &stop_idx) in route.iter().enumerate() {
        let stop = &stops[stop_idx];
        println!(
            "{}. Stop {} - ({:.4}, {:.4})",
            i + 1,
            stop_idx,
            stop.lat,
            stop.lng
        );
    }

    println!("\nTotal Distance: {:.2} km", total_distance);
    println!(
        "Estimated Fuel Savings: ${:.2}",
        calculate_fuel_savings(total_distance)
    );

    // Show comparison with naive route
    let naive_distance: f64 = (0..stops.len() - 1)
        .map(|i| distance_matrix[i * stops.len() + (i + 1)])
        .sum();
    let improvement = ((naive_distance - total_distance) / naive_distance) * 100.0;

    println!("\nNaive Route Distance: {:.2} km", naive_distance);
    println!("Improvement: {:.1}%", improvement);
}

fn calculate_fuel_savings(distance_km: f64) -> f64 {
    const MPG: f64 = 6.0; // Garbage truck fuel efficiency
    const FUEL_PRICE_PER_GALLON: f64 = 3.5;
    const KM_TO_MILES: f64 = 0.621371;

    let miles = distance_km * KM_TO_MILES;
    let gallons = miles / MPG;
    let baseline_savings = 0.15; // 15% typical savings from optimization

    gallons * FUEL_PRICE_PER_GALLON * baseline_savings
}
