// Rust FFI bindings for Zig route optimization kernels
// Pure C ABI interface

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GeoPoint {
    pub lat: f64,
    pub lng: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lng: f64) -> Self {
        Self { lat, lng }
    }
}

#[cfg(feature = "zig")]
extern "C" {
    fn haversine_distance(p1: *const GeoPoint, p2: *const GeoPoint) -> f64;
    fn route_total_distance(points: *const GeoPoint, n: usize) -> f64;
    fn distance_matrix(points: *const GeoPoint, n: usize, out_matrix: *mut f64);
    fn fast_distance_approx(p1: *const GeoPoint, p2: *const GeoPoint) -> f64;
    fn route_total_distance_fast(points: *const GeoPoint, n: usize) -> f64;
    
    fn tsp_nearest_neighbor(
        distance_matrix: *const f64,
        n: usize,
        start_idx: usize,
        out_route: *mut usize,
    ) -> f64;
    
    fn tsp_two_opt(
        distance_matrix: *const f64,
        n: usize,
        route: *mut usize,
        max_iterations: usize,
    ) -> f64;
    
    fn tsp_solve_optimized(
        distance_matrix: *const f64,
        n: usize,
        start_idx: usize,
        out_route: *mut usize,
        max_iterations: usize,
    ) -> f64;
}

/// Calculate Haversine distance between two geographic points
pub fn calculate_distance(p1: &GeoPoint, p2: &GeoPoint) -> f64 {
    #[cfg(feature = "zig")]
    unsafe {
        haversine_distance(p1 as *const GeoPoint, p2 as *const GeoPoint)
    }
    
    #[cfg(not(feature = "zig"))]
    haversine_distance_rust(p1, p2)
}

/// Calculate total route distance
pub fn calculate_route_distance(points: &[GeoPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    
    #[cfg(feature = "zig")]
    unsafe {
        route_total_distance(points.as_ptr(), points.len())
    }
    
    #[cfg(not(feature = "zig"))]
    {
        let mut total = 0.0;
        for i in 0..points.len() - 1 {
            total += haversine_distance_rust(&points[i], &points[i + 1]);
        }
        total
    }
}

/// Build distance matrix for all point pairs
pub fn build_distance_matrix(points: &[GeoPoint]) -> Vec<f64> {
    let n = points.len();
    let mut matrix = vec![0.0; n * n];
    
    #[cfg(feature = "zig")]
    unsafe {
        distance_matrix(points.as_ptr(), n, matrix.as_mut_ptr());
    }
    
    #[cfg(not(feature = "zig"))]
    {
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[i * n + j] = haversine_distance_rust(&points[i], &points[j]);
                }
            }
        }
    }
    
    matrix
}

/// Solve TSP using nearest neighbor heuristic
pub fn solve_tsp_nearest_neighbor(
    distance_matrix: &[f64],
    n: usize,
    start_idx: usize,
) -> (Vec<usize>, f64) {
    let mut route = vec![0; n];
    
    #[cfg(feature = "zig")]
    let distance = unsafe {
        tsp_nearest_neighbor(
            distance_matrix.as_ptr(),
            n,
            start_idx,
            route.as_mut_ptr(),
        )
    };
    
    #[cfg(not(feature = "zig"))]
    let distance = tsp_nearest_neighbor_rust(distance_matrix, n, start_idx, &mut route);
    
    (route, distance)
}

/// Optimize route using 2-opt local search
pub fn optimize_route_2opt(
    distance_matrix: &[f64],
    route: &mut [usize],
    max_iterations: usize,
) -> f64 {
    #[cfg(feature = "zig")]
    {
        let n = route.len();
        unsafe {
            tsp_two_opt(distance_matrix.as_ptr(), n, route.as_mut_ptr(), max_iterations)
        }
    }
    
    #[cfg(not(feature = "zig"))]
    tsp_two_opt_rust(distance_matrix, route, max_iterations)
}

/// Complete TSP solver: nearest neighbor + 2-opt
pub fn solve_tsp_optimized(
    distance_matrix: &[f64],
    n: usize,
    start_idx: usize,
    max_iterations: usize,
) -> (Vec<usize>, f64) {
    let mut route = vec![0; n];
    
    #[cfg(feature = "zig")]
    let distance = unsafe {
        tsp_solve_optimized(
            distance_matrix.as_ptr(),
            n,
            start_idx,
            route.as_mut_ptr(),
            max_iterations,
        )
    };
    
    #[cfg(not(feature = "zig"))]
    let distance = {
        tsp_nearest_neighbor_rust(distance_matrix, n, start_idx, &mut route);
        tsp_two_opt_rust(distance_matrix, &mut route, max_iterations)
    };
    
    (route, distance)
}

// Rust fallback implementations

fn haversine_distance_rust(p1: &GeoPoint, p2: &GeoPoint) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
    
    let lat1_rad = p1.lat * DEG_TO_RAD;
    let lat2_rad = p2.lat * DEG_TO_RAD;
    let dlat = (p2.lat - p1.lat) * DEG_TO_RAD;
    let dlng = (p2.lng - p1.lng) * DEG_TO_RAD;
    
    let a = (dlat / 2.0).sin().powi(2) +
        lat1_rad.cos() * lat2_rad.cos() *
        (dlng / 2.0).sin().powi(2);
    
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    
    EARTH_RADIUS_KM * c
}

#[cfg(not(feature = "zig"))]
fn tsp_nearest_neighbor_rust(
    distance_matrix: &[f64],
    n: usize,
    start_idx: usize,
    out_route: &mut [usize],
) -> f64 {
    if n == 0 {
        return 0.0;
    }
    
    let mut visited = vec![false; n];
    let mut current = start_idx;
    let mut route_len = 0;
    let mut total_distance = 0.0;
    
    out_route[route_len] = current;
    route_len += 1;
    visited[current] = true;
    
    while route_len < n {
        let mut nearest = 0;
        let mut min_dist = f64::INFINITY;
        
        for i in 0..n {
            if !visited[i] {
                let dist = distance_matrix[current * n + i];
                if dist < min_dist {
                    min_dist = dist;
                    nearest = i;
                }
            }
        }
        
        out_route[route_len] = nearest;
        route_len += 1;
        visited[nearest] = true;
        total_distance += min_dist;
        current = nearest;
    }
    
    // Return to start
    total_distance += distance_matrix[current * n + start_idx];
    
    total_distance
}

#[cfg(not(feature = "zig"))]
fn tsp_two_opt_rust(
    distance_matrix: &[f64],
    route: &mut [usize],
    max_iterations: usize,
) -> f64 {
    let n = route.len();
    if n < 4 {
        return calculate_route_distance_internal(distance_matrix, route);
    }
    
    let mut improved = true;
    let mut iterations = 0;
    
    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;
        
        for i in 0..n - 1 {
            for j in i + 2..n {
                let delta = calculate_2opt_delta_rust(distance_matrix, n, route, i, j);
                if delta < -0.001 {
                    reverse_segment_rust(route, i + 1, j);
                    improved = true;
                }
            }
        }
    }
    
    calculate_route_distance_internal(distance_matrix, route)
}

#[cfg(not(feature = "zig"))]
fn calculate_2opt_delta_rust(
    distance_matrix: &[f64],
    n: usize,
    route: &[usize],
    i: usize,
    j: usize,
) -> f64 {
    let a = route[i];
    let b = route[i + 1];
    let c = route[j];
    let d = if j + 1 < n { route[j + 1] } else { route[0] };
    
    let old_dist = distance_matrix[a * n + b] + distance_matrix[c * n + d];
    let new_dist = distance_matrix[a * n + c] + distance_matrix[b * n + d];
    
    new_dist - old_dist
}

#[cfg(not(feature = "zig"))]
fn reverse_segment_rust(route: &mut [usize], i: usize, j: usize) {
    let mut left = i;
    let mut right = j;
    
    while left < right {
        route.swap(left, right);
        left += 1;
        right -= 1;
    }
}

#[cfg(not(feature = "zig"))]
fn calculate_route_distance_internal(distance_matrix: &[f64], route: &[usize]) -> f64 {
    let n = route.len();
    if n < 2 {
        return 0.0;
    }
    
    let mut total = 0.0;
    for i in 0..n - 1 {
        let from = route[i];
        let to = route[i + 1];
        total += distance_matrix[from * n + to];
    }
    
    // Return to start
    let last = route[n - 1];
    let first = route[0];
    total += distance_matrix[last * n + first];
    
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_haversine_distance() {
        // New York to Philadelphia
        let ny = GeoPoint::new(40.7128, -74.0060);
        let philly = GeoPoint::new(39.9526, -75.1652);
        
        let dist = calculate_distance(&ny, &philly);
        assert!((dist - 130.0).abs() < 10.0); // ~130km
    }
    
    #[test]
    fn test_route_distance() {
        let points = vec![
            GeoPoint::new(40.7128, -74.0060), // NYC
            GeoPoint::new(39.9526, -75.1652), // Philadelphia
            GeoPoint::new(38.9072, -77.0369), // DC
        ];
        
        let dist = calculate_route_distance(&points);
        assert!(dist > 0.0);
        assert!(dist < 500.0); // Reasonable total
    }
    
    #[test]
    fn test_tsp_solver() {
        let points = vec![
            GeoPoint::new(40.7128, -74.0060),
            GeoPoint::new(39.9526, -75.1652),
            GeoPoint::new(38.9072, -77.0369),
            GeoPoint::new(40.4406, -79.9959), // Pittsburgh
        ];
        
        let matrix = build_distance_matrix(&points);
        let (route, distance) = solve_tsp_optimized(&matrix, points.len(), 0, 100);
        
        assert_eq!(route.len(), points.len());
        assert!(distance > 0.0);
    }
}
