// Route distance calculation kernel - Haversine formula for geographic distances
// Pure C ABI for Rust-Zig interop

const std = @import("std");
const math = std.math;

const EARTH_RADIUS_KM: f64 = 6371.0;
const DEG_TO_RAD: f64 = math.pi / 180.0;

/// Point in geographic coordinates
pub const GeoPoint = extern struct {
    lat: f64,
    lng: f64,
};

/// Calculate Haversine distance between two points in kilometers
/// C ABI export for Rust FFI
export fn haversine_distance(p1: *const GeoPoint, p2: *const GeoPoint) callconv(.c) f64 {
    const lat1_rad = p1.lat * DEG_TO_RAD;
    const lat2_rad = p2.lat * DEG_TO_RAD;
    const dlat = (p2.lat - p1.lat) * DEG_TO_RAD;
    const dlng = (p2.lng - p1.lng) * DEG_TO_RAD;

    const a = math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
        math.cos(lat1_rad) * math.cos(lat2_rad) *
        math.sin(dlng / 2.0) * math.sin(dlng / 2.0);

    const c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a));

    return EARTH_RADIUS_KM * c;
}

/// Calculate total route distance for array of points
/// Returns total distance in kilometers
export fn route_total_distance(points: [*]const GeoPoint, n: usize) callconv(.c) f64 {
    if (n < 2) return 0.0;

    var total: f64 = 0.0;
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        total += haversine_distance(&points[i], &points[i + 1]);
    }

    return total;
}

/// Calculate distance matrix for all point pairs
/// Matrix is stored in row-major order
/// matrix[i * n + j] = distance from point i to point j
export fn distance_matrix(
    points: [*]const GeoPoint,
    n: usize,
    out_matrix: [*]f64,
) callconv(.c) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            if (i == j) {
                out_matrix[i * n + j] = 0.0;
            } else {
                out_matrix[i * n + j] = haversine_distance(&points[i], &points[j]);
            }
        }
    }
}

/// Fast approximate distance using equirectangular projection
/// Much faster than Haversine, good for short distances
export fn fast_distance_approx(p1: *const GeoPoint, p2: *const GeoPoint) callconv(.c) f64 {
    const lat1_rad = p1.lat * DEG_TO_RAD;
    const lat2_rad = p2.lat * DEG_TO_RAD;
    const dlng = (p2.lng - p1.lng) * DEG_TO_RAD;
    const dlat = (p2.lat - p1.lat) * DEG_TO_RAD;

    const x = dlng * math.cos((lat1_rad + lat2_rad) / 2.0);
    const y = dlat;

    return EARTH_RADIUS_KM * math.sqrt(x * x + y * y);
}

/// Calculate route distance using fast approximation
export fn route_total_distance_fast(points: [*]const GeoPoint, n: usize) callconv(.c) f64 {
    if (n < 2) return 0.0;

    var total: f64 = 0.0;
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        total += fast_distance_approx(&points[i], &points[i + 1]);
    }

    return total;
}
