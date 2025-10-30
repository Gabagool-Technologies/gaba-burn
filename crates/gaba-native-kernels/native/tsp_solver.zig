// Traveling Salesman Problem solver for route optimization
// Nearest neighbor heuristic + 2-opt local search
// Pure C ABI for Rust-Zig interop

const std = @import("std");

/// Solve TSP using nearest neighbor heuristic
/// Input: dist_matrix (n x n, row-major), n (number of points), start_idx
/// Output: route (array of indices), returns total distance
export fn tsp_nearest_neighbor(
    dist_matrix: [*]const f64,
    n: usize,
    start_idx: usize,
    out_route: [*]usize,
) f64 {
    if (n == 0) return 0.0;
    if (n == 1) {
        out_route[0] = 0;
        return 0.0;
    }

    var visited = [_]bool{false} ** 256; // Max 256 points
    var current = start_idx;
    var route_len: usize = 0;
    var total_distance: f64 = 0.0;

    out_route[route_len] = current;
    route_len += 1;
    visited[current] = true;

    while (route_len < n) {
        var nearest: usize = 0;
        var min_dist: f64 = std.math.inf(f64);

        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (!visited[i]) {
                const dist = dist_matrix[current * n + i];
                if (dist < min_dist) {
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
    total_distance += dist_matrix[current * n + start_idx];

    return total_distance;
}

/// 2-opt improvement for TSP route
/// Iteratively improves route by reversing segments
/// Returns improved total distance
export fn tsp_two_opt(
    dist_matrix: [*]const f64,
    n: usize,
    route: [*]usize,
    max_iterations: usize,
) f64 {
    if (n < 4) return calculate_route_distance(dist_matrix, n, route);

    var improved = true;
    var iterations: usize = 0;

    while (improved and iterations < max_iterations) : (iterations += 1) {
        improved = false;

        var i: usize = 0;
        while (i < n - 1) : (i += 1) {
            var j: usize = i + 2;
            while (j < n) : (j += 1) {
                const delta = calculate_2opt_delta(dist_matrix, n, route, i, j);
                if (delta < -0.001) { // Improvement threshold
                    reverse_segment(route, i + 1, j);
                    improved = true;
                }
            }
        }
    }

    return calculate_route_distance(dist_matrix, n, route);
}

/// Calculate distance change from 2-opt swap
fn calculate_2opt_delta(
    dist_matrix: [*]const f64,
    n: usize,
    route: [*]const usize,
    i: usize,
    j: usize,
) f64 {
    const a = route[i];
    const b = route[i + 1];
    const c = route[j];
    const d = if (j + 1 < n) route[j + 1] else route[0];

    const old_dist = dist_matrix[a * n + b] + dist_matrix[c * n + d];
    const new_dist = dist_matrix[a * n + c] + dist_matrix[b * n + d];

    return new_dist - old_dist;
}

/// Reverse route segment [i, j]
fn reverse_segment(route: [*]usize, i: usize, j: usize) void {
    var left = i;
    var right = j;

    while (left < right) {
        const temp = route[left];
        route[left] = route[right];
        route[right] = temp;
        left += 1;
        right -= 1;
    }
}

/// Calculate total route distance
fn calculate_route_distance(
    dist_matrix: [*]const f64,
    n: usize,
    route: [*]const usize,
) f64 {
    if (n < 2) return 0.0;

    var total: f64 = 0.0;
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        const from = route[i];
        const to = route[i + 1];
        total += dist_matrix[from * n + to];
    }

    // Return to start
    const last = route[n - 1];
    const first = route[0];
    total += dist_matrix[last * n + first];

    return total;
}

/// Combined solver: nearest neighbor + 2-opt
/// Best of both worlds for quick, good solutions
export fn tsp_solve_optimized(
    dist_matrix: [*]const f64,
    n: usize,
    start_idx: usize,
    out_route: [*]usize,
    max_iterations: usize,
) f64 {
    // First pass: nearest neighbor
    _ = tsp_nearest_neighbor(dist_matrix, n, start_idx, out_route);

    // Second pass: 2-opt improvement
    return tsp_two_opt(dist_matrix, n, out_route, max_iterations);
}
