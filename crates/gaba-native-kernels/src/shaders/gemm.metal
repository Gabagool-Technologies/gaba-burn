#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

kernel void gemm_tiled_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    
    constexpr uint TILE_SIZE = 16;
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0;
    
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint tiledRow = t * TILE_SIZE + tid.y;
        uint tiledCol = t * TILE_SIZE + tid.x;
        
        As[tid.y][tid.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0;
        Bs[tid.y][tid.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
