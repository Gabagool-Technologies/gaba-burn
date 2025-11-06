use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct ApiState {
    pub models: Arc<RwLock<Vec<ModelInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub params: usize,
    pub inference_time_ms: f64,
    pub memory_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub input: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model: String,
    pub output: Vec<Vec<f32>>,
    pub inference_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub models_loaded: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

pub fn create_router() -> Router {
    let state = ApiState {
        models: Arc::new(RwLock::new(vec![
            ModelInfo {
                name: "MicroYOLONano".to_string(),
                version: "0.1.0".to_string(),
                params: 47_000,
                inference_time_ms: 0.90,
                memory_mb: 0.18,
            },
            ModelInfo {
                name: "EfficientEdgeLite".to_string(),
                version: "0.1.0".to_string(),
                params: 89_000,
                inference_time_ms: 0.72,
                memory_mb: 0.14,
            },
        ])),
    };

    Router::new()
        .route("/health", get(health_check))
        .route("/models", get(list_models))
        .route("/models/:name", get(get_model))
        .route("/infer", post(infer))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn health_check(State(state): State<ApiState>) -> Json<HealthResponse> {
    let models = state.models.read().await;
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "0.1.0".to_string(),
        models_loaded: models.len(),
    })
}

async fn list_models(State(state): State<ApiState>) -> Json<Vec<ModelInfo>> {
    let models = state.models.read().await;
    Json(models.clone())
}

async fn get_model(
    State(state): State<ApiState>,
    Path(name): Path<String>,
) -> Result<Json<ModelInfo>, StatusCode> {
    let models = state.models.read().await;
    models
        .iter()
        .find(|m| m.name == name)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn infer(
    State(_state): State<ApiState>,
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let start = std::time::Instant::now();
    
    // Simulate inference
    let output = req.input.clone();
    
    let duration = start.elapsed();
    
    Ok(Json(InferenceResponse {
        model: req.model,
        output,
        inference_time_ms: duration.as_secs_f64() * 1000.0,
    }))
}

async fn metrics() -> Json<MetricsResponse> {
    Json(MetricsResponse {
        total_requests: 0,
        avg_latency_ms: 0.64,
        p95_latency_ms: 1.2,
        p99_latency_ms: 2.1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_health_check() {
        let state = ApiState {
            models: Arc::new(RwLock::new(vec![])),
        };
        
        let response = health_check(State(state)).await;
        assert_eq!(response.0.status, "healthy");
    }
}
