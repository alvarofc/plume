use std::collections::HashMap;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use plume_core::types::{IndexResponse, QueryRequest, UpsertRequest, UpsertResponse};
use serde_json::json;
use tracing::{error, info};

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/ns/{ns}/upsert", post(upsert))
        .route("/ns/{ns}/query", post(query))
        .route("/ns/{ns}/index", post(build_index))
        .route("/ns/{ns}/fts-index", post(build_fts_index))
        .route("/ns/{ns}/warmup", post(warmup))
        .route("/ns/{ns}", delete(drop_namespace))
        .with_state(state)
}

/// GET /health — liveness check.
async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

/// GET /metrics — Prometheus exposition format.
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.cache.stats();

    let body = format!(
        "# HELP plume_cache_hits_total Total cache hits\n\
         # TYPE plume_cache_hits_total counter\n\
         plume_cache_hits_total {}\n\
         # HELP plume_cache_misses_total Total cache misses\n\
         # TYPE plume_cache_misses_total counter\n\
         plume_cache_misses_total {}\n\
         # HELP plume_object_requests_saved_total S3 requests saved by cache\n\
         # TYPE plume_object_requests_saved_total counter\n\
         plume_object_requests_saved_total {}\n",
        stats.hits,
        stats.misses,
        stats.hits,
    );

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}

/// POST /ns/{ns}/upsert — ingest documents.
async fn upsert(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<impl IntoResponse, AppError> {
    if req.rows.is_empty() {
        return Err(AppError::bad_request("rows must not be empty"));
    }

    let table = state.index.namespace(&ns).await?;

    let texts: Vec<&str> = req.rows.iter().map(|d| d.text.as_str()).collect();
    let multivectors = state.encoder.encode_batch(&texts)?;

    let ids: Vec<String> = req.rows.iter().map(|d| d.id.clone()).collect();
    let texts: Vec<String> = req.rows.iter().map(|d| d.text.clone()).collect();
    let metadata: Vec<HashMap<String, serde_json::Value>> =
        req.rows.iter().map(|d| d.metadata.clone()).collect();

    let count = table.upsert(&ids, &texts, &multivectors, &metadata).await?;

    // Invalidate cache for this namespace
    state.cache.invalidate(&ns);

    Ok(Json(UpsertResponse { upserted: count }))
}

/// POST /ns/{ns}/query — search documents.
async fn query(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Json(req): Json<QueryRequest>,
) -> Result<impl IntoResponse, AppError> {
    let table = state.index.namespace(&ns).await?;

    let query_vectors = state.encoder.encode_single(&req.query)?;

    let response = state
        .search
        .search(&table, &query_vectors, &req.query, req.k, req.mode)
        .await?;

    Ok(Json(response))
}

/// POST /ns/{ns}/index — build IVF_PQ vector index (async).
async fn build_index(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let table = state.index.namespace(&ns).await?;

    tokio::spawn(async move {
        if let Err(e) = table.build_vector_index().await {
            error!(namespace = %ns, error = %e, "vector index build failed");
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexResponse {
            status: "building".to_string(),
        }),
    ))
}

/// POST /ns/{ns}/fts-index — build BM25 full-text index (async).
async fn build_fts_index(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let table = state.index.namespace(&ns).await?;

    tokio::spawn(async move {
        if let Err(e) = table.build_fts_index().await {
            error!(namespace = %ns, error = %e, "FTS index build failed");
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexResponse {
            status: "building".to_string(),
        }),
    ))
}

/// POST /ns/{ns}/warmup — pre-warm cache for a namespace.
async fn warmup(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    // Pre-warm: run a dummy scan to pull data into OS page cache / LanceDB cache
    let table = state.index.namespace(&ns).await?;
    let count = table.count().await?;
    info!(namespace = %ns, docs = count, "warmup complete");
    Ok(Json(json!({"status": "ok", "namespace": ns, "docs": count})))
}

/// DELETE /ns/{ns} — drop a namespace.
async fn drop_namespace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    state.index.drop_namespace(&ns).await?;
    state.cache.invalidate(&ns);

    Ok(Json(json!({"status": "dropped", "namespace": ns})))
}

// --- Error handling ---

struct AppError(plume_core::error::PlumeError);

impl AppError {
    fn bad_request(msg: &str) -> Self {
        Self(plume_core::error::PlumeError::InvalidRequest(
            msg.to_string(),
        ))
    }
}

impl From<plume_core::error::PlumeError> for AppError {
    fn from(e: plume_core::error::PlumeError) -> Self {
        Self(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let status = self.0.status_code();
        let body = json!({
            "error": self.0.to_string(),
        });

        (
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(body),
        )
            .into_response()
    }
}
