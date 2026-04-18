use std::collections::HashMap;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use plume_core::types::{
    validate_namespace, IndexResponse, QueryRequest, UpsertRequest, UpsertResponse, MAX_K,
    MAX_ROWS_PER_UPSERT, MAX_TEXT_LENGTH,
};
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
        .route("/ns/{ns}/index-jobs/{job_id}", get(get_index_job))
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
         plume_cache_misses_total {}\n",
        stats.hits, stats.misses,
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
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;

    if req.rows.is_empty() {
        return Err(AppError::bad_request("rows must not be empty"));
    }
    if req.rows.len() > MAX_ROWS_PER_UPSERT {
        return Err(AppError::bad_request(&format!(
            "too many rows ({}, max {MAX_ROWS_PER_UPSERT})",
            req.rows.len()
        )));
    }
    for doc in &req.rows {
        if doc.text.len() > MAX_TEXT_LENGTH {
            return Err(AppError::bad_request(&format!(
                "document '{}' text too long ({} bytes, max {MAX_TEXT_LENGTH})",
                doc.id,
                doc.text.len()
            )));
        }
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
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;

    if req.k == 0 || req.k > MAX_K {
        return Err(AppError::bad_request(&format!(
            "k must be between 1 and {MAX_K}, got {}",
            req.k
        )));
    }
    if req.query.is_empty() {
        return Err(AppError::bad_request("query must not be empty"));
    }
    if req.query.len() > MAX_TEXT_LENGTH {
        return Err(AppError::bad_request("query text too long"));
    }

    let table = state.index.get_namespace(&ns).await?;

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
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;
    let table = state.index.get_namespace(&ns).await?;
    let job = state.jobs.create_job(&ns, "vector").await;
    let jobs = state.jobs.clone();
    let cache = state.cache.clone();
    let index_config = state.config.index.clone();
    let job_id = job.job_id.clone();
    let task_job_id = job_id.clone();
    let status_url = format!("/ns/{ns}/index-jobs/{job_id}");

    tokio::spawn(async move {
        jobs.mark_running(&task_job_id).await;
        if let Err(e) = table.build_vector_index(&index_config).await {
            error!(namespace = %ns, error = %e, job_id = %task_job_id, "vector index build failed");
            jobs.mark_failed(&task_job_id, e.to_string()).await;
            return;
        }
        // Invalidate cached results: entries written before the ANN
        // index existed came from the bounded-scan fallback, which has
        // degraded recall. Without this, a healthy post-build query
        // would still serve the old low-recall list.
        cache.invalidate(&ns);
        jobs.mark_completed(&task_job_id).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexResponse {
            status: "building".to_string(),
            job_id: Some(job_id),
            status_url: Some(status_url),
            index_type: Some("vector".to_string()),
        }),
    ))
}

/// POST /ns/{ns}/fts-index — build BM25 full-text index (async).
async fn build_fts_index(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;
    let table = state.index.get_namespace(&ns).await?;
    let job = state.jobs.create_job(&ns, "fts").await;
    let jobs = state.jobs.clone();
    let cache = state.cache.clone();
    let job_id = job.job_id.clone();
    let task_job_id = job_id.clone();
    let status_url = format!("/ns/{ns}/index-jobs/{job_id}");

    tokio::spawn(async move {
        jobs.mark_running(&task_job_id).await;
        if let Err(e) = table.build_fts_index().await {
            error!(namespace = %ns, error = %e, job_id = %task_job_id, "FTS index build failed");
            jobs.mark_failed(&task_job_id, e.to_string()).await;
            return;
        }
        // FTS + hybrid results cached before this point were produced
        // without the BM25 index; invalidate so they don't stick.
        cache.invalidate(&ns);
        jobs.mark_completed(&task_job_id).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexResponse {
            status: "building".to_string(),
            job_id: Some(job_id),
            status_url: Some(status_url),
            index_type: Some("fts".to_string()),
        }),
    ))
}

/// GET /ns/{ns}/index-jobs/{job_id} — poll an async index job.
async fn get_index_job(
    State(state): State<AppState>,
    Path((ns, job_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, AppError> {
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;

    let job = state
        .jobs
        .get(&job_id)
        .await
        .filter(|job| job.namespace == ns)
        .ok_or_else(|| {
            AppError(plume_core::error::PlumeError::NotFound(format!(
                "index job {job_id} for namespace {ns}"
            )))
        })?;

    Ok(Json(job))
}

/// POST /ns/{ns}/warmup — pre-warm cache for a namespace.
async fn warmup(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;
    // Pre-warm: run a dummy scan to pull data into OS page cache / LanceDB cache
    let table = state.index.get_namespace(&ns).await?;
    let count = table.count().await?;
    info!(namespace = %ns, docs = count, "warmup complete");
    Ok(Json(
        json!({"status": "ok", "namespace": ns, "docs": count}),
    ))
}

/// DELETE /ns/{ns} — drop a namespace.
async fn drop_namespace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    validate_namespace(&ns).map_err(|e| AppError::bad_request(&e))?;
    state.index.drop_namespace(&ns).await?;
    state.cache.remove_namespace(&ns);

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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use plume_cache::SearchCache;
    use plume_core::config::{
        CacheConfig, EncoderConfig, PlumeConfig, ServerConfig, StorageConfig,
    };
    use plume_core::types::{IndexJobResponse, IndexJobStatus, QueryResponse};
    use plume_encoder::build_encoder;
    use plume_index::IndexManager;
    use plume_search::SearchEngine;
    use tempfile::TempDir;
    use tower::ServiceExt;

    use crate::jobs::IndexJobRegistry;

    /// Build a test app on an ephemeral directory. The returned `TempDir`
    /// must stay alive for the duration of the test so that LanceDB and the
    /// NVMe cache can keep writing to it; dropping it removes everything.
    async fn test_app() -> (Router, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = StorageConfig {
            uri: dir.path().to_string_lossy().to_string(),
            region: None,
            endpoint: None,
        };
        let config = PlumeConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            storage: storage.clone(),
            cache: CacheConfig {
                ram_capacity_mb: 16,
                nvme_capacity_gb: 1,
                nvme_path: dir.path().join("cache").to_string_lossy().to_string(),
            },
            encoder: EncoderConfig {
                model: "mock".to_string(),
                pool_factor: 2,
                batch_size: 8,
            },
            index: Default::default(),
        };

        let index = IndexManager::connect(&storage).await.unwrap();
        let cache = Arc::new(SearchCache::new(&config.cache).await.unwrap());
        let search = Arc::new(SearchEngine::new(cache.clone(), config.index.clone()));
        let encoder = Arc::from(build_encoder(&config.encoder));
        let state = AppState {
            config,
            index: Arc::new(index),
            cache,
            search,
            encoder,
            jobs: Arc::new(IndexJobRegistry::new()),
        };

        (router(state), dir)
    }

    async fn response_json<T: serde::de::DeserializeOwned>(
        app: &Router,
        request: Request<Body>,
    ) -> (StatusCode, T) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let parsed = serde_json::from_slice(&body).unwrap();
        (status, parsed)
    }

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let (app, _tmp) = test_app().await;

        let (status, body): (StatusCode, serde_json::Value) = response_json(
            &app,
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn upsert_then_query_returns_results() {
        let (app, _tmp) = test_app().await;

        let upsert = json!({
            "rows": [
                {"id": "1", "text": "retry HTTP requests with exponential backoff", "metadata": {}},
                {"id": "2", "text": "binary search with generic comparator", "metadata": {}}
            ]
        });

        let (upsert_status, _): (StatusCode, UpsertResponse) = response_json(
            &app,
            Request::builder()
                .method("POST")
                .uri("/ns/code/upsert")
                .header("content-type", "application/json")
                .body(Body::from(upsert.to_string()))
                .unwrap(),
        )
        .await;
        assert_eq!(upsert_status, StatusCode::OK);

        let query = json!({"query": "exponential backoff", "k": 5, "mode": "semantic"});
        let (query_status, body): (StatusCode, QueryResponse) = response_json(
            &app,
            Request::builder()
                .method("POST")
                .uri("/ns/code/query")
                .header("content-type", "application/json")
                .body(Body::from(query.to_string()))
                .unwrap(),
        )
        .await;

        assert_eq!(query_status, StatusCode::OK);
        assert!(!body.results.is_empty());
    }

    #[tokio::test]
    async fn fts_index_build_returns_job_and_can_be_polled() {
        let (app, _tmp) = test_app().await;

        let upsert = json!({
            "rows": [
                {"id": "1", "text": "retry HTTP requests with exponential backoff", "metadata": {}}
            ]
        });

        let _: (StatusCode, UpsertResponse) = response_json(
            &app,
            Request::builder()
                .method("POST")
                .uri("/ns/code/upsert")
                .header("content-type", "application/json")
                .body(Body::from(upsert.to_string()))
                .unwrap(),
        )
        .await;

        let (status, response): (StatusCode, IndexResponse) = response_json(
            &app,
            Request::builder()
                .method("POST")
                .uri("/ns/code/fts-index")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::ACCEPTED);
        let job_id = response.job_id.expect("job id");

        let mut final_status = IndexJobStatus::Queued;
        for _ in 0..20 {
            let (_, job): (StatusCode, IndexJobResponse) = response_json(
                &app,
                Request::builder()
                    .uri(format!("/ns/code/index-jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;

            final_status = job.status;
            if matches!(
                final_status,
                IndexJobStatus::Completed | IndexJobStatus::Failed
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        assert_eq!(final_status, IndexJobStatus::Completed);
    }

    #[tokio::test]
    async fn unknown_index_job_returns_not_found() {
        let (app, _tmp) = test_app().await;

        let (status, body): (StatusCode, serde_json::Value) = response_json(
            &app,
            Request::builder()
                .uri("/ns/code/index-jobs/does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"].as_str().unwrap().contains("index job"));
    }
}
