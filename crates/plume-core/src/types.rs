use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A document to be indexed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Multi-vector representation of a document (ColBERT token embeddings).
pub type MultiVector = Vec<Vec<f32>>;

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Search mode: semantic (MaxSim), full-text (BM25), or hybrid (RRF fusion).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Semantic,
    Fts,
    #[default]
    Hybrid,
}

/// Query request from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub mode: SearchMode,
}

fn default_k() -> usize {
    10
}

/// Upsert request from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub rows: Vec<Document>,
}

/// Query response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<SearchResult>,
    pub cache_hit: bool,
}

/// Upsert response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub upserted: usize,
}

/// Index build response (async operation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_type: Option<String>,
}

/// Async index job status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

/// Poll response for an async index job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexJobResponse {
    pub job_id: String,
    pub namespace: String,
    pub index_type: String,
    pub status: IndexJobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Embedding dimensions for ColBERT-style models.
/// LateOn-Code-edge uses 48-dim output after the ColBERT linear projection.
pub const EMBEDDING_DIM: usize = 48;

// --- Input validation limits ---

/// Maximum number of results per query.
pub const MAX_K: usize = 1000;

/// Maximum documents per upsert batch.
pub const MAX_ROWS_PER_UPSERT: usize = 10_000;

/// Maximum text length per document (bytes).
pub const MAX_TEXT_LENGTH: usize = 1_000_000;

/// Maximum namespace name length.
pub const MAX_NAMESPACE_LENGTH: usize = 64;

/// Collapse a ColBERT multi-vector into a single normalized vector for ANN candidate retrieval.
///
/// This is used only to generate candidates. Final semantic ranking still uses
/// token-level MaxSim over the full multi-vector representation.
pub fn ann_vector(multivector: &MultiVector) -> Result<Vec<f32>, String> {
    if multivector.is_empty() {
        return Err("multivector must contain at least one token vector".into());
    }

    let mut pooled = vec![0.0f32; EMBEDDING_DIM];

    for token in multivector {
        if token.len() != EMBEDDING_DIM {
            return Err(format!(
                "expected embedding dim {EMBEDDING_DIM}, got {}",
                token.len()
            ));
        }

        for (dst, value) in pooled.iter_mut().zip(token.iter()) {
            *dst += *value;
        }
    }

    let count = multivector.len() as f32;
    for value in &mut pooled {
        *value /= count;
    }

    let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return Err("pooled ANN vector has zero norm".into());
    }

    for value in &mut pooled {
        *value /= norm;
    }

    Ok(pooled)
}

/// Validate that a namespace name is safe (alphanumeric, hyphens, underscores).
pub fn validate_namespace(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("namespace must not be empty".into());
    }
    if name.len() > MAX_NAMESPACE_LENGTH {
        return Err(format!(
            "namespace too long ({} chars, max {MAX_NAMESPACE_LENGTH})",
            name.len()
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("namespace must be alphanumeric, hyphens, or underscores".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ann_vector_normalizes_mean_embedding() {
        let pooled = ann_vector(&vec![vec![1.0; EMBEDDING_DIM], vec![1.0; EMBEDDING_DIM]])
            .expect("ann vector");

        let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(pooled.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn ann_vector_rejects_empty_multivector() {
        let err = ann_vector(&Vec::new()).unwrap_err();
        assert!(err.contains("at least one token vector"));
    }

    #[test]
    fn validate_namespace_accepts_safe_names() {
        assert!(validate_namespace("code").is_ok());
        assert!(validate_namespace("code-search_01").is_ok());
    }

    #[test]
    fn validate_namespace_rejects_invalid_names() {
        assert!(validate_namespace("").is_err());
        assert!(validate_namespace("contains space").is_err());
        assert!(validate_namespace("slash/name").is_err());
    }
}
