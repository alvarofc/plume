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
}

/// Embedding dimensions for ColBERT-style models.
pub const EMBEDDING_DIM: usize = 128;

// --- Input validation limits ---

/// Maximum number of results per query.
pub const MAX_K: usize = 1000;

/// Maximum documents per upsert batch.
pub const MAX_ROWS_PER_UPSERT: usize = 10_000;

/// Maximum text length per document (bytes).
pub const MAX_TEXT_LENGTH: usize = 1_000_000;

/// Maximum namespace name length.
pub const MAX_NAMESPACE_LENGTH: usize = 64;

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
