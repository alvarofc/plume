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
