mod fusion;
mod maxsim;

use std::sync::Arc;

use plume_cache::{hash_query, SearchCache};
use plume_core::error::PlumeError;
use plume_core::types::{MultiVector, QueryResponse, SearchMode};
use plume_index::NamespaceTable;

pub use fusion::rrf_fusion;
pub use maxsim::maxsim_score;

/// Orchestrates search: encode -> cache check -> scan + MaxSim score -> fuse -> cache store.
pub struct SearchEngine {
    cache: Arc<SearchCache>,
}

impl SearchEngine {
    pub fn new(cache: Arc<SearchCache>) -> Self {
        Self { cache }
    }

    /// Run a search query against a namespace.
    pub async fn search(
        &self,
        table: &NamespaceTable,
        query_vectors: &MultiVector,
        query_text: &str,
        k: usize,
        mode: SearchMode,
    ) -> Result<QueryResponse, PlumeError> {
        let mode_str = match mode {
            SearchMode::Semantic => "semantic",
            SearchMode::Fts => "fts",
            SearchMode::Hybrid => "hybrid",
        };

        // Check cache
        let query_hash = hash_query(query_text, mode_str);
        if let Some(cached) = self.cache.get(&table.name, query_hash) {
            return Ok(QueryResponse {
                results: cached,
                cache_hit: true,
            });
        }

        let results = match mode {
            SearchMode::Semantic => {
                self.semantic_search(table, query_vectors, k).await?
            }
            SearchMode::Fts => {
                table.fts_search(query_text, k).await?
            }
            SearchMode::Hybrid => {
                let (semantic, fts) = tokio::try_join!(
                    self.semantic_search(table, query_vectors, k * 2),
                    table.fts_search(query_text, k * 2),
                )?;
                rrf_fusion(&semantic, &fts, k)
            }
        };

        // Store in cache
        self.cache.insert(&table.name, query_hash, results.clone());

        Ok(QueryResponse {
            results,
            cache_hit: false,
        })
    }

    /// ColBERT-style semantic search: scan documents, score with MaxSim, rank.
    async fn semantic_search(
        &self,
        table: &NamespaceTable,
        query_vectors: &MultiVector,
        k: usize,
    ) -> Result<Vec<plume_core::types::SearchResult>, PlumeError> {
        // Scan all documents with their multivectors
        // For large datasets, this should be replaced with ANN candidate retrieval + re-rank
        let candidates = table.scan_with_vectors(k * 10).await?;

        // Score each candidate with MaxSim
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|(mut result, doc_vectors)| {
                result.score = maxsim_score(query_vectors, &doc_vectors);
                result
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored.truncate(k);
        Ok(scored)
    }
}
