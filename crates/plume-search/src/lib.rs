mod fusion;
mod maxsim;

use std::sync::Arc;

use plume_cache::{hash_query, SearchCache};
use plume_core::config::IndexConfig;
use plume_core::error::PlumeError;
use plume_core::types::{MultiVector, QueryResponse, SearchMode};
use plume_index::NamespaceTable;
use tracing::warn;

pub use fusion::rrf_fusion;
pub use maxsim::maxsim_score;

/// Orchestrates search: encode -> cache check -> scan + MaxSim score -> fuse -> cache store.
pub struct SearchEngine {
    cache: Arc<SearchCache>,
    index_config: IndexConfig,
}

impl SearchEngine {
    pub fn new(cache: Arc<SearchCache>, index_config: IndexConfig) -> Self {
        Self {
            cache,
            index_config,
        }
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

        // Check cache. A cache failure (disk IO, corruption, serialization)
        // is treated as a miss: the cache is an optimization, and a degraded
        // hot tier must not take search offline.
        let query_hash = hash_query(query_text, mode_str);
        match self.cache.get(&table.name, query_hash).await {
            Ok(Some(cached)) => {
                return Ok(QueryResponse {
                    results: cached,
                    cache_hit: true,
                });
            }
            Ok(None) => {}
            Err(e) => {
                warn!(
                    namespace = %table.name,
                    error = %e,
                    "search cache lookup failed; continuing as miss"
                );
            }
        }

        let results = match mode {
            SearchMode::Semantic => self.semantic_search(table, query_vectors, k).await?,
            SearchMode::Fts => table.fts_search(query_text, k).await?,
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

    /// ColBERT-style semantic search: ANN candidate retrieval, then exact MaxSim re-rank.
    async fn semantic_search(
        &self,
        table: &NamespaceTable,
        query_vectors: &MultiVector,
        k: usize,
    ) -> Result<Vec<plume_core::types::SearchResult>, PlumeError> {
        let candidate_limit = self
            .index_config
            .ann_candidate_multiplier
            .saturating_mul(k)
            .max(k);

        let candidates = if table.has_ann_index().await? {
            table
                .ann_search_with_vectors(
                    query_vectors,
                    candidate_limit,
                    self.index_config.nprobes as usize,
                    self.index_config.refine_factor,
                )
                .await?
        } else {
            // Fallback for early-stage namespaces (pre-`/index`): same
            // candidate cap as the ANN path. Scans only the first
            // `candidate_limit` rows, which keeps memory bounded on large
            // namespaces. Small namespaces (count <= candidate_limit) still
            // score every document, so recall is preserved while the index
            // is being built.
            table.scan_with_vectors(candidate_limit).await?
        };

        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|(mut result, doc_vectors)| {
                result.score = maxsim_score(query_vectors, &doc_vectors);
                result
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(k);
        Ok(scored)
    }
}
