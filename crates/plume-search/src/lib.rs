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
        let query_hash = hash_query(query_text, mode_str, k);
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
        // `max_candidates` caps memory usage: every candidate pulls its
        // full multivector for MaxSim rerank, so an unbounded
        // multiplier*k could OOM. We still guarantee at least `k`
        // candidates so the search can return the requested number of
        // hits even when the operator set a tight cap.
        let candidate_limit = self
            .index_config
            .ann_candidate_multiplier
            .saturating_mul(k)
            .max(k)
            .min(self.index_config.max_candidates.max(k));

        // A failing `list_indices` (transient metadata/object-store hiccup)
        // must not take search offline. Treat it as "no ANN index" so we
        // still return results via the bounded scan path.
        let has_ann = match table.has_ann_index().await {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    namespace = %table.name,
                    error = %e,
                    "ANN index check failed; falling back to bounded scan"
                );
                false
            }
        };

        let candidates = if has_ann {
            match table
                .ann_search_with_vectors(
                    query_vectors,
                    candidate_limit,
                    self.index_config.nprobes as usize,
                    self.index_config.refine_factor,
                )
                .await
            {
                Ok(c) => c,
                // An ANN index can transiently fail (mid-rebuild, object
                // store hiccup, corrupt manifest). Falling back to the
                // bounded scan keeps search available with the same
                // memory ceiling; operators will see the warn in logs.
                Err(e) => {
                    warn!(
                        namespace = %table.name,
                        error = %e,
                        "ANN search failed; falling back to bounded scan"
                    );
                    table.scan_with_vectors(candidate_limit).await?
                }
            }
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
