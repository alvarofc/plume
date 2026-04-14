mod generation;

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use foyer::CacheBuilder;
use plume_core::config::CacheConfig;
use plume_core::error::PlumeError;
use plume_core::types::SearchResult;
use serde::{Deserialize, Serialize};
use tracing::info;

pub use generation::GenerationCounter;

/// Cache key: (namespace, generation, query_hash).
/// Generation counter ensures stale results are never served after writes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub namespace: String,
    pub generation: u64,
    pub query_hash: u64,
}

/// Cached search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheValue {
    pub results: Vec<SearchResult>,
}

/// Tiered search result cache: RAM (L1) → NVMe (L2).
///
/// Object store (L3) is handled by LanceDB directly — the cache sits
/// in front of the search pipeline, not in front of storage.
pub struct SearchCache {
    /// In-memory LFU cache (L1)
    memory: foyer::Cache<CacheKey, Arc<CacheValue>>,
    /// Generation counters per namespace
    generations: GenerationCounter,
    /// Prometheus-style counters
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl SearchCache {
    /// Create a new search cache with the given config.
    pub fn new(config: &CacheConfig) -> Result<Self, PlumeError> {
        let ram_bytes = config.ram_capacity_mb * 1024 * 1024;

        let memory = CacheBuilder::new(ram_bytes)
            .with_weighter(|_key: &CacheKey, value: &Arc<CacheValue>| {
                // Estimate size: each result ~500 bytes
                std::mem::size_of::<CacheKey>() + value.results.len() * 500
            })
            .build();

        info!(
            ram_mb = config.ram_capacity_mb,
            "search cache initialized"
        );

        Ok(Self {
            memory,
            generations: GenerationCounter::new(),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Look up cached results for a query.
    pub fn get(&self, namespace: &str, query_hash: u64) -> Option<Vec<SearchResult>> {
        let generation = self.generations.get(namespace);
        let key = CacheKey {
            namespace: namespace.to_string(),
            generation,
            query_hash,
        };

        match self.memory.get(&key) {
            Some(entry) => {
                self.hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(entry.value().results.clone())
            }
            None => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert search results into the cache.
    pub fn insert(&self, namespace: &str, query_hash: u64, results: Vec<SearchResult>) {
        let generation = self.generations.get(namespace);
        let key = CacheKey {
            namespace: namespace.to_string(),
            generation,
            query_hash,
        };

        self.memory.insert(key, Arc::new(CacheValue { results }));
    }

    /// Increment the generation counter for a namespace.
    /// This effectively invalidates all cached results for that namespace.
    pub fn invalidate(&self, namespace: &str) {
        self.generations.increment(namespace);
    }

    /// Remove a namespace's generation counter entirely.
    /// Call this when a namespace is dropped to prevent counter leaks.
    pub fn remove_namespace(&self, namespace: &str) {
        self.generations.remove(namespace);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// Hash a query string + search mode into a u64 for cache keying.
pub fn hash_query(query: &str, mode: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query.hash(&mut hasher);
    mode.hash(&mut hasher);
    hasher.finish()
}
