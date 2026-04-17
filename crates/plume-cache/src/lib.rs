mod generation;

use std::hash::{Hash, Hasher};
use std::path::Path;

use foyer::{
    BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder,
    HybridCachePolicy,
};
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
    hybrid: HybridCache<CacheKey, CacheValue>,
    /// Generation counters per namespace
    generations: GenerationCounter,
    /// Application-level counters for cache lookups
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl SearchCache {
    /// Create a new RAM + NVMe search cache with the given config.
    pub async fn new(config: &CacheConfig) -> Result<Self, PlumeError> {
        let ram_bytes = config.ram_capacity_mb * 1024 * 1024;
        let nvme_bytes = config.nvme_capacity_gb * 1024 * 1024 * 1024;
        let nvme_path = Path::new(&config.nvme_path);

        if nvme_bytes == 0 {
            return Err(PlumeError::Cache(
                "nvme_capacity_gb must be greater than 0 for the hybrid cache".into(),
            ));
        }

        std::fs::create_dir_all(nvme_path).map_err(|e| {
            PlumeError::Cache(format!(
                "failed to create cache directory {}: {e}",
                nvme_path.display()
            ))
        })?;

        let device = FsDeviceBuilder::new(nvme_path)
            .with_capacity(nvme_bytes)
            .build()
            .map_err(|e| {
                PlumeError::Cache(format!(
                    "failed to initialize NVMe cache device at {}: {e}",
                    nvme_path.display()
                ))
            })?;

        let hybrid = HybridCacheBuilder::new()
            .with_name("plume-search-cache")
            // Persist entries on insert so the NVMe tier remains warm across restarts.
            .with_policy(HybridCachePolicy::WriteOnInsertion)
            .memory(ram_bytes)
            .with_weighter(|_key: &CacheKey, value: &CacheValue| {
                std::mem::size_of::<CacheKey>() + estimated_results_size(&value.results)
            })
            .storage()
            .with_engine_config(BlockEngineConfig::new(device))
            .build()
            .await
            .map_err(|e| PlumeError::Cache(format!("failed to build hybrid cache: {e}")))?;

        info!(
            ram_mb = config.ram_capacity_mb,
            nvme_gb = config.nvme_capacity_gb,
            nvme_path = %nvme_path.display(),
            "search cache initialized"
        );

        Ok(Self {
            hybrid,
            generations: GenerationCounter::new(),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Look up cached results for a query.
    pub async fn get(
        &self,
        namespace: &str,
        query_hash: u64,
    ) -> Result<Option<Vec<SearchResult>>, PlumeError> {
        let generation = self.generations.get(namespace);
        let key = CacheKey {
            namespace: namespace.to_string(),
            generation,
            query_hash,
        };

        match self.hybrid.get(&key).await {
            Ok(Some(entry)) => {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(Some(entry.value().results.clone()))
            }
            Ok(None) => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            }
            Err(e) => Err(PlumeError::Cache(format!("cache lookup failed: {e}"))),
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

        self.hybrid.insert(key, CacheValue { results });
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

    /// Close the cache and flush staged entries to disk.
    pub async fn close(&self) -> Result<(), PlumeError> {
        self.hybrid
            .close()
            .await
            .map_err(|e| PlumeError::Cache(format!("failed to close hybrid cache: {e}")))
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

fn estimated_results_size(results: &[SearchResult]) -> usize {
    results
        .iter()
        .map(|result| {
            result.id.len()
                + result.text.len()
                + result
                    .metadata
                    .iter()
                    .map(|(k, v)| k.len() + v.to_string().len())
                    .sum::<usize>()
                + std::mem::size_of::<SearchResult>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use uuid::Uuid;

    fn sample_results() -> Vec<SearchResult> {
        vec![SearchResult {
            id: "1".to_string(),
            text: "retry with exponential backoff".to_string(),
            score: 1.0,
            metadata: HashMap::new(),
        }]
    }

    fn test_config() -> CacheConfig {
        let path = std::env::temp_dir().join(format!("plume-cache-test-{}", Uuid::new_v4()));
        CacheConfig {
            ram_capacity_mb: 8,
            nvme_capacity_gb: 1,
            nvme_path: path.to_string_lossy().to_string(),
        }
    }

    #[tokio::test]
    async fn returns_cached_results() {
        let cache = SearchCache::new(&test_config()).await.unwrap();
        let query_hash = hash_query("retry", "semantic");

        cache.insert("code", query_hash, sample_results());

        let cached = cache.get("code", query_hash).await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[tokio::test]
    async fn generation_invalidation_skips_stale_entries() {
        let cache = SearchCache::new(&test_config()).await.unwrap();
        let query_hash = hash_query("retry", "semantic");

        cache.insert("code", query_hash, sample_results());
        cache.invalidate("code");

        let cached = cache.get("code", query_hash).await.unwrap();
        assert!(cached.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[tokio::test]
    async fn persists_entries_to_disk_across_reopen() {
        let config = test_config();
        let query_hash = hash_query("retry", "semantic");

        let cache = SearchCache::new(&config).await.unwrap();
        cache.insert("code", query_hash, sample_results());
        cache.close().await.unwrap();

        let reopened = SearchCache::new(&config).await.unwrap();
        let cached = reopened.get("code", query_hash).await.unwrap();
        assert!(cached.is_some());
    }
}
