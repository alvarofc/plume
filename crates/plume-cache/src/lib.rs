mod generation;

use std::hash::{Hash, Hasher};
use std::path::Path;

use foyer::{
    BlockEngineConfig, Cache, CacheBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache,
    HybridCacheBuilder, HybridCachePolicy,
};
use plume_core::config::CacheConfig;
use plume_core::error::PlumeError;
use plume_core::types::SearchResult;
use serde::{Deserialize, Serialize};
use tracing::info;

pub use generation::GenerationCounter;

/// Dedicated subdirectory under `CacheConfig::nvme_path` that plume owns and
/// wipes on startup. Keeping wipes scoped to this subdir means a
/// misconfigured parent path cannot take out unrelated files.
const CACHE_SUBDIR: &str = "plume-search-cache";

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
///
/// When `nvme_capacity_gb == 0`, the NVMe tier is skipped entirely and
/// only the in-memory tier is used. This keeps plume runnable on
/// read-only filesystems or ephemeral containers where no writable
/// cache directory is available.
///
/// The NVMe tier is cleared on every startup. `GenerationCounter` lives
/// in-memory, so persisted entries from a prior run would share keys
/// (namespace, generation=0) with fresh post-restart writes and could be
/// served as stale hits after the first `invalidate`. Wiping the tier at
/// boot trades warm-start latency for correctness.
pub struct SearchCache {
    backend: Backend,
    /// Generation counters per namespace
    generations: GenerationCounter,
    /// Application-level counters for cache lookups
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

enum Backend {
    Hybrid(HybridCache<CacheKey, CacheValue>),
    Memory(Cache<CacheKey, CacheValue>),
}

impl SearchCache {
    /// Create a new search cache. Uses a RAM + NVMe hybrid cache when
    /// `nvme_capacity_gb > 0`, or an in-memory-only cache otherwise.
    pub async fn new(config: &CacheConfig) -> Result<Self, PlumeError> {
        let ram_bytes = config.ram_capacity_mb * 1024 * 1024;

        if config.nvme_capacity_gb == 0 {
            let cache = CacheBuilder::new(ram_bytes)
                .with_name("plume-search-cache")
                .with_weighter(|_key: &CacheKey, value: &CacheValue| {
                    std::mem::size_of::<CacheKey>() + estimated_results_size(&value.results)
                })
                .build();

            info!(
                ram_mb = config.ram_capacity_mb,
                "search cache initialized (memory-only, NVMe tier disabled)"
            );

            return Ok(Self {
                backend: Backend::Memory(cache),
                generations: GenerationCounter::new(),
                hits: std::sync::atomic::AtomicU64::new(0),
                misses: std::sync::atomic::AtomicU64::new(0),
            });
        }

        let nvme_bytes = config.nvme_capacity_gb * 1024 * 1024 * 1024;
        let nvme_root = Path::new(&config.nvme_path);

        // Own a dedicated subdirectory under the configured path. We wipe
        // this on startup (generation counters are in-memory, so stale
        // persisted entries could collide with fresh keys after restart),
        // and we never touch anything outside it — a misconfigured
        // nvme_path (e.g. pointing at a shared directory) cannot cause
        // plume to recursively delete unrelated data.
        let cache_dir = nvme_root.join(CACHE_SUBDIR);

        std::fs::create_dir_all(nvme_root).map_err(|e| {
            PlumeError::Cache(format!(
                "failed to create cache parent {}: {e}",
                nvme_root.display()
            ))
        })?;

        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir).map_err(|e| {
                PlumeError::Cache(format!(
                    "failed to clear cache directory {}: {e}",
                    cache_dir.display()
                ))
            })?;
        }

        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            PlumeError::Cache(format!(
                "failed to create cache directory {}: {e}",
                cache_dir.display()
            ))
        })?;

        let device = FsDeviceBuilder::new(&cache_dir)
            .with_capacity(nvme_bytes)
            .build()
            .map_err(|e| {
                PlumeError::Cache(format!(
                    "failed to initialize NVMe cache device at {}: {e}",
                    cache_dir.display()
                ))
            })?;

        let hybrid = HybridCacheBuilder::new()
            .with_name("plume-search-cache")
            // Persist entries on insert so the NVMe tier absorbs hot RAM evictions.
            // The tier is wiped on startup (see above), so this only buys intra-run warmth.
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
            nvme_path = %cache_dir.display(),
            "search cache initialized"
        );

        Ok(Self {
            backend: Backend::Hybrid(hybrid),
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

        let found = match &self.backend {
            Backend::Hybrid(hybrid) => hybrid
                .get(&key)
                .await
                .map_err(|e| PlumeError::Cache(format!("cache lookup failed: {e}")))?
                .map(|entry| entry.value().results.clone()),
            Backend::Memory(cache) => cache.get(&key).map(|entry| entry.value().results.clone()),
        };

        if found.is_some() {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(found)
    }

    /// Insert search results into the cache.
    pub fn insert(&self, namespace: &str, query_hash: u64, results: Vec<SearchResult>) {
        let generation = self.generations.get(namespace);
        let key = CacheKey {
            namespace: namespace.to_string(),
            generation,
            query_hash,
        };

        match &self.backend {
            Backend::Hybrid(hybrid) => {
                hybrid.insert(key, CacheValue { results });
            }
            Backend::Memory(cache) => {
                cache.insert(key, CacheValue { results });
            }
        }
    }

    /// Increment the generation counter for a namespace.
    /// This effectively invalidates all cached results for that namespace.
    pub fn invalidate(&self, namespace: &str) {
        self.generations.increment(namespace);
    }

    /// Mark a namespace as dropped for cache purposes.
    ///
    /// We bump the generation instead of removing the counter: cache
    /// entries written under the old namespace are not actively evicted
    /// (foyer has no prefix-delete). If we reset the counter to 0 and
    /// the namespace was recreated and re-populated, the first write
    /// would set generation back to 1 and start matching those old
    /// entries, serving stale results. Keeping the counter monotonic
    /// guarantees every future generation is strictly greater than any
    /// previously-cached generation, so stale entries can never match.
    pub fn remove_namespace(&self, namespace: &str) {
        self.generations.increment(namespace);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Close the cache and flush staged entries to disk. No-op for memory-only.
    pub async fn close(&self) -> Result<(), PlumeError> {
        match &self.backend {
            Backend::Hybrid(hybrid) => hybrid
                .close()
                .await
                .map_err(|e| PlumeError::Cache(format!("failed to close hybrid cache: {e}"))),
            Backend::Memory(_) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// Hash a query string + search mode + k into a u64 for cache keying.
///
/// `k` is part of the key because the cache stores the truncated top-k
/// result list; reusing a k=5 entry for a k=50 request would silently
/// return fewer rows than the caller asked for.
pub fn hash_query(query: &str, mode: &str, k: usize) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query.hash(&mut hasher);
    mode.hash(&mut hasher);
    k.hash(&mut hasher);
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

    use tempfile::TempDir;

    fn sample_results() -> Vec<SearchResult> {
        vec![SearchResult {
            id: "1".to_string(),
            text: "retry with exponential backoff".to_string(),
            score: 1.0,
            metadata: HashMap::new(),
        }]
    }

    /// Build a cache config anchored on a TempDir so the NVMe tier is
    /// cleaned up automatically when the guard is dropped.
    fn test_config(dir: &TempDir) -> CacheConfig {
        CacheConfig {
            ram_capacity_mb: 8,
            nvme_capacity_gb: 1,
            nvme_path: dir.path().to_string_lossy().to_string(),
        }
    }

    #[tokio::test]
    async fn returns_cached_results() {
        let dir = TempDir::new().unwrap();
        let cache = SearchCache::new(&test_config(&dir)).await.unwrap();
        let query_hash = hash_query("retry", "semantic", 5);

        cache.insert("code", query_hash, sample_results());

        let cached = cache.get("code", query_hash).await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[tokio::test]
    async fn generation_invalidation_skips_stale_entries() {
        let dir = TempDir::new().unwrap();
        let cache = SearchCache::new(&test_config(&dir)).await.unwrap();
        let query_hash = hash_query("retry", "semantic", 5);

        cache.insert("code", query_hash, sample_results());
        cache.invalidate("code");

        let cached = cache.get("code", query_hash).await.unwrap();
        assert!(cached.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[tokio::test]
    async fn memory_only_cache_when_nvme_capacity_is_zero() {
        let config = CacheConfig {
            ram_capacity_mb: 8,
            nvme_capacity_gb: 0,
            nvme_path: "/nonexistent/unwritable".to_string(),
        };
        let cache = SearchCache::new(&config).await.unwrap();
        let query_hash = hash_query("retry", "semantic", 5);

        cache.insert("code", query_hash, sample_results());

        let cached = cache.get("code", query_hash).await.unwrap();
        assert!(cached.is_some());
        // close() should be a no-op for memory-only and never touch disk.
        cache.close().await.unwrap();
    }

    #[tokio::test]
    async fn clears_persisted_entries_on_startup() {
        // Generation counters live in memory, so persisted entries from
        // a previous run would share keys with fresh (gen=0) writes and
        // could serve stale data after the first invalidate. The cache
        // must therefore drop any prior NVMe contents on startup.
        let dir = TempDir::new().unwrap();
        let config = test_config(&dir);
        let query_hash = hash_query("retry", "semantic", 5);

        let cache = SearchCache::new(&config).await.unwrap();
        cache.insert("code", query_hash, sample_results());
        cache.close().await.unwrap();

        let reopened = SearchCache::new(&config).await.unwrap();
        let cached = reopened.get("code", query_hash).await.unwrap();
        assert!(cached.is_none());
        assert_eq!(reopened.stats().misses, 1);
    }
}
