//! Benchmark: cold vs warm query latency.
//!
//! Usage:
//!   PROTOC=/usr/local/opt/protobuf/bin/protoc cargo run --release -p plume-bench --bin bench-latency
//!
//! Measures:
//! - Ingest latency (encoding + LanceDB upsert)
//! - Cold query latency (first query, cache miss)
//! - Warm query latency (repeated query, cache hit)
//! - Cache invalidation + re-query latency

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use plume_cache::SearchCache;
use plume_core::config::{CacheConfig, IndexConfig, StorageConfig};
use plume_core::types::{Document, SearchMode};
use plume_encoder::{Encode, MockEncoder};
use plume_index::IndexManager;
use plume_search::SearchEngine;

fn generate_documents(n: usize) -> Vec<Document> {
    let texts = [
        "function that retries HTTP requests with exponential backoff and jitter",
        "binary search implementation in Rust with generic comparator trait",
        "parse JSON config file and validate schema against OpenAPI spec",
        "WebSocket server with heartbeat ping/pong and automatic reconnection",
        "rate limiter using token bucket algorithm with Redis backend store",
        "concurrent hash map with lock-free reads and sharded write locks",
        "gRPC service definition with streaming responses and deadline propagation",
        "merkle tree implementation for content-addressable storage verification",
        "async channel with backpressure and bounded capacity queue management",
        "B-tree index with prefix compression and page-level locking strategy",
        "TLS certificate rotation with graceful connection draining mechanism",
        "distributed consensus using Raft protocol with log compaction support",
        "memory allocator with thread-local caching and size class segregation",
        "HTTP/2 multiplexed connection pool with health checking and retry logic",
        "bloom filter for approximate set membership with configurable false positive rate",
        "write-ahead log with group commit and fsync batching for durability",
        "columnar storage engine with dictionary encoding and run-length compression",
        "circuit breaker pattern with half-open state and exponential backoff recovery",
        "consistent hashing ring with virtual nodes and minimal key redistribution",
        "lock-free skip list for concurrent sorted key-value storage operations",
    ];

    (0..n)
        .map(|i| Document {
            id: format!("doc-{i:06}"),
            text: texts[i % texts.len()].to_string(),
            metadata: HashMap::new(),
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let doc_counts = [100, 1_000, 10_000];
    let pool_factor = 2;

    println!("Plume Latency Benchmark");
    println!("=======================");
    println!("Encoder: MockEncoder (pool_factor={})", pool_factor);
    println!();

    for &n in &doc_counts {
        println!("--- {} documents ---", n);

        // Fresh LanceDB directory
        let data_dir = format!("/tmp/plume-bench-{n}");
        let _ = std::fs::remove_dir_all(&data_dir);
        std::fs::create_dir_all(&data_dir)?;

        let storage_config = StorageConfig {
            uri: data_dir.clone(),
            region: None,
            endpoint: None,
        };
        let cache_config = CacheConfig {
            ram_capacity_mb: 256,
            nvme_capacity_gb: 1,
            nvme_path: "/tmp/plume-bench-cache".into(),
        };
        let index_config = IndexConfig::default();

        let index = IndexManager::connect(&storage_config).await?;
        let cache = Arc::new(SearchCache::new(&cache_config).await?);
        let engine = SearchEngine::new(Arc::clone(&cache), index_config.clone());
        let encoder = MockEncoder::new(pool_factor);

        let docs = generate_documents(n);

        // --- Ingest benchmark ---
        let start = Instant::now();

        let batch_size = 100;
        for chunk in docs.chunks(batch_size) {
            let ids: Vec<String> = chunk.iter().map(|d| d.id.clone()).collect();
            let texts: Vec<String> = chunk.iter().map(|d| d.text.clone()).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let multivectors = encoder.encode_batch(&text_refs)?;
            let metadata: Vec<HashMap<String, serde_json::Value>> =
                chunk.iter().map(|d| d.metadata.clone()).collect();

            let table = index.namespace("bench").await?;
            table.upsert(&ids, &texts, &multivectors, &metadata).await?;
        }

        let ingest_ms = start.elapsed().as_millis();
        let ingest_per_doc = ingest_ms as f64 / n as f64;
        println!(
            "  Ingest:     {}ms total ({:.2}ms/doc)",
            ingest_ms, ingest_per_doc
        );

        let table = index.namespace("bench").await?;
        let start = Instant::now();
        table.build_vector_index(&index_config).await?;
        let index_build_ms = start.elapsed().as_millis();
        println!("  ANN index:  {}ms build", index_build_ms);

        // --- Cold query benchmark (cache miss) ---
        let query = "retry logic with exponential backoff";
        let query_vectors = encoder.encode_single(query)?;

        let start = Instant::now();
        let result = engine
            .search(&table, &query_vectors, query, 10, SearchMode::Semantic)
            .await?;
        let cold_us = start.elapsed().as_micros();
        assert!(!result.cache_hit, "expected cache miss");
        println!(
            "  Cold query: {}us ({} results, top: id={} score={:.2})",
            cold_us,
            result.results.len(),
            result.results[0].id,
            result.results[0].score,
        );

        // --- Warm query benchmark (cache hit) ---
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let result = engine
                .search(&table, &query_vectors, query, 10, SearchMode::Semantic)
                .await?;
            assert!(result.cache_hit, "expected cache hit");
        }
        let warm_total_us = start.elapsed().as_micros();
        let warm_per_query_ns = (warm_total_us * 1000) / iterations;
        println!(
            "  Warm query: {}ns/query ({}us for {} iterations)",
            warm_per_query_ns, warm_total_us, iterations,
        );

        // --- Cache invalidation + re-query ---
        cache.invalidate("bench");

        let start = Instant::now();
        let result = engine
            .search(&table, &query_vectors, query, 10, SearchMode::Semantic)
            .await?;
        let reinvalidate_us = start.elapsed().as_micros();
        assert!(!result.cache_hit, "expected cache miss after invalidation");
        println!("  Post-invalidation query: {}us", reinvalidate_us,);

        let stats = cache.stats();
        println!(
            "  Cache stats: {} hits, {} misses ({:.1}% hit rate)",
            stats.hits,
            stats.misses,
            100.0 * stats.hits as f64 / (stats.hits + stats.misses) as f64,
        );

        // Cleanup
        index.drop_namespace("bench").await?;
        let _ = std::fs::remove_dir_all(&data_dir);

        println!();
    }

    println!("Benchmark complete.");
    Ok(())
}
