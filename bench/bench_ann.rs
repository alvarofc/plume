//! Benchmark ANN tuning against exact MaxSim on a code-like synthetic corpus.
//!
//! Usage:
//!   PROTOC=/usr/local/opt/protobuf/bin/protoc cargo run --release -p plume-bench --bin bench-ann
//!
//! Optional env vars:
//!   PLUME_BENCH_DOCS=20000
//!   PLUME_BENCH_K=10
//!   PLUME_BENCH_PARTITIONS=auto,64,128
//!   PLUME_BENCH_NPROBES=8,16,32
//!   PLUME_BENCH_CANDIDATES=5,10,20
//!   PLUME_BENCH_REFINE=none,2

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use plume_cache::SearchCache;
use plume_core::config::{CacheConfig, IndexConfig, StorageConfig};
use plume_core::types::{Document, MultiVector, SearchResult};
use plume_encoder::{Encode, MockEncoder};
use plume_index::{IndexManager, NamespaceTable};
use plume_search::maxsim_score;

#[derive(Clone, Debug)]
struct BenchmarkRow {
    partitions: Option<u32>,
    nprobes: usize,
    candidate_multiplier: usize,
    refine_factor: Option<u32>,
    min_recall: f64,
    avg_recall_at_k: f64,
    avg_latency_ms: f64,
    p95_latency_ms: f64,
}

fn generate_documents(n: usize) -> Vec<Document> {
    let languages = ["Rust", "Go", "TypeScript", "Python", "C++", "Java", "Zig", "Elixir"];
    let components = [
        "HTTP retry middleware",
        "LSP symbol indexer",
        "incremental AST cache",
        "vector search worker",
        "raft snapshot pipeline",
        "S3 multipart uploader",
        "NVMe block cache",
        "BM25 ranking path",
        "token pooling stage",
        "schema validator",
        "gRPC streaming proxy",
        "WAL compactor",
        "connection pool manager",
        "rate limiter",
        "distributed lock service",
        "bloom filter index",
    ];
    let operations = [
        "with exponential backoff and jitter",
        "with cancellation and timeout propagation",
        "with bounded concurrency and backpressure",
        "with zero-copy parsing and arena allocation",
        "with page-level locking and optimistic reads",
        "with checkpointing and crash recovery",
        "with adaptive batching and queue draining",
        "with mmap prefetch and compaction",
        "with consistent hashing and rebalancing",
        "with circuit breaker and fallback",
        "with write-ahead logging and fsync",
        "with streaming deserialization and pipelining",
    ];
    let data_shapes = [
        "for code search over repository chunks",
        "for namespace-scoped document ingestion",
        "for late-interaction ANN candidate retrieval",
        "for full-text fusion and reranking",
        "for hybrid semantic and lexical ranking",
        "for multi-tenant metadata filtering",
        "for columnar scan and projection pushdown",
        "for incremental materialized view refresh",
    ];
    let qualifiers = [
        "production-hardened",
        "experimental",
        "high-throughput",
        "low-latency",
        "memory-efficient",
        "lock-free",
        "distributed",
        "single-node",
        "sharded",
        "replicated",
    ];

    // 8 * 16 * 12 * 8 * 10 = 122,880 unique combinations
    (0..n)
        .map(|i| {
            let language = languages[i % languages.len()];
            let component = components[(i / languages.len()) % components.len()];
            let operation = operations[(i / (languages.len() * components.len())) % operations.len()];
            let shape = data_shapes[(i / (languages.len() * components.len() * operations.len())) % data_shapes.len()];
            let qualifier = qualifiers[(i / (languages.len() * components.len() * operations.len() * data_shapes.len())) % qualifiers.len()];

            Document {
                id: format!("doc-{i:06}"),
                text: format!(
                    "{qualifier} {language} implementation of {component} {operation} {shape}; build_{:04}; variant_{i:06}",
                    i % 9973, // large prime avoids short cycles
                ),
                metadata: HashMap::new(),
            }
        })
        .collect()
}

fn benchmark_queries() -> Vec<&'static str> {
    vec![
        // Specific matches
        "retry middleware with exponential backoff and jitter",
        "nvme cache with mmap prefetch and compaction",
        "raft snapshot pipeline with crash recovery",
        "s3 multipart uploader with adaptive batching",
        // Cross-component queries (harder — similar ANN vectors, different MaxSim scores)
        "distributed lock with circuit breaker fallback",
        "sharded bloom filter for multi-tenant filtering",
        "low-latency gRPC proxy with streaming deserialization",
        "high-throughput WAL compactor with write-ahead logging",
        "connection pool with consistent hashing and rebalancing",
        "lock-free rate limiter with bounded concurrency",
        // Broad conceptual queries
        "memory efficient indexing for code search",
        "production hardened vector retrieval system",
        "replicated schema validation with pipelining",
        "incremental materialized view with optimistic reads",
        "experimental columnar scan and projection pushdown",
        "single node full text fusion and reranking",
    ]
}

fn parse_usize_env(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .with_context(|| format!("failed to parse {name}={value} as usize")),
        Err(_) => Ok(default),
    }
}

fn parse_usize_list_env(name: &str, default: &[usize]) -> Result<Vec<usize>> {
    match std::env::var(name) {
        Ok(value) => value
            .split(',')
            .map(|item| {
                item.trim()
                    .parse()
                    .with_context(|| format!("failed to parse {name} item '{item}' as usize"))
            })
            .collect(),
        Err(_) => Ok(default.to_vec()),
    }
}

fn parse_optional_u32_list_env(name: &str, default: &[Option<u32>]) -> Result<Vec<Option<u32>>> {
    match std::env::var(name) {
        Ok(value) => value
            .split(',')
            .map(|item| {
                let item = item.trim().to_ascii_lowercase();
                if item == "auto" || item == "none" {
                    Ok(None)
                } else {
                    item.parse()
                        .map(Some)
                        .with_context(|| format!("failed to parse {name} item '{item}' as u32"))
                }
            })
            .collect(),
        Err(_) => Ok(default.to_vec()),
    }
}

async fn ingest_documents(
    index: &IndexManager,
    namespace: &str,
    docs: &[Document],
    encoder: &impl Encode,
) -> Result<()> {
    let batch_size = 256;

    for chunk in docs.chunks(batch_size) {
        let ids: Vec<String> = chunk.iter().map(|doc| doc.id.clone()).collect();
        let texts: Vec<String> = chunk.iter().map(|doc| doc.text.clone()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|text| text.as_str()).collect();
        let multivectors = encoder.encode_batch(&text_refs)?;
        let metadata: Vec<HashMap<String, serde_json::Value>> =
            chunk.iter().map(|doc| doc.metadata.clone()).collect();

        let table = index.namespace(namespace).await?;
        table.upsert(&ids, &texts, &multivectors, &metadata).await?;
    }

    Ok(())
}

async fn exact_semantic_search(
    table: &NamespaceTable,
    query_vectors: &MultiVector,
    total_docs: usize,
    k: usize,
) -> Result<Vec<SearchResult>> {
    let mut scored: Vec<_> = table
        .scan_with_vectors(total_docs)
        .await?
        .into_iter()
        .map(|(mut result, doc_vectors)| {
            result.score = maxsim_score(query_vectors, &doc_vectors);
            result
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

async fn ann_semantic_search(
    table: &NamespaceTable,
    query_vectors: &MultiVector,
    k: usize,
    candidate_multiplier: usize,
    nprobes: usize,
    refine_factor: Option<u32>,
) -> Result<Vec<SearchResult>> {
    let candidate_limit = candidate_multiplier.saturating_mul(k).max(k);

    let mut scored: Vec<_> = table
        .ann_search_with_vectors(query_vectors, candidate_limit, nprobes, refine_factor)
        .await?
        .into_iter()
        .map(|(mut result, doc_vectors)| {
            result.score = maxsim_score(query_vectors, &doc_vectors);
            result
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

fn recall_at_k(exact: &[SearchResult], approx: &[SearchResult]) -> f64 {
    let exact_ids: HashSet<&str> = exact.iter().map(|result| result.id.as_str()).collect();
    let hits = approx
        .iter()
        .filter(|result| exact_ids.contains(result.id.as_str()))
        .count();

    hits as f64 / exact.len().max(1) as f64
}

fn percentile_ms(samples_ms: &mut [f64], percentile: f64) -> f64 {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let index = ((samples_ms.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    samples_ms[index]
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let docs = parse_usize_env("PLUME_BENCH_DOCS", 20_000)?;
    let k = parse_usize_env("PLUME_BENCH_K", 10)?;
    let partitions =
        parse_optional_u32_list_env("PLUME_BENCH_PARTITIONS", &[Some(128), Some(256)])?;
    let nprobes = parse_usize_list_env("PLUME_BENCH_NPROBES", &[4, 8, 16, 32])?;
    let candidate_multipliers = parse_usize_list_env("PLUME_BENCH_CANDIDATES", &[3, 5, 10, 20])?;
    let refine_factors = parse_optional_u32_list_env("PLUME_BENCH_REFINE", &[None, Some(2)])?;

    let encoder = MockEncoder::new(2);
    let corpus = generate_documents(docs);
    let queries = benchmark_queries();
    let namespace = "bench";

    println!("Plume ANN Benchmark");
    println!("===================");
    println!("Corpus: synthetic code-like namespace with {} docs", docs);
    println!("Queries: {}", queries.len());
    println!("k: {}", k);
    println!();
    println!(
        "partitions,nprobes,candidate_multiplier,refine_factor,index_build_ms,exact_avg_ms,ann_avg_ms,ann_p95_ms,min_recall,avg_recall_at_k"
    );

    let mut rows = Vec::new();

    for partition_count in partitions {
        let label = partition_count
            .map(|value| value.to_string())
            .unwrap_or_else(|| "auto".to_string());
        let data_dir = format!("/tmp/plume-ann-bench-{docs}-{label}");
        let cache_dir = format!("{data_dir}-cache");
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        let storage_config = StorageConfig {
            uri: data_dir.clone(),
            region: None,
            endpoint: None,
        };
        let cache_config = CacheConfig {
            ram_capacity_mb: 64,
            nvme_capacity_gb: 1,
            nvme_path: cache_dir,
        };
        let index_manager = IndexManager::connect(&storage_config).await?;
        let cache = Arc::new(SearchCache::new(&cache_config).await?);
        let table = index_manager.namespace(namespace).await?;

        ingest_documents(&index_manager, namespace, &corpus, &encoder).await?;

        let index_config = IndexConfig {
            nbits: 4,
            num_partitions: partition_count,
            nprobes: 32,
            refine_factor: None,
            ann_candidate_multiplier: 20,
            max_candidates: 10_000,
        };

        let index_start = Instant::now();
        table.build_vector_index(&index_config).await?;
        let index_build_ms = index_start.elapsed().as_secs_f64() * 1000.0;
        let total_docs = table.count().await?;

        let mut baselines = Vec::new();
        let mut exact_latencies_ms = Vec::new();
        for query in &queries {
            let query_vectors = encoder.encode_single(query)?;
            let start = Instant::now();
            let exact = exact_semantic_search(&table, &query_vectors, total_docs, k).await?;
            exact_latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            baselines.push((query_vectors, exact));
        }
        let exact_avg_ms =
            exact_latencies_ms.iter().sum::<f64>() / exact_latencies_ms.len().max(1) as f64;

        for &probe_count in &nprobes {
            for &candidate_multiplier in &candidate_multipliers {
                for &refine_factor in &refine_factors {
                    let mut recalls = Vec::new();
                    let mut latencies_ms = Vec::new();

                    for (query_vectors, exact) in &baselines {
                        let start = Instant::now();
                        let approx = ann_semantic_search(
                            &table,
                            query_vectors,
                            k,
                            candidate_multiplier,
                            probe_count,
                            refine_factor,
                        )
                        .await?;
                        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                        recalls.push(recall_at_k(exact, &approx));
                    }

                    let avg_recall_at_k = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
                    let min_recall = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
                    let avg_latency_ms =
                        latencies_ms.iter().sum::<f64>() / latencies_ms.len().max(1) as f64;
                    let p95_latency_ms = percentile_ms(&mut latencies_ms, 0.95);

                    println!(
                        "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3}",
                        label,
                        probe_count,
                        candidate_multiplier,
                        refine_factor
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "none".to_string()),
                        index_build_ms,
                        exact_avg_ms,
                        avg_latency_ms,
                        p95_latency_ms,
                        min_recall,
                        avg_recall_at_k,
                    );

                    rows.push(BenchmarkRow {
                        partitions: partition_count,
                        nprobes: probe_count,
                        candidate_multiplier,
                        refine_factor,
                        min_recall,
                        avg_recall_at_k,
                        avg_latency_ms,
                        p95_latency_ms,
                    });
                }
            }
        }

        cache.close().await?;
        index_manager.drop_namespace(namespace).await?;
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    rows.sort_by(|a, b| {
        b.avg_recall_at_k
            .partial_cmp(&a.avg_recall_at_k)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(Ordering::Equal)
            })
    });

    if let Some(best) = rows
        .iter()
        .find(|row| row.avg_recall_at_k >= 0.95)
        .or_else(|| rows.first())
    {
        println!();
        println!("Suggested profile");
        println!("-----------------");
        println!(
            "partitions={}, nprobes={}, candidate_multiplier={}, refine_factor={}, recall@{}={:.3}, avg_latency_ms={:.2}, p95_ms={:.2}",
            best.partitions
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            best.nprobes,
            best.candidate_multiplier,
            best.refine_factor
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            k,
            best.avg_recall_at_k,
            best.avg_latency_ms,
            best.p95_latency_ms,
        );
    }

    Ok(())
}
