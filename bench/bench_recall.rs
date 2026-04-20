//! Recall benchmark using BEIR SciFact with a real ColBERT encoder.
//!
//! Measures ANN candidate retrieval recall@k against exact MaxSim,
//! using real 128-dim ColBERT embeddings on the SciFact scientific
//! fact-checking corpus (5,183 docs, 300 test queries with relevance
//! judgments).
//!
//! Prerequisites:
//!   1. Download the model:  ./scripts/download-model.sh models/lateon-code-edge
//!   2. Download SciFact:    ./scripts/download-scifact.sh
//!
//! Usage:
//!   PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-recall \
//!     --features plume-encoder/onnx
//!
//! Optional env vars:
//!   PLUME_BENCH_MODEL=models/lateon-code-edge
//!   PLUME_BENCH_DATA=data/scifact
//!   PLUME_BENCH_K=10
//!   PLUME_BENCH_PARTITIONS=64,128,256
//!   PLUME_BENCH_NPROBES=4,8,16,32
//!   PLUME_BENCH_CANDIDATES=3,5,10,20
//!   PLUME_BENCH_REFINE=none,2
//!   PLUME_BENCH_NBITS=4         (PQ bits per sub-quantizer; 4 or 8)

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use plume_cache::SearchCache;
use plume_core::config::{CacheConfig, IndexConfig, StorageConfig};
use plume_core::types::{MultiVector, SearchResult};
use plume_encoder::build_encoder;
use plume_index::{IndexManager, NamespaceTable};
use plume_search::maxsim_score;

// --- Dataset loading ---

struct CorpusDoc {
    id: String,
    text: String,
}

struct Qrel {
    query_id: String,
    corpus_id: String,
    score: i32,
}

fn load_corpus(path: &str) -> Result<Vec<CorpusDoc>> {
    let file = std::fs::File::open(path).with_context(|| format!("cannot open {path}"))?;
    let reader = std::io::BufReader::new(file);
    let mut docs = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let id = v["_id"].as_str().unwrap_or("").to_string();
        let title = v["title"].as_str().unwrap_or("");
        let text = v["text"].as_str().unwrap_or("");
        // Combine title + text as the document content
        let combined = if title.is_empty() {
            text.to_string()
        } else {
            format!("{title}. {text}")
        };
        docs.push(CorpusDoc { id, text: combined });
    }
    Ok(docs)
}

fn load_queries(path: &str) -> Result<HashMap<String, String>> {
    let file = std::fs::File::open(path).with_context(|| format!("cannot open {path}"))?;
    let reader = std::io::BufReader::new(file);
    let mut queries = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let id = v["_id"].as_str().unwrap_or("").to_string();
        let text = v["text"].as_str().unwrap_or("").to_string();
        queries.insert(id, text);
    }
    Ok(queries)
}

fn load_qrels(path: &str) -> Result<Vec<Qrel>> {
    let file = std::fs::File::open(path).with_context(|| format!("cannot open {path}"))?;
    let reader = std::io::BufReader::new(file);
    let mut qrels = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("query-id") {
            continue; // skip header
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            qrels.push(Qrel {
                query_id: parts[0].to_string(),
                corpus_id: parts[1].to_string(),
                score: parts[2].parse().unwrap_or(0),
            });
        }
    }
    Ok(qrels)
}

/// Build a map: query_id -> set of relevant corpus_ids (score > 0).
fn relevance_map(qrels: &[Qrel]) -> HashMap<String, HashSet<String>> {
    let mut map: HashMap<String, HashSet<String>> = HashMap::new();
    for qrel in qrels {
        if qrel.score > 0 {
            map.entry(qrel.query_id.clone())
                .or_default()
                .insert(qrel.corpus_id.clone());
        }
    }
    map
}

// --- Env parsing (reuse patterns from bench_ann) ---

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .with_context(|| format!("failed to parse {name}={v}")),
        Err(_) => Ok(default),
    }
}

fn env_usize_list(name: &str, default: &[usize]) -> Result<Vec<usize>> {
    match std::env::var(name) {
        Ok(v) => v
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .with_context(|| format!("failed to parse {name} item '{s}'"))
            })
            .collect(),
        Err(_) => Ok(default.to_vec()),
    }
}

fn env_optional_u32_list(name: &str, default: &[Option<u32>]) -> Result<Vec<Option<u32>>> {
    match std::env::var(name) {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim().to_ascii_lowercase();
                if s == "auto" || s == "none" {
                    Ok(None)
                } else {
                    s.parse()
                        .map(Some)
                        .with_context(|| format!("failed to parse {name} item '{s}'"))
                }
            })
            .collect(),
        Err(_) => Ok(default.to_vec()),
    }
}

// --- Search helpers ---

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
    let exact_ids: HashSet<&str> = exact.iter().map(|r| r.id.as_str()).collect();
    let hits = approx
        .iter()
        .filter(|r| exact_ids.contains(r.id.as_str()))
        .count();
    hits as f64 / exact.len().max(1) as f64
}

/// End-to-end recall: how many ground-truth relevant docs appear in the top-k results.
fn e2e_recall(results: &[SearchResult], relevant: &HashSet<String>, k: usize) -> f64 {
    let top_k_ids: HashSet<&str> = results.iter().take(k).map(|r| r.id.as_str()).collect();
    let hits = relevant
        .iter()
        .filter(|id| top_k_ids.contains(id.as_str()))
        .count();
    hits as f64 / relevant.len().max(1) as f64
}

fn percentile_ms(samples: &mut [f64], pct: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((samples.len().saturating_sub(1)) as f64 * pct).round() as usize;
    samples[idx]
}

// --- Main ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let model_dir = env_str("PLUME_BENCH_MODEL", "models/lateon-code-edge");
    let data_dir = env_str("PLUME_BENCH_DATA", "data/scifact");
    let k = env_usize("PLUME_BENCH_K", 10)?;
    let partitions = env_optional_u32_list("PLUME_BENCH_PARTITIONS", &[Some(64), Some(128)])?;
    let nprobes = env_usize_list("PLUME_BENCH_NPROBES", &[4, 8, 16, 32])?;
    let candidates = env_usize_list("PLUME_BENCH_CANDIDATES", &[3, 5, 10, 20])?;
    let refine_factors = env_optional_u32_list("PLUME_BENCH_REFINE", &[None, Some(2)])?;
    let pool_factor = env_usize("PLUME_BENCH_POOL_FACTOR", 2)?;
    let nbits = env_usize("PLUME_BENCH_NBITS", 4)? as u32;

    // --- Load SciFact ---
    let corpus_path = format!("{data_dir}/corpus.jsonl");
    let queries_path = format!("{data_dir}/queries.jsonl");
    let qrels_path = format!("{data_dir}/qrels/test.tsv");

    println!("Loading SciFact from {data_dir}...");
    let corpus = load_corpus(&corpus_path)?;
    let all_queries = load_queries(&queries_path)?;
    let qrels = load_qrels(&qrels_path)?;
    let relevance = relevance_map(&qrels);

    // Only benchmark queries that have relevance judgments
    let test_queries: Vec<(String, String)> = relevance
        .keys()
        .filter_map(|qid| all_queries.get(qid).map(|text| (qid.clone(), text.clone())))
        .collect();

    println!(
        "Corpus: {} docs, Queries: {} (with qrels), k: {}",
        corpus.len(),
        test_queries.len(),
        k,
    );

    // --- Build encoder ---
    let encoder_config = plume_core::config::EncoderConfig {
        model: model_dir.clone(),
        pool_factor,
        batch_size: 32,
    };
    let encoder = build_encoder(&encoder_config);

    // Quick check: encode one doc to verify the encoder works
    let sample = encoder.encode_single(&corpus[0].text)?;
    println!(
        "Encoder: {} tokens/doc (sample), {} dims",
        sample.len(),
        sample[0].len(),
    );

    println!();
    println!("partitions,nprobes,candidates,refine,index_ms,exact_avg_ms,ann_avg_ms,ann_p95_ms,min_ann_recall,avg_ann_recall,e2e_exact_recall,e2e_ann_recall");

    for partition_count in &partitions {
        let label = partition_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "auto".to_string());

        let store_dir = format!("/tmp/plume-recall-bench-{label}");
        let cache_dir = format!("{store_dir}-cache");
        let _ = std::fs::remove_dir_all(&store_dir);
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&store_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        let storage_config = StorageConfig {
            uri: store_dir.clone(),
            region: None,
            endpoint: None,
        };
        let cache_config = CacheConfig {
            ram_capacity_mb: 256,
            nvme_capacity_gb: 1,
            nvme_path: cache_dir,
        };
        let index_manager = IndexManager::connect(&storage_config).await?;
        let cache = Arc::new(SearchCache::new(&cache_config).await?);
        let namespace = "scifact";
        let _table = index_manager.namespace(namespace).await?;

        // --- Ingest corpus ---
        println!("Ingesting {} docs (partitions={label})...", corpus.len());
        let ingest_start = Instant::now();
        let batch_size = 64;
        for chunk in corpus.chunks(batch_size) {
            let ids: Vec<String> = chunk.iter().map(|d| d.id.clone()).collect();
            let texts: Vec<String> = chunk.iter().map(|d| d.text.clone()).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let multivectors = encoder.encode_batch(&text_refs)?;
            let metadata: Vec<HashMap<String, serde_json::Value>> =
                chunk.iter().map(|_| HashMap::new()).collect();
            let table = index_manager.namespace(namespace).await?;
            table.upsert(&ids, &texts, &multivectors, &metadata).await?;
        }
        let ingest_ms = ingest_start.elapsed().as_secs_f64() * 1000.0;
        println!("  Ingested in {:.1}s", ingest_ms / 1000.0);

        // --- Build ANN index ---
        let index_config = IndexConfig {
            nbits,
            num_partitions: *partition_count,
            nprobes: 32,
            refine_factor: None,
            ann_candidate_multiplier: 20,
            max_candidates: 10_000,
            auto: Default::default(),
        };
        let table = index_manager.namespace(namespace).await?;
        let index_start = Instant::now();
        table.build_vector_index(&index_config).await?;
        let index_ms = index_start.elapsed().as_secs_f64() * 1000.0;
        println!("  Index built in {:.1}s", index_ms / 1000.0);
        let total_docs = table.count().await?;

        // --- Encode queries and compute exact baselines ---
        println!(
            "  Computing exact baselines for {} queries...",
            test_queries.len()
        );
        let mut baselines: Vec<(String, MultiVector, Vec<SearchResult>)> = Vec::new();
        let mut exact_latencies_ms = Vec::new();

        for (qid, qtext) in &test_queries {
            let qvecs = encoder.encode_single(qtext)?;
            let start = Instant::now();
            let exact = exact_semantic_search(&table, &qvecs, total_docs, k).await?;
            exact_latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            baselines.push((qid.clone(), qvecs, exact));
        }
        let exact_avg_ms =
            exact_latencies_ms.iter().sum::<f64>() / exact_latencies_ms.len().max(1) as f64;
        println!("  Exact baseline avg: {:.1}ms/query", exact_avg_ms);

        // --- Sweep ANN parameters ---
        for &probe_count in &nprobes {
            for &candidate_mult in &candidates {
                for &refine in &refine_factors {
                    let mut ann_recalls = Vec::new();
                    let mut latencies_ms = Vec::new();
                    let mut e2e_exact_recalls = Vec::new();
                    let mut e2e_ann_recalls = Vec::new();

                    for (qid, qvecs, exact_results) in &baselines {
                        let start = Instant::now();
                        let approx = ann_semantic_search(
                            &table,
                            qvecs,
                            k,
                            candidate_mult,
                            probe_count,
                            refine,
                        )
                        .await?;
                        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);

                        // ANN recall vs exact MaxSim (measures approximation quality)
                        ann_recalls.push(recall_at_k(exact_results, &approx));

                        // End-to-end recall vs ground truth qrels
                        if let Some(relevant) = relevance.get(qid) {
                            e2e_exact_recalls.push(e2e_recall(exact_results, relevant, k));
                            e2e_ann_recalls.push(e2e_recall(&approx, relevant, k));
                        }
                    }

                    let avg_ann_recall =
                        ann_recalls.iter().sum::<f64>() / ann_recalls.len().max(1) as f64;
                    let min_ann_recall = ann_recalls.iter().cloned().fold(f64::INFINITY, f64::min);
                    let avg_latency_ms =
                        latencies_ms.iter().sum::<f64>() / latencies_ms.len().max(1) as f64;
                    let p95_latency_ms = percentile_ms(&mut latencies_ms, 0.95);
                    let avg_e2e_exact = e2e_exact_recalls.iter().sum::<f64>()
                        / e2e_exact_recalls.len().max(1) as f64;
                    let avg_e2e_ann =
                        e2e_ann_recalls.iter().sum::<f64>() / e2e_ann_recalls.len().max(1) as f64;

                    println!(
                        "{},{},{},{},{:.0},{:.1},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3}",
                        label,
                        probe_count,
                        candidate_mult,
                        refine
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "none".to_string()),
                        index_ms,
                        exact_avg_ms,
                        avg_latency_ms,
                        p95_latency_ms,
                        min_ann_recall,
                        avg_ann_recall,
                        avg_e2e_exact,
                        avg_e2e_ann,
                    );
                }
            }
        }

        cache.close().await?;
        index_manager.drop_namespace(namespace).await?;
        let _ = std::fs::remove_dir_all(&store_dir);
    }

    println!();
    println!("Done. Columns:");
    println!("  min/avg_ann_recall = ANN recall vs exact MaxSim (measures approximation quality)");
    println!("  e2e_exact_recall   = exact MaxSim recall vs ground-truth qrels");
    println!("  e2e_ann_recall     = ANN recall vs ground-truth qrels");

    Ok(())
}
