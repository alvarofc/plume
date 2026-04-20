# Plume

Multi-vector semantic search engine with tiered caching. Combines ColBERT late-interaction retrieval (ANN candidate generation + exact MaxSim re-rank) with BM25 full-text search, backed by LanceDB for storage on S3, R2, GCS, or local disk.

## Architecture

```
                   ┌──────────────┐
  HTTP API ───────▶│  plume-api   │
                   └──────┬───────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐ ┌───────────┐ ┌──────────┐
      │  encoder │ │   search  │ │   cache  │
      └────┬─────┘ └─────┬─────┘ └────┬─────┘
           │             │             │
           │        ┌────▼────┐   RAM (LFU)
           │        │  index  │   NVMe tier
           │        └────┬────┘
           │             │
           ▼             ▼
        ONNX         LanceDB
       Runtime      (S3/R2/GCS/local)
```

**Crates:**

| Crate | Purpose |
|-------|---------|
| `plume-core` | Shared types, config, errors |
| `plume-encoder` | ColBERT encoding (ONNX or mock) |
| `plume-cache` | Tiered RAM/NVMe cache with generation-based invalidation |
| `plume-index` | LanceDB table management, ANN candidate retrieval, index build |
| `plume-search` | MaxSim scoring, ANN re-rank orchestration, BM25 FTS, RRF hybrid fusion |
| `plume-api` | Axum HTTP server and routes |

## Quickstart

### Build profiles

Plume now defaults to a **lean local build**:

- Local filesystem storage works out of the box.
- Cloud backends are opt-in at compile time so local development and tests do not pull in the full AWS/GCS stack.

Feature flags:

| Feature | Enables | Typical use |
|---------|---------|-------------|
| default | Local filesystem LanceDB only | Fast local development and tests |
| `storage-aws` | S3-compatible backends, including MinIO and Cloudflare R2 | Docker Compose, S3, R2 |
| `storage-gcs` | Google Cloud Storage | GCS deployments |

Examples:

```bash
# Lean local binary
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume

# S3 / MinIO / R2-enabled binary
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features storage-aws

# GCS-enabled binary
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features storage-gcs
```

### Local (filesystem storage)

```bash
# Build
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume

# Run with local config
PLUME_CONFIG=config.local.toml ./target/release/plume
```

### Docker Compose (MinIO S3)

```bash
docker compose up
```

This starts Plume on port 8787, MinIO on port 9000 (console on 9001), and auto-creates the `plume` bucket.
The Docker build uses the `storage-aws` feature because MinIO is S3-compatible storage.

### ONNX encoder (optional)

By default Plume uses a mock encoder for development. To use the real ColBERT model:

```bash
# Download model files (~70MB)
./scripts/download-model.sh models/lateon-code-edge

# Build with ONNX support
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features plume-encoder/onnx

# Build with ONNX + S3/MinIO support
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features plume-encoder/onnx,storage-aws

# Point config to model directory
# [encoder]
# model = "models/lateon-code-edge"
```

## CLI

The `plume` binary doubles as a CLI for a running server. It picks up
`PLUME_URL` from the environment (default `http://localhost:8787`).

```bash
# Start the server (also the default when no subcommand is given)
plume serve

# Ingest a JSONL file or a directory of .md files into a namespace
plume ingest ./docs --namespace docs
plume ingest ./corpus.jsonl -n code

# Or point directly at an S3 / GCS bucket (requires storage-aws / storage-gcs)
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
plume ingest s3://my-bucket/corpus -n corpus
plume ingest s3://my-bucket/corpus -n corpus --extensions md,txt,rst

# Semantic grep — local or remote
plume grep "rate limiting" ./src
plume grep "connection timeout" s3://my-bucket/runbooks

# Query via the daemon directly
plume search "exponential backoff" -n code -k 5
plume search "backoff retry logic" -n code --mode semantic --json

# Namespace management
plume ns list
plume ns create notes
plume ns delete notes

# Force a manual index rebuild (auto-index does this in the background
# after each upsert; this is only needed for bulk-load workflows)
plume index code
plume index code --fts-only
```

`plume ingest <dir>` walks the directory, picks up every file matching
`--extensions` (default `md,markdown,txt,rst,org`), and uses the relative
path (including the extension) as the document id, so re-running overwrites
existing documents in place and sibling files like `guide.md` and
`guide.txt` stay distinct. `s3://bucket/prefix` and `gs://bucket/prefix`
URLs behave identically, with the object key (relative to the prefix)
acting as the document id. Credentials follow the standard AWS / GCS
resolver chain; see `plume ingest --help` for the full flag list.

## Auto-indexing

Every upsert schedules a background ANN + FTS rebuild. Rebuilds are
debounced so rapid bursts coalesce into a single build per namespace.
Tune under `[index.auto]`:

```toml
[index.auto]
enabled = true          # set false to opt out and manage indexes manually
threshold_docs = 1000   # rebuild once pending writes cross this
debounce_ms = 5000      # or once writes have been idle this long
min_docs = 256          # skip the ANN build below this corpus size
```

The explicit `POST /ns/{ns}/index` and `POST /ns/{ns}/fts-index` endpoints
still work and are the right tool for bulk loads where the operator wants
to control when the rebuild happens.

## API

All document operations are scoped to a **namespace** (`/ns/{ns}/...`).

### Upsert documents

```bash
curl -X POST http://localhost:8787/ns/code/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "rows": [
      {"id": "1", "text": "retry HTTP requests with exponential backoff", "metadata": {}},
      {"id": "2", "text": "binary search with generic comparator", "metadata": {}},
      {"id": "3", "text": "parse JSON config and validate schema", "metadata": {}}
    ]
  }'
```

### Query

```bash
# Hybrid search (default: semantic + BM25 fused with RRF)
curl -X POST http://localhost:8787/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5}'

# Semantic only (MaxSim)
curl -X POST http://localhost:8787/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5, "mode": "semantic"}'

# Full-text only (BM25)
curl -X POST http://localhost:8787/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5, "mode": "fts"}'
```

### Build indexes

Vector and FTS indexes are built asynchronously. Call these after bulk ingestion:

```bash
# Build IVF_PQ ANN index over the per-document `multivector` column
curl -X POST http://localhost:8787/ns/code/index

# Build BM25 full-text index
curl -X POST http://localhost:8787/ns/code/fts-index
```

### Other endpoints

```bash
# Health check
curl http://localhost:8787/health

# Prometheus metrics
curl http://localhost:8787/metrics

# Warm up namespace (pull data into page cache)
curl -X POST http://localhost:8787/ns/code/warmup

# Drop namespace
curl -X DELETE http://localhost:8787/ns/code
```

## Configuration

Plume loads config from the path in `PLUME_CONFIG` env var, or falls back to defaults.

```toml
[server]
host = "0.0.0.0"
port = 8787

[storage]
# Local filesystem
uri = "./data/lancedb"

# S3 / MinIO (requires `storage-aws`; set AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars)
# uri = "s3://plume/data"
# region = "us-east-1"

# Cloudflare R2 (requires `storage-aws`)
# uri = "s3://bucket/plume"
# endpoint = "https://<account>.r2.cloudflarestorage.com"

# Google Cloud Storage (requires `storage-gcs`)
# uri = "gs://bucket/plume"

[cache]
ram_capacity_mb = 512       # In-memory LFU cache
nvme_capacity_gb = 1        # NVMe/disk spillover tier
nvme_path = "/tmp/plume-cache"

[encoder]
model = "lightonai/LateOn-Code-edge"   # Path to ONNX model dir, or name (mock fallback)
pool_factor = 2                         # Average adjacent token vectors (halves storage)
batch_size = 32

[index]
nbits = 4                     # IVF_PQ quantization bits
num_partitions = 64           # IVF partition count (omit to let LanceDB auto-select)
nprobes = 16                  # Number of IVF partitions to probe at query time
# refine_factor = 2           # Optional PQ refine stage (helps recall at scale, ~10% latency cost)
ann_candidate_multiplier = 10 # Retrieve k * multiplier ANN candidates before MaxSim rerank
max_candidates = 10000        # Hard cap on candidates materialized per query (OOM guard)
```

## Search modes

| Mode | Algorithm | Use case |
|------|-----------|----------|
| `semantic` | ANN candidate retrieval on a pooled document vector, then exact ColBERT MaxSim rerank on full token embeddings | Best for meaning-based retrieval |
| `fts` | LanceDB BM25 full-text search | Best for exact keyword matching |
| `hybrid` (default) | Reciprocal Rank Fusion (k=60) of semantic + FTS results | Best overall recall |

If the ANN index has not been built yet, semantic mode falls back to a full-recall scan so behavior stays correct while you are still ingesting or testing locally.

## Cache invalidation

The cache uses a **generation counter** per namespace. Every write (upsert) increments the generation, making all prior cache entries stale without needing to enumerate and delete them. This gives O(1) invalidation regardless of cache size.

## Benchmarks

```bash
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-latency

# Synthetic sweep against MockEncoder — fast wiring/smoke test
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-ann

# Real recall on BEIR SciFact with a ColBERT ONNX encoder
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-recall \
  --features plume-encoder/onnx
```

### SciFact recall (real encoder)

Measured on BEIR SciFact (5,183 docs, 300 test queries with qrels) with the 48-dim `lightonai/LateOn-Code-edge` ColBERT ONNX encoder, `pool_factor=2`, IVF_PQ with 128 partitions, `nbits=8`. `e2e_*_recall` is recall@10 against ground-truth qrels; `ann_recall` is ANN-vs-exact-MaxSim approximation quality.

| nprobes | cand | refine | Avg Latency | p95   | ann_recall | e2e_ann_recall |
|---------|------|--------|-------------|-------|------------|----------------|
| 20      | 50   | none   | **138ms**   | 195ms | 0.881      | **0.719**      |
| 50      | 5    | 10     | 451ms       | 734ms | 0.826      | 0.717          |
| 20      | 20   | 50     | 2,771ms     | 4,594ms | 0.998    | 0.720          |
| 20      | 50   | 10     | 2,953ms     | 4,701ms | 1.000    | 0.720          |
| —       | exact scan | —  | 1,547ms     | —     | baseline   | 0.720          |

Key takeaways:

- **`nprobes=20, candidate_multiplier=50, refine_factor=None`** hits 0.719 e2e recall at 138ms — essentially matching the exact-MaxSim ceiling of 0.720 at >10× the latency. This is the default.
- Probing more IVF partitions (20→50→100) does not improve recall — the candidate pool + client-side MaxSim re-rank in `plume-search` dominates once the ANN stage supplies enough candidates.
- `refine_factor` raises ANN-vs-exact approximation recall toward 1.000 but does not move e2e recall — the client-side MaxSim re-rank already recovers exact ordering from a wider candidate pool at a fraction of the latency.
- The e2e recall ceiling (~0.72) is the model's — not LanceDB's. `LateOn-Code-edge` is a code-oriented 48-dim ColBERT, not fine-tuned on biomedical scientific claims. Published ColBERTv2 numbers on BEIR SciFact are in the same 0.70–0.78 range. A domain-tuned encoder would raise the ceiling.

### Synthetic wiring sweep

`bench-ann` exercises the same code paths on a synthetic code-like corpus with a 2-dim `MockEncoder`. Useful for CI-speed smoke tests and ANN parameter sweeps, but the 2-dim vectors make retrieval trivial — **do not read recall numbers from `bench-ann` as representative**. Use `bench-recall` for real signal.

```bash
# Sweep LanceDB-recommended ranges against real SciFact + ColBERT
PLUME_BENCH_PARTITIONS=128 \
PLUME_BENCH_NPROBES=20,50,100 \
PLUME_BENCH_CANDIDATES=5,10,20,50 \
PLUME_BENCH_REFINE=none,10,50 \
PLUME_BENCH_NBITS=8 \
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-recall \
  --features plume-encoder/onnx
```

Prereqs for `bench-recall`:

```bash
./scripts/download-model.sh models/lateon-code-edge
./scripts/download-scifact.sh
```

## Development

```bash
# Run tests
PROTOC=$(which protoc) cargo test

# Run with debug logging
RUST_LOG=debug PLUME_CONFIG=config.local.toml cargo run --bin plume

# Format
cargo fmt

# Lint
cargo clippy
```

## Lean build notes

The workspace trims compile-time weight in three ways:

- Cloud storage support is opt-in instead of built into every local build.
- Unused direct dependencies such as `metrics-exporter-prometheus` and `object_store` have been removed.
- `tokio` uses a narrower feature set that matches the code paths in this repo.

## Related projects

- [`lightonai/next-plaid`](https://github.com/lightonai/next-plaid): the local-first multi-vector engine behind ColGREP. Its README is a useful reference for late-interaction retrieval, quantization, and memory-mapped indexing.
- [`gordonmurray/firnflow`](https://github.com/gordonmurray/firnflow): a close reference for the RAM -> NVMe -> object-storage architecture we are building toward on top of LanceDB and Foyer.

If you point `storage.uri` at `s3://...` or `gs://...` without the matching feature, Plume now fails fast with an explicit rebuild hint instead of a vague backend error.

On constrained machines, a leaner verification command helps reduce disk pressure during Rust builds:

```bash
PROTOC=$(which protoc) CARGO_INCREMENTAL=0 RUSTFLAGS='-Cdebuginfo=0' cargo test -p plume-api -j 1
```

## License

MIT
