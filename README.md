# Plume

Multi-vector semantic search engine with tiered caching. Combines ColBERT late-interaction retrieval (MaxSim scoring) with BM25 full-text search, backed by LanceDB for storage on S3, R2, GCS, or local disk.

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
| `plume-index` | LanceDB table management, upsert, scan |
| `plume-search` | MaxSim scoring, BM25 FTS, RRF hybrid fusion |
| `plume-api` | Axum HTTP server and routes |

## Quickstart

### Local (filesystem storage)

```bash
# Build
PROTOC=$(which protoc) cargo build --release --bin plume

# Run with local config
PLUME_CONFIG=config.local.toml ./target/release/plume
```

### Docker Compose (MinIO S3)

```bash
docker compose up
```

This starts Plume on port 3000, MinIO on port 9000 (console on 9001), and auto-creates the `plume` bucket.

### ONNX encoder (optional)

By default Plume uses a mock encoder for development. To use the real ColBERT model:

```bash
# Download model files (~70MB)
./scripts/download-model.sh models/lateon-code-edge

# Build with ONNX support
PROTOC=$(which protoc) cargo build --release --bin plume --features plume-encoder/onnx

# Point config to model directory
# [encoder]
# model = "models/lateon-code-edge"
```

## API

All document operations are scoped to a **namespace** (`/ns/{ns}/...`).

### Upsert documents

```bash
curl -X POST http://localhost:3000/ns/code/upsert \
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
curl -X POST http://localhost:3000/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5}'

# Semantic only (MaxSim)
curl -X POST http://localhost:3000/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5, "mode": "semantic"}'

# Full-text only (BM25)
curl -X POST http://localhost:3000/ns/code/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "backoff retry logic", "k": 5, "mode": "fts"}'
```

### Build indexes

Vector and FTS indexes are built asynchronously. Call these after bulk ingestion:

```bash
# Build IVF_PQ vector index
curl -X POST http://localhost:3000/ns/code/index

# Build BM25 full-text index
curl -X POST http://localhost:3000/ns/code/fts-index
```

### Other endpoints

```bash
# Health check
curl http://localhost:3000/health

# Prometheus metrics
curl http://localhost:3000/metrics

# Warm up namespace (pull data into page cache)
curl -X POST http://localhost:3000/ns/code/warmup

# Drop namespace
curl -X DELETE http://localhost:3000/ns/code
```

## Configuration

Plume loads config from the path in `PLUME_CONFIG` env var, or falls back to defaults.

```toml
[server]
host = "0.0.0.0"
port = 3000

[storage]
# Local filesystem
uri = "./data/lancedb"

# S3 / MinIO (set AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars)
# uri = "s3://plume/data"
# region = "us-east-1"

# Cloudflare R2
# uri = "s3://bucket/plume"
# endpoint = "https://<account>.r2.cloudflarestorage.com"

# Google Cloud Storage
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
nbits = 4       # IVF_PQ quantization bits
nprobes = 32    # Number of IVF partitions to probe at query time
```

## Search modes

| Mode | Algorithm | Use case |
|------|-----------|----------|
| `semantic` | ColBERT MaxSim — scores each query token against all document tokens, sums the max similarities | Best for meaning-based retrieval |
| `fts` | LanceDB BM25 full-text search | Best for exact keyword matching |
| `hybrid` (default) | Reciprocal Rank Fusion (k=60) of semantic + FTS results | Best overall recall |

## Cache invalidation

The cache uses a **generation counter** per namespace. Every write (upsert) increments the generation, making all prior cache entries stale without needing to enumerate and delete them. This gives O(1) invalidation regardless of cache size.

## Benchmarks

```bash
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-latency
```

Measured on local filesystem with MockEncoder:

| Metric | 100 docs | 1K docs | 10K docs |
|--------|----------|---------|----------|
| Ingest | ~0.5ms/doc | ~0.3ms/doc | ~0.2ms/doc |
| Cold query | ~5ms | ~17ms | ~19ms |
| Warm query | ~3us | ~3us | ~3us |
| Cache hit rate | 99.8% | 99.8% | 99.8% |

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

## License

MIT
