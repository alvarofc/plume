# Plume

Semantic `grep` for your code, docs, and buckets. Point it at a directory or an `s3://` / `gs://` URL, ask in plain English, and get back the chunks that actually match your intent.

```bash
plume grep "retry with exponential backoff" ./src
plume grep "connection pool leak"           s3://prod-runbooks
plume grep "where do we refresh auth tokens" gs://docs/engineering
```

No separate index step, no separate query service — `plume grep` spawns a local daemon on first use, indexes incrementally (only changed files re-embed), and subsequent queries are sub-second.

## Why

Regex finds what you typed. Classic single-vector embeddings find the vibe but blur away precision on short, code-like text. Plume uses **ColBERT-style multi-vector retrieval** — one embedding per token, scored with late-interaction MaxSim — so `"backoff retry logic"` can match `retry_with_jitter` and `sleep(backoff).await`, not just documents that happen to contain the word *retry*.

Under the hood it pairs four ideas that usually live in four different systems:

- **ANN candidate retrieval** on a pooled document vector — fast recall.
- **Exact MaxSim re-ranking** over full token embeddings — precision.
- **BM25 full-text** fused via **Reciprocal Rank Fusion** — a keyword safety net.
- **RAM + NVMe tiered cache** with O(1) generation-based invalidation — so re-asking is free.

Storage is **LanceDB** on local disk, S3, R2, or GCS. Swap backends by changing one config line.

## Quickstart

```bash
# 1. Build with the real ColBERT encoder (needs protoc on PATH).
#    Default features cover S3 + GCS; add `onnx` for real semantic search.
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features onnx

# 2. One-time: download the ColBERT encoder (~70MB)
./target/release/plume model pull

# 3. Semantic grep a local path or a bucket
./target/release/plume grep "rate limiting middleware" ./src
./target/release/plume grep "idempotency key reuse"   s3://my-bucket/code
```

That's it. First run builds the index; later runs only re-embed changed files.

> Building without `--features onnx` keeps the binary portable on hosts where pyke's prebuilt ONNX Runtime isn't available, but falls back to a mock encoder — semantic quality will be poor. See [Build profiles](#build-profiles).

### Docker

```bash
docker compose up   # Plume on :8787, MinIO on :9000, bucket auto-created
```

## CLI cheatsheet

```bash
plume serve                              # run the daemon (also the default)
plume grep "<query>" <path | s3:// | gs://>   # one-shot semantic grep
plume ingest <path> -n <namespace>       # explicit ingest for API use
plume search "<query>" -n <namespace>    # query an ingested namespace
plume push <src> <dst>                   # rsync-style upload to S3/GCS
plume ns list | create | delete
plume model pull | list | where
```

Environment: `PLUME_URL` (default `http://localhost:8787`), `PLUME_CONFIG` (TOML path), `AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`.

S3 credentials are read from the environment only (not `~/.aws/credentials`). On AWS SSO:

```bash
eval "$(aws configure export-credentials --profile <you> --format env)"
```

## HTTP API

All document ops are scoped to a namespace (`/ns/{ns}/...`).

```bash
# Upsert
curl -X POST :8787/ns/code/upsert -H 'content-type: application/json' -d '{
  "rows": [{"id":"1","text":"retry HTTP with exponential backoff","metadata":{}}]
}'

# Query — mode is "hybrid" (default), "semantic", or "fts"
curl -X POST :8787/ns/code/query -H 'content-type: application/json' -d '{
  "query": "backoff retry logic", "k": 5, "mode": "hybrid"
}'
```

Also: `GET /health`, `GET /metrics`, `POST /ns/{ns}/warmup`, `DELETE /ns/{ns}`, and explicit `POST /ns/{ns}/index` + `POST /ns/{ns}/fts-index`. Auto-indexing is on by default, so you almost never need the explicit build endpoints.

## Architecture

```
 HTTP API ─▶ plume-api
              │
        ┌─────┼─────┐
        ▼     ▼     ▼
    encoder search cache       RAM (LFU) + NVMe tier
        │     │
        ▼     ▼
      ONNX  LanceDB (local / S3 / R2 / GCS)
```

| Crate           | Purpose |
|-----------------|---------|
| `plume-core`    | Shared types, config, errors |
| `plume-encoder` | ColBERT encoding (ONNX or mock) |
| `plume-cache`   | Tiered RAM/NVMe cache, O(1) generation invalidation |
| `plume-index`   | LanceDB tables, ANN, FTS |
| `plume-search`  | MaxSim scoring, RRF fusion, orchestration |
| `plume-api`     | Axum HTTP server and CLI |

## Search modes

| Mode       | Algorithm                                                     | Best for |
|------------|---------------------------------------------------------------|----------|
| `semantic` | ANN candidates → exact ColBERT MaxSim re-rank                 | Meaning-based retrieval |
| `fts`      | LanceDB BM25                                                  | Exact keyword matching |
| `hybrid`   | RRF (k=60) over semantic + BM25 — **default**                 | Best overall recall |

If the ANN index is still building, `semantic` falls back to a bounded scan and `hybrid` degrades to semantic-only — queries never fail just because a background build is in flight.

## Configuration (`$PLUME_CONFIG`)

```toml
[server]  port = 8787
[storage] uri = "./data/lancedb"      # or "s3://bucket/plume" / "gs://bucket/plume"
[cache]   ram_capacity_mb  = 512
          nvme_capacity_gb = 1
          nvme_path        = "/tmp/plume-cache"
[encoder] model = "lightonai/LateOn-Code-edge"
[index]   nprobes                  = 20
          ann_candidate_multiplier = 50
```

See the annotated example in [`config.toml`](config.toml).

## Benchmarks

BEIR SciFact, 5,183 docs, 300 queries, recall@10, 48-dim `LateOn-Code-edge` ColBERT:

| Setup                                   | Avg latency | e2e recall |
|-----------------------------------------|-------------|------------|
| **Plume default** (nprobes=20, cand=50) | **138 ms**  | **0.719**  |
| Exact MaxSim scan (ceiling)             | 1,547 ms    | 0.720      |
| LanceDB-recommended (nprobes=50, refine=10) | 451 ms  | 0.717      |

Essentially the exact-MaxSim ceiling at ~10× less latency. The 0.72 ceiling is the encoder's, not LanceDB's — a domain-tuned ColBERT would raise it. Full sweep in [`plume-bench`](crates/plume-bench).

## Build profiles

Default features: `storage-aws`, `storage-gcs`. The ONNX encoder is opt-in so a vanilla `cargo build` stays portable on hosts where pyke's prebuilt ONNX Runtime isn't published:

```bash
# Cloud storage + real ColBERT encoder (recommended)
cargo build --release -p plume-api --features onnx

# Minimal: local-only, mock encoder
cargo build --release -p plume-api --no-default-features

# Intel Mac / anywhere pyke's ort prebuilts aren't published
cargo build --release -p plume-api --features onnx-system-ort
brew install onnxruntime
export ORT_DYLIB_PATH=/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib
```

Pointing `storage.uri` at `s3://` or `gs://` without the matching feature fails fast with a rebuild hint rather than a vague backend error.

## Prior art

Plume stands on the shoulders of two projects worth reading if you want to go deeper:

- [`lightonai/next-plaid`](https://github.com/lightonai/next-plaid) — the local-first multi-vector engine behind **ColGREP**. The canonical reference for late-interaction retrieval, quantization, and memory-mapped indexing.
- [`gordonmurray/firnflow`](https://github.com/gordonmurray/firnflow) — a close reference for the RAM → NVMe → object-storage tiering we're building on top of LanceDB and Foyer.

## More

- Operator setup, toolchain notes, and CI commands: [`AGENTS.md`](AGENTS.md)
- Late-interaction internals, LanceDB multi-vector notes, recall tuning: [`CLAUDE.md`](CLAUDE.md)

## License

MIT
