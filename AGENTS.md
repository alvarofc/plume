# AGENTS.md

Plume is a multi-vector (ColBERT-style) semantic search engine built on LanceDB, written in Rust.

For architecture, API, config, and deployment details see [`README.md`](README.md).
For Claude-specific context, benchmarking notes, and late-interaction indexing findings see [`CLAUDE.md`](CLAUDE.md).

## Setup

- Rust toolchain (workspace uses stable).
- `protoc` must be on `PATH` or exported as `PROTOC`. Every `cargo` command below assumes `PROTOC=$(which protoc)`.
  - On macOS, Homebrew's `protobuf` is keg-only: `brew install protobuf` doesn't put `protoc` on `PATH`, so `PROTOC=$(which protoc)` silently evaluates to empty and the build fails with a confusing error. Either `brew link --force protobuf` or point at it directly: `export PROTOC=/usr/local/opt/protobuf/bin/protoc` (Intel) / `/opt/homebrew/opt/protobuf/bin/protoc` (Apple Silicon).

## Common commands

```bash
# Build the server (default = S3 + GCS + ONNX)
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume

# Lean local build without cloud storage or the real encoder
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --no-default-features

# Run tests
PROTOC=$(which protoc) cargo test

# Lint + format
cargo clippy
cargo fmt

# Run locally against filesystem storage
PLUME_CONFIG=config.local.toml ./target/release/plume
```

## Feature flags

Defaults: `storage-aws`, `storage-gcs`, `onnx`. Drop with `--no-default-features`.

| Flag | Purpose | Default |
|------|---------|---------|
| `storage-aws` | S3 / MinIO / R2 | on |
| `storage-gcs` | Google Cloud Storage | on |
| `onnx` | Real ColBERT ONNX encoder (else mock) | on |

## Benchmarks

```bash
# Synthetic code-like corpus + MockEncoder
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-ann

# BEIR SciFact + real LateOn-Code-edge ColBERT (needs models/ + data/scifact/)
PROTOC=$(which protoc) cargo run --release -p plume-bench --bin bench-recall \
  --features plume-encoder/onnx
```

Prereqs for `bench-recall`:

```bash
plume model pull                    # or ./scripts/download-model.sh models/lateon-code-edge
./scripts/download-scifact.sh
```
