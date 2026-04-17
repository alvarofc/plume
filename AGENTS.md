# AGENTS.md

Plume is a multi-vector (ColBERT-style) semantic search engine built on LanceDB, written in Rust.

For architecture, API, config, and deployment details see [`README.md`](README.md).
For Claude-specific context, benchmarking notes, and late-interaction indexing findings see [`CLAUDE.md`](CLAUDE.md).

## Setup

- Rust toolchain (workspace uses stable).
- `protoc` must be on `PATH` or exported as `PROTOC`. Every `cargo` command below assumes `PROTOC=$(which protoc)`.

## Common commands

```bash
# Build the server (lean local build)
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume

# Run tests
PROTOC=$(which protoc) cargo test

# Lint + format
cargo clippy
cargo fmt

# Run locally against filesystem storage
PLUME_CONFIG=config.local.toml ./target/release/plume
```

## Feature flags

| Flag | Purpose |
|------|---------|
| default | Local filesystem LanceDB only |
| `storage-aws` | S3 / MinIO / R2 |
| `storage-gcs` | Google Cloud Storage |
| `plume-encoder/onnx` | Real ColBERT ONNX encoder (else mock) |

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
./scripts/download-model.sh models/lateon-code-edge
./scripts/download-scifact.sh
```
