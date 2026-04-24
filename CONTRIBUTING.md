# Contributing to Plume

Thanks for contributing.

## Development setup

Plume is a Rust workspace. You need:

- Stable Rust toolchain
- `protoc` on `PATH` or `PROTOC` set explicitly

Common commands:

```bash
PROTOC=$(which protoc) cargo fmt --all
PROTOC=$(which protoc) cargo clippy --workspace --all-targets -- -D warnings
PROTOC=$(which protoc) cargo test --workspace
```

Build the CLI/server locally:

```bash
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --features onnx
```

For a lean local-only build:

```bash
PROTOC=$(which protoc) cargo build --release -p plume-api --bin plume --no-default-features
```

More setup details and platform notes live in [`AGENTS.md`](AGENTS.md) and [`README.md`](README.md).

## Change guidelines

- Keep changes scoped and targeted.
- Add or update tests when behavior changes.
- Run `fmt`, `clippy`, and `test` before opening a PR.
- Preserve feature-gated behavior for `storage-aws`, `storage-gcs`, and ONNX paths.
- Update docs when user-facing commands, config, or build steps change.

## Benchmarks

If you change retrieval behavior, indexing, scoring, or ANN parameters, validate with the benchmark flow in [`CLAUDE.md`](CLAUDE.md) and prefer `bench-recall` numbers over synthetic-only results.

## Pull requests

PRs are easiest to review when they include:

- A short problem statement
- The concrete behavior change
- Any config, feature-flag, or migration impact
- Benchmark or test evidence when search quality or performance changes

## Releases

Tagged `v*.*.*` pushes build release artifacts in GitHub Actions. Keep `README.md`, `install.sh`, and the release workflow aligned when changing install or packaging behavior.
