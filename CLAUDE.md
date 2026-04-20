# CLAUDE.md

Project-specific context for Claude when working on Plume.
See [`AGENTS.md`](AGENTS.md) for setup and common commands, and [`README.md`](README.md) for architecture and deployment.

## Project shape

Rust workspace split into: `plume-core` (types/config), `plume-encoder` (ONNX or mock ColBERT), `plume-cache` (RAM + NVMe tiered cache, generation-based invalidation), `plume-index` (LanceDB tables, ANN, FTS), `plume-search` (MaxSim, RRF fusion, search orchestration), `plume-api` (Axum server), `plume-bench` (benchmarks).

Semantic-search flow: encode query into a multi-vector → ANN candidate retrieval on LanceDB → exact ColBERT MaxSim rerank in `plume-search` → cache result. FTS (BM25) and hybrid (RRF over semantic + FTS) are also supported.

## Ingest / grep source abstraction

`crates/plume-api/src/cli/source.rs` is the single place that knows how to read a corpus. It resolves either a local path or an `s3://bucket/prefix` / `gs://bucket/prefix` URL via [`Source::parse`], exposes `list()` and `read_text()`, and is consumed by both `plume ingest` and `plume grep`.

Two things to keep in mind when editing:

- **Feature gating.** Remote sources are behind `storage-aws` / `storage-gcs`, which activate `object_store/{aws,gcp}` in addition to the LanceDB backends. A default build with `s3://...` fails fast at parse time with a build hint — do not fall through to "local path not found".
- **Identity, not file path.** Manifests (for `plume grep`) and namespace names (`grep-{hash}`) are keyed off `Source::identity()`, not `&Path`. Local paths still hash their filesystem bytes (via `namespace_for_path`) for backward compatibility; remote sources hash the URL. When adding new source types, preserve that split so existing `.json` manifests aren't orphaned.

## LanceDB multi-vector

Verified against `lancedb-0.27.2` source; use this as the source of truth for this repo.

### Schema

Multi-vector column must be `List<FixedSizeList<Float32, dim>>`. The outer type is `List` (variable document length), the inner is `FixedSizeList` (fixed embedding dim). PyArrow equivalent: `pa.list_(pa.list_(pa.float32(), dim))`.

Defined in `crates/plume-index/src/schema.rs::plume_schema()`. Do not change the outer to `FixedSizeList` — that would force uniform token counts per document and break variable-length ColBERT outputs. Issue [lancedb/lancedb#2071](https://github.com/lancedb/lancedb/issues/2071) looked like it suggested otherwise but was just a version mismatch, resolved in LanceDB 0.18.1b1+.

### Query path

To hit the native multi-vector IVF_PQ path in LanceDB:

```rust
table.query()
    .nearest_to(first_token)?
    .column("multivector")
    .distance_type(DistanceType::Cosine)  // only Cosine is supported for multi-vector
    .nprobes(nprobes)
    .add_query_vector(token)?             // for each remaining token
    .refine_factor(refine_factor)         // optional; applies a MaxSim rerank on uncompressed vectors
    .limit(k)
    .execute()
```

The native path is triggered inside `lancedb::table::query::create_plan` (`src/table/query.rs:103`): when the vector column's type is `DataType::List`, all query vectors are concatenated into a single `FixedSizeList<FixedSizeList<_>>` and dispatched as one multi-vector scan. If the column were any other type, LanceDB would fan out into one plan per query token and UNION them — which is much slower and does not use late-interaction MaxSim at the index.

### Parameter tuning (late-interaction)

LanceDB's GIST-1M docs suggest `nprobes ~ 50`, `refine_factor ~ 50` as starting points. Our actual bench-recall sweep on BEIR SciFact landed on a different spot: with the wider client-side MaxSim rerank pool in `plume-search`, raising `nprobes` past 20 or enabling `refine_factor` at all did not improve e2e recall and cost >10× latency. Current `config.toml` defaults (`nprobes=20`, `refine_factor=None`, `ann_candidate_multiplier=50`) hit ~0.719 e2e recall at ~138ms — essentially matching the exact-MaxSim ceiling of 0.720. Before raising these "because LanceDB said so", rerun `bench-recall` on the target corpus and compare e2e recall, not ANN-vs-exact approximation recall.

Known trade-off: on small corpora, pooled single-vector ANN + client-side MaxSim rerank (what we used pre-native-multivector) can beat the native multi-vector IVF_PQ index on both recall and latency. LanceDB's own ColPali tutorial uses pooled+padded retrieval plus client-side MaxSim rather than the native index for the same reason. Before committing to one or the other, run `bench-recall` with both paths on the target corpus size.

## Benchmarking

Two benchmarks in `bench/`:

- `bench-ann` — synthetic code-like corpus with `MockEncoder` (2-dim vectors). Fast, good for wiring/smoke-testing ANN parameter sweeps.
- `bench-recall` — BEIR SciFact + real LateOn-Code-edge ColBERT ONNX encoder (48-dim). This is the one that measures real recall.

`bench-recall` reports both:

- `min/avg_ann_recall` — ANN-vs-exact-MaxSim recall (approximation quality of the ANN stage).
- `e2e_ann_recall` / `e2e_exact_recall` — recall against ground-truth qrels (end-to-end retrieval quality, bounded by what the encoder can actually retrieve).

Ingesting 5,183 SciFact docs on CPU takes ~14 minutes. Computing 300 exact baselines takes several minutes. Budget accordingly.

Env vars (both benches): `PLUME_BENCH_DOCS`, `PLUME_BENCH_K`, `PLUME_BENCH_PARTITIONS`, `PLUME_BENCH_NPROBES`, `PLUME_BENCH_CANDIDATES`, `PLUME_BENCH_REFINE`, `PLUME_BENCH_NBITS`, `PLUME_BENCH_POOL_FACTOR`, `PLUME_BENCH_MODEL`, `PLUME_BENCH_DATA`.

## Things to watch out for

- **Empty table merge-insert panics.** `NamespaceTable::upsert` uses `add()` on the first batch and `merge_insert` after — don't collapse the two paths.
- **Distance metric must match index.** We build IVF_PQ with `DistanceType::Cosine`; queries must also use Cosine or results are invalid.
- **`nprobes(n)`** sets both min and max partitions. For adaptive probing, call `minimum_nprobes` and `maximum_nprobes` separately.
- **`refine_factor`** always incurs a full-vector fetch, even at `refine_factor=1`. Without it, `_distance` is an approximate PQ distance.
- **ANN fallback.** `SearchEngine::semantic_search` falls back to a bounded scan (capped at `min(ann_candidate_multiplier * k, max_candidates)`) when `has_ann_index()` is false *and* when an ANN query errors transiently mid-flight. Early-stage or mid-rebuild namespaces still return results; once the corpus exceeds the candidate cap the operator needs a healthy ANN index to keep recall intact.
- **Cache tiering.** `SearchCache` uses a RAM+NVMe hybrid cache by default, wiped on startup (generation counters are in-memory, so persisted entries could otherwise collide with fresh keys after restart). The tier lives at `{cache.nvme_path}/plume-search-cache/`. Setting `cache.nvme_capacity_gb = 0` switches to an in-memory-only cache — useful on read-only filesystems, ephemeral containers, or CI.
- **Cache key shape.** `hash_query(query, mode, k)` includes `k`; the cache stores the truncated top-k list, so reusing an entry at a different `k` would return the wrong number of rows.

## README numbers

The recall/latency table in `README.md` uses `MockEncoder` (2-dim vectors), which is unrealistically easy — perfect recall there does not generalize. Real numbers live in `bench-recall` output. When updating README benchmark tables, pull from `bench-recall` on SciFact, not `bench-ann`.
