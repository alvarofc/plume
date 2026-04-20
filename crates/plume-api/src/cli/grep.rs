//! `plume grep <query> [path]` — colgrep-style one-shot semantic search.
//!
//! First invocation against a path transparently:
//!   1. Spawns a `plume serve` daemon if `--url` isn't reachable.
//!   2. Derives a stable namespace name from the canonical path.
//!   3. Scans supported files (filtered by --include/--exclude/--exclude-dir),
//!      chunks them with a sliding line window, and upserts everything new
//!      or changed.
//!   4. Prunes chunks belonging to files that were deleted or modified
//!      since the last run (DELETE by id).
//!   5. Queries and prints results grep-style.
//!
//! The server-side auto-indexer debounces the writes into a single ANN +
//! FTS rebuild. See `AutoIndexer::recover` for restart durability.

use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use globset::{Glob, GlobSet, GlobSetBuilder};
use plume_core::types::{
    DeleteDocsRequest, DeleteDocsResponse, Document, QueryRequest, QueryResponse, SearchMode,
    SearchResult, UpsertRequest, UpsertResponse, MAX_ROWS_PER_UPSERT, MAX_TEXT_LENGTH,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::client::{Client, DEFAULT_URL};
use super::source::{Filter, Source, SourceFile};

#[derive(Parser)]
pub struct Args {
    /// Natural language query. Combine with `-e` for multi-query fusion.
    pub query: Option<String>,

    /// Files or directories to search. Defaults to the current directory.
    /// Accepts local paths and `s3://bucket/prefix` / `gs://bucket/prefix`
    /// URLs (with the `storage-aws` / `storage-gcs` features).
    #[arg(default_value = ".")]
    pub path: String,

    /// Additional query. Repeat `-e` to fuse multiple queries; scores merge
    /// by max across queries per chunk.
    #[arg(short = 'e', long = "query", value_name = "QUERY")]
    pub extra_queries: Vec<String>,

    /// Top-k results.
    #[arg(short, long, default_value_t = 15)]
    pub k: usize,

    /// Glob(s) to include. Repeatable. Disables the default extension
    /// allowlist: only files matching at least one include are scanned.
    #[arg(long, value_name = "GLOB")]
    pub include: Vec<String>,

    /// Glob(s) to exclude. Repeatable. Applied to every candidate path.
    #[arg(long, value_name = "GLOB")]
    pub exclude: Vec<String>,

    /// Extra directory basenames to skip, on top of the built-in list
    /// (.git, node_modules, target, ...). Repeatable.
    #[arg(long, value_name = "NAME")]
    pub exclude_dir: Vec<String>,

    /// List matching file paths only, like `grep -l`.
    #[arg(short = 'l', long = "files-with-matches")]
    pub files_with_matches: bool,

    /// Count matches per file, like `grep -c`. Implies path grouping.
    #[arg(short = 'c', long = "count")]
    pub count: bool,

    /// Show chunk line numbers in output (on by default; pass `--no-line-number` to hide).
    #[arg(short = 'n', long = "line-number", default_value_t = true)]
    pub line_number: bool,

    /// Hide chunk line numbers.
    #[arg(long = "no-line-number")]
    pub no_line_number: bool,

    /// How many preview lines to show per hit (default 3).
    #[arg(long, default_value_t = 3)]
    pub preview: usize,

    /// Search mode. Hybrid (semantic + BM25 RRF fusion) is the default;
    /// semantic-only or fts-only are available for ablations.
    #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
    pub mode: Mode,

    /// When to colorize output. `auto` (default) colors when stdout is a TTY.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// Emit raw JSON instead of the grep-style table.
    #[arg(long, conflicts_with_all = ["files_with_matches", "count"])]
    pub json: bool,

    /// Skip auto-ingest; query the existing namespace as-is.
    #[arg(long)]
    pub no_index: bool,

    /// Don't spawn a background daemon if the URL is unreachable — fail instead.
    #[arg(long)]
    pub no_daemon: bool,

    /// Print progress/debug info on stderr.
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Plume server URL. When unreachable, `plume grep` spawns a daemon
    /// here and waits for it to come up (unless --no-daemon).
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum Mode {
    Semantic,
    Fts,
    Hybrid,
}

impl From<Mode> for SearchMode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Semantic => SearchMode::Semantic,
            Mode::Fts => SearchMode::Fts,
            Mode::Hybrid => SearchMode::Hybrid,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// Default file extensions. Biased toward code + notes. Disabled when
/// --include is passed: the user is explicitly telling us what to scan.
const DEFAULT_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "rst", "org", "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "c",
    "cc", "cpp", "h", "hpp", "rb", "swift", "kt", "kts", "scala", "lua", "ex", "exs", "hs", "ml",
    "mli", "r", "zig", "jl", "sh", "bash", "zsh", "fish", "sql", "toml", "yaml", "yml", "json",
];

/// Directories we never descend into.
const EXCLUDED_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".turbo",
    ".cache",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "vendor",
    "Pods",
    "DerivedData",
];

/// Window size for the line-sliding chunker. Picked to comfortably fit in
/// a ColBERT context while keeping enough surrounding lines for semantic
/// coherence. Tune by corpus via env var if we need to.
const CHUNK_WINDOW_LINES: usize = 40;
/// Overlap between adjacent chunks. 25% is enough to keep a match near a
/// window boundary from getting chopped in half.
const CHUNK_OVERLAP_LINES: usize = 10;

/// Per-file entry in the on-disk manifest. Tracks the set of chunk ids so
/// we can issue targeted DELETEs when a file changes or is removed.
#[derive(Serialize, Deserialize, Clone, Default)]
struct FileEntry {
    mtime_nanos: u128,
    size: u64,
    /// Deterministic chunk ids we pushed into the namespace. Empty for
    /// manifests written by older versions of `plume grep` — treated as
    /// "we don't know what's in the server", which forces a re-ingest
    /// (but can't prune orphan rows; `plume clear <path>` does that).
    #[serde(default)]
    chunk_ids: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct Manifest {
    files: HashMap<String, FileEntry>,
}

pub async fn run(args: Args) -> Result<()> {
    let queries = collect_queries(&args)?;

    let source = Source::parse(&args.path)?;
    // For local sources we want the fully-resolved path so the same working
    // directory reached via a symlink shares a namespace / manifest.
    let identity = source.identity();
    if let Some(local) = source.as_local_root() {
        if !local.exists() {
            bail!("path does not exist: {}", local.display());
        }
    }

    let namespace = namespace_for_identity(&identity);
    let manifest_path = grep_state_dir()?.join(format!("{namespace}.json"));

    ensure_daemon(&args.url, args.no_daemon, args.verbose).await?;

    let client = Client::new(args.url.clone());

    warn_on_mock_encoder(&client).await;

    if !args.no_index {
        sync_index(
            &client,
            &namespace,
            &source,
            &manifest_path,
            &args,
            args.verbose,
        )
        .await?;
    }

    let results = run_queries(&client, &namespace, &queries, args.k, args.mode.into()).await?;

    if results.is_empty() {
        if !args.json {
            eprintln!("no results");
        }
        return Ok(());
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    let palette = Palette::from_choice(args.color);
    let show_line_numbers = args.line_number && !args.no_line_number;
    let display_root = DisplayRoot::for_source(&source);

    if args.files_with_matches {
        print_files_with_matches(&results, &display_root, &palette);
    } else if args.count {
        print_counts(&results, &display_root, &palette);
    } else {
        print_hits(
            &results,
            &display_root,
            show_line_numbers,
            args.preview,
            &palette,
        );
    }
    Ok(())
}

/// Collect non-empty queries from the positional argument and any `-e`
/// repetitions. Accepts `-e`-only invocations so the CLI honors its own
/// promise that `-e` can stand in as an alternate query source.
fn collect_queries(args: &Args) -> Result<Vec<String>> {
    let mut queries: Vec<String> = Vec::new();
    if let Some(q) = args.query.clone().filter(|q| !q.is_empty()) {
        queries.push(q);
    }
    queries.extend(
        args.extra_queries
            .iter()
            .filter(|q| !q.is_empty())
            .cloned(),
    );
    if queries.is_empty() {
        bail!("a query is required (positional or via -e)");
    }
    Ok(queries)
}

/// Run N queries against the namespace and fuse by max score per chunk id.
/// For a single query we just forward; for multi-query we merge.
async fn run_queries(
    client: &Client,
    namespace: &str,
    queries: &[String],
    k: usize,
    mode: SearchMode,
) -> Result<Vec<SearchResult>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    // Over-fetch per query so fusion has more candidates to choose from;
    // cap top output back to `k` at the end.
    let per_query_k = if queries.len() == 1 { k } else { k * 3 };
    let mut merged: HashMap<String, SearchResult> = HashMap::new();

    for q in queries {
        let resp: QueryResponse = client
            .post_json(
                &format!("/ns/{namespace}/query"),
                &QueryRequest {
                    query: q.clone(),
                    k: per_query_k,
                    mode,
                },
            )
            .await
            .with_context(|| format!("query {q:?}"))?;
        for hit in resp.results {
            merged
                .entry(hit.id.clone())
                .and_modify(|existing| {
                    if hit.score > existing.score {
                        existing.score = hit.score;
                    }
                })
                .or_insert(hit);
        }
    }

    let mut out: Vec<SearchResult> = merged.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(k);
    Ok(out)
}

/// If /health reports a mock encoder, warn once to stderr — the user is
/// about to get nonsense embeddings and we'd rather be loud about it than
/// silent.
async fn warn_on_mock_encoder(client: &Client) {
    let Ok(health) = client.get_json::<Value>("/health").await else {
        return;
    };
    let kind = health.get("encoder").and_then(|v| v.as_str()).unwrap_or("");
    if kind == "mock" || kind.starts_with("mock:") {
        eprintln!(
            "plume: warning: server is using the MOCK encoder — embeddings are deterministic \
             hashes, not semantic vectors. Set `[encoder] model = \"/path/to/onnx\"` in \
             config.toml and restart the daemon for real search quality."
        );
    }
}

/// Detect whether a Plume server is listening at `url`; spawn one if not.
async fn ensure_daemon(url: &str, no_daemon: bool, verbose: bool) -> Result<()> {
    let client = Client::new(url.to_string());
    if probe_health(url, Duration::from_secs(2)).await {
        return Ok(());
    }

    if no_daemon {
        bail!("plume daemon not reachable at {url} (--no-daemon set)");
    }

    // Parse the URL so the spawned daemon binds the same host:port the CLI
    // is polling. Without this, a child `plume serve` would use config /
    // defaults (0.0.0.0:8787) and the parent's health poll would loop on an
    // address nothing is listening at.
    let bind = parse_bind_target(url)?;
    if !bind.is_local() {
        bail!(
            "plume daemon not reachable at {url} and auto-spawn only works for local URLs \
             (got host {host:?}); start the daemon on that host manually or pass --no-daemon",
            host = bind.host
        );
    }

    let data_dir = plume_data_dir()?;
    fs::create_dir_all(&data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    let log_path = data_dir.join("daemon.log");
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open {}", log_path.display()))?;
    let log_err = log.try_clone()?;
    let exe = std::env::current_exe().context("find plume binary for daemon spawn")?;

    eprintln!(
        "plume: starting background daemon on {}:{} (logs: {})",
        bind.host,
        bind.port,
        log_path.display()
    );
    if verbose {
        eprintln!("plume: data dir = {}", data_dir.display());
    }

    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve")
        .current_dir(&data_dir)
        .env("PLUME_BIND_HOST", &bind.host)
        .env("PLUME_BIND_PORT", bind.port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().context("spawn plume daemon")?;

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(15) {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if client.get_json::<Value>("/health").await.is_ok() {
            return Ok(());
        }
    }
    bail!(
        "plume daemon failed to come up within 15s (see logs at {})",
        log_path.display()
    );
}

/// Short-timeout health probe used to decide whether to spawn a daemon.
/// The shared `Client` has no timeout configured, so probing a remote URL
/// that isn't accepting connections would hang on TCP connect for many
/// seconds before the `is_local` bail could fire.
async fn probe_health(url: &str, timeout: Duration) -> bool {
    let Ok(http) = reqwest::Client::builder().timeout(timeout).build() else {
        return false;
    };
    let url = format!("{}/health", url.trim_end_matches('/'));
    matches!(http.get(&url).send().await, Ok(r) if r.status().is_success())
}

/// Host/port extracted from `--url`, used to force the spawned daemon to
/// bind the address the CLI will actually poll.
struct BindTarget {
    host: String,
    port: u16,
}

impl BindTarget {
    /// Whether the target is a loopback or wildcard address — the only
    /// shapes where spawning a local daemon makes sense.
    fn is_local(&self) -> bool {
        // Strip IPv6 brackets that `url` preserves in `host_str` (e.g.
        // `[::1]`) before handing the string to `IpAddr::parse`.
        let bare = self
            .host
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(&self.host);
        if matches!(bare, "localhost" | "0.0.0.0" | "::" | "::0") {
            return true;
        }
        if let Ok(addr) = bare.parse::<std::net::IpAddr>() {
            return addr.is_loopback() || addr.is_unspecified();
        }
        false
    }
}

fn parse_bind_target(raw: &str) -> Result<BindTarget> {
    let parsed = url::Url::parse(raw).with_context(|| format!("parse --url {raw:?}"))?;
    let host = parsed
        .host_str()
        .with_context(|| format!("--url {raw:?} has no host"))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .with_context(|| format!("--url {raw:?} has no port and scheme has no default"))?;
    Ok(BindTarget { host, port })
}

/// Scan `source`, diff against the manifest, and upsert changed content
/// while pruning stale chunks. Returns after the final HTTP batch.
async fn sync_index(
    client: &Client,
    namespace: &str,
    source: &Source,
    manifest_path: &Path,
    args: &Args,
    verbose: bool,
) -> Result<()> {
    let filters = FileFilters::from_args(args)?;
    let mut manifest = load_manifest(manifest_path);
    let scanned = scan_source(source, &filters).await?;
    let scanned_keys: HashSet<String> = scanned.iter().map(|f| f.key.clone()).collect();

    // Files present in the manifest but not in the source → prune.
    let mut to_delete_ids: Vec<String> = Vec::new();
    let mut to_delete_rels: Vec<String> = Vec::new();
    for (rel, entry) in &manifest.files {
        if !scanned_keys.contains(rel) {
            to_delete_ids.extend(entry.chunk_ids.iter().cloned());
            to_delete_rels.push(rel.clone());
        }
    }

    // Changed files: re-chunk, queue old chunk ids for deletion, push new.
    let mut new_chunks: Vec<Document> = Vec::new();
    let mut new_entries: HashMap<String, FileEntry> = HashMap::new();

    for file in &scanned {
        let prev = manifest.files.get(&file.key);
        let unchanged = prev.is_some_and(|p| {
            p.mtime_nanos == file.mtime_nanos && p.size == file.size && !p.chunk_ids.is_empty()
        });
        if unchanged {
            continue;
        }
        let Some(text) = source
            .read_text(&file.key)
            .await
            .with_context(|| format!("read {}", file.key))?
        else {
            continue;
        };
        let chunks = chunk_file(&file.key, &text);
        if chunks.is_empty() {
            // Empty file: still record it so we don't keep re-chunking.
            if let Some(prev) = prev {
                to_delete_ids.extend(prev.chunk_ids.iter().cloned());
            }
            new_entries.insert(
                file.key.clone(),
                FileEntry {
                    mtime_nanos: file.mtime_nanos,
                    size: file.size,
                    chunk_ids: Vec::new(),
                },
            );
            continue;
        }

        if let Some(prev) = prev {
            to_delete_ids.extend(prev.chunk_ids.iter().cloned());
        }

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        for ch in chunks {
            chunk_ids.push(ch.id.clone());
            new_chunks.push(Document {
                id: ch.id,
                text: ch.text,
                metadata: HashMap::from([
                    ("path".to_string(), json!(&file.key)),
                    ("start_line".to_string(), json!(ch.start_line)),
                    ("end_line".to_string(), json!(ch.end_line)),
                ]),
            });
        }
        new_entries.insert(
            file.key.clone(),
            FileEntry {
                mtime_nanos: file.mtime_nanos,
                size: file.size,
                chunk_ids,
            },
        );
    }

    if to_delete_ids.is_empty() && new_chunks.is_empty() {
        if verbose {
            eprintln!("plume: nothing changed since last run");
        }
        return Ok(());
    }

    // Ensure the namespace exists before the first write.
    let _: Value = client
        .post_empty(&format!("/ns/{namespace}"))
        .await
        .unwrap_or(Value::Null);

    if !to_delete_ids.is_empty() {
        if verbose {
            eprintln!(
                "plume: pruning {} stale chunk(s) across {} removed/changed file(s)",
                to_delete_ids.len(),
                to_delete_rels.len() + new_entries.len()
            );
        }
        for batch in to_delete_ids.chunks(MAX_ROWS_PER_UPSERT) {
            let req = DeleteDocsRequest {
                ids: batch.to_vec(),
            };
            let _: DeleteDocsResponse = client
                .delete_with_body(&format!("/ns/{namespace}/docs"), &req)
                .await
                .with_context(|| format!("delete batch from namespace {namespace}"))?;
        }
    }

    if !new_chunks.is_empty() {
        eprintln!(
            "plume: indexing {} chunk(s) from {} changed file(s)",
            new_chunks.len(),
            new_entries.len()
        );
        for batch in new_chunks.chunks(MAX_ROWS_PER_UPSERT) {
            let req = UpsertRequest {
                rows: batch.to_vec(),
            };
            let _: UpsertResponse = client
                .post_json(&format!("/ns/{namespace}/upsert"), &req)
                .await
                .with_context(|| format!("upsert batch to namespace {namespace}"))?;
        }
    }

    // Commit manifest changes: drop removed rels, insert new entries.
    for rel in &to_delete_rels {
        manifest.files.remove(rel);
    }
    for (rel, entry) in new_entries {
        manifest.files.insert(rel, entry);
    }
    save_manifest(manifest_path, &manifest)?;
    Ok(())
}

// --- Chunker ---

#[derive(Debug, Clone)]
struct Chunk {
    id: String,
    text: String,
    start_line: usize, // 1-indexed
    end_line: usize,   // 1-indexed, inclusive
}

/// Split `content` into overlapping line-window chunks. Each chunk id is
/// `{rel}:{start_line}` so it's both human-readable and deterministic
/// across runs (safe to re-upsert or DELETE by id).
///
/// Big chunks above `MAX_TEXT_LENGTH` are truncated at byte boundary
/// (on a char boundary) rather than skipped — better to index what we
/// can than to drop the whole file.
fn chunk_file(rel: &str, content: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let step = CHUNK_WINDOW_LINES
        .saturating_sub(CHUNK_OVERLAP_LINES)
        .max(1);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + CHUNK_WINDOW_LINES).min(lines.len());
        let mut text = lines[start..end].join("\n");
        if text.len() > MAX_TEXT_LENGTH {
            // Walk back to a char boundary before truncating.
            let mut cut = MAX_TEXT_LENGTH;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
        }
        if !text.trim().is_empty() {
            out.push(Chunk {
                id: format!("{rel}:{}", start + 1),
                text,
                start_line: start + 1,
                end_line: end,
            });
        }
        if end >= lines.len() {
            break;
        }
        start += step;
    }
    out
}

// --- Scan + filters ---

struct FileFilters {
    include: Option<GlobSet>,
    exclude: Option<GlobSet>,
    excluded_dirs: HashSet<String>,
    /// Preserved for UX messaging when `--include` disables the extension
    /// allowlist.
    #[allow(dead_code)]
    include_overrides_extensions: bool,
}

impl FileFilters {
    fn from_args(args: &Args) -> Result<Self> {
        let include = build_globset(&args.include).context("compile --include glob")?;
        let exclude = build_globset(&args.exclude).context("compile --exclude glob")?;
        let mut excluded_dirs: HashSet<String> =
            EXCLUDED_DIRS.iter().map(|s| s.to_string()).collect();
        for extra in &args.exclude_dir {
            excluded_dirs.insert(extra.clone());
        }
        let include_overrides_extensions = include.is_some();
        Ok(Self {
            include,
            exclude,
            excluded_dirs,
            include_overrides_extensions,
        })
    }

    /// `rel` is the key relative to the scan root, always forward-slash
    /// normalized. Globs are matched against that relative form.
    fn accepts(&self, rel: &str, ext: &str) -> bool {
        if let Some(ex) = &self.exclude {
            if ex.is_match(rel) {
                return false;
            }
        }
        if let Some(inc) = &self.include {
            return inc.is_match(rel);
        }
        DEFAULT_EXTENSIONS.contains(&ext)
    }

    /// Project onto a [`super::source::Filter`] — the Source layer uses
    /// this to pre-filter before we ever issue a GET.
    fn to_source_filter(&self) -> Filter {
        let mut f = if self.include.is_none() {
            Filter::with_exts(DEFAULT_EXTENSIONS.iter().copied())
        } else {
            // With --include the extension allowlist is disabled; let the
            // source return everything and glob-match in `accepts`.
            Filter::any()
        };
        f.excluded_dirs = self.excluded_dirs.clone();
        f
    }
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        let glob = Glob::new(p).with_context(|| format!("invalid glob {p:?}"))?;
        builder.add(glob);
    }
    Ok(Some(builder.build()?))
}

async fn scan_source(source: &Source, filters: &FileFilters) -> Result<Vec<SourceFile>> {
    let source_filter = filters.to_source_filter();
    let raw = source.list(&source_filter).await?;
    // Second pass for include/exclude globs the source layer doesn't know
    // about, plus binary-probe short circuits that read_text handles later.
    let mut out = Vec::with_capacity(raw.len());
    for file in raw {
        if !filters.accepts(&file.key, &file.ext) {
            continue;
        }
        out.push(file);
    }
    Ok(out)
}

// --- Display ---

struct Palette {
    enabled: bool,
}

impl Palette {
    fn from_choice(choice: ColorChoice) -> Self {
        let enabled = match choice {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => std::io::stdout().is_terminal(),
        };
        Self { enabled }
    }

    fn path(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[1;36m{s}\x1b[0m")
        } else {
            s.into()
        }
    }

    fn line(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.into()
        }
    }

    fn score(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.into()
        }
    }
}

/// How to render a relative source key back to the user. Local sources
/// prepend the directory we were pointed at; remote sources prepend the
/// `s3://bucket/prefix/` URL so hits can be pasted into `aws s3 cp`.
enum DisplayRoot {
    LocalDir(PathBuf),
    LocalFile(PathBuf),
    Url(String),
}

impl DisplayRoot {
    fn for_source(source: &Source) -> Self {
        match source {
            Source::Local { root } => {
                let canonical = fs::canonicalize(root).unwrap_or_else(|_| root.clone());
                if canonical.is_file() {
                    Self::LocalFile(canonical)
                } else {
                    Self::LocalDir(canonical)
                }
            }
            Source::Remote(_) => Self::Url(source.display()),
        }
    }

    fn render(&self, key: &str) -> String {
        match self {
            Self::LocalDir(root) => root.join(key).display().to_string(),
            Self::LocalFile(path) => path.display().to_string(),
            Self::Url(url) => {
                if url.ends_with('/') {
                    format!("{url}{key}")
                } else {
                    format!("{url}/{key}")
                }
            }
        }
    }
}

fn print_hits(
    results: &[SearchResult],
    root: &DisplayRoot,
    show_line: bool,
    preview: usize,
    palette: &Palette,
) {
    for hit in results {
        let (path, line) = split_chunk_id(&hit.id);
        let path_str = palette.path(&root.render(&path));
        let score_str = palette.score(&format!("{:.4}", hit.score));
        if show_line {
            let line_str = palette.line(&line.to_string());
            println!("{path_str}:{line_str}  {score_str}");
        } else {
            println!("{path_str}  {score_str}");
        }
        if preview > 0 {
            for line in hit.text.lines().take(preview) {
                println!("    {line}");
            }
        }
    }
}

fn print_files_with_matches(results: &[SearchResult], root: &DisplayRoot, palette: &Palette) {
    let mut seen: HashSet<String> = HashSet::new();
    for hit in results {
        let (path, _) = split_chunk_id(&hit.id);
        if seen.insert(path.clone()) {
            println!("{}", palette.path(&root.render(&path)));
        }
    }
}

fn print_counts(results: &[SearchResult], root: &DisplayRoot, palette: &Palette) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for hit in results {
        let (path, _) = split_chunk_id(&hit.id);
        if !counts.contains_key(&path) {
            order.push(path.clone());
        }
        *counts.entry(path).or_insert(0) += 1;
    }
    for path in order {
        let path_str = palette.path(&root.render(&path));
        let c = counts[&path];
        println!("{path_str}:{c}");
    }
}

/// Chunk ids are `{rel}:{start_line}`. Split on the last `:` so paths
/// with embedded `:` (e.g. Windows-only) don't confuse us.
fn split_chunk_id(id: &str) -> (String, usize) {
    match id.rsplit_once(':') {
        Some((path, tail)) => {
            let line = tail.parse().unwrap_or(1);
            (path.to_string(), line)
        }
        None => (id.to_string(), 1),
    }
}

fn load_manifest(path: &Path) -> Manifest {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Manifest>(&raw).ok())
        .unwrap_or_default()
}

fn save_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string(manifest)?;
    fs::write(&tmp, body)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Stable namespace name for a local path. Thin wrapper around
/// [`namespace_for_identity`] that preserves byte-for-byte compatibility
/// with the pre-s3 manifest naming so old manifests keep working.
pub(super) fn namespace_for_path(abs_path: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    abs_path.hash(&mut hasher);
    format!("grep-{:016x}", hasher.finish())
}

/// Stable namespace name for an arbitrary source identity string. Used by
/// remote sources where we hash the URL rather than a filesystem path.
pub(super) fn namespace_for_identity(identity: &str) -> String {
    // Local paths still hash via the Path bytes so upgrading to a
    // Source-aware codebase doesn't orphan existing manifests.
    if !identity.starts_with("s3://") && !identity.starts_with("gs://") {
        return namespace_for_path(Path::new(identity));
    }
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    format!("grep-{:016x}", hasher.finish())
}

/// Root for plume's daemon working directory.
pub(super) fn plume_data_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("PLUME_DATA_DIR") {
        return Ok(PathBuf::from(explicit));
    }
    let base = if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg)
    } else {
        home_dir()?.join(".local").join("share")
    };
    Ok(base.join("plume"))
}

/// Root for per-path grep manifests.
pub(super) fn grep_state_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("PLUME_STATE_DIR") {
        return Ok(PathBuf::from(explicit).join("grep"));
    }
    let base = if let Ok(xdg) = std::env::var("XDG_STATE_HOME") {
        PathBuf::from(xdg)
    } else {
        home_dir()?.join(".local").join("state")
    };
    Ok(base.join("plume").join("grep"))
}

fn home_dir() -> Result<PathBuf> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

/// Manifest path for a given absolute directory. Kept for the
/// `namespace_for_path` backward-compat wrapper; `status` and `clear`
/// now use [`resolve_source_identity`] to handle remote sources too.
#[allow(dead_code)]
pub(super) fn manifest_path_for(abs_path: &Path) -> Result<PathBuf> {
    Ok(grep_state_dir()?.join(format!("{}.json", namespace_for_path(abs_path))))
}

/// Resolve a user-supplied input (local path or `s3://…` / `gs://…` URL)
/// into the (namespace, manifest_path, display_identity) triple that the
/// `status` / `clear` commands need.
pub(super) fn resolve_source_identity(input: &str) -> Result<SourceIdentity> {
    let source = Source::parse(input)?;
    let identity = source.identity();
    let namespace = namespace_for_identity(&identity);
    let manifest = grep_state_dir()?.join(format!("{namespace}.json"));
    Ok(SourceIdentity {
        namespace,
        manifest,
        display: identity,
    })
}

pub(super) struct SourceIdentity {
    pub namespace: String,
    pub manifest: PathBuf,
    pub display: String,
}

/// Public view onto a loaded manifest, for `status` / `clear` to count
/// files + chunks without re-deriving the format.
pub(super) struct ManifestSummary {
    pub files: usize,
    pub chunks: usize,
}

pub(super) fn summarize_manifest(path: &Path) -> ManifestSummary {
    let m = load_manifest(path);
    let chunks = m.files.values().map(|e| e.chunk_ids.len()).sum();
    ManifestSummary {
        files: m.files.len(),
        chunks,
    }
}

pub(super) fn remove_manifest(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn empty_filters() -> FileFilters {
        FileFilters {
            include: None,
            exclude: None,
            excluded_dirs: EXCLUDED_DIRS.iter().map(|s| s.to_string()).collect(),
            include_overrides_extensions: false,
        }
    }

    fn local_source(root: &Path) -> Source {
        Source::Local {
            root: root.to_path_buf(),
        }
    }

    #[test]
    fn namespace_is_deterministic_and_path_scoped() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        let a = fs::canonicalize(&a).unwrap();
        let b = fs::canonicalize(&b).unwrap();
        assert_eq!(namespace_for_path(&a), namespace_for_path(&a));
        assert_ne!(namespace_for_path(&a), namespace_for_path(&b));
        assert!(namespace_for_path(&a).starts_with("grep-"));
    }

    #[tokio::test]
    async fn scan_skips_excluded_dirs_and_unknown_extensions() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("node_modules/foo")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("notes.md"), "# notes").unwrap();
        fs::write(root.join("image.png"), b"not text").unwrap();
        fs::write(root.join("node_modules/foo/x.js"), "console.log(1)").unwrap();

        let results = scan_source(&local_source(root), &empty_filters())
            .await
            .unwrap();
        let names: Vec<&str> = results.iter().map(|f| f.key.as_str()).collect();

        assert!(names.iter().any(|n| n.ends_with("main.rs")));
        assert!(names.iter().any(|n| n.ends_with("notes.md")));
        assert!(!names.iter().any(|n| n.ends_with(".png")));
        assert!(!names.iter().any(|n| n.starts_with("node_modules")));
    }

    #[tokio::test]
    async fn include_glob_disables_default_extension_list() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::write(root.join("a.rs"), "x").unwrap();
        fs::write(root.join("b.log"), "x").unwrap();
        fs::write(root.join("c.md"), "x").unwrap();

        let filters = FileFilters {
            include: build_globset(&["*.log".into()]).unwrap(),
            exclude: None,
            excluded_dirs: HashSet::new(),
            include_overrides_extensions: true,
        };
        let results = scan_source(&local_source(root), &filters).await.unwrap();
        let names: Vec<&str> = results.iter().map(|f| f.key.as_str()).collect();
        assert_eq!(names, vec!["b.log"]);
    }

    #[tokio::test]
    async fn exclude_glob_filters_otherwise_accepted_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "x").unwrap();
        fs::write(root.join("src/generated.rs"), "x").unwrap();

        let filters = FileFilters {
            include: None,
            exclude: build_globset(&["**/generated.rs".into()]).unwrap(),
            excluded_dirs: HashSet::new(),
            include_overrides_extensions: false,
        };
        let results = scan_source(&local_source(root), &filters).await.unwrap();
        let names: Vec<String> = results.iter().map(|f| f.key.clone()).collect();
        assert!(names.iter().any(|n| n == "src/lib.rs"));
        assert!(!names.iter().any(|n| n.ends_with("generated.rs")));
    }

    #[tokio::test]
    async fn exclude_dir_adds_to_builtin_list() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("custom_skip")).unwrap();
        fs::write(root.join("custom_skip/file.rs"), "x").unwrap();
        fs::write(root.join("keep.rs"), "x").unwrap();

        let mut excluded = EXCLUDED_DIRS
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>();
        excluded.insert("custom_skip".to_string());
        let filters = FileFilters {
            include: None,
            exclude: None,
            excluded_dirs: excluded,
            include_overrides_extensions: false,
        };
        let results = scan_source(&local_source(root), &filters).await.unwrap();
        let names: Vec<&str> = results.iter().map(|f| f.key.as_str()).collect();
        assert!(names.iter().any(|n| n.ends_with("keep.rs")));
        assert!(!names.iter().any(|n| n.contains("custom_skip")));
    }

    #[tokio::test]
    async fn source_read_text_rejects_binary_and_reads_utf8() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.bin"), [0x00, 0x01, 0x02, 0x03]).unwrap();
        fs::write(tmp.path().join("a.txt"), "hello world").unwrap();

        let src = local_source(tmp.path());
        assert!(src.read_text("a.bin").await.unwrap().is_none());
        assert_eq!(
            src.read_text("a.txt").await.unwrap().as_deref(),
            Some("hello world")
        );
    }

    #[test]
    fn namespace_for_identity_is_s3_aware() {
        let local_ns = namespace_for_identity("/tmp/foo");
        let s3_ns = namespace_for_identity("s3://bucket/prefix");
        let gs_ns = namespace_for_identity("gs://bucket/prefix");
        assert!(local_ns.starts_with("grep-"));
        assert!(s3_ns.starts_with("grep-"));
        assert!(gs_ns.starts_with("grep-"));
        // Different identities produce different namespaces.
        assert_ne!(local_ns, s3_ns);
        assert_ne!(s3_ns, gs_ns);
        // Deterministic for the same identity.
        assert_eq!(s3_ns, namespace_for_identity("s3://bucket/prefix"));
    }

    #[test]
    fn manifest_roundtrip_preserves_chunk_ids() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("m.json");
        let mut m = Manifest::default();
        m.files.insert(
            "a.rs".into(),
            FileEntry {
                mtime_nanos: 123,
                size: 4,
                chunk_ids: vec!["a.rs:1".into(), "a.rs:31".into()],
            },
        );
        save_manifest(&path, &m).unwrap();
        let loaded = load_manifest(&path);
        let e = loaded.files.get("a.rs").unwrap();
        assert_eq!(e.size, 4);
        assert_eq!(
            e.chunk_ids,
            vec!["a.rs:1".to_string(), "a.rs:31".to_string()]
        );
    }

    #[test]
    fn manifest_missing_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let m = load_manifest(&tmp.path().join("none.json"));
        assert!(m.files.is_empty());
    }

    #[test]
    fn chunk_file_emits_overlapping_windows() {
        let content = (1..=100)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_file("f.rs", &content);
        // 100 lines, window=40, overlap=10, step=30:
        //   [0..40)  → lines 1..=40
        //   [30..70) → lines 31..=70
        //   [60..100) → lines 61..=100  (hits end, break)
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].id, "f.rs:1");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 40);
        assert_eq!(chunks[1].id, "f.rs:31");
        assert_eq!(chunks[1].start_line, 31);
        // Last window clamps at the line count.
        assert_eq!(chunks[2].end_line, 100);
        // Consecutive starts differ by step (window - overlap).
        assert_eq!(chunks[1].start_line - chunks[0].start_line, 30);
    }

    #[test]
    fn chunk_file_on_short_input_emits_one_chunk() {
        let chunks = chunk_file("small.md", "a\nb\nc");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn chunk_file_on_empty_yields_nothing() {
        assert!(chunk_file("e.md", "").is_empty());
        assert!(chunk_file("e.md", "   \n   ").is_empty());
    }

    fn args_with(query: Option<&str>, extras: &[&str]) -> Args {
        Args {
            query: query.map(String::from),
            path: ".".into(),
            extra_queries: extras.iter().map(|s| s.to_string()).collect(),
            k: 15,
            include: vec![],
            exclude: vec![],
            exclude_dir: vec![],
            files_with_matches: false,
            count: false,
            line_number: true,
            no_line_number: false,
            preview: 3,
            mode: Mode::Hybrid,
            color: ColorChoice::Never,
            json: false,
            no_index: false,
            no_daemon: true,
            verbose: false,
            url: DEFAULT_URL.into(),
        }
    }

    #[test]
    fn collect_queries_accepts_positional_alone() {
        let qs = collect_queries(&args_with(Some("hello"), &[])).unwrap();
        assert_eq!(qs, vec!["hello".to_string()]);
    }

    #[test]
    fn collect_queries_accepts_only_extra_queries() {
        let qs = collect_queries(&args_with(None, &["alpha", "beta"])).unwrap();
        assert_eq!(qs, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn collect_queries_fuses_positional_and_extras() {
        let qs = collect_queries(&args_with(Some("hello"), &["alpha"])).unwrap();
        assert_eq!(qs, vec!["hello".to_string(), "alpha".to_string()]);
    }

    #[test]
    fn collect_queries_rejects_all_empty() {
        let err = collect_queries(&args_with(None, &[])).unwrap_err();
        assert!(format!("{err}").contains("query is required"));
        let err = collect_queries(&args_with(Some(""), &[""])).unwrap_err();
        assert!(format!("{err}").contains("query is required"));
    }

    #[test]
    fn parse_bind_target_extracts_host_and_port() {
        let b = parse_bind_target("http://127.0.0.1:9999").unwrap();
        assert_eq!(b.host, "127.0.0.1");
        assert_eq!(b.port, 9999);

        let b = parse_bind_target("http://localhost:8787/").unwrap();
        assert_eq!(b.host, "localhost");
        assert_eq!(b.port, 8787);

        let b = parse_bind_target("http://[::1]:8080").unwrap();
        // url::Url::host_str strips brackets for IPv6 literals.
        assert_eq!(b.host, "[::1]");
        assert_eq!(b.port, 8080);
        assert!(b.is_local());
    }

    #[test]
    fn bind_target_is_local_recognizes_loopback_and_wildcard() {
        let cases_local = [
            "127.0.0.1",
            "127.1.2.3",
            "0.0.0.0",
            "localhost",
            "[::1]",
            "::1",
            "::",
        ];
        for host in cases_local {
            let b = BindTarget {
                host: host.into(),
                port: 1234,
            };
            assert!(b.is_local(), "expected {host:?} to be local");
        }

        let cases_remote = ["example.com", "10.0.0.5", "192.168.1.10", "8.8.8.8"];
        for host in cases_remote {
            let b = BindTarget {
                host: host.into(),
                port: 1234,
            };
            assert!(!b.is_local(), "expected {host:?} to NOT be local");
        }
    }

    #[test]
    fn split_chunk_id_round_trips() {
        assert_eq!(split_chunk_id("src/main.rs:42"), ("src/main.rs".into(), 42));
        assert_eq!(
            split_chunk_id("C:/Users/me/x.rs:12"),
            ("C:/Users/me/x.rs".into(), 12)
        );
        // No colon: everything is the path, line defaults to 1.
        assert_eq!(split_chunk_id("orphan"), ("orphan".into(), 1));
    }
}
