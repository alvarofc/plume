//! `plume ingest` — upload a corpus into a namespace.
//!
//! Accepts three shapes:
//!   - a `.jsonl` file (local only)
//!   - a local directory of `.md` / `.markdown` files
//!   - an `s3://bucket/prefix` or `gs://bucket/prefix` URL pointing at a
//!     collection of text files (default `.md` / `.markdown` / `.txt`)
//!
//! All paths stream: documents are yielded one at a time, batched up to
//! `batch_size`, and sent incrementally. Memory stays bounded by the
//! batch rather than the corpus size.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use plume_core::types::{Document, UpsertRequest, UpsertResponse, MAX_ROWS_PER_UPSERT};

use super::client::{Client, DEFAULT_URL};
use super::source::{Filter, Source};

/// Text-file extensions the directory / object-store walker will pick up.
/// Intentionally narrower than grep's default — ingest is about whole-file
/// docs, so code files belong in `grep` land instead.
const DEFAULT_TEXT_EXTS: &[&str] = &["md", "markdown", "txt", "rst", "org"];

#[derive(Parser)]
pub struct Args {
    /// Path to ingest. One of:
    ///   - `./corpus.jsonl` (local JSONL file)
    ///   - `./notes/` (local directory of .md files)
    ///   - `s3://bucket/prefix` or `gs://bucket/prefix` (remote)
    pub path: String,

    /// Namespace to upsert into.
    #[arg(short, long)]
    pub namespace: String,

    /// Plume server URL.
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,

    /// Documents per HTTP batch.
    #[arg(long, default_value_t = 500)]
    pub batch_size: usize,

    /// Comma-separated file extensions to pick up from directory / bucket
    /// sources (ignored for `.jsonl`). Default: md,markdown,txt,rst,org.
    #[arg(long, value_name = "CSV")]
    pub extensions: Option<String>,
}

pub async fn run(args: Args) -> Result<()> {
    if args.batch_size == 0 || args.batch_size > MAX_ROWS_PER_UPSERT {
        bail!(
            "batch-size must be between 1 and {MAX_ROWS_PER_UPSERT}, got {}",
            args.batch_size
        );
    }

    // JSONL is local-only; everything else goes through Source.
    if is_jsonl_path(&args.path) {
        return ingest_jsonl(&args).await;
    }

    let source = Source::parse(&args.path)?;
    if source.is_single_file() {
        bail!(
            "unsupported local file: {} (only .jsonl is supported as a single file; \
             for a directory of text files, point `plume ingest` at the directory)",
            args.path
        );
    }

    let exts = resolve_extensions(args.extensions.as_deref());
    let filter = Filter::with_exts(exts);

    let files = source
        .list(&filter)
        .await
        .with_context(|| format!("list {}", source.display()))?;
    if files.is_empty() {
        bail!(
            "no ingestable files under {} (filter: extensions={:?})",
            source.display(),
            resolve_extensions(args.extensions.as_deref())
        );
    }

    eprintln!(
        "plume: ingesting {} file(s) from {} into namespace '{}'",
        files.len(),
        source.display(),
        args.namespace
    );

    let client = Client::new(args.url.clone());
    let ns = &args.namespace;
    let mut batch: Vec<Document> = Vec::with_capacity(args.batch_size);
    let mut uploaded = 0usize;

    for file in files {
        let text = match source
            .read_text(&file.key)
            .await
            .with_context(|| format!("read {}", file.key))?
        {
            Some(t) => t,
            None => continue,
        };
        let id = id_from_key(&file.key);
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert("path".into(), serde_json::Value::String(file.key.clone()));
        batch.push(Document {
            id,
            text,
            metadata,
        });
        if batch.len() >= args.batch_size {
            uploaded += flush(&client, ns, &mut batch, args.batch_size).await?;
            eprintln!("plume: upserted {uploaded} docs into {ns}");
        }
    }
    if !batch.is_empty() {
        uploaded += flush(&client, ns, &mut batch, args.batch_size).await?;
        eprintln!("plume: upserted {uploaded} docs into {ns}");
    }

    if uploaded == 0 {
        bail!("no documents ingested from {}", source.display());
    }

    println!(
        "ingested {uploaded} documents into namespace '{ns}'. Indexes will build automatically."
    );
    Ok(())
}

async fn ingest_jsonl(args: &Args) -> Result<()> {
    let client = Client::new(args.url.clone());
    let ns = &args.namespace;
    let path = PathBuf::from(&args.path);
    let iter = jsonl_iter(&path)?;
    let mut batch: Vec<Document> = Vec::with_capacity(args.batch_size);
    let mut uploaded = 0usize;

    for doc in iter {
        batch.push(doc?);
        if batch.len() >= args.batch_size {
            uploaded += flush(&client, ns, &mut batch, args.batch_size).await?;
            eprintln!("plume: upserted {uploaded} docs into {ns}");
        }
    }
    if !batch.is_empty() {
        uploaded += flush(&client, ns, &mut batch, args.batch_size).await?;
        eprintln!("plume: upserted {uploaded} docs into {ns}");
    }

    if uploaded == 0 {
        bail!("no documents found at {}", args.path);
    }

    println!(
        "ingested {uploaded} documents into namespace '{ns}'. Indexes will build automatically."
    );
    Ok(())
}

async fn flush(
    client: &Client,
    namespace: &str,
    batch: &mut Vec<Document>,
    batch_size: usize,
) -> Result<usize> {
    let rows = std::mem::replace(batch, Vec::with_capacity(batch_size));
    let req = UpsertRequest { rows };
    let resp: UpsertResponse = client
        .post_json(&format!("/ns/{namespace}/upsert"), &req)
        .await?;
    Ok(resp.upserted)
}

type DocIter = Box<dyn Iterator<Item = Result<Document>>>;

fn is_jsonl_path(input: &str) -> bool {
    Path::new(input)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false)
}

fn jsonl_iter(path: &Path) -> Result<DocIter> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let path_display = path.display().to_string();
    let reader = BufReader::new(file);
    let iter = reader.lines().enumerate().filter_map(move |(i, line)| {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                return Some(Err(anyhow::Error::from(e)
                    .context(format!("read line {} of {path_display}", i + 1))));
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        let doc = serde_json::from_str::<Document>(trimmed)
            .with_context(|| format!("parse JSON on line {} of {path_display}", i + 1));
        Some(doc)
    });
    Ok(Box::new(iter))
}

fn id_from_key(key: &str) -> String {
    // Use the full key — including extension — so mixed-extension files
    // (`guide.md` vs `guide.txt`) upsert as distinct documents. An earlier
    // version stripped extensions for rename-tolerance, but that silently
    // collapsed two distinct files onto one id and clobbered data.
    key.to_string()
}

fn resolve_extensions(spec: Option<&str>) -> Vec<String> {
    match spec {
        Some(csv) => csv
            .split(',')
            .map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect(),
        None => DEFAULT_TEXT_EXTS.iter().map(|s| s.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_detection_is_extension_based() {
        assert!(is_jsonl_path("foo.jsonl"));
        assert!(is_jsonl_path("/abs/path.JSONL"));
        assert!(!is_jsonl_path("foo.md"));
        assert!(!is_jsonl_path("s3://bucket/data"));
    }

    #[test]
    fn id_from_key_preserves_extension() {
        // Distinct extensions must produce distinct ids so mixed-extension
        // files don't clobber each other on upsert.
        assert_eq!(id_from_key("foo/bar.md"), "foo/bar.md");
        assert_eq!(id_from_key("foo/bar.markdown"), "foo/bar.markdown");
        assert_ne!(id_from_key("guide.md"), id_from_key("guide.txt"));
        assert_eq!(id_from_key("weird/name"), "weird/name");
    }

    #[test]
    fn resolve_extensions_defaults_and_parses() {
        let d = resolve_extensions(None);
        assert!(d.iter().any(|e| e == "md"));
        assert!(d.iter().any(|e| e == "txt"));

        let parsed = resolve_extensions(Some(" .log, txt,  .JSON"));
        assert_eq!(parsed, vec!["log", "txt", "json"]);
    }
}
