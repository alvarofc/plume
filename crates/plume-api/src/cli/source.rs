//! Unified read source for `plume ingest` and `plume grep`.
//!
//! Callers pass a user-supplied path; [`Source::parse`] decides whether it's
//! a local filesystem path or a remote object-store URL (`s3://bucket/prefix`,
//! `gs://bucket/prefix`) and returns a [`Source`] that can list files and
//! read their contents with a single API shape.
//!
//! Remote sources are gated behind the `storage-aws` / `storage-gcs` Cargo
//! features so a default build stays lean. Passing an `s3://` URL to a binary
//! compiled without `storage-aws` fails fast with a build hint rather than a
//! runtime authentication mystery.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg_attr(
    not(any(feature = "storage-aws", feature = "storage-gcs")),
    allow(unused_imports)
)]
use anyhow::{bail, Context, Result};

/// A scannable file — what both ingest (whole-file docs) and grep (chunked)
/// iterate over. `key` is the source-relative identifier we use for:
///   - document / chunk ids (so re-ingest is idempotent),
///   - pretty-printing hits,
///   - the grep manifest keyed lookup.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Source-relative identifier. For local sources this is the relative
    /// path under the scan root (forward-slash normalized). For S3 it's
    /// the object key relative to the configured prefix.
    pub key: String,
    /// Mtime as nanos since the UNIX epoch. Used by grep to skip unchanged
    /// files across runs. S3 LastModified has ms precision; we convert.
    pub mtime_nanos: u128,
    /// Size in bytes. Used by grep as a cheap mtime tiebreaker.
    pub size: u64,
    /// File extension (lower-cased, no leading dot). Empty when the key
    /// has no extension.
    pub ext: String,
}

/// A resolved ingest / grep input.
pub enum Source {
    Local {
        root: PathBuf,
    },
    // Only constructed via parse_s3 / parse_gcs (feature-gated) or test
    // helpers. The default-features build has no way to reach this variant,
    // which is deliberate — the warning would flag a real regression.
    #[cfg_attr(
        not(any(feature = "storage-aws", feature = "storage-gcs", test)),
        allow(dead_code)
    )]
    Remote(RemoteSource),
}

impl Source {
    /// Parse a user-supplied path. Relative or absolute local paths return
    /// [`Source::Local`]; `s3://bucket/prefix` / `gs://bucket/prefix` URLs
    /// return [`Source::Remote`] when the matching cargo feature is
    /// enabled, otherwise error with a build hint.
    pub fn parse(input: &str) -> Result<Self> {
        if let Some(rest) = strip_scheme(input, "s3") {
            return parse_s3(rest);
        }
        if let Some(rest) = strip_scheme(input, "gs") {
            return parse_gcs(rest);
        }
        Ok(Self::Local {
            root: PathBuf::from(input),
        })
    }

    /// Human-readable source root for logs and the grep namespace hash.
    /// Local sources canonicalize; remote sources echo their URL.
    pub fn display(&self) -> String {
        match self {
            Self::Local { root } => stable_local_path(root).display().to_string(),
            Self::Remote(r) => r.display_url(),
        }
    }

    /// Stable identity for namespace hashing. For local sources this is
    /// the canonical filesystem path; for remote sources it's the URL.
    /// Either way it's stable across invocations from the same host.
    pub fn identity(&self) -> String {
        self.display()
    }

    /// `true` when the source points at a single-file local path (e.g.
    /// `./corpus.jsonl`). Remote and directory sources are `false`.
    pub fn is_single_file(&self) -> bool {
        match self {
            Self::Local { root } => root.is_file(),
            Self::Remote(_) => false,
        }
    }

    /// Local-only accessor for the resolved filesystem path. Used by
    /// callers that need to walk the tree with platform-native APIs
    /// (e.g. grep's WalkDir scan).
    pub fn as_local_root(&self) -> Option<&Path> {
        match self {
            Self::Local { root } => Some(root.as_path()),
            Self::Remote(_) => None,
        }
    }

    /// List every file under the source matching the provided filter.
    /// Implementations are paginated / streaming under the hood; the
    /// returned `Vec` is bounded by the source itself, not a slice.
    pub async fn list(&self, filter: &Filter) -> Result<Vec<SourceFile>> {
        match self {
            Self::Local { root } => local_list(root, filter),
            Self::Remote(r) => r.list(filter).await,
        }
    }

    /// Read a file's text content. Returns `Ok(None)` when the file looks
    /// binary (null byte in the first 4 KiB) or can't be decoded as UTF-8.
    /// Read errors propagate as `Err`.
    pub async fn read_text(&self, key: &str) -> Result<Option<String>> {
        match self {
            Self::Local { root } => {
                let path = root.join(key);
                // tokio::fs delegates to a blocking thread pool; keeps the
                // async caller from blocking the reactor on large files.
                let bytes = match tokio::fs::read(&path).await {
                    Ok(b) => b,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => {
                        return Err(anyhow::Error::from(e)
                            .context(format!("read {}", path.display())))
                    }
                };
                Ok(decode_text(&bytes))
            }
            Self::Remote(r) => r.read_text(key).await,
        }
    }
}

/// Predicate passed to [`Source::list`]. The source decides extension /
/// exclude-dir filtering so remote sources can skip unwanted objects before
/// we ever pay the LIST-after-GET round trip.
pub struct Filter {
    pub allowed_exts: Option<Vec<String>>, // None = accept any extension
    pub excluded_dirs: std::collections::HashSet<String>,
}

impl Filter {
    pub fn any() -> Self {
        Self {
            allowed_exts: None,
            excluded_dirs: std::collections::HashSet::new(),
        }
    }

    pub fn with_exts(exts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_exts: Some(exts.into_iter().map(Into::into).collect()),
            excluded_dirs: std::collections::HashSet::new(),
        }
    }

    pub fn accepts_key(&self, key: &str, ext: &str) -> bool {
        if !self.excluded_dirs.is_empty() {
            for seg in key.split('/') {
                if self.excluded_dirs.contains(seg) {
                    return false;
                }
            }
        }
        match &self.allowed_exts {
            Some(list) => list.iter().any(|e| e == ext),
            None => true,
        }
    }
}

// ---------- URL parsing ----------

fn strip_scheme<'a>(input: &'a str, scheme: &str) -> Option<&'a str> {
    let prefix = format!("{scheme}://");
    input.strip_prefix(&prefix)
}

/// `bucket[/prefix...]` → (bucket, prefix). Prefix never has a leading `/`.
/// Empty prefix means "root of bucket".
#[cfg_attr(
    not(any(feature = "storage-aws", feature = "storage-gcs", test)),
    allow(dead_code)
)]
pub(crate) fn split_bucket_and_prefix(raw: &str) -> Result<(String, String)> {
    let raw = raw.trim_end_matches('/');
    if raw.is_empty() {
        bail!("missing bucket name");
    }
    let (bucket, prefix) = match raw.split_once('/') {
        Some((b, p)) => (b.to_string(), p.trim_start_matches('/').to_string()),
        None => (raw.to_string(), String::new()),
    };
    if bucket.is_empty() {
        bail!("missing bucket name");
    }
    Ok((bucket, prefix))
}

#[cfg(feature = "storage-aws")]
fn parse_s3(rest: &str) -> Result<Source> {
    let (bucket, prefix) = split_bucket_and_prefix(rest).context("parse s3:// URL")?;
    let store = remote::build_s3(&bucket)?;
    Ok(Source::Remote(RemoteSource {
        scheme: RemoteScheme::S3,
        bucket,
        prefix,
        store,
    }))
}

#[cfg(not(feature = "storage-aws"))]
fn parse_s3(_rest: &str) -> Result<Source> {
    bail!(
        "s3:// sources require the `storage-aws` Cargo feature — \
         rebuild with: cargo install --path crates/plume-api --features storage-aws"
    );
}

#[cfg(feature = "storage-gcs")]
fn parse_gcs(rest: &str) -> Result<Source> {
    let (bucket, prefix) = split_bucket_and_prefix(rest).context("parse gs:// URL")?;
    let store = remote::build_gcs(&bucket)?;
    Ok(Source::Remote(RemoteSource {
        scheme: RemoteScheme::Gcs,
        bucket,
        prefix,
        store,
    }))
}

#[cfg(not(feature = "storage-gcs"))]
fn parse_gcs(_rest: &str) -> Result<Source> {
    bail!(
        "gs:// sources require the `storage-gcs` Cargo feature — \
         rebuild with: cargo install --path crates/plume-api --features storage-gcs"
    );
}

// ---------- Local implementation ----------

fn local_list(root: &Path, filter: &Filter) -> Result<Vec<SourceFile>> {
    use walkdir::WalkDir;

    if !root.exists() {
        bail!("path does not exist: {}", root.display());
    }

    let mut out = Vec::new();
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        if e.path() == root {
            return true;
        }
        if e.file_type().is_dir() {
            let name = e.file_name().to_str().unwrap_or_default();
            return !filter.excluded_dirs.contains(name);
        }
        true
    });
    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        if !filter.accepts_key(&rel, &ext) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        out.push(SourceFile {
            key: rel,
            mtime_nanos: mtime,
            size: metadata.len(),
            ext,
        });
    }
    Ok(out)
}

/// Stable absolute path for a local input, whether or not the path
/// currently exists on disk. Canonicalizes the longest existing
/// ancestor (resolving symlinks in that prefix) and reattaches the
/// missing tail, so `plume status` / `plume clear` keep pointing at
/// the same namespace after the directory has been removed.
pub(crate) fn stable_local_path(root: &Path) -> PathBuf {
    let abs = absolute_path(root);
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor = abs.clone();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&cursor) {
            let mut resolved = canonical;
            for component in tail.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        match cursor.file_name() {
            Some(name) => tail.push(name.to_os_string()),
            None => return abs,
        }
        if !cursor.pop() {
            return abs;
        }
    }
}

fn absolute_path(root: &Path) -> PathBuf {
    if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(root))
            .unwrap_or_else(|_| root.to_path_buf())
    }
}

/// Decode bytes as UTF-8 text, skipping anything that looks binary
/// (null byte in the first 4 KiB). Returning `None` here means "this
/// file isn't indexable", which is distinct from an I/O error.
fn decode_text(bytes: &[u8]) -> Option<String> {
    let probe_len = bytes.len().min(4096);
    if bytes[..probe_len].contains(&0) {
        return None;
    }
    String::from_utf8(bytes.to_vec()).ok()
}

// ---------- Remote implementation ----------

#[derive(Copy, Clone, Debug)]
pub enum RemoteScheme {
    #[cfg_attr(not(feature = "storage-aws"), allow(dead_code))]
    S3,
    #[cfg_attr(not(feature = "storage-gcs"), allow(dead_code))]
    Gcs,
    /// Used by unit tests that back [`RemoteSource`] with an in-memory
    /// `ObjectStore`. Never produced by [`Source::parse`].
    #[cfg(test)]
    Memory,
}

impl RemoteScheme {
    fn prefix(&self) -> &'static str {
        match self {
            Self::S3 => "s3",
            Self::Gcs => "gs",
            #[cfg(test)]
            Self::Memory => "mem",
        }
    }
}

/// Object-store-backed corpus. Always compiled so tests can inject an
/// in-memory store; only `parse_s3` / `parse_gcs` are feature-gated so
/// production binaries without the flag can't accidentally depend on
/// real-cloud credentials.
pub struct RemoteSource {
    scheme: RemoteScheme,
    pub bucket: String,
    pub prefix: String,
    store: std::sync::Arc<dyn object_store::ObjectStore>,
}

impl RemoteSource {
    #[cfg(test)]
    pub(crate) fn new_in_memory(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        store: std::sync::Arc<dyn object_store::ObjectStore>,
    ) -> Self {
        Self {
            scheme: RemoteScheme::Memory,
            bucket: bucket.into(),
            prefix: prefix.into(),
            store,
        }
    }

    pub fn display_url(&self) -> String {
        format!("{}://{}/{}", self.scheme.prefix(), self.bucket, self.prefix)
    }

    async fn list(&self, filter: &Filter) -> Result<Vec<SourceFile>> {
        use futures::StreamExt;
        use object_store::path::Path as OsPath;

        let prefix_owned;
        let prefix = if self.prefix.is_empty() {
            None
        } else {
            prefix_owned = OsPath::from(self.prefix.clone());
            Some(&prefix_owned)
        };

        let mut stream = self.store.list(prefix);
        let mut out = Vec::new();
        while let Some(meta) = stream.next().await {
            let meta = meta.context("list objects")?;
            let abs_key = meta.location.as_ref().to_string();
            // Relative to the configured prefix so keys survive as doc ids.
            let rel_key = if self.prefix.is_empty() {
                abs_key.clone()
            } else {
                abs_key
                    .strip_prefix(&self.prefix)
                    .unwrap_or(&abs_key)
                    .trim_start_matches('/')
                    .to_string()
            };
            if rel_key.is_empty() {
                continue;
            }
            let ext = std::path::Path::new(&rel_key)
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            if !filter.accepts_key(&rel_key, &ext) {
                continue;
            }
            let mtime_nanos = meta
                .last_modified
                .timestamp_nanos_opt()
                .map(|n| n as u128)
                .unwrap_or(0);
            out.push(SourceFile {
                key: rel_key,
                mtime_nanos,
                size: meta.size,
                ext,
            });
        }
        Ok(out)
    }

    async fn read_text(&self, key: &str) -> Result<Option<String>> {
        use object_store::path::Path as OsPath;

        let full = if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), key)
        };
        let path = OsPath::from(full);
        let bytes = self
            .store
            .get(&path)
            .await
            .with_context(|| format!("get {key}"))?
            .bytes()
            .await
            .with_context(|| format!("read body {key}"))?;
        Ok(decode_text(&bytes))
    }
}

#[cfg(any(feature = "storage-aws", feature = "storage-gcs"))]
mod remote {
    use super::*;
    use std::sync::Arc;

    #[cfg(feature = "storage-aws")]
    pub(super) fn build_s3(bucket: &str) -> Result<Arc<dyn object_store::ObjectStore>> {
        use object_store::aws::AmazonS3Builder;
        let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
        if let Ok(endpoint) =
            std::env::var("AWS_ENDPOINT").or_else(|_| std::env::var("AWS_ENDPOINT_URL"))
        {
            builder = builder.with_endpoint(endpoint).with_allow_http(true);
        }
        if let Ok(region) =
            std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        {
            builder = builder.with_region(region);
        }
        let s3 = builder
            .build()
            .context("build S3 client (check AWS_REGION and credentials)")?;
        Ok(Arc::new(s3))
    }

    #[cfg(feature = "storage-gcs")]
    pub(super) fn build_gcs(bucket: &str) -> Result<Arc<dyn object_store::ObjectStore>> {
        use object_store::gcp::GoogleCloudStorageBuilder;
        let gcs = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .context(
                "build GCS client (check GOOGLE_SERVICE_ACCOUNT or GOOGLE_APPLICATION_CREDENTIALS)",
            )?;
        Ok(Arc::new(gcs))
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_local_returns_local_source() {
        let s = Source::parse("./corpus").unwrap();
        assert!(matches!(s, Source::Local { .. }));
        let s = Source::parse("/abs/path").unwrap();
        assert!(matches!(s, Source::Local { .. }));
    }

    #[test]
    fn split_bucket_and_prefix_handles_bucket_only() {
        let (b, p) = split_bucket_and_prefix("my-bucket").unwrap();
        assert_eq!(b, "my-bucket");
        assert_eq!(p, "");
    }

    #[test]
    fn split_bucket_and_prefix_handles_nested_prefix() {
        let (b, p) = split_bucket_and_prefix("my-bucket/docs/2024").unwrap();
        assert_eq!(b, "my-bucket");
        assert_eq!(p, "docs/2024");
    }

    #[test]
    fn split_bucket_and_prefix_strips_trailing_slash() {
        let (b, p) = split_bucket_and_prefix("b/pre/").unwrap();
        assert_eq!(b, "b");
        assert_eq!(p, "pre");
    }

    #[test]
    fn split_bucket_and_prefix_rejects_empty() {
        assert!(split_bucket_and_prefix("").is_err());
        assert!(split_bucket_and_prefix("/").is_err());
    }

    #[cfg(not(feature = "storage-aws"))]
    #[test]
    fn s3_url_errors_without_feature() {
        let err = match Source::parse("s3://bucket/prefix") {
            Err(e) => e,
            Ok(_) => panic!("expected s3:// without storage-aws to fail"),
        };
        let msg = format!("{err}");
        assert!(msg.contains("storage-aws"), "got: {msg}");
    }

    #[test]
    fn filter_rejects_unallowed_ext() {
        let f = Filter::with_exts(["md", "rs"]);
        assert!(f.accepts_key("a/b.md", "md"));
        assert!(f.accepts_key("src/lib.rs", "rs"));
        assert!(!f.accepts_key("a.png", "png"));
    }

    #[test]
    fn filter_rejects_excluded_dir_segment() {
        let mut f = Filter::any();
        f.excluded_dirs.insert("node_modules".into());
        assert!(f.accepts_key("src/lib.rs", "rs"));
        assert!(!f.accepts_key("node_modules/foo/bar.js", "js"));
        assert!(!f.accepts_key("app/node_modules/foo.js", "js"));
    }

    #[test]
    fn local_list_walks_a_directory() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("a")).unwrap();
        std::fs::write(tmp.path().join("a/one.md"), "hi").unwrap();
        std::fs::write(tmp.path().join("a/two.txt"), "bye").unwrap();
        std::fs::write(tmp.path().join("binary.png"), [0u8, 1, 2]).unwrap();
        let filter = Filter::with_exts(["md", "txt"]);
        let files = local_list(tmp.path(), &filter).unwrap();
        let mut keys: Vec<String> = files.into_iter().map(|f| f.key).collect();
        keys.sort();
        assert_eq!(keys, vec!["a/one.md", "a/two.txt"]);
    }

    #[test]
    fn stable_local_path_survives_leaf_deletion() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("repo");
        std::fs::create_dir_all(&target).unwrap();
        let before = Source::Local {
            root: target.clone(),
        }
        .identity();
        std::fs::remove_dir_all(&target).unwrap();
        assert!(!target.exists());
        let after = Source::Local {
            root: target.clone(),
        }
        .identity();
        assert_eq!(
            before, after,
            "identity should be stable across leaf deletion"
        );
    }

    #[test]
    fn stable_local_path_is_absolute_for_relative_input() {
        let rel = PathBuf::from("definitely-does-not-exist-xyz");
        let p = stable_local_path(&rel);
        assert!(
            p.is_absolute(),
            "stable_local_path must return an absolute path, got {}",
            p.display()
        );
    }

    #[tokio::test]
    async fn remote_source_lists_and_reads_from_in_memory_store() {
        use object_store::memory::InMemory;
        use object_store::path::Path as OsPath;
        use object_store::ObjectStore;
        use std::sync::Arc;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        store
            .put(&OsPath::from("docs/a.md"), b"alpha".to_vec().into())
            .await
            .unwrap();
        store
            .put(&OsPath::from("docs/nested/b.txt"), b"beta".to_vec().into())
            .await
            .unwrap();
        store
            .put(&OsPath::from("docs/skip.png"), vec![0u8, 1, 2].into())
            .await
            .unwrap();
        store
            .put(&OsPath::from("other/c.md"), b"outside prefix".to_vec().into())
            .await
            .unwrap();

        let remote = RemoteSource::new_in_memory("bucket", "docs", store);
        let source = Source::Remote(remote);

        let files = source.list(&Filter::with_exts(["md", "txt"])).await.unwrap();
        let mut keys: Vec<String> = files.into_iter().map(|f| f.key).collect();
        keys.sort();
        assert_eq!(keys, vec!["a.md", "nested/b.txt"]);

        let text = source.read_text("a.md").await.unwrap();
        assert_eq!(text.as_deref(), Some("alpha"));

        let nested = source.read_text("nested/b.txt").await.unwrap();
        assert_eq!(nested.as_deref(), Some("beta"));
    }

    #[tokio::test]
    async fn remote_source_read_text_rejects_binary() {
        use object_store::memory::InMemory;
        use object_store::path::Path as OsPath;
        use object_store::ObjectStore;
        use std::sync::Arc;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        store
            .put(&OsPath::from("bin.dat"), vec![0u8, 1, 2, 3].into())
            .await
            .unwrap();
        let source = Source::Remote(RemoteSource::new_in_memory("bucket", "", store));
        assert!(source.read_text("bin.dat").await.unwrap().is_none());
    }
}
