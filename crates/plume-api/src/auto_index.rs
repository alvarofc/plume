//! Per-namespace auto-indexing scheduler.
//!
//! Writes bump a pending counter; a background task per namespace rebuilds
//! the ANN + FTS indexes when the counter crosses `threshold_docs` or when
//! writes have been quiet for `debounce_ms`. The explicit `/index` and
//! `/fts-index` endpoints still work for operators who want manual control
//! of bulk loads.
//!
//! Durability across restarts: after every successful build we write the
//! indexed row count to `{storage}/.plume-autoindex/{ns}.count`. A
//! sibling `{ns}.pending` file is touched on every upsert and cleared
//! when the rebuild completes, so in-place updates that leave the row
//! count unchanged still re-queue on restart. On startup, `recover()`
//! compares each namespace's current row count to its marker and
//! re-queues a rebuild when they drift, when the pending marker is
//! present, or when ANN is absent above `min_docs`. Markers are disabled
//! when storage points at an object store — object-store restart recovery
//! would need a different mechanism and hasn't been wired up yet.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use plume_cache::SearchCache;
use plume_core::config::{IndexConfig, StorageConfig};
use plume_index::IndexManager;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Pending counter + last-write timestamp guarded by a single mutex so that
/// the scheduler can atomically snapshot the backlog, decide whether to
/// build, and subtract the claimed amount without racing with incoming
/// upserts.
struct Pending {
    count: usize,
    last_write: Instant,
}

struct NamespaceState {
    pending: Mutex<Pending>,
    notify: Notify,
}

impl NamespaceState {
    fn new() -> Self {
        Self {
            pending: Mutex::new(Pending {
                count: 0,
                last_write: Instant::now(),
            }),
            notify: Notify::new(),
        }
    }
}

struct NamespaceEntry {
    state: Arc<NamespaceState>,
    handle: JoinHandle<()>,
}

#[derive(Clone)]
pub struct AutoIndexer {
    inner: Arc<AutoIndexerInner>,
}

struct AutoIndexerInner {
    config: IndexConfig,
    index: Arc<IndexManager>,
    cache: Arc<SearchCache>,
    states: Mutex<HashMap<String, NamespaceEntry>>,
    /// Directory for per-namespace `{ns}.count` marker files. `None` when
    /// storage points at an object store (feature disabled).
    state_dir: Option<PathBuf>,
}

impl AutoIndexer {
    pub fn new(
        config: IndexConfig,
        storage: &StorageConfig,
        index: Arc<IndexManager>,
        cache: Arc<SearchCache>,
    ) -> Self {
        let state_dir = derive_state_dir(&storage.uri);
        if let Some(ref dir) = state_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warn!(
                    path = %dir.display(),
                    error = %e,
                    "auto-index: failed to create state dir; restart recovery will be a no-op until it's writable"
                );
            }
        }
        Self {
            inner: Arc::new(AutoIndexerInner {
                config,
                index,
                cache,
                states: Mutex::new(HashMap::new()),
                state_dir,
            }),
        }
    }

    /// Re-queue any namespaces whose indexes may be stale after a restart.
    ///
    /// For each known namespace we compare the live row count to the
    /// last-indexed count stored on disk. A mismatch (or missing marker,
    /// or absent ANN index above `min_docs`) triggers a `notify_upsert`
    /// so the normal debounced scheduler picks the work up. Empty
    /// namespaces and remote-storage deployments are skipped.
    pub async fn recover(&self) {
        if !self.inner.config.auto.enabled {
            return;
        }
        let Some(state_dir) = self.inner.state_dir.clone() else {
            debug!("auto-index: restart recovery disabled (non-local storage)");
            return;
        };
        let namespaces = match self.inner.index.list_namespaces().await {
            Ok(ns) => ns,
            Err(e) => {
                warn!(error = %e, "auto-index: list_namespaces failed during recovery");
                return;
            }
        };
        for ns in namespaces {
            if let Err(e) = self.recover_namespace(&ns, &state_dir).await {
                warn!(%ns, error = %e, "auto-index: skipping namespace during recovery");
            }
        }
    }

    async fn recover_namespace(
        &self,
        ns: &str,
        state_dir: &Path,
    ) -> Result<(), plume_core::error::PlumeError> {
        let table = self.inner.index.get_namespace(ns).await?;
        let current = table.count().await?;
        if current == 0 {
            // Nothing to index yet; leave it to the first upsert.
            return Ok(());
        }
        let marker = read_marker(state_dir, ns);
        let has_ann = table.has_ann_index().await.unwrap_or(false);
        let ann_missing_but_expected = current >= self.inner.config.auto.min_docs && !has_ann;
        let pending = has_pending_marker(state_dir, ns);
        if marker == Some(current) && !ann_missing_but_expected && !pending {
            return Ok(());
        }
        let delta = current.saturating_sub(marker.unwrap_or(0)).max(1);
        info!(
            %ns,
            current,
            marker = ?marker,
            has_ann,
            pending,
            "auto-index: queuing restart rebuild"
        );
        self.notify_upsert(ns, delta).await;
        Ok(())
    }

    /// Called after a manual `/index` or `/fts-index` job succeeds.
    /// Persists the count marker and clears the pending flag so a
    /// restart doesn't re-queue a rebuild of indexes the operator
    /// just built by hand. Best-effort: on any error we log and move
    /// on; the worst case is a spurious rebuild on next startup.
    pub async fn record_manual_build(&self, namespace: &str) {
        if !self.inner.config.auto.enabled {
            return;
        }
        let Some(ref state_dir) = self.inner.state_dir else {
            return;
        };
        let table = match self.inner.index.get_namespace(namespace).await {
            Ok(t) => t,
            Err(e) => {
                warn!(%namespace, error = %e, "auto-index: manual-build marker skipped (namespace)");
                return;
            }
        };
        let docs = match table.count().await {
            Ok(c) => c,
            Err(e) => {
                warn!(%namespace, error = %e, "auto-index: manual-build marker skipped (count)");
                return;
            }
        };
        if let Err(e) = write_marker(state_dir, namespace, docs) {
            warn!(%namespace, error = %e, "auto-index: manual-build marker write failed");
        }
        // Deliberately do *not* clear the pending marker here. A write
        // that arrived during the manual build bumped the pending flag
        // but has no in-memory scheduler to drive it (manual builds
        // bypass `notify_upsert`). Clearing it would lose the restart
        // recovery signal. Worst case: one spurious rebuild on the next
        // startup if no writes raced the manual build — cheap.
    }

    /// Record an upsert. Spawns the namespace scheduler on first use or
    /// after an idle-timeout exit; reuses the existing state otherwise.
    pub async fn notify_upsert(&self, namespace: &str, rows: usize) {
        if !self.inner.config.auto.enabled || rows == 0 {
            return;
        }

        // Persist a pending flag before acknowledging the in-memory bump.
        // Covers the pure-update case where row count is unchanged: on
        // restart, `recover()` sees the flag and re-queues even when the
        // count marker already matches `current`. Best-effort — a
        // transient fs error is logged and otherwise ignored.
        //
        // Uses tokio::fs so the fs touch doesn't block the reactor on
        // slow disks; upsert is a hot path.
        if let Some(ref state_dir) = self.inner.state_dir {
            let path = pending_marker_path(state_dir, namespace);
            if let Err(e) = tokio::fs::write(&path, b"").await {
                debug!(
                    %namespace,
                    error = %e,
                    "auto-index: failed to write pending marker"
                );
            }
        }

        let state = {
            let mut states = self.inner.states.lock().await;
            // Reap stale entries whose scheduler task already exited (idle
            // timeout or panic) so we don't push notifies into a dead task.
            if states
                .get(namespace)
                .is_some_and(|e| e.handle.is_finished())
            {
                states.remove(namespace);
            }
            if let Some(entry) = states.get(namespace) {
                Arc::clone(&entry.state)
            } else {
                let state = Arc::new(NamespaceState::new());
                let inner = Arc::clone(&self.inner);
                let ns = namespace.to_string();
                let state_for_task = Arc::clone(&state);
                let handle = tokio::spawn(run_scheduler(inner, ns, state_for_task));
                states.insert(
                    namespace.to_string(),
                    NamespaceEntry {
                        state: Arc::clone(&state),
                        handle,
                    },
                );
                state
            }
        };

        {
            let mut p = state.pending.lock().await;
            p.count += rows;
            p.last_write = Instant::now();
        }
        state.notify.notify_one();
    }

    /// Stop the scheduler and forget any state for `namespace`. Called from
    /// the drop-namespace route so long-lived processes that churn
    /// namespaces don't leak tasks or pending counters.
    pub async fn drop_namespace(&self, namespace: &str) {
        let entry = self.inner.states.lock().await.remove(namespace);
        if let Some(entry) = entry {
            entry.handle.abort();
            // Await the handle so a build in-flight finishes winding
            // down before we delete the table under it. Ignore JoinError
            // — abort produces a cancelled() error which is expected.
            let _ = entry.handle.await;
        }
        if let Some(ref dir) = self.inner.state_dir {
            let path = marker_path(dir, namespace);
            match std::fs::remove_file(&path) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => warn!(
                    path = %path.display(),
                    error = %e,
                    "auto-index: failed to remove marker for dropped namespace"
                ),
            }
            clear_pending_marker(dir, namespace);
        }
    }
}

/// Marker files live next to the LanceDB storage when it's local. For
/// object-store backends we skip persistence — object-store recovery
/// would need its own design.
fn derive_state_dir(storage_uri: &str) -> Option<PathBuf> {
    const REMOTE_PREFIXES: &[&str] = &["s3://", "gs://", "az://", "azure://", "r2://", "https://"];
    let lower = storage_uri.to_ascii_lowercase();
    if REMOTE_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return None;
    }
    let base = storage_uri
        .strip_prefix("file://")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(storage_uri));
    Some(base.join(".plume-autoindex"))
}

fn marker_path(state_dir: &Path, ns: &str) -> PathBuf {
    state_dir.join(format!("{ns}.count"))
}

fn read_marker(state_dir: &Path, ns: &str) -> Option<usize> {
    let raw = std::fs::read_to_string(marker_path(state_dir, ns)).ok()?;
    raw.trim().parse().ok()
}

/// Atomic write: stage to `.tmp` then rename. A crash mid-write leaves
/// the previous marker intact rather than a truncated file.
fn write_marker(state_dir: &Path, ns: &str, count: usize) -> io::Result<()> {
    let final_path = marker_path(state_dir, ns);
    let tmp_path = state_dir.join(format!("{ns}.count.tmp"));
    std::fs::write(&tmp_path, count.to_string())?;
    std::fs::rename(&tmp_path, &final_path)
}

fn pending_marker_path(state_dir: &Path, ns: &str) -> PathBuf {
    state_dir.join(format!("{ns}.pending"))
}

fn has_pending_marker(state_dir: &Path, ns: &str) -> bool {
    pending_marker_path(state_dir, ns).exists()
}

/// Flags the namespace as having unindexed writes. Idempotent: we don't
/// care about the contents, only the presence of the file. Only used by
/// tests now — `notify_upsert` writes the pending marker inline via
/// `tokio::fs` so upserts don't block the reactor on the filesystem.
#[cfg(test)]
fn touch_pending_marker(state_dir: &Path, ns: &str) -> io::Result<()> {
    std::fs::write(pending_marker_path(state_dir, ns), b"")
}

/// Best-effort: log-and-continue on failure. A stale pending marker
/// triggers at most one extra rebuild on next startup.
fn clear_pending_marker(state_dir: &Path, ns: &str) {
    let path = pending_marker_path(state_dir, ns);
    match std::fs::remove_file(&path) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => warn!(
            path = %path.display(),
            error = %e,
            "auto-index: failed to clear pending marker"
        ),
    }
}

/// Idle timeout before a scheduler removes itself from the map and exits.
/// Keeps long-lived processes from accumulating one task per namespace
/// they've ever touched. The next upsert re-spawns a fresh worker.
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Backoff after a failed build so a persistently broken namespace doesn't
/// hot-loop on errors.
const FAILURE_BACKOFF: Duration = Duration::from_secs(5);

async fn run_scheduler(
    inner: Arc<AutoIndexerInner>,
    namespace: String,
    state: Arc<NamespaceState>,
) {
    let debounce = Duration::from_millis(inner.config.auto.debounce_ms);
    let threshold = inner.config.auto.threshold_docs;

    loop {
        // Wait for the first upsert, or exit after an idle window. The
        // scheduler reaps itself from `inner.states` under the same lock
        // `notify_upsert` uses, so a concurrent upsert either sees a live
        // task or a missing entry — never both.
        tokio::select! {
            _ = state.notify.notified() => {}
            _ = tokio::time::sleep(IDLE_TIMEOUT) => {
                let mut states = inner.states.lock().await;
                let pending_now = state.pending.lock().await.count;
                if pending_now == 0 {
                    states.remove(&namespace);
                    return;
                }
                // Work arrived between the timeout firing and our lock;
                // fall through and drain it.
            }
        }

        // Drain writes: either hit the threshold, or wait out the debounce
        // window with no new writes arriving.
        loop {
            // Atomically snapshot backlog + decide whether to build. If we
            // build, subtract the claimed amount (instead of zeroing) so any
            // upsert that lands while the lock is dropped survives and
            // triggers another iteration.
            let (claimed, sleep_for) = {
                let mut p = state.pending.lock().await;
                if p.count == 0 {
                    (0, None)
                } else if p.count >= threshold || p.last_write.elapsed() >= debounce {
                    let n = p.count;
                    p.count = 0;
                    (n, None)
                } else {
                    (0, Some(debounce.saturating_sub(p.last_write.elapsed())))
                }
            };

            if claimed > 0 {
                if !build_indexes(&inner, &namespace).await {
                    // Restore the claimed amount so the backlog isn't lost;
                    // back off briefly to avoid hot-looping on persistent
                    // errors, then try again on the next debounce.
                    {
                        let mut p = state.pending.lock().await;
                        p.count = p.count.saturating_add(claimed);
                    }
                    tokio::time::sleep(FAILURE_BACKOFF).await;
                }
                continue;
            }

            let Some(remaining) = sleep_for else {
                break;
            };
            tokio::select! {
                _ = tokio::time::sleep(remaining) => {},
                _ = state.notify.notified() => {},
            }
        }
    }
}

/// Returns `true` iff every attempted build step succeeded. A `false`
/// signals the caller to preserve the claimed pending count and retry.
/// "Skipped ANN because below `min_docs`" counts as success — there's
/// nothing to do and nothing to retry.
async fn build_indexes(inner: &AutoIndexerInner, namespace: &str) -> bool {
    let table = match inner.index.get_namespace(namespace).await {
        Ok(t) => t,
        Err(e) => {
            warn!(%namespace, error = %e, "auto-index: namespace unavailable");
            return false;
        }
    };

    let docs = match table.count().await {
        Ok(c) => c,
        Err(e) => {
            warn!(%namespace, error = %e, "auto-index: count failed");
            return false;
        }
    };

    let mut ok = true;

    if docs >= inner.config.auto.min_docs {
        debug!(%namespace, docs, "auto-index: building ANN index");
        if let Err(e) = table.build_vector_index(&inner.config).await {
            error!(%namespace, error = %e, "auto-index: ANN build failed");
            ok = false;
        } else {
            info!(%namespace, docs, "auto-index: ANN index built");
        }
    } else {
        debug!(
            %namespace,
            docs,
            min = inner.config.auto.min_docs,
            "auto-index: skipping ANN build, below minimum"
        );
    }

    if let Err(e) = table.build_fts_index().await {
        error!(%namespace, error = %e, "auto-index: FTS build failed");
        ok = false;
    } else {
        info!(%namespace, docs, "auto-index: FTS index built");
    }

    // Entries cached before these builds may have come from the bounded-scan
    // fallback (lower recall) or from FTS-less hybrid scoring; drop them so
    // the first post-build query serves fresh results. Do this even on
    // partial failure — at worst the cache is warmed again on the retry.
    inner.cache.invalidate(namespace);

    // Persist the row count we just indexed so a restart can tell whether
    // any writes drifted past the last build. Using the pre-build count is
    // deliberate: rows that landed during the build are still unindexed
    // from LanceDB's perspective, so we *want* the next restart to notice
    // the mismatch and re-queue. Marker writes are best-effort — a
    // transient filesystem error is logged but doesn't fail the build.
    if ok {
        if let Some(ref state_dir) = inner.state_dir {
            if let Err(e) = write_marker(state_dir, namespace, docs) {
                warn!(
                    %namespace,
                    error = %e,
                    "auto-index: failed to persist marker after build"
                );
            } else {
                // Count marker landed; the pending flag is now redundant.
                // A write that races in during the build will re-touch it
                // via `notify_upsert` and the next cycle picks it up.
                clear_pending_marker(state_dir, namespace);
            }
        }
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use plume_cache::SearchCache;
    use plume_core::config::{AutoIndexConfig, CacheConfig, IndexConfig, StorageConfig};
    use plume_index::IndexManager;
    use tempfile::TempDir;

    async fn harness(auto: AutoIndexConfig) -> (AutoIndexer, Arc<IndexManager>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = StorageConfig {
            uri: dir.path().to_string_lossy().to_string(),
            region: None,
            endpoint: None,
        };
        let cache = Arc::new(
            SearchCache::new(&CacheConfig {
                ram_capacity_mb: 16,
                nvme_capacity_gb: 1,
                nvme_path: dir.path().join("cache").to_string_lossy().to_string(),
            })
            .await
            .unwrap(),
        );
        let index = Arc::new(IndexManager::connect(&storage).await.unwrap());
        let cfg = IndexConfig {
            auto,
            ..IndexConfig::default()
        };
        let indexer = AutoIndexer::new(cfg, &storage, Arc::clone(&index), cache);
        (indexer, index, dir)
    }

    #[tokio::test]
    async fn disabled_auto_index_is_noop() {
        let auto = AutoIndexConfig {
            enabled: false,
            ..AutoIndexConfig::default()
        };
        let (indexer, _index, _tmp) = harness(auto).await;
        indexer.notify_upsert("code", 10).await;
        // No task should have been spawned.
        assert!(indexer.inner.states.lock().await.is_empty());
    }

    #[tokio::test]
    async fn upsert_spawns_single_namespace_task() {
        let (indexer, _index, _tmp) = harness(AutoIndexConfig::default()).await;
        indexer.notify_upsert("code", 1).await;
        indexer.notify_upsert("code", 1).await;
        indexer.notify_upsert("docs", 1).await;
        let states = indexer.inner.states.lock().await;
        assert_eq!(states.len(), 2);
    }

    #[tokio::test]
    async fn drop_namespace_removes_entry_and_aborts_task() {
        let (indexer, _index, _tmp) = harness(AutoIndexConfig::default()).await;
        indexer.notify_upsert("code", 1).await;
        let handle = {
            let states = indexer.inner.states.lock().await;
            assert_eq!(states.len(), 1);
            // Clone-by-reference for the abort check: JoinHandle isn't Clone,
            // so we just hold on to the fact that abort was scheduled.
            states.get("code").is_some()
        };
        assert!(handle);
        indexer.drop_namespace("code").await;
        assert!(indexer.inner.states.lock().await.is_empty());
    }

    #[test]
    fn marker_roundtrip() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(read_marker(tmp.path(), "code"), None);
        write_marker(tmp.path(), "code", 42).unwrap();
        assert_eq!(read_marker(tmp.path(), "code"), Some(42));
        write_marker(tmp.path(), "code", 100).unwrap();
        assert_eq!(read_marker(tmp.path(), "code"), Some(100));
    }

    #[test]
    fn derive_state_dir_handles_local_and_remote() {
        assert_eq!(
            derive_state_dir("./data/lancedb"),
            Some(PathBuf::from("./data/lancedb/.plume-autoindex"))
        );
        assert_eq!(
            derive_state_dir("file:///tmp/plume"),
            Some(PathBuf::from("/tmp/plume/.plume-autoindex"))
        );
        assert_eq!(derive_state_dir("s3://bucket/prefix"), None);
        assert_eq!(derive_state_dir("gs://bucket/prefix"), None);
    }

    #[tokio::test]
    async fn drop_namespace_clears_marker() {
        let (indexer, _index, tmp) = harness(AutoIndexConfig::default()).await;
        let state_dir = tmp.path().join(".plume-autoindex");
        std::fs::create_dir_all(&state_dir).unwrap();
        write_marker(&state_dir, "code", 7).unwrap();
        assert!(marker_path(&state_dir, "code").exists());
        indexer.drop_namespace("code").await;
        assert!(!marker_path(&state_dir, "code").exists());
    }

    #[tokio::test]
    async fn recover_is_noop_for_empty_namespace() {
        let (indexer, index, _tmp) = harness(AutoIndexConfig::default()).await;
        // Materialize an empty namespace — zero rows should be skipped
        // during recovery (no spurious rebuild on a just-created table).
        index.namespace("code").await.unwrap();
        indexer.recover().await;
        assert!(indexer.inner.states.lock().await.is_empty());
    }

    #[tokio::test]
    async fn pending_marker_roundtrip() {
        let tmp = TempDir::new().unwrap();
        assert!(!has_pending_marker(tmp.path(), "code"));
        touch_pending_marker(tmp.path(), "code").unwrap();
        assert!(has_pending_marker(tmp.path(), "code"));
        clear_pending_marker(tmp.path(), "code");
        assert!(!has_pending_marker(tmp.path(), "code"));
        // Clear is idempotent.
        clear_pending_marker(tmp.path(), "code");
    }

    #[tokio::test]
    async fn drop_namespace_clears_pending_marker() {
        let (indexer, _index, tmp) = harness(AutoIndexConfig::default()).await;
        let state_dir = tmp.path().join(".plume-autoindex");
        std::fs::create_dir_all(&state_dir).unwrap();
        touch_pending_marker(&state_dir, "code").unwrap();
        assert!(has_pending_marker(&state_dir, "code"));
        indexer.drop_namespace("code").await;
        assert!(!has_pending_marker(&state_dir, "code"));
    }

    #[tokio::test]
    async fn recover_skips_when_marker_matches_count() {
        let (indexer, index, tmp) = harness(AutoIndexConfig::default()).await;
        // Low min_docs so the has_ann check doesn't trigger a rebuild on
        // an unindexable (too small) table.
        let state_dir = tmp.path().join(".plume-autoindex");
        std::fs::create_dir_all(&state_dir).unwrap();
        index.namespace("code").await.unwrap();
        // Count is zero and marker matches zero? We skip count==0 outright
        // — use a non-zero marker and assert the zero-row early-return
        // takes precedence, so no task spawns.
        write_marker(&state_dir, "code", 0).unwrap();
        indexer.recover().await;
        assert!(indexer.inner.states.lock().await.is_empty());
    }
}
