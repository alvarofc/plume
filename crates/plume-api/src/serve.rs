//! HTTP server entry point for `plume serve`.

use std::sync::Arc;

use anyhow::Context;
use plume_cache::SearchCache;
use plume_core::config::PlumeConfig;
use plume_encoder::build_encoder;
use plume_index::IndexManager;
use plume_search::SearchEngine;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

use crate::auto_index::AutoIndexer;
use crate::jobs::IndexJobRegistry;
use crate::routes;
use crate::state::AppState;

pub async fn run() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let mut config = PlumeConfig::from_env_or_default()?;
    info!("starting Plume v{}", env!("CARGO_PKG_VERSION"));

    // Auto-resolve the default encoder model: if nothing is configured
    // (still the shipped HF repo id) and no override env var is set,
    // download it into the canonical local dir so `plume serve` out of
    // the box lights up the real ONNX encoder instead of silently
    // falling back to the mock.
    if let Some(local) = maybe_autoresolve_model(&config).await? {
        config.encoder.model = local;
    }

    let index_manager = Arc::new(IndexManager::connect(&config.storage).await?);
    let cache = Arc::new(SearchCache::new(&config.cache).await?);
    let search_engine = SearchEngine::new(Arc::clone(&cache), config.index.clone());
    let encoder = build_encoder(&config.encoder);
    let auto_index = AutoIndexer::new(
        config.index.clone(),
        &config.storage,
        Arc::clone(&index_manager),
        Arc::clone(&cache),
    );
    // Replay any pending rebuilds from before this process started. Drift
    // is detected from on-disk markers; empty namespaces and namespaces
    // already in sync are skipped.
    auto_index.recover().await;

    let state = AppState {
        config: config.clone(),
        index: index_manager,
        cache,
        search: Arc::new(search_engine),
        encoder: Arc::from(encoder),
        jobs: Arc::new(IndexJobRegistry::new()),
        auto_index,
    };

    let app = routes::router(state);

    // Go through the tuple form so DNS hostnames and bracketed IPv6
    // literals work the same as bare IPs. `SocketAddr::parse` only
    // accepts raw numeric addresses, which means `host = "localhost"`
    // or `host = "[::1]"` from config would blow up at startup.
    let bare_host = config
        .server
        .host
        .trim_start_matches('[')
        .trim_end_matches(']');
    let listener = tokio::net::TcpListener::bind((bare_host, config.server.port))
        .await
        .with_context(|| format!("bind {}:{}", config.server.host, config.server.port))?;
    let addr = listener.local_addr().context("read listener local addr")?;
    info!(%addr, "listening");
    axum::serve(listener, app).await?;

    Ok(())
}

/// When the configured encoder model is still the shipped default HF repo
/// id (i.e. not a filesystem path the user wired up themselves), download
/// it into the canonical local dir and return the path. Returns `None`
/// when the user has supplied their own path or when the binary was built
/// without ONNX support — in both cases `build_encoder` already does the
/// right thing.
#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
async fn maybe_autoresolve_model(config: &PlumeConfig) -> anyhow::Result<Option<String>> {
    use std::path::Path;

    // Any value that looks like a filesystem path is the user's choice —
    // leave it alone so pointing at a bespoke model keeps working.
    let configured = &config.encoder.model;
    if Path::new(configured).is_absolute() || configured.starts_with('.') {
        return Ok(None);
    }

    let (_, default_repo, default_target) = crate::cli::model::default_local_model()?;
    // Only auto-pull the bundled default. Surprising a user who typed
    // `model = "org/custom"` with a mystery download would be worse than
    // letting `build_encoder` fall back and log a warning.
    if configured != &default_repo {
        return Ok(None);
    }

    if !crate::cli::model::is_model_ready(&default_target) {
        crate::cli::model::ensure_default_model().await?;
    }
    Ok(Some(default_target.to_string_lossy().into_owned()))
}

#[cfg(not(any(feature = "onnx", feature = "onnx-system-ort")))]
async fn maybe_autoresolve_model(_config: &PlumeConfig) -> anyhow::Result<Option<String>> {
    Ok(None)
}
