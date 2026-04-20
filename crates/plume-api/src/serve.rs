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

    let config = PlumeConfig::from_env_or_default()?;
    info!("starting Plume v{}", env!("CARGO_PKG_VERSION"));

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
