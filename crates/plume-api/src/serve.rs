//! HTTP server entry point for `plume serve`.

use std::net::SocketAddr;
use std::sync::Arc;

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

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("invalid server address");

    info!(%addr, "listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
