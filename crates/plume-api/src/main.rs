mod routes;
mod state;

use std::net::SocketAddr;
use std::sync::Arc;

use plume_cache::SearchCache;
use plume_core::config::PlumeConfig;
use plume_encoder::build_encoder;
use plume_index::IndexManager;
use plume_search::SearchEngine;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = PlumeConfig::from_env_or_default()?;
    info!("starting Plume v{}", env!("CARGO_PKG_VERSION"));

    let index_manager = IndexManager::connect(&config.storage).await?;
    let cache = Arc::new(SearchCache::new(&config.cache)?);
    let search_engine = SearchEngine::new(Arc::clone(&cache));
    let encoder = build_encoder(&config.encoder);

    let state = AppState {
        config: config.clone(),
        index: Arc::new(index_manager),
        cache,
        search: Arc::new(search_engine),
        encoder: Arc::from(encoder),
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
