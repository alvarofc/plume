use crate::auto_index::AutoIndexer;
use crate::jobs::IndexJobRegistry;

use std::sync::Arc;

use plume_cache::SearchCache;
use plume_core::config::PlumeConfig;
use plume_encoder::Encode;
use plume_index::IndexManager;
use plume_search::SearchEngine;

/// Shared application state passed to all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: PlumeConfig,
    pub index: Arc<IndexManager>,
    pub cache: Arc<SearchCache>,
    pub search: Arc<SearchEngine>,
    pub encoder: Arc<dyn Encode>,
    pub jobs: Arc<IndexJobRegistry>,
    pub auto_index: AutoIndexer,
}
