//! `plume index` — force a manual index rebuild (bulk-load workflows).

use anyhow::Result;
use clap::Parser;
use plume_core::types::IndexResponse;

use super::client::{Client, DEFAULT_URL};

#[derive(Parser)]
pub struct Args {
    /// Namespace to rebuild indexes for.
    pub namespace: String,

    /// Plume server URL.
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,

    /// Rebuild only the FTS (BM25) index.
    #[arg(long, conflicts_with = "ann_only")]
    pub fts_only: bool,

    /// Rebuild only the ANN (IVF_PQ) index.
    #[arg(long, conflicts_with = "fts_only")]
    pub ann_only: bool,
}

pub async fn run(args: Args) -> Result<()> {
    let client = Client::new(args.url.clone());
    let ns = &args.namespace;

    if !args.fts_only {
        let resp: IndexResponse = client.post_empty(&format!("/ns/{ns}/index")).await?;
        println!(
            "ANN index build queued (job {}): {}",
            resp.job_id.as_deref().unwrap_or("?"),
            resp.status
        );
    }

    if !args.ann_only {
        let resp: IndexResponse = client.post_empty(&format!("/ns/{ns}/fts-index")).await?;
        println!(
            "FTS index build queued (job {}): {}",
            resp.job_id.as_deref().unwrap_or("?"),
            resp.status
        );
    }

    Ok(())
}
