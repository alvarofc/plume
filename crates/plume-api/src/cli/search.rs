//! `plume search` — one-shot query.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use plume_core::types::{QueryRequest, QueryResponse, SearchMode};

use super::client::{Client, DEFAULT_URL};

#[derive(Parser)]
pub struct Args {
    /// Query text.
    pub query: String,

    /// Namespace to search.
    #[arg(short, long)]
    pub namespace: String,

    /// Number of results.
    #[arg(short, long, default_value_t = 10)]
    pub k: usize,

    /// Search mode. Defaults to `semantic` so `plume ingest && plume search`
    /// works immediately on fresh namespaces — hybrid needs the BM25 index,
    /// which the auto-indexer builds in the background.
    #[arg(short, long, value_enum, default_value_t = Mode::Semantic)]
    pub mode: Mode,

    /// Print raw JSON instead of a table.
    #[arg(long)]
    pub json: bool,

    /// Plume server URL.
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,
}

#[derive(Copy, Clone, ValueEnum)]
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

pub async fn run(args: Args) -> Result<()> {
    let client = Client::new(args.url.clone());
    let req = QueryRequest {
        query: args.query.clone(),
        k: args.k,
        mode: args.mode.into(),
    };
    let resp: QueryResponse = client
        .post_json(&format!("/ns/{}/query", args.namespace), &req)
        .await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&resp)?);
        return Ok(());
    }

    if resp.results.is_empty() {
        println!("no results (cache_hit={})", resp.cache_hit);
        return Ok(());
    }

    println!(
        "{} results (cache_hit={}):",
        resp.results.len(),
        resp.cache_hit
    );
    for (i, r) in resp.results.iter().enumerate() {
        let snippet = snippet(&r.text, 120);
        println!("{:>3}. [{:.4}] {}  {}", i + 1, r.score, r.id, snippet);
    }
    Ok(())
}

fn snippet(text: &str, max: usize) -> String {
    let one_line: String = text
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    if one_line.chars().count() <= max {
        one_line
    } else {
        let truncated: String = one_line.chars().take(max).collect();
        format!("{truncated}…")
    }
}
