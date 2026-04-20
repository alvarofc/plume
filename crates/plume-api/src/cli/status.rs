//! `plume status [path]` — inspect daemon + per-path grep state.

use anyhow::Result;
use clap::Parser;
use serde_json::Value;

use super::client::{Client, DEFAULT_URL};
use super::grep::{resolve_source_identity, summarize_manifest};

#[derive(Parser)]
pub struct Args {
    /// Directory or `s3://bucket/prefix` URL to report grep-indexing
    /// state for. Defaults to the current working directory.
    #[arg(default_value = ".")]
    pub path: String,

    /// Plume server URL.
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,
}

pub async fn run(args: Args) -> Result<()> {
    let ident = resolve_source_identity(&args.path)?;
    let summary = summarize_manifest(&ident.manifest);

    println!("path:       {}", ident.display);
    println!("namespace:  {}", ident.namespace);
    println!(
        "manifest:   {} ({} file(s), {} chunk(s))",
        ident.manifest.display(),
        summary.files,
        summary.chunks
    );

    let client = Client::new(args.url.clone());
    match client.get_json::<Value>("/health").await {
        Ok(health) => {
            let encoder = health
                .get("encoder")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let version = health
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            println!("daemon:     up @ {} (plume {version})", args.url);
            println!("encoder:    {encoder}");
        }
        Err(_) => {
            println!("daemon:     not reachable at {}", args.url);
        }
    }
    Ok(())
}
