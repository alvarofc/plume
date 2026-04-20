//! `plume clear [path]` — drop the namespace + manifest for a path.
//!
//! Use this when the per-path grep index is out of sync (e.g. after
//! upgrading from an older manifest format) or when you want to reclaim
//! the disk it's using. Fully wipes server-side rows and local state.

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;

use super::client::{Client, DEFAULT_URL};
use super::grep::{remove_manifest, resolve_source_identity};

#[derive(Parser)]
pub struct Args {
    /// Directory or `s3://bucket/prefix` URL whose grep index should be
    /// cleared. Defaults to the current working directory.
    #[arg(default_value = ".")]
    pub path: String,

    /// Don't prompt; assume yes.
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Plume server URL.
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL)]
    pub url: String,
}

pub async fn run(args: Args) -> Result<()> {
    let ident = resolve_source_identity(&args.path)?;

    if !args.yes {
        eprintln!(
            "plume: about to drop namespace '{}' and manifest at {}\n\
             plume: re-run with --yes to confirm (or pipe `yes` in)",
            ident.namespace,
            ident.manifest.display()
        );
        use std::io::{self, BufRead};
        let mut line = String::new();
        io::stdin()
            .lock()
            .read_line(&mut line)
            .context("read confirmation from stdin")?;
        if !matches!(line.trim(), "y" | "yes" | "Y" | "YES") {
            eprintln!("plume: aborted");
            return Ok(());
        }
    }

    let client = Client::new(args.url.clone());
    // Server-side delete — ok if the namespace doesn't exist.
    match client
        .delete_json::<Value>(&format!("/ns/{}", ident.namespace))
        .await
    {
        Ok(_) => eprintln!("plume: dropped namespace '{}'", ident.namespace),
        Err(e) => eprintln!("plume: server delete skipped: {e}"),
    }

    remove_manifest(&ident.manifest)?;
    eprintln!("plume: removed manifest at {}", ident.manifest.display());
    Ok(())
}
