//! `plume ns` — list, create, delete namespaces.

use anyhow::Result;
use clap::{Parser, Subcommand};
use plume_core::types::NamespacesResponse;
use serde_json::Value;

use super::client::{Client, DEFAULT_URL};

#[derive(Parser)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,

    /// Plume server URL (shared across subcommands).
    #[arg(long, env = "PLUME_URL", default_value = DEFAULT_URL, global = true)]
    pub url: String,
}

#[derive(Subcommand)]
pub enum Command {
    /// List all namespaces.
    List,
    /// Create a namespace (materializes the backing table).
    Create { name: String },
    /// Delete a namespace and all its documents.
    Delete { name: String },
}

pub async fn run(args: Args) -> Result<()> {
    let client = Client::new(args.url.clone());
    match args.command {
        Command::List => {
            let resp: NamespacesResponse = client.get_json("/ns").await?;
            if resp.namespaces.is_empty() {
                println!("(no namespaces)");
            } else {
                for ns in resp.namespaces {
                    println!("{ns}");
                }
            }
        }
        Command::Create { name } => {
            let _: Value = client.post_empty(&format!("/ns/{name}")).await?;
            println!("created namespace '{name}'");
        }
        Command::Delete { name } => {
            let _: Value = client.delete_json(&format!("/ns/{name}")).await?;
            println!("deleted namespace '{name}'");
        }
    }
    Ok(())
}
