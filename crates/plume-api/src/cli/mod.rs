//! `plume` CLI subcommands. Each module talks to a running Plume server
//! over HTTP, so the same binary can serve and operate against remote
//! deployments.

pub mod clear;
pub mod client;
pub mod completions;
pub mod grep;
pub mod index;
pub mod ingest;
pub mod model;
pub mod ns;
pub mod progress;
pub mod push;
pub mod search;
pub mod source;
pub mod status;
