//! `plume` CLI subcommands. Each module talks to a running Plume server
//! over HTTP, so the same binary can serve and operate against remote
//! deployments.

pub mod clear;
pub mod client;
pub mod grep;
pub mod index;
pub mod ingest;
pub mod ns;
pub mod search;
pub mod source;
pub mod status;
