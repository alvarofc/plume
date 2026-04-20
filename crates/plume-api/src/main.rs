mod auto_index;
mod cli;
mod jobs;
mod routes;
mod serve;
mod state;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "plume",
    version,
    about = "Multi-vector semantic search engine",
    long_about = "Multi-vector semantic search engine.\n\n\
                  With no subcommand, `plume` runs the server.\n\
                  With a query string, `plume \"auth token\"` runs a semantic grep\n\
                  against the current directory (auto-spawning a daemon if needed)."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the HTTP server (default when no args and no implicit query).
    Serve,
    /// Semantic grep: auto-index a path and search it (colgrep-style).
    Grep(cli::grep::Args),
    /// Ingest documents from a JSONL file or a directory of .md files.
    Ingest(cli::ingest::Args),
    /// Upload a local directory to S3/GCS (or another local path).
    Push(cli::push::Args),
    /// Run a query and print results.
    Search(cli::search::Args),
    /// Manage namespaces.
    Ns(cli::ns::Args),
    /// Force a manual index rebuild.
    Index(cli::index::Args),
    /// Show daemon + per-path grep index state.
    Status(cli::status::Args),
    /// Drop the per-path grep index (namespace + manifest).
    Clear(cli::clear::Args),
    /// Download and manage encoder models (HuggingFace ONNX).
    Model(cli::model::Args),
}

/// Subcommand names we recognize. Anything else in argv[1] falls through
/// to implicit `grep`, so `plume "auth token" src/` works without typing
/// `grep`.
const KNOWN_SUBCOMMANDS: &[&str] = &[
    "serve", "grep", "ingest", "push", "search", "ns", "index", "status", "clear", "model", "help",
];

/// argv[1] values clap handles natively; don't rewrite these into `grep`.
const PASS_THROUGH_FLAGS: &[&str] = &["--help", "-h", "--version", "-V"];

/// If the user invoked `plume <not-a-subcommand> ...`, rewrite argv to
/// route through the `grep` subcommand. Preserves `plume`, `plume serve`,
/// `plume --help`, etc.
fn rewrite_argv_for_implicit_grep(argv: Vec<String>) -> Vec<String> {
    if argv.len() < 2 {
        return argv;
    }
    let first = argv[1].as_str();
    if KNOWN_SUBCOMMANDS.contains(&first) || PASS_THROUGH_FLAGS.contains(&first) {
        return argv;
    }
    let mut rewritten = Vec::with_capacity(argv.len() + 1);
    rewritten.push(argv[0].clone());
    rewritten.push("grep".to_string());
    rewritten.extend(argv.into_iter().skip(1));
    rewritten
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let argv = rewrite_argv_for_implicit_grep(argv);
    let cli = Cli::parse_from(argv);

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve::run().await,
        Command::Grep(args) => cli::grep::run(args).await,
        Command::Ingest(args) => cli::ingest::run(args).await,
        Command::Push(args) => cli::push::run(args).await,
        Command::Search(args) => cli::search::run(args).await,
        Command::Ns(args) => cli::ns::run(args).await,
        Command::Index(args) => cli::index::run(args).await,
        Command::Status(args) => cli::status::run(args).await,
        Command::Clear(args) => cli::clear::run(args).await,
        Command::Model(args) => cli::model::run(args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &[&str]) -> Vec<String> {
        std::iter::once("plume")
            .chain(s.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn rewrite_leaves_known_subcommands_alone() {
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["serve"])),
            argv(&["serve"])
        );
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["grep", "foo", "."])),
            argv(&["grep", "foo", "."])
        );
    }

    #[test]
    fn rewrite_leaves_help_and_version_alone() {
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["--help"])),
            argv(&["--help"])
        );
        assert_eq!(rewrite_argv_for_implicit_grep(argv(&["-V"])), argv(&["-V"]));
    }

    #[test]
    fn rewrite_injects_grep_for_positional_query() {
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["authenticate user"])),
            argv(&["grep", "authenticate user"])
        );
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["auth", "src/"])),
            argv(&["grep", "auth", "src/"])
        );
    }

    #[test]
    fn rewrite_injects_grep_for_implicit_flag_before_query() {
        // `plume --json "query"` → `plume grep --json "query"`.
        assert_eq!(
            rewrite_argv_for_implicit_grep(argv(&["--json", "query"])),
            argv(&["grep", "--json", "query"])
        );
    }

    #[test]
    fn rewrite_noop_on_empty_argv() {
        let base: Vec<String> = vec!["plume".into()];
        assert_eq!(rewrite_argv_for_implicit_grep(base.clone()), base);
    }
}
