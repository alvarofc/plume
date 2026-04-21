//! `plume model pull` — fetch an ONNX encoder model from HuggingFace.
//!
//! Replaces the manual `scripts/download-model.sh` + `config.local.toml`
//! edit combo with a single self-contained command. Files land in
//! `{data_dir}/models/{name}` where `data_dir` is `PLUME_DATA_DIR` or
//! `$XDG_DATA_HOME/plume`, so config can point at a stable location that
//! survives target-dir wipes.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use tokio::io::AsyncWriteExt;

use super::grep::plume_data_dir;
use super::progress::{new_bytes, new_spinner};

/// Curated model aliases. Keeps `plume model pull` ergonomic for the
/// "please just set up the encoder" case while still letting power users
/// pass a raw HuggingFace repo id.
///
/// (alias, hf-repo). The first entry is the default when no id is given.
const BUILTIN_MODELS: &[(&str, &str)] = &[("lateon-code-edge", "lightonai/LateOn-Code-edge")];

/// Files ColBERT-style encoders need at runtime. The ONNX model is the
/// only byte-heavy one; the rest are tokenizer + config small files.
const REQUIRED_FILES: &[&str] = &[
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

#[derive(Parser)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Download an encoder model from HuggingFace into the local model dir.
    Pull(PullArgs),
    /// List known model aliases and whether they're already downloaded.
    List,
    /// Print the resolved local directory for an alias or HF id.
    Where(WhereArgs),
}

#[derive(Parser)]
pub struct PullArgs {
    /// Either a known alias (e.g. `lateon-code-edge`) or a full
    /// HuggingFace repo id (`org/repo`). Defaults to the first entry
    /// in the builtin list.
    pub model: Option<String>,

    /// Override the local target directory. Defaults to
    /// `{PLUME_DATA_DIR}/models/{name}`.
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,

    /// Re-download even if all required files already exist locally.
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser)]
pub struct WhereArgs {
    /// Alias or HuggingFace repo id.
    pub model: Option<String>,
}

pub async fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Pull(a) => pull(a).await,
        Command::List => list(),
        Command::Where(a) => print_where(a),
    }
}

fn list() -> Result<()> {
    let data = plume_data_dir()?.join("models");
    for (alias, repo) in BUILTIN_MODELS {
        let dir = data.join(alias);
        let status = if model_is_complete(&dir) {
            "downloaded"
        } else {
            "not present"
        };
        println!("{alias:24} {repo:40} {status}");
    }
    Ok(())
}

fn print_where(args: WhereArgs) -> Result<()> {
    let (name, _) = resolve_model(args.model.as_deref())?;
    let dir = plume_data_dir()?.join("models").join(&name);
    println!("{}", dir.display());
    Ok(())
}

async fn pull(args: PullArgs) -> Result<()> {
    let (name, repo) = resolve_model(args.model.as_deref())?;
    let target = match args.dir {
        Some(explicit) => explicit,
        None => plume_data_dir()?.join("models").join(&name),
    };
    ensure_model(&name, &repo, &target, args.force).await?;
    print_config_hint(&name, &target);
    Ok(())
}

/// Download `repo`'s REQUIRED_FILES into `target_dir`. Idempotent: returns
/// early when every required file is present and non-empty (unless `force`).
/// Exposed so other CLI paths (e.g. `plume grep`'s auto-daemon-spawn) can
/// make sure a model is on disk before starting the encoder.
pub(super) async fn ensure_model(
    name: &str,
    repo: &str,
    target_dir: &Path,
    force: bool,
) -> Result<()> {
    tokio::fs::create_dir_all(target_dir)
        .await
        .with_context(|| format!("create {}", target_dir.display()))?;

    if !force && model_is_complete(target_dir) {
        eprintln!(
            "plume: model {name} already present at {} (re-run with --force to refetch)",
            target_dir.display()
        );
        return Ok(());
    }

    eprintln!(
        "plume: downloading {repo} → {} ({} file(s))",
        target_dir.display(),
        REQUIRED_FILES.len()
    );

    let http = reqwest::Client::builder()
        .user_agent(concat!("plume-cli/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build HTTP client")?;

    for file in REQUIRED_FILES {
        let dest = target_dir.join(file);
        if !force && dest.exists() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            eprintln!("  skip  {file} (already present)");
            continue;
        }
        download_file(&http, repo, file, &dest).await?;
    }

    eprintln!("plume: model ready.");
    Ok(())
}

/// Default model alias + repo (first entry in BUILTIN_MODELS). The local
/// target directory is derived so `plume grep` can pre-stage the model
/// before spawning its daemon — keeping download progress visible on the
/// caller's terminal rather than buried in daemon.log.
///
/// Only compiled into the binary when ONNX is enabled — mock-only builds
/// never auto-pull anyway.
#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
pub(crate) fn default_local_model() -> Result<(String, String, PathBuf)> {
    let (alias, repo) = BUILTIN_MODELS[0];
    let target = plume_data_dir()?.join("models").join(alias);
    Ok((alias.to_string(), repo.to_string(), target))
}

/// True when every required model file already exists on disk.
/// Used by callers that want to skip the auto-pull fast path.
#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
pub(crate) fn is_model_ready(dir: &Path) -> bool {
    model_is_complete(dir)
}

/// Download the default model into its canonical local path if it isn't
/// already complete. Idempotent; returns the resolved directory regardless.
#[cfg(any(feature = "onnx", feature = "onnx-system-ort"))]
pub(crate) async fn ensure_default_model() -> Result<PathBuf> {
    let (name, repo, target) = default_local_model()?;
    if !model_is_complete(&target) {
        ensure_model(&name, &repo, &target, false).await?;
    }
    Ok(target)
}

/// Resolve either a registered alias or a bare HF repo id. Bare ids
/// (anything containing a `/`) are passed through untouched so users can
/// pull random models without a code change here.
fn resolve_model(input: Option<&str>) -> Result<(String, String)> {
    let Some(input) = input else {
        let (alias, repo) = BUILTIN_MODELS[0];
        return Ok((alias.into(), repo.into()));
    };
    if let Some((alias, repo)) = BUILTIN_MODELS.iter().find(|(a, _)| *a == input) {
        return Ok((alias.to_string(), repo.to_string()));
    }
    if input.contains('/') {
        // HF repo ids look like `org/name`. Pick the basename as the local
        // directory name so `plume model pull lightonai/Foo-Bar` lands in
        // `.../models/Foo-Bar` instead of `.../models/lightonai/Foo-Bar`.
        let local = input.rsplit('/').next().unwrap_or(input);
        if local.is_empty() || input.split('/').any(str::is_empty) {
            bail!("invalid HuggingFace repo id '{input}'; expected `org/name`");
        }
        return Ok((local.to_string(), input.to_string()));
    }
    bail!(
        "unknown model '{input}'. Known aliases: {}. \
         Or pass a full HuggingFace repo id like `org/name`.",
        BUILTIN_MODELS
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn model_is_complete(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|f| {
        let p = dir.join(f);
        p.exists() && p.metadata().map(|m| m.len() > 0).unwrap_or(false)
    })
}

async fn download_file(http: &reqwest::Client, repo: &str, file: &str, dest: &Path) -> Result<()> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");

    let resp = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("download {file} failed: HTTP {}", resp.status());
    }
    let total = resp.content_length();

    // Stage to `.tmp` and rename on success — a ctrl-C mid-download
    // shouldn't leave a zero-byte `model.onnx` that looks complete to the
    // `model_is_complete` check on the next run.
    let tmp = dest.with_extension("part");
    let mut out = tokio::fs::File::create(&tmp)
        .await
        .with_context(|| format!("create {}", tmp.display()))?;

    let bar = match total {
        Some(n) => new_bytes(n, "downloading"),
        None => new_spinner("downloading"),
    };
    bar.set_message(file.to_string());

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read body {file}"))?;
        out.write_all(&chunk)
            .await
            .with_context(|| format!("write {}", tmp.display()))?;
        bar.inc(chunk.len() as u64);
    }
    out.flush()
        .await
        .with_context(|| format!("flush {}", tmp.display()))?;
    drop(out);
    bar.finish_and_clear();

    // Windows' `rename` is not atomic-replace: it fails if `dest` exists.
    // A partial-download recovery run (model.onnx present but zero bytes)
    // would trip that path, so pre-remove any stale dest before renaming.
    // On Unix the `rename` below is already atomic-replace; the extra
    // remove is a no-op when the file isn't there.
    if tokio::fs::try_exists(dest)
        .await
        .with_context(|| format!("check {}", dest.display()))?
    {
        tokio::fs::remove_file(dest)
            .await
            .with_context(|| format!("remove stale {}", dest.display()))?;
    }

    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("rename {} → {}", tmp.display(), dest.display()))?;
    // Catch truncated downloads that somehow slipped past the streaming
    // writer so `model_is_complete` on the next run can trust the file.
    let len = tokio::fs::metadata(dest)
        .await
        .with_context(|| format!("stat {}", dest.display()))?
        .len();
    if len == 0 {
        bail!("download {file} produced an empty file");
    }
    eprintln!("  ok    {file}");
    Ok(())
}

fn print_config_hint(name: &str, target: &Path) {
    eprintln!();
    eprintln!("To use this model, point config.toml at it:");
    eprintln!();
    eprintln!("  [encoder]");
    eprintln!("  model = \"{}\"", target.display());
    eprintln!();
    eprintln!(
        "(or set via env: PLUME_ENCODER_MODEL=\"{}\")",
        target.display()
    );
    let _ = name; // kept for potential future use (e.g. tag in output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_default_picks_first_builtin() {
        let (name, repo) = resolve_model(None).unwrap();
        assert_eq!(name, "lateon-code-edge");
        assert_eq!(repo, "lightonai/LateOn-Code-edge");
    }

    #[test]
    fn resolve_known_alias_returns_repo() {
        let (name, repo) = resolve_model(Some("lateon-code-edge")).unwrap();
        assert_eq!(name, "lateon-code-edge");
        assert!(repo.contains("LateOn-Code-edge"));
    }

    #[test]
    fn resolve_raw_hf_id_uses_basename_as_local_name() {
        let (name, repo) = resolve_model(Some("org/My-Model")).unwrap();
        assert_eq!(name, "My-Model");
        assert_eq!(repo, "org/My-Model");
    }

    #[test]
    fn resolve_unknown_bare_name_errors() {
        let err = resolve_model(Some("not-a-thing")).unwrap_err();
        assert!(format!("{err}").contains("unknown model"));
    }

    #[test]
    fn model_is_complete_requires_all_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!model_is_complete(tmp.path()));
        for f in REQUIRED_FILES.iter().take(REQUIRED_FILES.len() - 1) {
            std::fs::write(tmp.path().join(f), b"x").unwrap();
        }
        assert!(!model_is_complete(tmp.path()));
        std::fs::write(
            tmp.path().join(REQUIRED_FILES[REQUIRED_FILES.len() - 1]),
            b"x",
        )
        .unwrap();
        assert!(model_is_complete(tmp.path()));
    }
}
