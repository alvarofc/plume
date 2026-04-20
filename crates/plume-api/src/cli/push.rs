//! `plume push <local-dir> <dest>` — upload a local directory to a remote
//! object store (or a sibling local dir, for symmetry with the rest of
//! the Source abstraction).
//!
//! Replaces the manual `aws s3 sync` step in the dogfood flow so the whole
//! "bootstrap my repo into plume" loop lives inside the plume binary.
//! Uses the same include/exclude/exclude-dir flag shape as `plume grep`
//! so a user who narrowed a grep scope can reuse the same flags to upload
//! exactly those files.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use globset::{Glob, GlobSet, GlobSetBuilder};

use super::progress::{new_bytes, new_spinner};
use super::source::{Filter, Source, SourceFile};

/// Default set of excluded directory basenames when `--exclude-dir` isn't
/// passed. Mirrors `plume grep`'s builtin list so a grep-then-push workflow
/// uploads the same files the grep scan would have seen.
const DEFAULT_EXCLUDED_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".turbo",
    ".cache",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "vendor",
    "Pods",
    "DerivedData",
    "data",
    "models",
];

#[derive(Parser)]
pub struct Args {
    /// Local directory to upload from. Must exist.
    pub source: PathBuf,

    /// Destination. `s3://bucket/prefix`, `gs://bucket/prefix`, or a
    /// local path for same-machine copies.
    pub dest: String,

    /// Glob(s) of files to include. Repeatable. When set, the source is
    /// filtered to *only* files matching at least one include glob.
    #[arg(long, value_name = "GLOB")]
    pub include: Vec<String>,

    /// Glob(s) of files to exclude. Repeatable. Applied before include
    /// filtering so `--exclude "**/secret*"` wins.
    #[arg(long, value_name = "GLOB")]
    pub exclude: Vec<String>,

    /// Extra directory basenames to skip on top of the builtin list.
    #[arg(long, value_name = "NAME")]
    pub exclude_dir: Vec<String>,

    /// Skip the directory-name allowlist (`.git`, `target`, etc.). Useful
    /// when uploading build artifacts or a `target/` dir on purpose.
    #[arg(long)]
    pub no_default_excludes: bool,

    /// List what would be uploaded without writing anything. Useful to
    /// preview filters before paying egress on a large tree.
    #[arg(long)]
    pub dry_run: bool,
}

pub async fn run(args: Args) -> Result<()> {
    if !args.source.exists() {
        bail!("source does not exist: {}", args.source.display());
    }
    if !args.source.is_dir() {
        bail!(
            "source must be a directory, got file: {} (use `aws s3 cp` for single-file uploads)",
            args.source.display()
        );
    }

    let dest = Source::parse(&args.dest)?;
    let local = Source::Local {
        root: args.source.clone(),
    };

    let include = build_globset(&args.include).context("compile --include glob")?;
    let exclude = build_globset(&args.exclude).context("compile --exclude glob")?;

    let mut excluded_dirs: HashSet<String> = if args.no_default_excludes {
        HashSet::new()
    } else {
        DEFAULT_EXCLUDED_DIRS
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    for extra in &args.exclude_dir {
        excluded_dirs.insert(extra.clone());
    }

    let scan_spinner = new_spinner("scanning");
    scan_spinner.set_message(local.display());
    let mut filter = Filter::any();
    filter.excluded_dirs = excluded_dirs;
    let raw = local.list(&filter).await?;
    scan_spinner.finish_and_clear();

    let files: Vec<SourceFile> = raw
        .into_iter()
        .filter(|f| {
            if let Some(ex) = &exclude {
                if ex.is_match(&f.key) {
                    return false;
                }
            }
            if let Some(inc) = &include {
                return inc.is_match(&f.key);
            }
            true
        })
        .collect();

    if files.is_empty() {
        bail!(
            "no files to upload from {} (check --include/--exclude)",
            args.source.display()
        );
    }

    let total_bytes: u64 = files.iter().map(|f| f.size).sum();
    eprintln!(
        "plume: {} file(s), {} from {} → {}",
        files.len(),
        humanize_bytes(total_bytes),
        local.display(),
        dest.display(),
    );

    if args.dry_run {
        for f in &files {
            println!("{}", f.key);
        }
        eprintln!("plume: dry-run — no files written");
        return Ok(());
    }

    let bar = new_bytes(total_bytes, "uploading");
    let mut uploaded_files = 0usize;
    for file in &files {
        bar.set_message(file.key.clone());
        // Stream the file straight into the destination rather than
        // materializing it in RAM — multi-GiB artifacts would otherwise OOM.
        dest.write_file(&file.key, &args.source.join(&file.key))
            .await
            .with_context(|| format!("write {}", file.key))?;
        bar.inc(file.size);
        uploaded_files += 1;
    }
    bar.finish_and_clear();

    eprintln!(
        "plume: uploaded {uploaded_files} file(s) ({}) to {}",
        humanize_bytes(total_bytes),
        dest.display()
    );
    Ok(())
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        let g = Glob::new(p).with_context(|| format!("invalid glob {p:?}"))?;
        b.add(g);
    }
    Ok(Some(b.build()?))
}

fn humanize_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_bytes_scales_binary_prefixes() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(512), "512 B");
        assert_eq!(humanize_bytes(1024), "1.0 KiB");
        assert_eq!(humanize_bytes(1536), "1.5 KiB");
        assert_eq!(humanize_bytes(1024 * 1024), "1.0 MiB");
    }

    #[tokio::test]
    async fn push_local_to_local_copies_tree_filtered() {
        let tmp_src = tempfile::TempDir::new().unwrap();
        let tmp_dst = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp_src.path().join("a")).unwrap();
        std::fs::create_dir_all(tmp_src.path().join("target/ignored")).unwrap();
        std::fs::write(tmp_src.path().join("a/one.md"), b"hello").unwrap();
        std::fs::write(tmp_src.path().join("a/two.txt"), b"bye").unwrap();
        std::fs::write(
            tmp_src.path().join("target/ignored/should-skip.rs"),
            b"nope",
        )
        .unwrap();

        let args = Args {
            source: tmp_src.path().to_path_buf(),
            dest: tmp_dst.path().to_string_lossy().to_string(),
            include: vec!["**/*.md".into()],
            exclude: vec![],
            exclude_dir: vec![],
            no_default_excludes: false,
            dry_run: false,
        };

        run(args).await.unwrap();

        assert!(tmp_dst.path().join("a/one.md").exists());
        assert!(!tmp_dst.path().join("a/two.txt").exists());
        assert!(!tmp_dst
            .path()
            .join("target/ignored/should-skip.rs")
            .exists());
    }
}
