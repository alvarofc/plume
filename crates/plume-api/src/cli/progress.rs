//! Progress bars for long-running CLI work (upload, download, encode).
//!
//! Always goes to stderr so grep-style stdout pipelines stay clean. Falls
//! back to a no-op bar when stderr isn't a TTY, when `NO_COLOR` / `CI` is
//! set, or when `PLUME_NO_PROGRESS=1` — callers can still call the same
//! `inc`/`set_message`/`finish` methods without branching.

use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Whether a visible progress bar makes sense in the current shell. Keeps
/// the progress bar out of log files, CI captures, and anything else
/// redirecting stderr to a non-TTY.
pub fn progress_enabled() -> bool {
    if std::env::var_os("PLUME_NO_PROGRESS").is_some() {
        return false;
    }
    // De-facto standards for opting out of decorated CLI output — honored
    // here so CI jobs and no-color shells don't get escape codes in logs.
    if std::env::var_os("NO_COLOR").is_some() || std::env::var_os("CI").is_some() {
        return false;
    }
    std::io::stderr().is_terminal()
}

/// Build a finite-step bar (files uploaded / chunks indexed / docs
/// flushed). Returns a hidden bar when progress is disabled, so callers
/// can call `.inc()` / `.finish()` unconditionally.
pub fn new_steps(total: u64, label: &'static str) -> ProgressBar {
    let bar = if progress_enabled() {
        ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr())
    } else {
        ProgressBar::hidden()
    };
    // `{wide_bar}` eats remaining width so long labels don't stack on top
    // of the bar. `{msg}` is the per-step hint (e.g. current filename).
    let tmpl = format!(
        "  {label:<10} {{wide_bar:.cyan/blue}} {{pos}}/{{len}} {{msg}}",
        label = label
    );
    bar.set_style(
        ProgressStyle::with_template(&tmpl)
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓░"),
    );
    bar
}

/// Finite bar measured in bytes (file download, upload). Uses byte-aware
/// formatters so 1.2 GiB reads as `1.2 GiB` instead of a raw integer.
pub fn new_bytes(total: u64, label: &'static str) -> ProgressBar {
    let bar = if progress_enabled() {
        ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr())
    } else {
        ProgressBar::hidden()
    };
    let tmpl = format!(
        "  {label:<10} {{wide_bar:.green/white}} {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}) {{msg}}",
        label = label
    );
    bar.set_style(
        ProgressStyle::with_template(&tmpl)
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓░"),
    );
    bar
}

/// Open-ended spinner for phases where we don't know the total up front
/// (e.g. waiting for the server-side ANN + FTS rebuild to settle).
pub fn new_spinner(label: &'static str) -> ProgressBar {
    let bar = if progress_enabled() {
        ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr())
    } else {
        ProgressBar::hidden()
    };
    let tmpl = format!("  {label:<10} {{spinner:.cyan}} {{msg}}", label = label);
    bar.set_style(
        ProgressStyle::with_template(&tmpl)
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    if progress_enabled() {
        bar.enable_steady_tick(Duration::from_millis(120));
    }
    bar
}
