//! `plume completions <shell>` — emit shell completion script to stdout.
//!
//! Lets users wire up tab-completion with one line in their rc file, e.g.
//! `plume completions zsh > ~/.zfunc/_plume`. Avoids having to ship
//! separate completion files per distribution.

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::{generate, Shell};

#[derive(Parser)]
pub struct Args {
    /// Shell to emit a completion script for.
    pub shell: Shell,
}

pub fn run<C: CommandFactory>(args: Args) -> Result<()> {
    let mut cmd = C::command();
    let bin = cmd.get_name().to_string();
    generate(args.shell, &mut cmd, bin, &mut std::io::stdout());
    Ok(())
}
