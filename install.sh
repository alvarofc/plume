#!/usr/bin/env sh
# install.sh — one-liner installer for Plume prebuilt binaries.
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/<owner>/<repo>/main/install.sh | sh
#   curl -sSf .../install.sh | sh -s -- --version v0.1.0 --bin-dir ~/.local/bin
#
# Detects OS/arch, downloads the matching release tarball from GitHub
# Releases, verifies its SHA-256 against the published `.sha256` sidecar,
# and drops `plume` into `--bin-dir` (default `~/.local/bin`).
#
# Stays POSIX sh so macOS's /bin/sh works without bash-isms.

set -eu

# --- defaults ---------------------------------------------------------------

REPO="${PLUME_REPO:-alvarofc/plume}"
VERSION="${PLUME_VERSION:-latest}"
BIN_DIR_DEFAULT="${HOME}/.local/bin"
BIN_DIR=""
QUIET=0

# Accept the usual shell truthy spellings for PLUME_SKIP_VERIFY; anything
# else is off. We compare numerically later (-eq/-ne), so letting a
# non-numeric value through would crash under `set -e`.
case "$(printf '%s' "${PLUME_SKIP_VERIFY:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) SKIP_VERIFY=1 ;;
    *)             SKIP_VERIFY=0 ;;
esac

say()  { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }
warn() { printf '%s\n' "plume-install: warning: $*" >&2; }
die()  { printf '%s\n' "plume-install: error: $*" >&2; exit 1; }

usage() {
    cat <<EOF
Plume installer

Usage: install.sh [--version TAG] [--bin-dir DIR] [--repo OWNER/NAME] [--quiet] [--skip-verify]

  --version TAG     Release tag to install (default: latest).
  --bin-dir DIR     Where to drop the \`plume\` binary (default: ~/.local/bin).
  --repo OWNER/NAME GitHub repo to pull from (default: $REPO).
  --quiet           Suppress progress output.
  --skip-verify     Skip SHA-256 checksum verification (not recommended).
  -h, --help        Show this help.

Env vars: PLUME_REPO, PLUME_VERSION, PLUME_SKIP_VERIFY.
EOF
}

# --- arg parsing ------------------------------------------------------------

# `set -u` would crash on `install.sh --version` (no value) when the case
# body touches `$2`. Guard `$#` first.
while [ $# -gt 0 ]; do
    case "$1" in
        --version)     [ $# -ge 2 ] || die "--version requires a value (see --help)"
                       VERSION="$2"; shift 2 ;;
        --bin-dir)     [ $# -ge 2 ] || die "--bin-dir requires a value (see --help)"
                       BIN_DIR="$2"; shift 2 ;;
        --repo)        [ $# -ge 2 ] || die "--repo requires a value (see --help)"
                       REPO="$2"; shift 2 ;;
        --quiet)       QUIET=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) die "unknown arg: $1 (see --help)" ;;
    esac
done

[ -n "$BIN_DIR" ] || BIN_DIR="$BIN_DIR_DEFAULT"

# --- tool detection ---------------------------------------------------------

need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need uname
need mkdir
need tar
need mv
need rm
need sed

DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl -fsSL"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget -qO-"
else
    die "need curl or wget on PATH"
fi

SHA=""
if command -v shasum >/dev/null 2>&1; then
    SHA="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    SHA="sha256sum"
fi

if [ "$SKIP_VERIFY" -ne 1 ] && [ -z "$SHA" ]; then
    die "no shasum/sha256sum available; install one or re-run with --skip-verify"
fi

# --- target detection -------------------------------------------------------

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
    Darwin) os_tag="apple-darwin" ;;
    Linux)  os_tag="unknown-linux-gnu" ;;
    *) die "unsupported OS: $os (Plume ships prebuilts for macOS and Linux)" ;;
esac

case "$arch" in
    arm64|aarch64) arch_tag="aarch64" ;;
    x86_64|amd64)  arch_tag="x86_64" ;;
    *) die "unsupported arch: $arch" ;;
esac

TARGET="${arch_tag}-${os_tag}"

# --- resolve version --------------------------------------------------------

if [ "$VERSION" = "latest" ]; then
    say "plume-install: resolving latest release from $REPO"
    if command -v curl >/dev/null 2>&1; then
        VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
                   | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
                   | head -n1)"
    else
        VERSION="$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
                   | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
                   | head -n1)"
    fi
    [ -n "$VERSION" ] || die "could not resolve latest release from GitHub. If no release is published yet, build from source or pass --version <tag>."
fi

ARCHIVE="plume-${VERSION}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ARCHIVE}"
SUM_URL="${URL}.sha256"

say "plume-install: target   = $TARGET"
say "plume-install: version  = $VERSION"
say "plume-install: source   = $URL"
say "plume-install: bin dir  = $BIN_DIR"

# --- download ---------------------------------------------------------------

# Use a mktemp'd workdir so a mid-install interrupt doesn't leave a stale
# half-extracted binary on PATH. We trap-cleanup even on error.
TMP="$(mktemp -d 2>/dev/null || mktemp -d -t plume)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

say "plume-install: downloading archive"
$DOWNLOADER "$URL" > "$TMP/$ARCHIVE" || die "download failed: $URL"

if [ "$SKIP_VERIFY" -eq 1 ]; then
    warn "skipping checksum verification (--skip-verify / PLUME_SKIP_VERIFY=1)"
else
    say "plume-install: verifying checksum"
    $DOWNLOADER "$SUM_URL" > "$TMP/$ARCHIVE.sha256" \
        || die "could not fetch checksum sidecar at $SUM_URL"
    [ -s "$TMP/$ARCHIVE.sha256" ] || die "checksum sidecar is empty at $SUM_URL"
    (cd "$TMP" && $SHA -c "$ARCHIVE.sha256") \
        || die "checksum verification failed"
fi

# --- extract and install ---------------------------------------------------

say "plume-install: extracting"
tar -xzf "$TMP/$ARCHIVE" -C "$TMP"
STAGE_DIR="$TMP/plume-${VERSION}-${TARGET}"
[ -x "$STAGE_DIR/plume" ] || die "archive did not contain plume binary at $STAGE_DIR/plume"

mkdir -p "$BIN_DIR"

# Intel-mac builds ship a bundled libonnxruntime. `ort`'s load-dynamic
# path does NOT auto-discover a dylib next to the executable — the user
# has to point `ORT_DYLIB_PATH` at it. Make that plug-and-play: stash
# the real binary as `plume-bin` and put a tiny launcher at `plume` that
# sets the env var. Everything else (plain macOS/Linux) just drops the
# binary straight in.
if [ -f "$STAGE_DIR/libonnxruntime.dylib" ]; then
    LIB_DIR="$BIN_DIR/../lib/plume"
    mkdir -p "$LIB_DIR"
    # Resolve to an absolute path so the wrapper works no matter where
    # it's invoked from. `cd` + `pwd -P` is the POSIX way; `realpath`
    # isn't available on every mac.
    LIB_ABS="$(cd "$LIB_DIR" && pwd -P)"
    mv "$STAGE_DIR/libonnxruntime.dylib" "$LIB_ABS/libonnxruntime.dylib"
    mv "$STAGE_DIR/plume" "$BIN_DIR/plume-bin"
    chmod +x "$BIN_DIR/plume-bin"
    cat > "$BIN_DIR/plume" <<WRAPPER
#!/bin/sh
# Launcher for Intel-mac prebuilt: point ort's load-dynamic path at the
# bundled libonnxruntime. Users can override by exporting ORT_DYLIB_PATH
# themselves before invoking plume.
: "\${ORT_DYLIB_PATH:=$LIB_ABS/libonnxruntime.dylib}"
export ORT_DYLIB_PATH
exec "$(cd "$BIN_DIR" && pwd -P)/plume-bin" "\$@"
WRAPPER
    chmod +x "$BIN_DIR/plume"
    say "plume-install: installed libonnxruntime → $LIB_ABS/libonnxruntime.dylib"
    say "plume-install: installed launcher    → $BIN_DIR/plume"
    say "plume-install: installed binary      → $BIN_DIR/plume-bin"
else
    mv "$STAGE_DIR/plume" "$BIN_DIR/plume"
    chmod +x "$BIN_DIR/plume"
    say "plume-install: installed to $BIN_DIR/plume"
fi

# Warn if $BIN_DIR isn't on PATH — otherwise `plume` won't resolve and the
# user will think install failed. Point them at the conventional fix.
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on PATH. Add it to your shell rc, e.g.:"
       warn "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc" ;;
esac

say ""
say "Next:"
say "  plume grep \"rate limiting middleware\" ./src"
say "  (model auto-downloads on first run)"
