#!/usr/bin/env bash
#
# Build and install Voice Transcriber for EN-KR as a /Applications .app.
#
# Safe to re-run: rebuilds the bundle, replaces the existing /Applications
# copy, and resets TCC grants for the bundle (every rebuild gets a new
# ad-hoc signature, so old grants are orphaned anyway).

set -euo pipefail

APP_NAME="Voice Transcriber for EN-KR"
APP_PATH="/Applications/${APP_NAME}.app"
BUNDLE_ID="com.noos.voicetranscriberenkr"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# ---- output helpers -------------------------------------------------------

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; CYAN=$'\033[36m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; CYAN=""; YELLOW=""; RED=""; RESET=""
fi

step() { printf "\n${BOLD}${CYAN}==> %s${RESET}\n" "$*"; }
info() { printf "    %s\n" "$*"; }
warn() { printf "${YELLOW}!!  %s${RESET}\n" "$*" >&2; }
die()  { printf "${RED}!!  %s${RESET}\n" "$*" >&2; exit 1; }

confirm() {
    local prompt="$1"
    local reply
    read -r -p "    ${prompt} [y/N] " reply
    [[ "${reply}" =~ ^[Yy]$ ]]
}

# ---- prerequisite checks --------------------------------------------------

require_macos_arm64() {
    [[ "$(uname -s)" == "Darwin" ]] || die "This script is macOS-only."
    [[ "$(uname -m)" == "arm64" ]]  || die "MLX requires Apple Silicon (M-series). Detected: $(uname -m)."
}

require_command() {
    local cmd="$1" hint="$2"
    command -v "${cmd}" >/dev/null 2>&1 || die "${cmd} not found. ${hint}"
}

require_brew_pkg() {
    local pkg="$1"
    if brew list --formula "${pkg}" >/dev/null 2>&1; then
        info "${pkg} already installed"
        return
    fi
    warn "${pkg} is not installed via Homebrew."
    if confirm "Install ${pkg} now with 'brew install ${pkg}'?"; then
        brew install "${pkg}"
    else
        die "${pkg} is required. Run 'brew install ${pkg}' and re-run this script."
    fi
}

# ---- build steps ----------------------------------------------------------

setup_venv() {
    if [[ ! -d ".venv" ]]; then
        info "creating .venv with Python 3.12"
        uv venv --python 3.12
    else
        info ".venv already exists"
    fi
    info "installing runtime + build dependencies"
    uv pip install --quiet -r requirements.txt
    uv pip install --quiet py2app
}

stop_running_app() {
    if pgrep -f "${APP_NAME}" >/dev/null 2>&1; then
        info "quitting running bundle"
        pkill -f "${APP_NAME}" || true
        sleep 1
    else
        info "no running bundle to stop"
    fi
}

build_bundle() {
    info "removing previous build/dist + /Applications copy"
    rm -rf build dist
    rm -rf "${APP_PATH}"
    info "running py2app (this is the slow step — 5-10 min)"
    uv run python setup.py py2app >/tmp/voice-transcriber-py2app.log 2>&1 || {
        warn "py2app failed. Last 20 lines of log:"
        tail -20 /tmp/voice-transcriber-py2app.log >&2
        die "Build failed. Full log: /tmp/voice-transcriber-py2app.log"
    }
    [[ -d "dist/${APP_NAME}.app" ]] || die "py2app returned 0 but no .app produced."
}

install_bundle() {
    info "moving bundle to /Applications"
    mv "dist/${APP_NAME}.app" /Applications/
    info "stripping macOS quarantine flag"
    xattr -cr "${APP_PATH}"
}

reset_tcc() {
    info "clearing stale TCC grants for ${BUNDLE_ID} (rebuild changes the signature)"
    tccutil reset All "${BUNDLE_ID}" >/dev/null 2>&1 || true
}

launch_bundle() {
    info "launching ${APP_NAME}"
    open "${APP_PATH}"
}

print_next_steps() {
    cat <<EOF

${BOLD}${CYAN}==> Done. The app is launching now.${RESET}

The 🎤 icon should appear in your menu bar momentarily. Three permissions
are required, none of which can be granted programmatically by this script:

${BOLD}1. Accessibility${RESET} — for the right-shift hotkey
   System Settings → Privacy & Security → Accessibility →  +
   Add: ${APP_PATH}

${BOLD}2. Input Monitoring${RESET} — same reason
   System Settings → Privacy & Security → Input Monitoring →  +
   Add: ${APP_PATH}

${BOLD}3. Microphone + Automation${RESET} — auto-prompted on first use
   Just click "Allow" when macOS asks the first time you record + paste.

After granting Accessibility and Input Monitoring, ${BOLD}quit and re-launch${RESET}
the bundle (🎤 → 종료, then double-click in /Applications) so pynput
picks up the new grants.

If anything misbehaves, the bundle's debug log lives at:
    ~/Library/Logs/voice-transcriber.log
EOF
}

# ---- main ----------------------------------------------------------------

step "Checking prerequisites"
require_macos_arm64
require_command brew "Install from https://brew.sh"
require_command uv   "Install with: brew install uv"
info "macOS arm64, brew + uv present"

step "Checking system libraries"
require_brew_pkg portaudio
require_brew_pkg ffmpeg

step "Setting up Python environment"
setup_venv

step "Stopping any running copy of ${APP_NAME}"
stop_running_app

step "Building the bundle"
build_bundle

step "Installing to /Applications"
install_bundle

step "Resetting TCC grants for the bundle"
reset_tcc

step "Launching"
launch_bundle

print_next_steps
