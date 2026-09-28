#!/usr/bin/env bash
#
# Set up this repo for development on a fresh Linux/macOS machine.
#
# Installs uv if missing, creates the virtualenv from uv.lock, and runs the
# test suite. Safe to re-run: every step is idempotent.
#
# uv provisions its own Python 3.10-3.13 interpreter, so you do NOT need a
# system Python of any particular version.
#
# Usage:
#   ./scripts/setup.sh              # install + unit tests
#   ./scripts/setup.sh --skip-tests # install only
#   ./scripts/setup.sh --stress     # also run the live-network stress suite
#
set -euo pipefail

SKIP_TESTS=0
RUN_STRESS=0
for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    --stress)     RUN_STRESS=1 ;;
    -h|--help)    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  C='\033[36m'; G='\033[32m'; Y='\033[33m'; R='\033[0m'
else
  C=''; G=''; Y=''; R=''
fi
step() { printf "\n${C}==> %s${R}\n" "$1"; }
ok()   { printf "    ${G}%s${R}\n" "$1"; }
warn() { printf "    ${Y}%s${R}\n" "$1"; }

# Run from the repo root regardless of where this was invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
step "Repo root: $REPO_ROOT"

# ── 1. uv ─────────────────────────────────────────────────────────────────────
# The installer drops uv in ~/.local/bin, which is not always on PATH in a
# non-login shell, so add it before checking.
export PATH="$HOME/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
  step "uv already installed"
  ok "$(uv --version)"
else
  step "Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Need curl or wget to install uv." >&2
    echo "Install it manually: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    echo "uv installed but not on PATH. Open a new shell and re-run." >&2; exit 1; }
  ok "$(uv --version)"
  warn "Add \$HOME/.local/bin to your PATH in your shell profile to keep uv available."
fi

# ── 2. Dependencies ───────────────────────────────────────────────────────────
# `uv sync` downloads a matching interpreter if needed, creates .venv, and
# installs exactly what uv.lock pins (prod + dev groups).
step "Syncing dependencies from uv.lock"
uv sync --dev
ok "Interpreter: $(uv run python --version)"

# ── 3. Import check ───────────────────────────────────────────────────────────
# Catches the failure mode plain pytest can mask: the server module itself not
# importing (a missing service module, a broken tool registration).
step "Verifying the server imports and registers its tools"
uv run python scripts/verify_server.py

# ── 4. Tests ──────────────────────────────────────────────────────────────────
if [ "$SKIP_TESTS" -eq 1 ]; then
  step "Skipping tests (--skip-tests)"
else
  step "Running the unit suite"
  uv run pytest -q

  if [ "$RUN_STRESS" -eq 1 ]; then
    step "Running the network stress suite (--stress)"
    warn "These call live endpoints; failures here may be upstream, not yours."
    uv run pytest -m stress -q || warn "stress suite failed - see note above"
  fi
fi

# ── 5. Optional config ────────────────────────────────────────────────────────
step "Optional configuration"
if [ -z "${MARKETAUX_API_TOKEN:-}" ]; then
  warn "MARKETAUX_API_TOKEN is not set."
  warn "  Without it, news/sentiment tools return no articles:"
  warn "  financial_news, market_sentiment, combined_analysis, news_price_lag_detector."
  warn "  Everything else works unaffected. Get a key at https://www.marketaux.com/"
else
  ok "MARKETAUX_API_TOKEN is set."
fi

printf "\n${G}Setup complete.${R}\n"
echo "  Run the server:  uv run tradingview-mcp"
echo "  Run the tests:   uv run pytest -q"
echo "  Python REPL:     uv run python"
