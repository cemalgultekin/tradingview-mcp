<#
.SYNOPSIS
    Set up this repo for development on a fresh Windows machine.

.DESCRIPTION
    Installs uv (via winget) if missing, creates the virtualenv from uv.lock,
    and runs the test suite. Safe to re-run: every step is idempotent.

    uv provisions its own Python 3.10-3.13 interpreter, so you do NOT need a
    system Python. A pre-existing Python 3.9 on PATH is fine and is ignored.

.PARAMETER SkipTests
    Install dependencies but do not run the suite.

.PARAMETER Stress
    Also run the opt-in network stress tests (tests/stress). These hit live
    Yahoo/TradingView endpoints, take much longer, and can fail for reasons
    that have nothing to do with your changes.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -SkipTests
    .\scripts\setup.ps1 -Stress
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$Stress
)

$ErrorActionPreference = 'Stop'

function Write-Step { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "    $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "    $m" -ForegroundColor Yellow }

# Run from the repo root regardless of where the script was invoked from.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Step "Repo root: $repoRoot"

# ── 1. uv ─────────────────────────────────────────────────────────────────────
# winget modifies PATH for *new* shells only, so after a fresh install we have
# to reach the shim directory directly for the rest of this session.
$wingetLinks = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
if (Test-Path $wingetLinks -PathType Container) {
    $env:Path = "$wingetLinks;$env:Path"
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Step "uv already installed"
    Write-Ok (uv --version)
} else {
    Write-Step "Installing uv via winget"
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget not found. Install uv manually from https://docs.astral.sh/uv/getting-started/installation/ then re-run this script."
    }
    winget install --id astral-sh.uv -e `
        --accept-source-agreements --accept-package-agreements --disable-interactivity
    if (Test-Path $wingetLinks -PathType Container) {
        $env:Path = "$wingetLinks;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installed but is still not on PATH. Open a new terminal and re-run this script."
    }
    Write-Ok (uv --version)
    Write-Warn "PATH was updated for future shells; this session uses the shim directly."
}

# ── 2. Dependencies ───────────────────────────────────────────────────────────
# `uv sync` downloads a matching interpreter if needed, creates .venv, and
# installs exactly what uv.lock pins (prod + dev groups).
Write-Step "Syncing dependencies from uv.lock"
uv sync --dev
if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit $LASTEXITCODE)" }
Write-Ok "Interpreter: $(uv run python --version)"

# ── 3. Import check ───────────────────────────────────────────────────────────
# Catches the failure mode plain `pytest` can mask: the server module itself
# not importing (a missing service module, a broken tool registration).
Write-Step "Verifying the server imports and registers its tools"
uv run python scripts/verify_server.py
if ($LASTEXITCODE -ne 0) { throw "server import check failed (exit $LASTEXITCODE)" }

# ── 4. Tests ──────────────────────────────────────────────────────────────────
if ($SkipTests) {
    Write-Step "Skipping tests (-SkipTests)"
} else {
    Write-Step "Running the unit suite"
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed (exit $LASTEXITCODE)" }

    if ($Stress) {
        Write-Step "Running the network stress suite (-Stress)"
        Write-Warn "These call live endpoints; failures here may be upstream, not yours."
        uv run pytest -m stress -q
        if ($LASTEXITCODE -ne 0) { Write-Warn "stress suite failed (exit $LASTEXITCODE) - see note above" }
    }
}

# ── 5. Optional config ────────────────────────────────────────────────────────
Write-Step "Optional configuration"
if (-not $env:MARKETAUX_API_TOKEN) {
    Write-Warn "MARKETAUX_API_TOKEN is not set."
    Write-Warn "  Without it, news/sentiment tools return no articles:"
    Write-Warn "  financial_news, market_sentiment, combined_analysis, news_price_lag_detector."
    Write-Warn "  Everything else works unaffected. Get a key at https://www.marketaux.com/"
} else {
    Write-Ok "MARKETAUX_API_TOKEN is set."
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "  Run the server:  uv run tradingview-mcp" -ForegroundColor Gray
Write-Host "  Run the tests:   uv run pytest -q" -ForegroundColor Gray
Write-Host "  Python REPL:     uv run python" -ForegroundColor Gray
