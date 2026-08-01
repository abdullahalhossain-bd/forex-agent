# ════════════════════════════════════════════════════════════════
# Forex AI — Decision Layer Diagnostic (PowerShell Wrapper)
# ════════════════════════════════════════════════════════════════
# Tests every decision layer in the pipeline and reports status.
#
# Usage:
#   .\scripts\check_layers.ps1                    # default (XAUUSD)
#   .\scripts\check_layers.ps1 -Pair EURUSD       # specific pair
#   .\scripts\check_layers.ps1 -Verbose           # show full errors
#   .\scripts\check_layers.ps1 -Quick             # skip slow tests
# ════════════════════════════════════════════════════════════════

param(
    [string]$Pair = "XAUUSD",
    [switch]$Verbose,
    [switch]$Quick
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Forex AI — Decision Layer Diagnostic" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Project : $ProjectRoot"
Write-Host "  Pair    : $Pair"
Write-Host "  Mode    : $(if ($Quick) {'Quick'} else {'Full'})"
Write-Host ""

# ── Step 1: Check Python ──────────────────────────────────────
Write-Host "[1/3] Checking Python..." -ForegroundColor Yellow
try {
    $pyVer = python --version 2>&1
    Write-Host "  OK: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# ── Step 2: Check script exists ───────────────────────────────
Write-Host ""
Write-Host "[2/3] Checking diagnostic script..." -ForegroundColor Yellow
$scriptPath = Join-Path $ProjectRoot "scripts\diagnose_layers.py"
if (-not (Test-Path $scriptPath)) {
    Write-Host "  FAIL: Script not found: $scriptPath" -ForegroundColor Red
    Write-Host "  Make sure you extracted the full package." -ForegroundColor Yellow
    exit 1
}
Write-Host "  OK: $scriptPath" -ForegroundColor Green

# ── Step 3: Run diagnostic ────────────────────────────────────
Write-Host ""
Write-Host "[3/3] Running diagnostic..." -ForegroundColor Yellow
Write-Host ""

$args = @("scripts\diagnose_layers.py", "--pair", $Pair)
if ($Verbose) { $args += "--verbose" }

python @args
$exitCode = $LASTEXITCODE

# ── Summary ───────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "  RESULT: ALL LAYERS OK" -ForegroundColor Green
    Write-Host "  Next: python main.py" -ForegroundColor Green
} else {
    Write-Host "  RESULT: SOME LAYERS NEED ATTENTION" -ForegroundColor Yellow
    Write-Host "  Fix the failures above before running the bot." -ForegroundColor Yellow
}
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

exit $exitCode
