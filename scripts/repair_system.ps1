# Full System Repair Script (PowerShell)
# Usage: .\scripts\repair_system.ps1
#        .\scripts\repair_system.ps1 -DryRun

param([switch]$DryRun)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Forex AI — Full System Repair" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
if ($DryRun) { Write-Host "  MODE: DRY RUN" -ForegroundColor Yellow }
Write-Host ""

# Fix 1: NumPy/Numba
Write-Host "[Fix 1/4] NumPy/Numba compatibility..." -ForegroundColor Yellow
$numpyVer = python -c "import numpy; print(numpy.__version__)" 2>$null
if ($numpyVer) {
    $parts = $numpyVer.Split('.')
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -gt 2 -or ($major -eq 2 -and $minor -gt 2)) {
        Write-Host "  NumPy $numpyVer too new — downgrading to 2.2.6..." -ForegroundColor Yellow
        if (-not $DryRun) {
            pip install "numpy==2.2.6" --upgrade 2>&1 | Out-Null
            pip install --upgrade numba 2>&1 | Out-Null
            $newVer = python -c "import numpy; print(numpy.__version__)" 2>$null
            Write-Host "  NumPy now: $newVer" -ForegroundColor Green
        }
    } else {
        Write-Host "  NumPy $numpyVer compatible" -ForegroundColor Green
    }
}

# Fix 2: ML dependencies
Write-Host ""
Write-Host "[Fix 2/4] ML dependencies..." -ForegroundColor Yellow
$deps = @("xgboost", "lightgbm", "catboost", "scikit-learn", "pandas", "joblib", "MetaTrader5")
$missing = @()
foreach ($d in $deps) {
    $modName = if ($d -eq "scikit-learn") { "sklearn" } else { $d }
    $ver = python -c "import $modName; print($modName.__version__)" 2>$null
    if ($ver) {
        Write-Host "  [OK]   $d $ver" -ForegroundColor Green
    } else {
        Write-Host "  [MISS] $d" -ForegroundColor Red
        $missing += $d
    }
}
if ($missing.Count -gt 0 -and -not $DryRun) {
    Write-Host "  Installing: $($missing -join ', ')" -ForegroundColor Yellow
    pip install $missing 2>&1 | Out-Null
}

# Fix 3: .env file
Write-Host ""
Write-Host "[Fix 3/4] .env file..." -ForegroundColor Yellow
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    $envExample = Join-Path $ProjectRoot ".env.example"
    if (Test-Path $envExample) {
        if (-not $DryRun) {
            Copy-Item $envExample $envFile
            Write-Host "  Created .env from .env.example — EDIT IT!" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  .env exists" -ForegroundColor Green
    $envContent = Get-Content $envFile -Raw
    if ($envContent -notmatch "MT5_FALLBACK_TO_SIMULATION") {
        if (-not $DryRun) {
            Add-Content $envFile "`n# Auto-added by repair script`nMT5_FALLBACK_TO_SIMULATION=true`n"
        }
        Write-Host "  Added MT5_FALLBACK_TO_SIMULATION=true" -ForegroundColor Green
    }
}

# Fix 4: Boot smoke test
Write-Host ""
Write-Host "[Fix 4/4] Boot smoke test..." -ForegroundColor Yellow
$bootTest = python -c "
import sys
sys.path.insert(0, '.')
try:
    from config import EXECUTION_MODE, SIMULATION_MODE, MT5_FALLBACK_TO_SIMULATION
    print(f'  EXECUTION_MODE={EXECUTION_MODE}')
    print(f'  SIMULATION_MODE={SIMULATION_MODE}')
    print(f'  MT5_FALLBACK_TO_SIMULATION={MT5_FALLBACK_TO_SIMULATION}')
    from execution.execution_router import ExecutionRouter
    print('  ExecutionRouter import: OK')
    from ml.model_predictor import ModelPredictor
    p = ModelPredictor()
    print(f'  ModelPredictor.is_ready()={p.is_ready()}')
    print('  ALL IMPORTS OK')
except Exception as e:
    print(f'  IMPORT FAIL: {e}')
    sys.exit(1)
" 2>&1
Write-Host $bootTest

if ($bootTest -match "ALL IMPORTS OK") {
    Write-Host "  Boot test PASSED" -ForegroundColor Green
} else {
    Write-Host "  Boot test FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Repair Complete. Next: python main.py" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
