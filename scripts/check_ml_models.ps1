# ML Model Diagnostic & Repair Script (PowerShell)
# Usage:
#   .\scripts\check_ml_models.ps1           # diagnose only
#   .\scripts\check_ml_models.ps1 -Fix      # diagnose + auto-fix paths
#   .\scripts\check_ml_models.ps1 -Train    # train new models

param(
    [switch]$Fix,
    [switch]$Train
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ModelsDir = Join-Path $ProjectRoot "memory\ml_models"
$RegistryPath = Join-Path $ModelsDir "_registry.json"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ML Model Diagnostic — Forex AI Trading System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Project Root : $ProjectRoot"
Write-Host "  Models Dir   : $ModelsDir"
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  OK: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: Python not found" -ForegroundColor Red
    exit 1
}

# Check models directory
Write-Host ""
Write-Host "[2/5] Checking models directory..." -ForegroundColor Yellow
if (-not (Test-Path $ModelsDir)) {
    Write-Host "  FAIL: Models directory not found: $ModelsDir" -ForegroundColor Red
    exit 1
}

$modelFiles = Get-ChildItem -Path $ModelsDir -Filter "*.pkl" -Recurse -ErrorAction SilentlyContinue
Write-Host "  Found $($modelFiles.Count) .pkl model files"

if ($modelFiles.Count -eq 0) {
    Write-Host "  WARNING: No model files — ML ensemble is disabled" -ForegroundColor Red
} else {
    $modelFiles | ForEach-Object {
        $rel = $_.FullName.Replace($ProjectRoot + "\", "")
        Write-Host "    $rel ($('{0:N1}' -f ($_.Length/1KB)) KB)"
    }
}

# Check registry
Write-Host ""
Write-Host "[3/5] Checking registry..." -ForegroundColor Yellow
if (Test-Path $RegistryPath) {
    $registry = Get-Content $RegistryPath -Raw | ConvertFrom-Json
    $modelCount = ($registry.models.PSObject.Properties | Measure-Object).Count
    Write-Host "  Registry has $modelCount model entries" -ForegroundColor Green

    $broken = 0
    foreach ($prop in $registry.models.PSObject.Properties) {
        $model = $prop.Value
        $latest = $model.latest
        $version = $model.versions | Where-Object { $_.version -eq $latest } | Select-Object -First 1
        if ($version) {
            $modelPath = $version.model_path
            $exists = Test-Path $modelPath
            $status = if ($exists) { "OK" } else { "BROKEN" }
            $color = if ($exists) { "Green" } else { "Red" }
            Write-Host "    [$status] $($prop.Name)" -ForegroundColor $color
            if (-not $exists) { $broken++ }
        }
    }
    if ($broken -gt 0 -and $Fix) {
        Write-Host "  Note: paths auto-heal on load (model_store.py fix)" -ForegroundColor Yellow
    }
}

# Check ML dependencies
Write-Host ""
Write-Host "[4/5] Checking ML dependencies..." -ForegroundColor Yellow
$mlDeps = @("xgboost", "lightgbm", "catboost", "sklearn", "pandas", "numpy", "joblib")
foreach ($dep in $mlDeps) {
    $result = python -c "import $dep; print($dep.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK]   $dep $result" -ForegroundColor Green
    } else {
        Write-Host "  [MISS] $dep — pip install $dep" -ForegroundColor Red
    }
}

# Check NumPy/Numba
Write-Host ""
Write-Host "[5/5] NumPy/Numba compatibility..." -ForegroundColor Yellow
$numpyVer = python -c "import numpy; print(numpy.__version__)" 2>$null
if ($numpyVer) {
    $parts = $numpyVer.Split('.')
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    Write-Host "  NumPy: $numpyVer" -ForegroundColor Green
    if ($major -gt 2 -or ($major -eq 2 -and $minor -gt 2)) {
        Write-Host "  WARNING: NumPy too new for Numba (needs <= 2.2)" -ForegroundColor Red
        Write-Host "  Fix: pip install 'numpy==2.2.6' --upgrade" -ForegroundColor Yellow
    } else {
        Write-Host "  NumPy compatible with Numba" -ForegroundColor Green
    }
}

# Train
Write-Host ""
if ($Train) {
    Write-Host "Training models..." -ForegroundColor Yellow
    python (Join-Path $ProjectRoot "scripts\train_models.py")
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Done. Next: python main.py" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
