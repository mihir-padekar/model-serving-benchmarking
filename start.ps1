# PowerShell quick launcher for Model Serving Benchmark
$ErrorActionPreference = "Stop"

Write-Host "⚡ Starting Model Serving Benchmark Fleet..." -ForegroundColor Cyan

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

& $Python (Join-Path $PSScriptRoot "run.py")

