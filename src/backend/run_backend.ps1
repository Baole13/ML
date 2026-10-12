# Run the Animal Visual Search backend locally.
#
# This creates/reuses a Python venv (reusing system torch/faiss to avoid a
# multi-GB download), generates synthetic serving artifacts if missing, and
# starts the FastAPI server on http://127.0.0.1:8000.
#
# Usage (from src/backend):
#   .\run_backend.ps1              # mock encoder + synthetic artifacts (offline demo)
#   .\run_backend.ps1 -Encoder dinov2 -ArtifactsDir .\artifacts   # real model
#
# Stop with Ctrl+C.

param(
    [string]$Encoder = "mock",
    [string]$ArtifactsDir = "",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$be = $PSScriptRoot
$venvPython = Join-Path $be ".venv\Scripts\python.exe"

# 1) Create the venv (reusing system site-packages for torch/faiss) if needed.
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv (reusing system torch/faiss)..."
    python -m venv (Join-Path $be ".venv") --system-site-packages
}

# 2) Resolve artifacts dir (default: .\artifacts next to this script).
if ([string]::IsNullOrWhiteSpace($ArtifactsDir)) {
    $ArtifactsDir = Join-Path $be "artifacts"
}

# 3) Generate synthetic artifacts if the package is missing.
$manifest = Join-Path $ArtifactsDir "manifest.json"
if (-not (Test-Path $manifest)) {
    Write-Host "No artifacts found -> generating synthetic package..."
    $env:PYTHONPATH = $be
    & $venvPython (Join-Path $be "scripts\make_synthetic_artifacts.py") $ArtifactsDir
}

# 4) Start the server with the correct environment.
$env:PYTHONPATH = $be
$env:ENCODER = $Encoder
$env:ARTIFACTS_DIR = $ArtifactsDir

Write-Host "Starting backend: encoder=$Encoder artifacts=$ArtifactsDir port=$Port"
Write-Host "Docs:  http://127.0.0.1:$Port/docs"
Write-Host "Health: http://127.0.0.1:$Port/health"
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
