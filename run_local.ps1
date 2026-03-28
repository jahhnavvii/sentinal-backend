# Sentinel AI — Local Backend Starter
# Run this in a PowerShell window to start the FastAPI server.

Write-Host "[Sentinel] Starting local FastAPI backend..." -ForegroundColor Cyan

# Ensure you have the virtual environment activated if you use one
# If you don't use a venv, just run uvicorn directly
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "[Sentinel] Activating virtual environment..." -ForegroundColor Gray
    . .\venv\Scripts\Activate.ps1
}

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
